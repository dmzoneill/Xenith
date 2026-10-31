"""Unit tests for voice input and the modules extracted from it"""

import importlib
import importlib.util
import types
import unittest
from unittest import mock

audio_deps = importlib.import_module("src.audio.audio_deps")
audio_devices = importlib.import_module("src.audio.audio_devices")
voice_state = importlib.import_module("src.audio.voice_state")
voice_stt = importlib.import_module("src.audio.voice_stt")
voice_input = importlib.import_module("src.audio.voice_input")

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
np = importlib.import_module("numpy") if HAVE_NUMPY else None
NEEDS_NUMPY = unittest.skipUnless(HAVE_NUMPY, "numpy not installed")

DEVICES = [
    {"name": "out", "max_input_channels": 0, "default_samplerate": 48000.0},
    {"name": "mic", "max_input_channels": 2, "default_samplerate": 44100.0},
]


class AudioDevicesTest(unittest.TestCase):
    """audio_devices"""

    def setUp(self):
        self.sd = mock.MagicMock()
        self.sd.query_devices.return_value = DEVICES
        self.sd.default.device = (1, 0)
        patches = [
            mock.patch.object(audio_devices, "sd", self.sd),
            mock.patch.object(audio_devices, "AUDIO_AVAILABLE", True),
            mock.patch("builtins.print"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_list_only_inputs(self):
        """Output-only devices are skipped"""
        self.assertEqual(
            audio_devices.list_input_devices(), [{"index": 1, "name": "mic", "channels": 2, "sample_rate": 44100.0}]
        )

    def test_list_error(self):
        """Device query errors give an empty list"""
        self.sd.query_devices.side_effect = OSError("boom")
        self.assertEqual(audio_devices.list_input_devices(), [])
        self.assertIsNone(audio_devices.print_input_devices())

    def test_select_device(self):
        """Interactive selection handles bad input, defaults, a valid index and quit"""
        with mock.patch("builtins.input", side_effect=["x", "5", "0"]):
            self.assertEqual(audio_devices.select_device_interactive(), 1)
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(audio_devices.select_device_interactive(), 1)
        with mock.patch("builtins.input", return_value="q"):
            self.assertIsNone(audio_devices.select_device_interactive())
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertIsNone(audio_devices.select_device_interactive())

    def test_voice_input_static_wrappers(self):
        """VoiceInput keeps its static device helpers"""
        self.assertEqual(len(voice_input.VoiceInput.list_input_devices()), 1)


class TranscriberTest(unittest.TestCase):
    """voice_stt.Transcriber"""

    def setUp(self):
        patch = mock.patch("builtins.print")
        patch.start()
        self.addCleanup(patch.stop)

    def test_backend_preferred(self):
        """The new backend is used before legacy Whisper"""
        stt = voice_stt.Transcriber()
        self.assertFalse(stt.available)
        self.assertIsNone(stt.transcribe([0.0]))
        stt.whisper_model = mock.MagicMock()
        stt.whisper_model.transcribe.return_value = {"text": " legacy "}
        self.assertEqual(stt.transcribe([0.0]), "legacy")
        stt.backend = mock.MagicMock()
        stt.backend.transcribe.return_value = types.SimpleNamespace(text="hi there")
        self.assertEqual(stt.transcribe([0.0]), "hi there")

    def test_load_backend(self):
        """Loads through stt_backends.get_stt_backend"""
        stt = voice_stt.Transcriber("openvino", "npu", "tiny")
        backend = mock.MagicMock()
        backend.load_model.return_value = True
        module = types.SimpleNamespace(get_stt_backend=mock.MagicMock(return_value=backend))
        with mock.patch.object(voice_stt.importlib, "import_module", return_value=module):
            self.assertTrue(stt.load())
        module.get_stt_backend.assert_called_once_with(backend="openvino", model_name="tiny", device="npu")

    def test_load_backend_import_error_falls_back(self):
        """Import errors fall back to legacy Whisper, which may also be missing"""
        stt = voice_stt.Transcriber()
        with mock.patch.object(voice_stt.importlib, "import_module", side_effect=ImportError("nope")):
            self.assertFalse(stt.load_backend())

    def test_unload_and_warning(self):
        """Unload swallows backend errors; the missing-engine warning prints once"""
        stt = voice_stt.Transcriber()
        stt.backend = mock.MagicMock()
        stt.backend.unload_model.side_effect = RuntimeError
        stt.unload()
        self.assertIsNone(stt.backend)
        stt.print_missing_warning()
        self.assertTrue(stt.warning_printed)


class VoiceStateTest(unittest.TestCase):
    """voice_state"""

    def test_wake_buffer_is_bounded(self):
        """Only the newest samples are kept"""
        buffers = voice_state.AudioBuffers(max_wake_samples=3)
        buffers.add_wake_audio([1, 2, 4, 5])
        self.assertEqual(buffers.wake, [2, 4, 5])
        buffers.command = [1]
        buffers.clear()
        self.assertEqual((buffers.wake, buffers.command), ([], []))

    def test_callbacks(self):
        """State notifications are optional"""
        callbacks = voice_state.VoiceCallbacks()
        callbacks.state("idle")
        callbacks.on_state_change = mock.MagicMock()
        callbacks.state("listening")
        callbacks.on_state_change.assert_called_once_with("listening")

    @NEEDS_NUMPY
    @mock.patch.object(voice_state, "np", np)
    def test_recent_wake_audio(self):
        """Too short or too quiet audio is skipped; loud audio is normalized"""
        self.assertIsNone(voice_state.recent_wake_audio([0.5] * 5, 10))
        self.assertIsNone(voice_state.recent_wake_audio([0.0] * 20, 10))
        audio = voice_state.recent_wake_audio([0.25] * 30, 10)
        self.assertEqual(len(audio), 20)
        self.assertAlmostEqual(float(audio.max()), 1.0)


@NEEDS_NUMPY
class VoiceInputTest(unittest.TestCase):
    """VoiceInput behaviour with mocked audio, STT, threads and time"""

    def setUp(self):
        self.now = 100.0
        self.sd = mock.MagicMock()
        self.stt = mock.MagicMock()
        self.stt.available = True
        self.stt.whisper_model = None
        patches = [
            mock.patch.object(voice_input.time_module, "time", side_effect=lambda: self.now),
            mock.patch.object(voice_input.time_module, "sleep"),
            mock.patch.object(voice_input.threading, "Thread"),
            mock.patch.object(voice_input, "sd", self.sd),
            mock.patch.object(voice_input, "np", np),
            mock.patch.object(voice_state, "np", np),
            mock.patch.object(voice_input, "AUDIO_AVAILABLE", True),
            mock.patch.object(voice_input, "Transcriber", return_value=self.stt),
            mock.patch("builtins.print"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.voice = voice_input.VoiceInput(sample_rate=10, chunk_size=5, wake_word="Hi", device=2)
        self.states = []
        self.voice.on_state_change = self.states.append
        self.transcripts = []
        self.voice.on_transcript = self.transcripts.append

    def _thread_target(self, name):
        for call in voice_input.threading.Thread.call_args_list:
            target = call.kwargs.get("target")
            if target is not None and target.__name__ == name:
                return target
        raise AssertionError(f"no thread started for {name}")

    def _feed(self, level, chunks=1):
        callback = self.sd.InputStream.call_args.kwargs["callback"]
        for _ in range(chunks):
            callback(np.full((5, 1), level, dtype=np.float32), 5, None, None)

    def test_callbacks_are_properties(self):
        """The public callback attributes still work"""
        self.assertEqual(self.voice.on_transcript, self.transcripts.append)
        self.assertEqual(self.voice.on_state_change, self.states.append)
        self.voice.on_wake_word_detected = None
        self.assertIsNone(self.voice.on_wake_word_detected)

    def test_wake_word_command_round_trip(self):
        """Wake word -> listening -> command captured after silence -> transcript"""
        self.stt.transcribe.side_effect = ["Hi Xenith", "turn on the lights"]
        self.voice.start_listening()
        self.stt.load.assert_called_once()
        self.assertEqual(self.sd.InputStream.call_args.kwargs["device"], 2)

        self._feed(0.5, chunks=3)  # 1.5s of wake word audio
        self._thread_target("_wake_word_detection_loop")()
        self.assertEqual(self.states, ["idle", "listening"])

        self.now += 2.0  # past the post-wake-word ignore window
        self._feed(0.5, chunks=2)  # 1s of command audio
        self.now += 1.0
        self._feed(0.0)  # silence triggers processing
        self.assertEqual(self.states[-1], "processing")
        self._thread_target("_process_worker")()
        self.assertEqual(self.transcripts, ["turn on the lights"])
        self.assertEqual(self.states[-1], "responding")

    def test_short_command_handling(self):
        """Too-short audio keeps listening within 5s of the wake word, then returns to idle"""
        self.stt.transcribe.return_value = "hi"
        self.voice.start_listening()
        self._feed(0.5, chunks=3)
        self._thread_target("_wake_word_detection_loop")()
        callback = self.sd.InputStream.call_args.kwargs["callback"]
        blip = np.full((2, 1), 0.5, dtype=np.float32)  # 0.2s of audio

        self.now += 2.0
        callback(blip, 2, None, None)
        self.now += 1.0
        self._feed(0.0)
        self.assertEqual(self.states, ["idle", "listening"])  # still waiting for more

        self.now += 3.0
        callback(blip, 2, None, None)
        self.now += 1.0
        self._feed(0.0)
        self.assertEqual(self.states, ["idle", "listening", "idle"])

    def test_mute_ignores_audio(self):
        """Muted input is dropped"""
        self.voice.start_listening()
        self.voice.mute()
        self._feed(0.5, chunks=3)
        self.stt.transcribe.assert_not_called()

    def test_no_backend_transcript(self):
        """Without an engine the command reports that no backend is available"""
        self.stt.available = False
        self.voice.start_listening()
        self.voice.cleanup()
        self.stt.unload.assert_called_once()
        self.assertEqual(self.states[-1], "idle")
