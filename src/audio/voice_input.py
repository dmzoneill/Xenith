"""Voice input handler with wake word detection

Supports multiple STT backends:
- OpenAI Whisper (CUDA/CPU) - Original, high accuracy
- OpenVINO (NPU/GPU/CPU) - Ultra-low power on Intel hardware
"""

import threading
import time as time_module
import traceback
from collections.abc import Callable
from typing import Any

from . import audio_devices
from .audio_deps import AUDIO_AVAILABLE, AUDIO_ERRORS, np, sd
from .voice_state import (
    AudioBuffers,
    AudioConfig,
    ListenState,
    VoiceCallbacks,
    WakeWordMonitor,
    normalize,
    recent_wake_audio,
)
from .voice_stt import STT_ERRORS, Transcriber

WAKE_WORD_BUFFER_SECONDS = 3.0  # Keep 3 seconds of audio for wake word analysis
TRANSCRIPTION_ERROR_TEXT = "Voice input detected (transcription error)"
NO_BACKEND_TEXT = "Voice input detected (no STT backend available)"


class VoiceInput:
    """Handles voice input with wake word detection

    Supports configurable STT backends for transcription:
    - "whisper": OpenAI Whisper with PyTorch (CUDA/CPU)
    - "openvino": OpenVINO Whisper (NPU/GPU/CPU) - ultra-low power
    - "auto": Automatically select best available backend
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, wake_word: str = "hi", **options):
        """Initialize VoiceInput

        Args:
            sample_rate: Audio sample rate (default 16kHz for Whisper)
            chunk_size: Audio chunk size for processing
            wake_word: Wake word to listen for
            **options: ``device`` (audio input device index), ``stt_backend`` ("whisper", "openvino"
                or "auto"), ``stt_device`` ("auto", "cpu", "cuda", "npu", "gpu") and ``stt_model``
                ("tiny", "base", "small", "medium", "large")
        """
        self._config = AudioConfig(sample_rate, chunk_size, wake_word.lower(), options.get("device"))
        self._state = ListenState()
        self._buffers = AudioBuffers(int(sample_rate * WAKE_WORD_BUFFER_SECONDS))
        self._wake = WakeWordMonitor()
        self._callbacks = VoiceCallbacks()
        self._stt = Transcriber(
            options.get("stt_backend", "auto"), options.get("stt_device", "auto"), options.get("stt_model", "base")
        )
        self._stream: Any = None

    @property
    def on_state_change(self) -> Callable[[str], None] | None:
        """Callback invoked with the new state name"""
        return self._callbacks.on_state_change

    @on_state_change.setter
    def on_state_change(self, callback: Callable[[str], None] | None) -> None:
        self._callbacks.on_state_change = callback

    @property
    def on_transcript(self) -> Callable[[str], None] | None:
        """Callback invoked with each command transcript"""
        return self._callbacks.on_transcript

    @on_transcript.setter
    def on_transcript(self, callback: Callable[[str], None] | None) -> None:
        self._callbacks.on_transcript = callback

    @property
    def on_wake_word_detected(self) -> Callable[[], None] | None:
        """Callback invoked when the wake word is heard"""
        return self._callbacks.on_wake_word_detected

    @on_wake_word_detected.setter
    def on_wake_word_detected(self, callback: Callable[[], None] | None) -> None:
        self._callbacks.on_wake_word_detected = callback

    @staticmethod
    def list_input_devices():
        """List all available input devices"""
        return audio_devices.list_input_devices()

    @staticmethod
    def print_input_devices():
        """Print all available input devices in a formatted way"""
        return audio_devices.print_input_devices()

    @staticmethod
    def select_device_interactive() -> int | None:
        """Interactively select an input device"""
        return audio_devices.select_device_interactive()

    def _ensure_wake_thread(self, message: str | None = None) -> None:
        """Start the wake word detection thread if it is not running"""
        thread = self._wake.thread
        if thread is not None and thread.is_alive():
            return
        if message:
            print(message)
        thread = threading.Thread(target=self._wake_word_detection_loop, daemon=True)
        self._wake.thread = thread
        thread.start()

    def start_listening(self):
        """Start listening for wake word (always-on mode)"""
        if not AUDIO_AVAILABLE:
            print("Audio libraries not available. Cannot start voice input.")
            return
        if self._stream is not None:
            return  # Already listening

        self._state.wake_word = True
        self._state.command = False
        self._buffers.clear()
        self._stt.load()
        # Start in idle state - waiting for wake word
        self._callbacks.state("idle")

        try:
            self._open_stream()
            self._ensure_wake_thread()
            print("[INIT] ✓ Wake word detection thread started")
        except AUDIO_ERRORS as e:
            print(f"Error starting audio stream: {e}")
            self._state.wake_word = False
            self._callbacks.state("idle")

    def _open_stream(self) -> None:
        """Open and start the audio stream used for wake word detection"""
        config = self._config
        stream_kwargs = {
            "samplerate": config.sample_rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": config.chunk_size,
            "callback": self._audio_callback,
        }
        if config.device is not None:
            stream_kwargs["device"] = config.device
            try:
                device_info = sd.query_devices(config.device)
                print(f"Using input device: [{config.device}] {device_info['name']}")
            except AUDIO_ERRORS as e:
                print(f"Warning: Could not get device info: {e}")

        self._stream = sd.InputStream(**stream_kwargs)
        self._stream.start()
        print("[INIT] ✓ Audio stream started")
        print(f"[INIT] Listening for wake word: '{config.wake_word}'")
        print(f"[INIT] Sample rate: {config.sample_rate} Hz, Chunk size: {config.chunk_size}")
        print(f"[INIT] Whisper model: {'Loaded' if self._stt.whisper_model else 'Not available'}")
        print("[INIT] Audio callback active - you should see [AUDIO] messages every 2 seconds")

    def stop_listening(self):
        """Stop listening for voice input"""
        self._state.wake_word = False
        self._state.command = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except AUDIO_ERRORS:  # nosec B110
                pass
            self._stream = None
        self._callbacks.state("idle")

    def mute(self):
        """Mute microphone input (use during TTS playback to prevent feedback)"""
        print("[VOICE] Muting microphone...")
        self._state.muted = True
        self._state.wake_word = False
        self._state.command = False
        # Stop the audio stream to prevent any processing
        if self._stream:
            try:
                self._stream.stop()
            except AUDIO_ERRORS:
                pass
        self._buffers.clear()

    def unmute(self):
        """Unmute microphone and resume wake word detection"""
        print("[VOICE] Unmuting microphone, resuming wake word detection...")
        self._state.muted = False
        # Clear buffers again to remove any residual audio
        self._buffers.clear()

        # Restart audio stream if it was stopped
        if self._stream and not self._stream.active:
            try:
                self._stream.start()
            except AUDIO_ERRORS as e:
                print(f"[VOICE] Error restarting stream: {e}")
                self._create_audio_stream()

        self._state.wake_word = True
        self._state.command = False
        self._ensure_wake_thread()
        self._callbacks.state("idle")

    def _create_audio_stream(self):
        """Create or recreate the audio input stream"""
        try:
            if self._stream:
                try:
                    self._stream.close()
                except AUDIO_ERRORS:
                    pass
            self._stream = sd.InputStream(
                device=self._config.device,
                channels=1,
                samplerate=self._config.sample_rate,
                blocksize=self._config.chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
        except AUDIO_ERRORS as e:
            print(f"[VOICE] Error creating audio stream: {e}")

    def _audio_callback(self, indata, _frames, _time_info, status):
        """Callback for audio input"""
        if status:
            print(f"[AUDIO] Callback status: {status}")
        if not AUDIO_AVAILABLE or self._state.muted:
            return

        audio_data = indata[:, 0]
        energy = np.abs(audio_data).mean()
        current_time = time_module.time()
        self._debug_audio_level(energy, current_time)

        # Always maintain wake word buffer (for wake word detection)
        if self._state.wake_word:
            self._buffers.add_wake_audio(audio_data.tolist())

        # If listening for command (after wake word detected)
        if self._state.command:
            self._handle_command_audio(audio_data, energy, current_time)

    def _debug_audio_level(self, energy: float, current_time: float) -> None:
        """Print audio levels every 10 seconds, only if there is significant activity"""
        if self._wake.last_debug_print is None:
            self._wake.last_debug_print = current_time
            self._wake.debug_counter = 0
        self._wake.debug_counter += 1
        if current_time - self._wake.last_debug_print < 10.0:
            return
        if energy > 0.001:
            buffer_info = ""
            if self._state.wake_word:
                buffer_info = f" | Buffer: {len(self._buffers.wake) / self._config.sample_rate:.1f}s"
            if self._state.command:
                buffer_info = f" | Recording: {len(self._buffers.command) / self._config.sample_rate:.1f}s"
            print(f"[AUDIO] Level: {energy:.4f}{buffer_info}")
        self._wake.last_debug_print = current_time
        self._wake.debug_counter = 0

    def _handle_command_audio(self, audio_data, energy: float, current_time: float) -> None:
        """Capture command audio and process it after enough silence"""
        # Ignore audio briefly after wake word detection so the wake word isn't processed as a command
        if current_time < self._state.ignore_audio_until:
            return
        if energy > self._config.energy_threshold:
            self._state.last_voice_time = current_time
            self._buffers.command.extend(audio_data.tolist())
            # Only print when starting to capture
            if len(self._buffers.command) == len(audio_data):
                print("[COMMAND] Listening...")
            return

        silence_duration = current_time - self._state.last_voice_time
        # Prevent multiple rapid processing calls
        if silence_duration <= self._config.silence_duration or not self._buffers.command or self._state.processing:
            return
        duration = len(self._buffers.command) / self._config.sample_rate
        # Only process meaningful audio (at least 0.5 seconds); shorter clips may be noise or wake word echo
        if duration >= 0.5:
            print(f"[COMMAND] Processing {duration:.1f}s of audio...")
            self._process_audio()
        else:
            self._handle_short_command(duration, current_time)

    def _handle_short_command(self, duration: float, current_time: float) -> None:
        """Too-short audio: keep waiting for up to 5 seconds after the wake word, then give up"""
        time_since_listening_start = current_time - self._state.command_start_time
        self._buffers.command = []
        if time_since_listening_start < 5.0:
            # Too early to give up - user might still be speaking
            print(f"[COMMAND] Audio too short ({duration:.1f}s), waiting for more...")
            self._state.last_voice_time = current_time
            return
        waited = f"{time_since_listening_start:.1f}s"
        print(f"[COMMAND] Audio too short ({duration:.1f}s) after {waited}, returning to idle...")
        self._state.command = False
        self._state.wake_word = True
        self._callbacks.state("idle")

    def _wake_word_detection_loop(self):
        """Continuously check for wake word in background thread"""
        print(f"[WAKE WORD] Detection active - listening for '{self._config.wake_word}'")
        check_count = 0
        while self._state.wake_word and self._stream is not None:
            try:
                current_time = time_module.time()
                if current_time - self._wake.last_check >= self._wake.check_interval:
                    self._wake.last_check = current_time
                    check_count += 1
                    self._wake_word_check_step(check_count)
                time_module.sleep(0.25)  # Fast polling (250ms) for quick wake word response
            except (*STT_ERRORS, *AUDIO_ERRORS) as e:
                print(f"[WAKE WORD] Error: {e}")
                traceback.print_exc()
                time_module.sleep(0.25)

    def _wake_word_check_step(self, check_count: int) -> None:
        """One periodic wake word check"""
        sample_rate = self._config.sample_rate
        buffer_duration = len(self._buffers.wake) / sample_rate if sample_rate > 0 else 0
        # Only print status every 10 checks or on first check
        if check_count == 1 or check_count % 10 == 0:
            print(f"[WAKE WORD] Status: {buffer_duration:.1f}s buffer ready")
        if len(self._buffers.wake) > sample_rate:  # At least 1 second of audio
            if self._check_wake_word():
                self._on_wake_word_detected()
        elif check_count == 1:
            print(f"[WAKE WORD] Waiting for audio... (need 1.0s, have {buffer_duration:.1f}s)")

    def _check_wake_word(self) -> bool:
        """Check if wake word is present in buffer"""
        if not AUDIO_AVAILABLE:
            return False
        audio_array = recent_wake_audio(self._buffers.wake, self._config.sample_rate)
        if audio_array is None:
            return False
        if not self._stt.available:
            self._stt.print_missing_warning()
            return False
        error_label = "STT error" if self._stt.backend is not None else "Transcription error"
        try:
            transcript = self._stt.transcribe(audio_array) or ""
        except STT_ERRORS as e:
            print(f"[WAKE WORD] {error_label}: {e}")
            traceback.print_exc()
            return False
        detected = self._config.wake_word in transcript.lower().strip()
        if not transcript.strip():
            print("[WAKE WORD] (silence or unclear)")
        elif detected:
            print(f"[WAKE WORD] ✓ Heard: '{transcript}' ← WAKE WORD DETECTED!")
        else:
            print(f"[WAKE WORD] Heard: '{transcript}'")
        return detected

    def _on_wake_word_detected(self):
        """Handle wake word detection"""
        # Prevent multiple rapid detections (3 second debounce)
        current_time = time_module.time()
        if self._wake.last_detection is not None and current_time - self._wake.last_detection < 3.0:
            return
        self._wake.last_detection = current_time
        # Don't process if we're already listening for a command
        if self._state.command:
            return

        print(f"Wake word '{self._config.wake_word}' detected!")
        self._state.wake_word = False
        self._state.command = True
        # Clear audio buffer and reset voice time to prevent processing the wake word as a command
        self._buffers.command = []
        self._state.last_voice_time = time_module.time()
        self._state.command_start_time = time_module.time()
        self._state.ignore_audio_until = time_module.time() + 1.0  # Ignore audio for 1 second
        # Clear the wake word buffer to prevent immediate re-detection
        self._buffers.wake = []
        print("[WAKE WORD] Buffer cleared to prevent re-detection")

        self._callbacks.wake_word_detected()
        self._callbacks.state("listening")
        threading.Thread(target=self._return_to_wake_word_listening, daemon=True).start()

    def _wait_for_quiet(self, wait_time: float = 10.0, check_interval: float = 1.0) -> None:
        """Wait up to ``wait_time``, extending while the user keeps speaking"""
        elapsed = 0.0
        while elapsed < wait_time:
            time_module.sleep(check_interval)
            elapsed += check_interval
            time_since_last_voice = time_module.time() - self._state.last_voice_time
            if time_since_last_voice < 3.0:
                # User is actually speaking (recent voice activity), reset the timer
                elapsed = 0.0
                print("[WAKE WORD] User still speaking, extending wait time...")
            elif not self._state.command and not self._state.processing and time_since_last_voice > 3.0:
                # Command processing is done and the user has been quiet for 3+ seconds
                break

    def _return_to_wake_word_listening(self) -> None:
        """After a command, return to wake word listening once the user is quiet"""
        self._wait_for_quiet()
        time_since_last_voice = time_module.time() - self._state.last_voice_time
        if self._state.processing or (self._state.command and time_since_last_voice <= 3.0):
            print("[WAKE WORD] User still active, not returning to idle yet")
            return

        self._state.wake_word = True
        self._ensure_wake_thread("[WAKE WORD] Restarting detection thread...")
        # Clear wake word buffer so old audio can't trigger a new detection
        self._buffers.wake = []
        print("[WAKE WORD] Buffer cleared when returning to wake word listening")
        self._wake.last_check = time_module.time()
        self._callbacks.state("idle")
        print(f"Returned to listening for wake word: '{self._config.wake_word}'")

    def _process_audio(self):
        """Process captured audio command"""
        if self._state.processing or len(self._buffers.command) == 0:
            return
        self._state.processing = True
        self._callbacks.state("processing")
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _transcribe_command(self, audio_array) -> str:
        """Transcribe captured command audio"""
        if not self._stt.available:
            time_module.sleep(0.5)
            return NO_BACKEND_TEXT
        try:
            transcript = self._stt.transcribe(audio_array)
        except STT_ERRORS as e:
            print(f"[TRANSCRIBE] Error: {e}")
            traceback.print_exc()
            return TRANSCRIPTION_ERROR_TEXT
        if transcript:
            print(f"[TRANSCRIBE] '{transcript}'")
        else:
            print("[TRANSCRIBE] (no speech detected)")
        return transcript or ""

    def _process_worker(self) -> None:
        """Transcribe the command buffer and report it (runs in a background thread)"""
        try:
            audio_array = normalize(np.array(self._buffers.command, dtype=np.float32))
            transcript = self._transcribe_command(audio_array)
            self._callbacks.state("responding")
            time_module.sleep(1.0)  # Simulate response time
            self._callbacks.transcript(transcript)
            # The return-to-wake-word thread decides when to go back to idle
        except (*STT_ERRORS, *AUDIO_ERRORS) as e:
            print(f"Error processing audio: {e}")
            self._callbacks.state("idle")
            # Still return to wake word listening even on error
            self._state.command = False
            self._state.wake_word = True
            self._ensure_wake_thread()
            self._wake.last_check = time_module.time()
        finally:
            self._state.processing = False
            self._buffers.command = []

    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()
        self._stt.unload()
