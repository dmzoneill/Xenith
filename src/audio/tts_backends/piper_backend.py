"""
Piper TTS Backend - Fast neural TTS with persistent model loading

Piper keeps models in memory for instant synthesis (~50-100ms per sentence).
Much faster than MeloTTS which reloads models every call.
"""

import tempfile
import threading
import urllib.request
import wave
from pathlib import Path
from typing import ClassVar

import numpy as np

from .base import TTSBackend, TTSResult

try:
    from piper import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


class PiperTTSBackend(TTSBackend):
    """Fast TTS using Piper with persistent model loading"""

    # Available voices (will be downloaded on first use)
    VOICES: ClassVar[dict[str, str]] = {
        "en_US-amy-medium": "Amy (US, Medium quality)",
        "en_US-lessac-medium": "Lessac (US, Medium quality)",
        "en_GB-alan-medium": "Alan (UK, Medium quality)",
    }

    # Map MeloTTS voice names to Piper voices
    # Ryan only has "high", lessac has "medium" (faster)
    VOICE_MAP: ClassVar[dict[str, str]] = {
        "EN-Default": "en_US-ryan-high",  # Male voice (quality)
        "EN-Fast": "en_US-lessac-medium",  # Fast synthesis (~100ms)
        "EN-US": "en_US-ryan-high",
        "EN-BR": "en_US-lessac-medium",
        "EN-AU": "en_GB-alan-medium",
        "EN-INDIA": "en_US-lessac-medium",
    }

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        device: str = "auto",  # Ignored - Piper uses CPU
        output_dir: str | None = None,
        **kwargs,
    ):
        del kwargs  # accepted for factory compatibility; unused by Piper
        # Map MeloTTS-style voice names to Piper voices
        self._voice_name = self.VOICE_MAP.get(voice, voice)
        super().__init__(voice=self._voice_name, device=device)
        # Default to a private (0700) per-instance temp directory
        self._output_dir = Path(output_dir or tempfile.mkdtemp(prefix="xenith_tts_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._voice: PiperVoice | None = None
        self._counter = 0
        self._lock = threading.Lock()
        self._loaded = False
        self._model_path: Path | None = None

    @classmethod
    def is_available(cls) -> bool:
        return PIPER_AVAILABLE

    def _get_model_path(self) -> Path | None:
        """Get or download the voice model"""
        cache_dir = Path.home() / ".cache" / "piper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / f"{self._voice_name}.onnx"
        config_path = cache_dir / f"{self._voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            return model_path

        # Download model
        print(f"[PiperTTS] Downloading voice model: {self._voice_name}...")
        try:
            # Get model URL from Piper voices
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            lang = self._voice_name.split("-")[0]  # en_US -> en
            # en_US-lessac-medium -> en_US/lessac-medium
            voice_dir = self._voice_name.replace("-", "/", 1)

            model_url = f"{base_url}/{lang}/{voice_dir}/{self._voice_name}.onnx"
            config_url = f"{base_url}/{lang}/{voice_dir}/{self._voice_name}.onnx.json"

            urllib.request.urlretrieve(model_url, model_path)  # nosec B310
            urllib.request.urlretrieve(config_url, config_path)  # nosec B310

            print(f"[PiperTTS] ✓ Downloaded {self._voice_name}")
            return model_path

        except (OSError, ValueError) as e:
            print(f"[PiperTTS] Failed to download model: {e}")
            return None

    def load(self) -> bool:
        if not PIPER_AVAILABLE:
            print("[PiperTTS] piper-tts not installed")
            return False

        try:
            self._model_path = self._get_model_path()
            if not self._model_path:
                return False

            # Load voice model (stays in memory)
            print("[PiperTTS] Loading voice model...")
            self._voice = PiperVoice.load(str(self._model_path))
            self._loaded = True
            print(f"[PiperTTS] ✓ Loaded: {self._voice_name}")
            return True

        except (KeyError, OSError, RuntimeError, ValueError) as e:
            print(f"[PiperTTS] Load failed: {e}")
            return False

    def unload(self):
        self._voice = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(
        self,
        text: str,
        output_path: Path | str | None = None,
        speed: float = 1.0,
        **kwargs,
    ) -> TTSResult:
        """Synthesize text to audio

        Args:
            text: Text to synthesize
            output_path: Optional path to save WAV file (if None, only returns audio_data)
            speed: Ignored (Piper synthesizes at the model's native rate)
            **kwargs: Additional options (ignored)

        Returns:
            TTSResult with audio_data (numpy array) and optionally audio_path
        """
        del speed, kwargs  # interface compatibility with TTSBackend.synthesize
        if (not self._loaded or not self._voice) and not self.load():
            return TTSResult(audio_path=None, duration=0.0, error="Not loaded")
        voice = self._voice
        if voice is None:
            return TTSResult(audio_path=None, duration=0.0, error="Not loaded")

        try:
            # Synthesize audio chunks directly to memory
            audio_chunks = list(voice.synthesize(text))
            if not audio_chunks:
                return TTSResult(audio_path=None, duration=0.0, error="No audio generated")

            # Combine all chunks into bytes
            all_audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
            sample_rate = audio_chunks[0].sample_rate

            # Convert to numpy float32 array (what sounddevice expects)
            pcm = np.frombuffer(all_audio_bytes, dtype=np.int16)
            audio_data = pcm.astype(np.float32) / 32768.0

            audio_duration = len(audio_data) / sample_rate

            # Only write file if explicitly requested
            result_path = None
            if output_path:
                result_path = Path(output_path)
                with wave.Wave_write(str(result_path)) as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(all_audio_bytes)

            return TTSResult(
                audio_path=result_path,
                audio_data=audio_data,
                duration=audio_duration,
                sample_rate=sample_rate,
            )

        except (wave.Error, OSError, RuntimeError, ValueError) as e:
            return TTSResult(audio_path=None, duration=0.0, error=str(e))

    def get_available_voices(self) -> list[str]:
        return list(self.VOICES.keys())

    def __repr__(self):
        return f"PiperTTSBackend(voice={self._voice_name}, loaded={self._loaded})"
