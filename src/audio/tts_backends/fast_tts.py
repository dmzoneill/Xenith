"""
Fast TTS Backend using espeak-ng

Provides near-instant TTS for low-latency first response (~50ms),
while MeloTTS handles high-quality audio for subsequent sentences.
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from .base import TTSBackend, TTSResult

# Check if espeak-ng is available
ESPEAK_PATH = shutil.which("espeak-ng") or shutil.which("espeak")
ESPEAK_AVAILABLE = ESPEAK_PATH is not None


class FastTTSBackend(TTSBackend):
    """Ultra-fast TTS using espeak-ng (saves to file without playing)"""

    def __init__(self, rate: int = 175, output_dir: str = "/dev/shm/xenith_tts"):
        self._rate = rate
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        return ESPEAK_AVAILABLE

    @classmethod
    def get_available_voices(cls) -> list:
        """Get available voices"""
        if not ESPEAK_AVAILABLE:
            return []
        try:
            result = subprocess.run(
                [ESPEAK_PATH, "--voices"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse voice list
            voices = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    voices.append(parts[3])  # Voice name
            return voices[:10] if voices else ["en"]
        except Exception:
            return ["en"]

    def load(self) -> bool:
        if not ESPEAK_AVAILABLE:
            print("[FastTTS] espeak-ng not installed")
            return False

        self._loaded = True
        print(f"[FastTTS] Loaded espeak-ng (rate={self._rate})")
        return True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def synthesize(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> TTSResult:
        """Synthesize text to audio file (no audio playback)"""
        if not self._loaded:
            if not self.load():
                return TTSResult(audio_path="", duration=0.0, error="Not loaded")

        with self._lock:
            self._counter += 1
            if not output_path:
                output_path = str(self._output_dir / f"fast_tts_{self._counter}.wav")

        try:
            import time

            start = time.time()

            # Use espeak-ng to save to file (no audio output)
            result = subprocess.run(
                [
                    ESPEAK_PATH,
                    "-w",
                    output_path,  # Write to WAV file
                    "-s",
                    str(self._rate),  # Speed
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            duration = time.time() - start

            if result.returncode == 0 and Path(output_path).exists():
                return TTSResult(audio_path=output_path, duration=duration)
            else:
                return TTSResult(
                    audio_path="",
                    duration=0.0,
                    error=result.stderr or "espeak-ng failed",
                )

        except Exception as e:
            return TTSResult(audio_path="", duration=0.0, error=str(e))

    def __repr__(self):
        return f"FastTTSBackend(espeak-ng, rate={self._rate}, loaded={self._loaded})"


# Keep pyttsx3 check for backwards compatibility
try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


def get_fast_tts() -> Optional[FastTTSBackend]:
    """Get a fast TTS backend if available"""
    if ESPEAK_AVAILABLE:
        backend = FastTTSBackend()
        if backend.load():
            return backend
    return None
