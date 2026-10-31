"""
Fast TTS Backend using espeak-ng

Provides near-instant TTS for low-latency first response (~50ms),
while MeloTTS handles high-quality audio for subsequent sentences.
"""

import importlib.util
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .base import TTSBackend, TTSResult

# Check if espeak-ng is available
ESPEAK_PATH = shutil.which("espeak-ng") or shutil.which("espeak")
ESPEAK_AVAILABLE = ESPEAK_PATH is not None


class FastTTSBackend(TTSBackend):
    """Ultra-fast TTS using espeak-ng (saves to file without playing)"""

    def __init__(self, rate: int = 175, output_dir: str | None = None):
        super().__init__()
        self._rate = rate
        # Default to a private (0700) per-instance temp directory
        self._output_dir = Path(output_dir or tempfile.mkdtemp(prefix="xenith_tts_"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._lock = threading.Lock()
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        return ESPEAK_AVAILABLE

    @classmethod
    def get_available_voices(cls) -> list[str]:  # pylint: disable=W0221
        """Get available voices"""
        if ESPEAK_PATH is None:
            return []
        try:
            result = subprocess.run(  # nosec B603 # nosemgrep
                [ESPEAK_PATH, "--voices"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            # Parse voice list
            voices = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    voices.append(parts[3])  # Voice name
            return voices[:10] if voices else ["en"]
        except (OSError, subprocess.SubprocessError):
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

    def synthesize(  # pylint: disable=W0221
        self,
        text: str,
        output_path: Path | str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSResult:
        """Synthesize text to audio file (no audio playback)"""
        if ESPEAK_PATH is None or (not self._loaded and not self.load()):
            return TTSResult(duration=0.0, error="Not loaded")

        with self._lock:
            self._counter += 1
            target = Path(output_path) if output_path else self._output_dir / f"fast_tts_{self._counter}.wav"

        try:
            start = time.time()

            # Use espeak-ng to save to file (no audio output)
            result = subprocess.run(
                [
                    ESPEAK_PATH,
                    "-w",
                    str(target),  # Write to WAV file
                    "-s",
                    str(round(self._rate * speed)),  # Speed
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            duration = time.time() - start

            if result.returncode == 0 and target.exists():
                return TTSResult(audio_path=target, duration=duration)
            return TTSResult(duration=0.0, error=result.stderr or "espeak-ng failed")

        except (OSError, subprocess.SubprocessError) as e:
            return TTSResult(duration=0.0, error=str(e))

    def __repr__(self):
        return f"FastTTSBackend(espeak-ng, rate={self._rate}, loaded={self._loaded})"


# Keep pyttsx3 check for backwards compatibility
PYTTSX3_AVAILABLE = importlib.util.find_spec("pyttsx3") is not None


def get_fast_tts() -> FastTTSBackend | None:
    """Get a fast TTS backend if available"""
    if ESPEAK_AVAILABLE:
        backend = FastTTSBackend()
        if backend.load():
            return backend
    return None
