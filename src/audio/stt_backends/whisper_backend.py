"""OpenAI Whisper backend for Speech-to-Text

Uses the original OpenAI Whisper model with PyTorch.
Supports CPU and CUDA (NVIDIA GPU) acceleration.
"""

import importlib
import importlib.util
import traceback
from typing import Any

import numpy as np

from .base import STTBackend, STTResult

# Errors PyTorch / Whisper raise at runtime (CUDA, model-file and tensor
# problems). Anything else is a programming error and should surface.
RUNTIME_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    AttributeError,
    KeyError,
)


def _module_available(name: str) -> bool:
    """True when an optional module can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


class WhisperBackend(STTBackend):
    """OpenAI Whisper backend using PyTorch

    This is the original Whisper implementation from OpenAI.
    Supports CUDA acceleration on NVIDIA GPUs (like your RTX 4060).

    Advantages:
        - High accuracy
        - Full feature support (language detection, timestamps, etc.)
        - Well-tested and stable

    Disadvantages:
        - Higher power consumption on GPU
        - Larger memory footprint
    """

    def __init__(self, model_name: str = "base", device: str = "auto"):
        """Initialize Whisper backend

        Args:
            model_name: Model size - "tiny", "base", "small", "medium", "large"
            device: "auto", "cpu", or "cuda"
        """
        super().__init__(model_name, device)
        self._model: Any = None
        self._torch: Any = None
        self._whisper: Any = None
        self._actual_device: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Whisper and PyTorch are available"""
        return _module_available("whisper") and _module_available("torch")

    @classmethod
    def get_device_info(cls) -> dict:
        """Get information about available devices"""
        devices = ["cpu"]
        info: dict[str, Any] = {
            "devices": devices,
            "default": "cpu",
            "cuda_available": False,
        }

        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return info

        if torch.cuda.is_available():
            devices.append("cuda")
            info["default"] = "cuda"
            info["cuda_available"] = True
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda

        return info

    def load_model(self) -> bool:
        """Load the Whisper model"""
        try:
            whisper = importlib.import_module("whisper")
            torch = importlib.import_module("torch")

            self._whisper = whisper
            self._torch = torch

            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._actual_device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[WHISPER] GPU acceleration: {gpu_name}")
                    print(f"[WHISPER] CUDA version: {torch.version.cuda}")
                else:
                    self._actual_device = "cpu"
                    print("[WHISPER] GPU not available, using CPU")
            else:
                self._actual_device = self.device

            # Load model
            device = self._actual_device
            print(f"[WHISPER] Loading '{self.model_name}' model on {device}...")
            self._model = whisper.load_model(self.model_name, device=device)
            self._is_loaded = True

            print(f"[WHISPER] Model loaded successfully on {device}")
            return True

        except ImportError as e:
            print(f"[WHISPER] Import error: {e}")
            print("[WHISPER] Install with: pip install openai-whisper torch")
            return False
        except RUNTIME_ERRORS as e:
            print(f"[WHISPER] Failed to load model: {e}")
            traceback.print_exc()
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "en",
        **kwargs,
    ) -> STTResult:
        """Transcribe audio using Whisper

        Args:
            audio: Audio data (float32, mono, 16kHz sample rate)
            language: Language code
            **kwargs: Additional Whisper options (fp16, beam_size, etc.)
        """
        if not self._is_loaded and not self.load_model():
            return STTResult(text="", language=language)

        try:
            # Ensure audio is the right format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Whisper options
            fp16 = kwargs.get("fp16", self._actual_device == "cuda")

            # Transcribe
            result = self._model.transcribe(
                audio,
                language=language,
                fp16=fp16,
                **{k: v for k, v in kwargs.items() if k != "fp16"},
            )

            # Extract segments if available
            segments = None
            if "segments" in result:
                segments = result["segments"]

            return STTResult(
                text=result["text"].strip(),
                language=result.get("language", language),
                segments=segments,
                duration=len(audio) / 16000,  # Assuming 16kHz
            )

        except RUNTIME_ERRORS as e:
            print(f"[WHISPER] Transcription error: {e}")
            traceback.print_exc()
            return STTResult(text="", language=language)

    def unload_model(self) -> None:
        """Unload model and free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None

            # Clear CUDA cache if using GPU
            if self._torch is not None and self._actual_device == "cuda":
                try:
                    self._torch.cuda.empty_cache()
                except RuntimeError as e:
                    # Freeing the cache is best-effort; the model is already gone.
                    print(f"[WHISPER] Could not empty CUDA cache: {e}")

        self._is_loaded = False
        print("[WHISPER] Model unloaded")
