"""Base class for Speech-to-Text backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class STTResult:
    """Result from speech-to-text transcription"""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[dict]] = None  # For word-level timestamps
    duration: Optional[float] = None  # Audio duration in seconds


class STTBackend(ABC):
    """Abstract base class for STT backends

    All STT backends must implement this interface to work with Xenith's
    voice input system.
    """

    def __init__(self, model_name: str = "base", device: str = "auto"):
        """Initialize the STT backend

        Args:
            model_name: Name/size of the model (e.g., "tiny", "base", "small", "medium", "large")
            device: Device to run inference on ("auto", "cpu", "cuda", "npu", "gpu")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Return the backend name"""
        return self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> bool:
        """Load the STT model

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, language: str = "en", **kwargs
    ) -> STTResult:
        """Transcribe audio to text

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language code for transcription
            **kwargs: Backend-specific options

        Returns:
            STTResult with transcription
        """
        pass

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        self._model = None
        self._is_loaded = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available (dependencies installed)

        Returns:
            True if backend can be used, False otherwise
        """
        return False  # Override in subclasses

    @classmethod
    def get_device_info(cls) -> dict:
        """Get information about available devices for this backend

        Returns:
            Dict with device information
        """
        return {"devices": [], "default": "cpu"}

    def __repr__(self) -> str:
        return f"{self.name}(model={self.model_name}, device={self.device}, loaded={self._is_loaded})"
