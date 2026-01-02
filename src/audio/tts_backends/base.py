"""Base class for Text-to-Speech backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis"""

    audio_path: Optional[Path] = None  # Path to generated audio file
    audio_data: Optional[np.ndarray] = None  # Audio data as numpy array
    sample_rate: int = 22050
    duration: Optional[float] = None  # Audio duration in seconds
    voice: Optional[str] = None  # Voice/speaker used
    error: Optional[str] = None  # Error message if synthesis failed


class TTSBackend(ABC):
    """Abstract base class for TTS backends

    All TTS backends must implement this interface to work with Xenith's
    voice output system.
    """

    def __init__(
        self,
        voice: str = "default",
        device: str = "auto",
        language: str = "EN",
    ):
        """Initialize the TTS backend

        Args:
            voice: Voice/speaker to use (e.g., "EN-US", "EN-BR", "EN-Default")
            device: Device to run inference on ("auto", "cpu", "gpu", "npu")
            language: Language for synthesis ("EN", "ZH")
        """
        self.voice = voice
        self.device = device
        self.language = language
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Return the backend name"""
        return self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        """Check if backend is loaded and ready"""
        return self._is_loaded

    @abstractmethod
    def load(self) -> bool:
        """Load the TTS backend/models

        Returns:
            True if loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> TTSResult:
        """Synthesize speech from text

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            speed: Speech speed multiplier (1.0 = normal)
            **kwargs: Backend-specific options

        Returns:
            TTSResult with audio data or file path
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Get list of available voices/speakers

        Returns:
            List of voice identifiers
        """
        pass

    def unload(self) -> None:
        """Unload backend to free resources"""
        self._is_loaded = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available

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
        return f"{self.name}(voice={self.voice}, device={self.device}, loaded={self._is_loaded})"
