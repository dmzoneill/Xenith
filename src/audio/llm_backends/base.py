"""Base class for Language Model backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Generator, Dict, Any


@dataclass
class LLMResult:
    """Result from LLM generation"""

    text: str
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"  # stop, length, error
    model: Optional[str] = None


@dataclass
class Message:
    """Chat message"""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMConfig:
    """Configuration for LLM generation"""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)


class LLMBackend(ABC):
    """Abstract base class for LLM backends

    All LLM backends must implement this interface to work with Xenith's
    voice assistant pipeline.
    """

    # Available models for this backend
    MODELS: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        model: str = "qwen2.5-1.5b",
        device: str = "auto",
    ):
        """Initialize the LLM backend

        Args:
            model: Model identifier (e.g., "qwen2.5-1.5b", "phi-3-mini")
            device: Device to run inference on ("auto", "cpu", "gpu", "npu")
        """
        self.model = model
        self.device = device
        self._is_loaded = False
        self._pipeline = None

    @property
    def name(self) -> str:
        """Return the backend name"""
        return self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded

    @abstractmethod
    def load(self) -> bool:
        """Load the LLM model

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResult:
        """Generate text from a prompt

        Args:
            prompt: User input/prompt
            config: Generation configuration
            system_prompt: Optional system prompt

        Returns:
            LLMResult with generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming output

        Args:
            prompt: User input/prompt
            config: Generation configuration
            system_prompt: Optional system prompt

        Yields:
            Generated text tokens
        """
        pass

    def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResult:
        """Chat with conversation history

        Args:
            messages: List of chat messages
            config: Generation configuration

        Returns:
            LLMResult with assistant response
        """
        # Default implementation - format messages into prompt
        prompt = self._format_messages(messages)
        return self.generate(prompt, config)

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages into a prompt string"""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        formatted.append("Assistant:")
        return "\n".join(formatted)

    def unload(self) -> None:
        """Unload the model to free memory"""
        self._pipeline = None
        self._is_loaded = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available

        Returns:
            True if backend can be used, False otherwise
        """
        return False  # Override in subclasses

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models for this backend

        Returns:
            List of model identifiers
        """
        return list(cls.MODELS.keys())

    @classmethod
    def get_device_info(cls) -> dict:
        """Get information about available devices

        Returns:
            Dict with device information
        """
        return {"devices": [], "default": "cpu"}

    def __repr__(self) -> str:
        return f"{self.name}(model={self.model}, device={self.device}, loaded={self._is_loaded})"

