"""OpenVINO LLM backend for NPU/GPU/CPU inference"""

import os
from pathlib import Path
from typing import Optional, List, Generator, Dict, Any

from .base import LLMBackend, LLMResult, LLMConfig, Message

# Model registry with HuggingFace paths
OPENVINO_MODELS: Dict[str, Dict[str, Any]] = {
    # Ultra-small models (NPU optimized)
    "qwen3-0.6b": {
        "hf_id": "OpenVINO/Qwen3-0.6B-int4-ov",
        "size": "0.6B",
        "ram": "1GB",
        "npu_compatible": True,
        "description": "Smallest model, basic commands",
    },
    # Small models (NPU optimized)
    "tinyllama-1.1b": {
        "hf_id": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "size": "1.1B",
        "ram": "2GB",
        "npu_compatible": True,
        "description": "Small chat model",
    },
    "qwen2.5-1.5b": {
        "hf_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "size": "1.5B",
        "ram": "2GB",
        "npu_compatible": True,
        "description": "Recommended for voice assistant",
    },
    "deepseek-r1-1.5b": {
        "hf_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
        "size": "1.5B",
        "ram": "2GB",
        "npu_compatible": True,
        "description": "Reasoning-focused model",
    },
    # Medium models (NPU/GPU)
    "phi-3-mini": {
        "hf_id": "OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
        "size": "3.8B",
        "ram": "4GB",
        "npu_compatible": True,
        "description": "High quality, larger model",
    },
    "phi-3.5-mini": {
        "hf_id": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        "size": "3.8B",
        "ram": "4GB",
        "npu_compatible": True,
        "description": "Latest Phi model",
    },
    # Large models (GPU recommended)
    "mistral-7b": {
        "hf_id": "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov",
        "size": "7B",
        "ram": "8GB",
        "npu_compatible": False,
        "description": "High quality, GPU recommended",
    },
    "deepseek-r1-7b": {
        "hf_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov",
        "size": "7B",
        "ram": "8GB",
        "npu_compatible": False,
        "description": "Reasoning model, GPU recommended",
    },
}

# Default model cache directory
MODEL_CACHE_DIR = Path.home() / ".cache" / "xenith" / "models"


class OpenVINOLLMBackend(LLMBackend):
    """OpenVINO-based LLM backend supporting NPU, GPU, and CPU

    Uses openvino_genai.LLMPipeline for efficient inference.
    """

    MODELS = OPENVINO_MODELS

    def __init__(
        self,
        model: str = "qwen2.5-1.5b",
        device: str = "auto",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize OpenVINO LLM backend

        Args:
            model: Model identifier from MODELS registry
            device: Device ("auto", "cpu", "gpu", "npu", "GPU.0", "GPU.1")
            cache_dir: Directory for model cache
        """
        super().__init__(model=model, device=device)
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        self._tokenizer = None
        self._streamer = None

    def _resolve_device(self) -> str:
        """Resolve 'auto' device to best available"""
        if self.device != "auto":
            return self.device.upper()

        try:
            import openvino as ov

            core = ov.Core()
            available = core.available_devices

            # Check model NPU compatibility
            model_info = self.MODELS.get(self.model, {})
            npu_compatible = model_info.get("npu_compatible", False)

            # Priority: NPU → Intel GPU → CPU
            if "NPU" in available and npu_compatible:
                return "NPU"
            if "GPU.0" in available:  # Intel iGPU
                return "GPU.0"
            if "GPU" in available:
                return "GPU"
            return "CPU"
        except ImportError:
            return "CPU"

    def _get_model_path(self) -> Path:
        """Get local path to model, downloading if needed"""
        model_info = self.MODELS.get(self.model)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model}")

        hf_id = model_info["hf_id"]
        model_name = hf_id.split("/")[-1]
        local_path = self.cache_dir / model_name

        if local_path.exists():
            return local_path

        # Download from HuggingFace
        print(f"[LLM] Downloading {hf_id}...")
        self._download_model(hf_id, local_path)
        return local_path

    def _download_model(self, hf_id: str, local_path: Path) -> None:
        """Download model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=hf_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
            )
            print(f"[LLM] Downloaded to {local_path}")
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )

    def load(self) -> bool:
        """Load the LLM model"""
        if self._is_loaded:
            return True

        try:
            import openvino_genai as ov_genai

            model_path = self._get_model_path()
            device = self._resolve_device()

            print(f"[LLM] Loading {self.model} on {device}...")

            # Create LLM pipeline
            self._pipeline = ov_genai.LLMPipeline(str(model_path), device)

            self._is_loaded = True
            print(f"[LLM] ✓ Loaded {self.model} on {device}")
            return True

        except ImportError as e:
            print(f"[LLM] openvino_genai not available: {e}")
            return False
        except Exception as e:
            print(f"[LLM] Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResult:
        """Generate text from prompt"""
        if not self._is_loaded:
            if not self.load():
                return LLMResult(text="", finish_reason="error")

        config = config or LLMConfig()

        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"

        try:
            import openvino_genai as ov_genai

            # Configure generation
            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = config.max_tokens
            gen_config.temperature = config.temperature
            gen_config.top_p = config.top_p
            gen_config.top_k = config.top_k
            gen_config.repetition_penalty = config.repetition_penalty

            # Generate
            result = self._pipeline.generate(full_prompt, gen_config)

            # Extract text (handle different return types)
            if hasattr(result, "texts"):
                text = result.texts[0] if result.texts else ""
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

            # Clean up response
            text = text.strip()

            return LLMResult(
                text=text,
                model=self.model,
                finish_reason="stop",
            )

        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return LLMResult(text="", finish_reason="error")

    def generate_stream(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Generate text with streaming"""
        if not self._is_loaded:
            if not self.load():
                return

        config = config or LLMConfig()

        # Build full prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"

        try:
            import openvino_genai as ov_genai

            # Configure generation
            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = config.max_tokens
            gen_config.temperature = config.temperature
            gen_config.top_p = config.top_p
            gen_config.top_k = config.top_k

            # Create streamer
            def streamer_callback(token: str) -> bool:
                return True  # Continue generation

            # Use TextStreamer for streaming
            streamer = ov_genai.TextStreamer(self._pipeline.get_tokenizer())

            # Generate with streaming
            for token in self._pipeline.generate(full_prompt, gen_config, streamer):
                yield token

        except Exception as e:
            print(f"[LLM] Streaming error: {e}")

    def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResult:
        """Chat with conversation history"""
        # Format messages for chat
        system_prompt = None
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                conversation.append(f"{msg.role.capitalize()}: {msg.content}")

        # Get last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt = msg.content
                break

        # Build context from history
        history = "\n".join(conversation[:-1]) if len(conversation) > 1 else ""
        if history:
            full_prompt = f"{history}\n\nUser: {prompt}"
        else:
            full_prompt = prompt

        return self.generate(full_prompt, config, system_prompt)

    def unload(self) -> None:
        """Unload model"""
        self._pipeline = None
        self._tokenizer = None
        self._is_loaded = False
        print(f"[LLM] Unloaded {self.model}")

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenVINO GenAI is available"""
        try:
            import openvino_genai

            return True
        except ImportError:
            return False

    @classmethod
    def get_device_info(cls) -> dict:
        """Get available OpenVINO devices"""
        devices = ["cpu"]
        try:
            import openvino as ov

            core = ov.Core()
            available = core.available_devices
            if "NPU" in available:
                devices.append("npu")
            if any("GPU" in d for d in available):
                devices.append("gpu")
        except ImportError:
            pass

        return {
            "devices": devices,
            "default": "npu" if "npu" in devices else "cpu",
        }


def download_model(model: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a model from HuggingFace

    Args:
        model: Model identifier (e.g., "qwen2.5-1.5b")
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded model
    """
    backend = OpenVINOLLMBackend(model=model, cache_dir=cache_dir)
    return backend._get_model_path()


def list_models() -> Dict[str, Dict[str, Any]]:
    """List all available OpenVINO models"""
    return OPENVINO_MODELS

