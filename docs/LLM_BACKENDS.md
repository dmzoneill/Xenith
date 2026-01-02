# LLM Backends Reference

This document describes the Language Model (LLM) backend system in Xenith.

## Overview

Xenith uses a pluggable LLM backend system with **OpenVINO GenAI**, which runs quantized models efficiently on CPU, NPU, or GPU.

## ⚡ Critical: CPU vs NPU for Response Time

| Device | First Token Latency | Power | Recommended For |
|--------|---------------------|-------|-----------------|
| **CPU** | **~200ms** ⚡ | ~15-30W | **Fast voice response** |
| NPU | ~2,300ms | ~3-5W | Battery life (slower) |
| GPU | ~300ms | ~30-50W | Larger models |

> **Recommendation:** Use `device: "CPU"` for fast voice assistant responses (~1.5s total). NPU is 12x slower due to per-query recompilation overhead.

## Available Backends

### OpenVINO (Default)

Runs INT4 quantized models via OpenVINO GenAI on Intel hardware.

#### Supported Devices

| Device | First Token | Power | Best For |
|--------|-------------|-------|----------|
| CPU | ~200ms | ~15-30W | **Fast response** ⭐ |
| NPU | ~2,300ms | ~3-5W | Power savings |
| GPU.0 (Intel iGPU) | ~300ms | ~15-25W | Balance |
| GPU.1 (NVIDIA) | ~100ms | ~50-100W | Fastest |

#### Available Models

##### NPU-Compatible (Recommended)

| Model ID | Size | RAM | Power | Quality | Use Case |
|----------|------|-----|-------|---------|----------|
| `qwen3-0.6b` | 0.6B | 1GB | ~1-2W | Basic | Simple commands |
| `tinyllama-1.1b` | 1.1B | 2GB | ~2-3W | Good | Basic chat |
| `qwen2.5-1.5b` | 1.5B | 2GB | ~3-5W | Better | **Voice assistant** ⭐ |
| `deepseek-r1-1.5b` | 1.5B | 2GB | ~3-5W | Better | Reasoning tasks |
| `phi-3-mini` | 3.8B | 4GB | ~5-10W | Excellent | Complex tasks |
| `phi-3.5-mini` | 3.8B | 4GB | ~5-10W | Excellent | Latest Phi |

##### GPU-Only (Higher Power)

| Model ID | Size | RAM | Power | Quality |
|----------|------|-----|-------|---------|
| `mistral-7b` | 7B | 8GB | ~15-25W | Best |
| `deepseek-r1-7b` | 7B | 8GB | ~15-25W | Best |

## Configuration

LLM settings are in `config/config.yaml`:

```yaml
llm:
  # Backend: "auto", "openvino", "openai", "anthropic"
  default_backend: "auto"
  
  # Device: "cpu", "npu", "gpu", "auto"
  #   - cpu: RECOMMENDED for fast response (~200ms warmup)
  #   - npu: Low power but slow (~2.3s warmup per query!)
  #   - gpu: Good balance
  #   - auto: NPU → GPU → CPU priority
  device: "CPU"  # CPU recommended for voice assistant
  
  # Model for local inference
  model: "qwen2.5-1.5b"
  
  # System prompt
  system_prompt: "You are Xenith, a helpful voice assistant. Be concise."
  
  # Generation settings
  max_tokens: 256
  temperature: 0.7
  
  # Cloud API fallback
  fallback_to_cloud: false
```

## Python API

### Basic Usage

```python
from src.audio.llm_backends import get_llm_backend, print_llm_status

# Print status
print_llm_status()

# Get backend (auto-selects best available)
llm = get_llm_backend(model="qwen2.5-1.5b")

if llm and llm.load():
    result = llm.generate("What is the capital of France?")
    print(result.text)  # "The capital of France is Paris."
```

### Specifying Device

```python
# Force NPU
llm = get_llm_backend(model="qwen2.5-1.5b", device="npu")

# Force GPU
llm = get_llm_backend(model="phi-3-mini", device="gpu")

# Force CPU
llm = get_llm_backend(model="tinyllama-1.1b", device="cpu")
```

### With System Prompt

```python
result = llm.generate(
    "What's the weather?",
    system_prompt="You are a helpful weather assistant."
)
```

### Generation Configuration

```python
from src.audio.llm_backends import LLMConfig

config = LLMConfig(
    max_tokens=512,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
)

result = llm.generate("Tell me a story", config=config)
```

### Streaming Output

```python
for token in llm.generate_stream("Explain quantum physics"):
    print(token, end="", flush=True)
```

### Chat with History

```python
from src.audio.llm_backends import Message

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
    Message(role="assistant", content="Hi there! How can I help?"),
    Message(role="user", content="What's 2+2?"),
]

result = llm.chat(messages)
print(result.text)  # "2+2 equals 4."
```

## Model Download

Models are downloaded automatically from HuggingFace on first use.

### Manual Download

```bash
# Download specific model
python test_llm_backends.py --download --model qwen2.5-1.5b

# Download all recommended models
python -c "
from src.audio.llm_backends.openvino_backend import download_model
for model in ['qwen2.5-1.5b', 'tinyllama-1.1b', 'phi-3-mini']:
    download_model(model)
"
```

### Cache Location

Models are cached at: `~/.cache/xenith/models/`

## LLMBackend Base Class

```python
class LLMBackend(ABC):
    def __init__(self, model: str, device: str): ...
    
    @property
    def name(self) -> str: ...
    
    @property
    def is_loaded(self) -> bool: ...
    
    @abstractmethod
    def load(self) -> bool: ...
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResult: ...
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]: ...
    
    def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResult: ...
    
    def unload(self) -> None: ...
    
    @classmethod
    def is_available(cls) -> bool: ...
    
    @classmethod
    def get_available_models(cls) -> List[str]: ...
```

## LLMResult Dataclass

```python
@dataclass
class LLMResult:
    text: str
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"  # stop, length, error
    model: Optional[str] = None
```

## Performance Benchmarks

Tested on Intel Core Ultra 7 165H:

| Model | Device | First Token | Total Response | Power |
|-------|--------|-------------|----------------|-------|
| qwen2.5-1.5b | **CPU** | **~200ms** ⚡ | ~700ms | ~15-25W |
| qwen2.5-1.5b | NPU | ~2,300ms | ~3,500ms | ~3-5W |
| qwen2.5-1.5b | GPU.0 | ~300ms | ~900ms | ~15-20W |
| phi-3-mini | CPU | ~400ms | ~1,500ms | ~25-35W |
| phi-3-mini | NPU | ~3,000ms | ~5,000ms | ~5-10W |

### Key Finding

**NPU has ~2.3s overhead per query** due to model recompilation. This makes it unsuitable for real-time voice responses. Use **CPU for voice assistant** (12x faster first token).

## Troubleshooting

### Model Not Loading

```bash
# Check OpenVINO GenAI is installed
python -c "import openvino_genai; print('OK')"

# Check available devices
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

### NPU Not Available

Ensure NPU drivers are installed. See [INTEL_NPU_SETUP.md](INTEL_NPU_SETUP.md).

### Out of Memory

Try a smaller model:
```python
llm = get_llm_backend(model="tinyllama-1.1b")  # Uses less RAM
```

### Slow Generation

1. Check you're using NPU or GPU, not CPU
2. Try a smaller model
3. Reduce `max_tokens` in generation config

## See Also

- [Voice Pipeline Architecture](VOICE_PIPELINE.md)
- [Intel NPU Setup Guide](INTEL_NPU_SETUP.md)
- [STT Backends Reference](STT_BACKENDS.md)
- [TTS Backends Reference](TTS_BACKENDS.md)


