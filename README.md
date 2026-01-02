# Xenith

**Ultra-fast, on-device voice assistant for Linux**

Xenith is a fully local voice assistant that runs entirely on your hardware with no cloud dependencies. Optimized for Intel Core Ultra processors, it achieves **~1.5-2.5 second response times** from wake word to audio output.

![Response Time](https://img.shields.io/badge/Response%20Time-1.5--2.5s-brightgreen)
![Power](https://img.shields.io/badge/Power-15--30W%20Active-blue)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green)

## Features

- 🎤 **Wake Word Detection** - Always listening with ultra-low power (~2-3W)
- 🧠 **Local LLM** - Qwen2.5-1.5B runs entirely on-device
- 🔊 **Natural TTS** - High-quality Piper neural voices
- ⚡ **Streaming Response** - Audio starts playing as LLM generates
- 🔒 **100% Private** - No data leaves your device
- 🎨 **Beautiful UI** - Animated plasma widget shows voice state

## Quick Start

```bash
# Install dependencies
make install

# Run Xenith
make run
```

Say **"Hi"** to activate, then speak your command!

## Performance

| Stage | Time |
|-------|------|
| Wake word → Detection | ~500ms |
| STT Processing | ~300ms |
| LLM First Token | ~200ms |
| TTS → Audio | ~100ms |
| **Total to First Audio** | **~1.5s** |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     XENITH VOICE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   🎤 Microphone                                                     │
│        │ (continuous audio stream)                                  │
│        ▼                                                            │
│   ┌──────────────────┐                                              │
│   │   Wake Word      │  "Hi" detection                              │
│   │   Detection      │  • Whisper on NPU (~2-3W)                    │
│   │                  │  • 0.5s check interval                       │
│   └────────┬─────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   ┌──────────────────┐                                              │
│   │   Speech-to-Text │  Whisper STT                                 │
│   │   (STT)          │  • OpenVINO on NPU                           │
│   │                  │  • 0.3s silence threshold                    │
│   └────────┬─────────┘                                              │
│            │ text                                                   │
│            ▼                                                        │
│   ┌──────────────────┐                                              │
│   │   LLM Brain      │  Qwen2.5-1.5B                                │
│   │                  │  • OpenVINO on CPU (fast, ~200ms warmup)     │
│   │                  │  • Token streaming enabled                   │
│   └────────┬─────────┘                                              │
│            │ streaming tokens                                       │
│            ▼                                                        │
│   ┌──────────────────┐                                              │
│   │   Sentence       │  Buffers tokens until sentence complete      │
│   │   Buffer         │  • Min 3 chars, ends on .!?;,:               │
│   └────────┬─────────┘                                              │
│            │ sentences                                              │
│            ▼                                                        │
│   ┌──────────────────┐                                              │
│   │   TTS (Piper)    │  Neural text-to-speech                       │
│   │                  │  • ~100ms per sentence                       │
│   │                  │  • In-memory audio (no file I/O)             │
│   └────────┬─────────┘                                              │
│            │ numpy audio                                            │
│            ▼                                                        │
│   ┌──────────────────┐                                              │
│   │   Audio Player   │  Real-time playback                          │
│   │                  │  • Direct sounddevice output                 │
│   │                  │  • 10ms queue polling                        │
│   └────────┬─────────┘                                              │
│            │                                                        │
│            ▼                                                        │
│   🔊 Speakers                                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow (Optimized)

All audio data flows **in-memory** with zero file I/O in the critical path:

```
Mic → numpy → STT(NPU) → text → LLM(CPU) → tokens → TTS(CPU) → numpy → speakers
      ↑                                                              ↑
      └── in-memory ─────────────────────────────────────────────────┘
```

## Hardware Requirements

### Minimum
- Intel Core Ultra (Meteor Lake) or newer
- 8GB RAM
- 5GB disk space

### Recommended
- Intel Core Ultra 7/9
- 16GB RAM
- Intel Arc GPU (optional, for larger models)

## Configuration

Edit `config/config.yaml`:

```yaml
llm:
  # CPU recommended for fast response (~200ms warmup)
  # NPU is low power but slow (~2.3s warmup per query)
  device: "CPU"
  model: "qwen2.5-1.5b"

audio:
  stt:
    device: "auto"  # NPU → Intel GPU → CPU
    model: "base"
  tts:
    voice: "EN-Default"  # Ryan male voice (high quality)
```

## Device Trade-offs

| Device | LLM Warmup | Power | Best For |
|--------|------------|-------|----------|
| CPU | ~200ms | ~15-30W | **Fast response (recommended)** |
| NPU | ~2,300ms | ~3-5W | Battery life |
| GPU | ~300ms | ~30-50W | Larger models |

## Project Structure

```
src/
├── app.py                  # Main GTK application
├── main.py                 # Entry point
├── widgets/
│   └── plasma_widget.py    # Animated voice indicator
└── audio/
    ├── voice_input.py      # Wake word & STT handling
    ├── streaming_pipeline.py  # LLM + TTS streaming
    ├── pipeline_metrics.py    # Performance tracking
    ├── stt_backends/       # Speech-to-Text
    │   ├── openvino_backend.py
    │   └── whisper_backend.py
    ├── llm_backends/       # Language Models
    │   └── openvino_backend.py
    └── tts_backends/       # Text-to-Speech
        ├── piper_backend.py
        └── melotts_backend.py
```

## Testing

```bash
# Test STT backends
python test_stt_backends.py

# Test TTS backends
python test_tts_backends.py

# Test LLM backends
python test_llm_backends.py

# Test full pipeline
python test_streaming_pipeline.py
```

## Documentation

- [Voice Pipeline Architecture](docs/VOICE_PIPELINE.md)
- [Intel NPU Setup Guide](docs/INTEL_NPU_SETUP.md)
- [STT Backends Reference](docs/STT_BACKENDS.md)
- [TTS Backends Reference](docs/TTS_BACKENDS.md)
- [LLM Backends Reference](docs/LLM_BACKENDS.md)
- [Scripts Reference](docs/SCRIPTS.md)

## Performance Tuning

### For Fastest Response (~1.5s)
```yaml
llm:
  device: "CPU"  # 12x faster than NPU
audio:
  tts:
    voice: "EN-Fast"  # Medium quality, faster synthesis
```

### For Lowest Power (~3-5W active)
```yaml
llm:
  device: "NPU"  # Slower but efficient
audio:
  stt:
    device: "NPU"
```

## License

MIT License - See LICENSE file for details.

