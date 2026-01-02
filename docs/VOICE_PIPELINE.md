# Xenith Voice Pipeline Architecture

This document describes the complete voice processing pipeline in Xenith, optimized for ultra-fast response times (~1.5-2.5 seconds from wake word to audio).

## Overview

Xenith implements a fully on-device, streaming voice assistant pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        XENITH VOICE PIPELINE                            │
│                    (Optimized for ~1.5s Response)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   🎤 Audio Input                                                        │
│        │ (continuous 16kHz stream)                                      │
│        ▼                                                                │
│   ┌──────────────┐                                                      │
│   │  Wake Word   │  "Hi" detection                                      │
│   │  Detector    │  • Whisper on NPU (~2-3W)                            │
│   │              │  • 0.5s check interval, 0.25s poll                   │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │   Whisper    │  Speech-to-Text (STT)                                │
│   │   OpenVINO   │  • Runs on NPU (~2-3W)                               │
│   │              │  • 0.3s silence threshold                            │
│   └──────┬───────┘                                                      │
│          │ text                                                         │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │   LLM Brain  │  Qwen2.5-1.5B                                        │
│   │   OpenVINO   │  • Runs on CPU (~200ms warmup) ⚡                    │
│   │              │  • Streaming token output                            │
│   └──────┬───────┘                                                      │
│          │ streaming tokens                                             │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  Sentence    │  Token buffer                                        │
│   │  Buffer      │  • Min 3 chars before sentence end                   │
│   │              │  • Splits on .!?;,:                                  │
│   └──────┬───────┘                                                      │
│          │ sentences (parallel)                                         │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  Piper TTS   │  Neural text-to-speech                               │
│   │              │  • ~100ms per sentence                               │
│   │              │  • In-memory audio (no file I/O)                     │
│   └──────┬───────┘                                                      │
│          │ numpy audio (direct)                                         │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │ Audio Player │  Real-time playback                                  │
│   │              │  • 10ms queue polling                                │
│   │              │  • sounddevice output                                │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   🔊 Audio Output                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Metrics

### Optimized Latency Breakdown

| Stage | Time | Notes |
|-------|------|-------|
| Wake word detection | ~500-750ms | 0.5s check interval |
| STT processing | ~300-500ms | 0.3s silence threshold |
| LLM first token | **~200ms** | CPU mode (was 2.3s on NPU!) |
| TTS first sentence | ~100ms | Piper neural TTS |
| Audio queue delay | ~0ms | In-memory transfer |
| **Total to first audio** | **~1.5-2.0s** | ⚡ |

### Speed Optimizations Applied

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Wake word check interval | 2.0s | 0.5s | 4x faster |
| Wake word loop sleep | 1.0s | 0.25s | 4x faster |
| Silence detection | 0.5s | 0.3s | 40% faster |
| LLM device | NPU (2.3s) | CPU (0.2s) | **12x faster!** |
| TTS data transfer | File I/O | In-memory | ~50ms saved |
| Audio queue polling | 100ms | 10ms | 10x faster |
| Token decoding | Full re-decode | Incremental | ~2x faster |

## Data Flow

### Optimized In-Memory Pipeline

All audio data flows in-memory with **zero file I/O** in the critical path:

```
🎤 Mic → numpy array (16kHz float32)
    ↓
📝 STT → text string
    ↓
🧠 LLM → streaming tokens → sentence buffer
    ↓
🔊 TTS → numpy array (22kHz float32) [NO FILE!]
    ↓
🔈 sounddevice → speakers
```

### Key Optimization: No File I/O

Previously:
```
TTS → WAV file → read file → numpy → sounddevice  (adds ~50ms)
```

Now:
```
TTS → numpy array → sounddevice (direct, ~0ms delay)
```

## Device Recommendations

### For Fastest Response (Recommended)

```yaml
llm:
  device: "CPU"  # ~200ms warmup vs 2.3s on NPU
audio:
  stt:
    device: "auto"  # NPU for low power
```

**Result:** ~1.5-2.0s response time

### For Lowest Power

```yaml
llm:
  device: "NPU"  # Slow but efficient
audio:
  stt:
    device: "NPU"
```

**Result:** ~3.5-4.5s response time, but only 3-5W active power

### Device Comparison

| Device | LLM Warmup | Power (Active) | Best For |
|--------|------------|----------------|----------|
| CPU | ~200ms | ~15-30W | **Fast response** |
| NPU | ~2,300ms | ~3-5W | Battery/power savings |
| GPU.0 (Intel) | ~300ms | ~15-25W | Balance |
| GPU.1 (NVIDIA) | ~100ms | ~50-100W | Fastest, high power |

## Streaming Architecture

### Parallel TTS Processing

While LLM generates tokens, TTS synthesizes sentences in parallel:

```
Time →
LLM:  [token][token][token][SENTENCE 1][token][token][SENTENCE 2]...
                          ↓                        ↓
TTS:                [synthesize S1]         [synthesize S2]
                          ↓                        ↓
Audio:              [PLAY S1]              [PLAY S2]
```

This overlap means audio starts playing **before the LLM finishes** generating the full response.

### Sentence Buffer

Tokens are buffered until a complete sentence is detected:

- Minimum length: 3 characters
- Sentence end characters: `.!?;,:`
- Flush remaining on LLM complete

## Power Consumption

### By Component

| Component | Device | Idle | Active | Notes |
|-----------|--------|------|--------|-------|
| Wake Word | NPU | ~1W | ~2-3W | Always listening |
| STT (Whisper) | NPU | 0W | ~2-3W | Only during speech |
| LLM (Qwen2.5-1.5B) | CPU | 0W | ~15-25W | Only during inference |
| TTS (Piper) | CPU | 0W | ~5-8W | Only during synthesis |
| **Total Active** | | | **~25-35W** | During response |
| **Idle** | | | **~1-3W** | Waiting for wake word |

### Comparison

| Solution | Response Time | Power (Active) |
|----------|--------------|----------------|
| **Xenith (CPU LLM)** | ~1.5-2s | ~25-35W |
| **Xenith (NPU LLM)** | ~3.5-4.5s | ~10-15W |
| Cloud API | ~1-2s | Network only |
| GPU (RTX 4060) | ~0.5-1s | ~80-120W |

## Supported Models

### Speech-to-Text (STT)

| Model | Size | Device | Power | Accuracy |
|-------|------|--------|-------|----------|
| whisper-tiny | 39MB | NPU | ~1W | Basic |
| whisper-base | 74MB | NPU | ~2W | Good ⭐ |
| whisper-small | 244MB | NPU | ~3W | Better |
| whisper-medium | 769MB | NPU/GPU | ~5W | High |

### Language Models (LLM)

| Model | Parameters | NPU Compatible | Power (CPU) | Quality |
|-------|------------|----------------|-------------|---------|
| qwen3-0.6b | 0.6B | ✅ | ~10W | Basic |
| tinyllama-1.1b | 1.1B | ✅ | ~12W | Good |
| qwen2.5-1.5b | 1.5B | ✅ | ~15W | Better ⭐ |
| phi-3-mini | 3.8B | ✅ | ~25W | Excellent |
| mistral-7b | 7B | ❌ | ~40W | Best |

### Text-to-Speech (TTS)

| Backend | Speed | Quality | Device |
|---------|-------|---------|--------|
| Piper (ryan-high) | ~100ms | High ⭐ | CPU |
| Piper (lessac-medium) | ~80ms | Medium | CPU |
| MeloTTS | ~300ms | Highest | CPU + NPU BERT |
| espeak-ng | ~20ms | Low | CPU |

## Configuration

All settings in `config/config.yaml`:

```yaml
llm:
  # Device: CPU for speed, NPU for power
  device: "CPU"
  model: "qwen2.5-1.5b"
  max_tokens: 256
  temperature: 0.7

audio:
  stt:
    backend: "auto"  # openvino preferred
    device: "auto"   # NPU → GPU → CPU
    model: "base"

  tts:
    backend: "auto"  # piper preferred
    voice: "EN-Default"  # Ryan male voice
```

## Hardware Requirements

### Minimum
- Intel Core Ultra (Meteor Lake) or newer with NPU
- 8GB RAM
- 5GB disk space for models

### Recommended
- Intel Core Ultra 7/9 (Lunar Lake)
- 16GB RAM
- 10GB disk space

## Background NPU Warmup

When using NPU for LLM, a background thread periodically warms the model to reduce cold-start latency:

```python
# Warmup runs every 20 seconds when idle
# Helps keep NPU caches warm (marginal improvement)
```

Note: NPU warmup has limited effectiveness due to per-query recompilation. **CPU is still recommended for fast response.**

## Metrics & Instrumentation

The pipeline includes built-in timing instrumentation:

```
============================================================
📊 PIPELINE TIMING METRICS
============================================================

⚡ Time to first audio: 1.127s

--- Stage Latencies ---
  🟡   └─ LLM warmup: 0.184s
  🟡 First token → First audio: 0.127s
  🟢   └─ TTS first sentence: 0.093s
  🟢   └─ Audio queue delay: 0.000s
  🟠 LLM total generation: 0.680s
============================================================
```

## See Also

- [README](../README.md) - Project overview
- [Intel NPU Setup Guide](INTEL_NPU_SETUP.md)
- [STT Backends Reference](STT_BACKENDS.md)
- [TTS Backends Reference](TTS_BACKENDS.md)
- [LLM Backends Reference](LLM_BACKENDS.md)
- [Scripts Reference](SCRIPTS.md)
