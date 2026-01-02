# Xenith Scripts & Commands Reference

This document describes all scripts, test utilities, and Makefile targets available in the Xenith project.

## Quick Reference

| Command | Description |
|---------|-------------|
| `make run` | Run the main Xenith application |
| `make test` | Run all tests |
| `make install` | Install dependencies |
| `python test_stt_backends.py` | Test speech-to-text backends |
| `python test_tts_backends.py` | Test text-to-speech backends (MeloTTS) |
| `python test_llm_backends.py` | Test LLM backends (NPU inference) |
| `python test_voice_pipeline.py` | Test complete voice pipeline |
| `python test_audio_devices.py` | Test audio input devices |
| `python test_voice_states.py` | Test voice state UI transitions |

---

## Makefile Targets

The Makefile provides convenient shortcuts for common operations.

### Installation

```bash
# Full installation (system deps + pipenv + project)
make install

# Install system dependencies only (Python 3.13, cairo, etc.)
make install-system-deps

# Install in development mode
make install-dev

# Install OpenAI Whisper for CUDA-based STT
make install-whisper

# Install Python 3.13 if not available
make install-python313
```

### Running

```bash
# Run the main application
make run

# Run without audio device selection prompt
make run-quick
```

### Testing

```bash
# Run all tests
make test

# Run voice state UI test
make test-voice

# Run audio device test
make test-audio
```

### Development

```bash
# Activate pipenv shell
make shell

# Check if dependencies are installed
make check-deps

# Show pipenv environment info
make env-info

# Run linter (pylint + flake8)
make lint

# Format code (black + isort)
make format

# Clean build artifacts and cache
make clean
```

---

## Test Scripts

### `test_stt_backends.py` - Speech-to-Text Backend Testing

Tests and diagnoses STT backend availability including OpenVINO NPU support.

**Usage:**
```bash
# Check all available backends
python test_stt_backends.py

# Test specific backend and device
python test_stt_backends.py --backend openvino --device NPU
python test_stt_backends.py --backend whisper --device cuda

# Test transcription with audio file
python test_stt_backends.py --test-audio sample.wav

# Show NPU setup instructions
python test_stt_backends.py --setup
```

**Options:**
| Option | Description |
|--------|-------------|
| `--backend` | Backend to test: `auto`, `whisper`, `openvino` |
| `--device` | Device: `auto`, `cpu`, `cuda`, `npu`, `gpu`, `GPU.0`, `GPU.1` |
| `--model` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--test-audio FILE` | Transcribe an audio file |
| `--setup` | Print Intel NPU setup instructions |

**Example Output:**
```
============================================================
Checking STT Backend Availability
============================================================

1. OpenAI Whisper (PyTorch/CUDA)
----------------------------------------
   ✓ whisper: installed
   ✓ torch: installed
   ✓ CUDA: NVIDIA GeForce RTX 4060 Laptop GPU

2. OpenVINO (Intel NPU/GPU/CPU)
----------------------------------------
   ✓ openvino: installed (version 2025.4.1)
   ✓ Available devices: CPU, GPU.0, GPU.1, NPU
   ✓ NPU: Intel(R) AI Boost
   ✓ openvino_genai: installed
```

---

### `test_tts_backends.py` - Text-to-Speech Backend Testing

Tests TTS backend availability including MeloTTS with NPU acceleration.

**Usage:**
```bash
# Check all available backends
python test_tts_backends.py
```

**Features:**
1. Lists available TTS backends
2. Shows device support (CPU, GPU, NPU)
3. Tests synthesis with sample text
4. Generates audio file for verification

**Example Output:**
```
============================================================
TTS Backend Test
============================================================

✓ TTS backends module imported successfully
[TTS] MeloTTS backend available

=== TTS Backend Status ===

melotts:
  Available: True
  Devices: ['cpu', 'gpu', 'npu']
    npu: NPU only used for BERT preprocessing, TTS runs on CPU/GPU
    gpu: Intel Arc or compatible GPU

Available backends: ['melotts']

Testing synthesis...
  ✓ Audio generated: /tmp/tts_test_output.wav
    Voice: EN-Default
    Sample rate: 22050

Available voices:
  - EN-US
  - EN-BR
  - EN-INDIA
  - EN-AU
  - EN-Default
```

---

### `test_llm_backends.py` - Language Model Backend Testing

Tests LLM backends for NPU inference.

**Usage:**
```bash
# Check available backends and models
python test_llm_backends.py

# Download a specific model
python test_llm_backends.py --download --model qwen2.5-1.5b

# Test inference on specific device
python test_llm_backends.py --model phi-3-mini --device npu --prompt "What is AI?"
```

**Options:**
| Option | Description |
|--------|-------------|
| `--model` | Model to test (default: qwen2.5-1.5b) |
| `--device` | Device: auto, cpu, gpu, npu (default: auto) |
| `--download` | Download model without testing |
| `--prompt` | Test prompt (default: "What is 2 + 2?") |

**Available Models (NPU-compatible):**
- `qwen3-0.6b` - Smallest (~1-2W)
- `tinyllama-1.1b` - Small chat (~2-3W)
- `qwen2.5-1.5b` - Recommended (~3-5W)
- `phi-3-mini` - High quality (~5-10W)

---

### `test_voice_pipeline.py` - Complete Voice Pipeline Testing

Tests the full STT → LLM → TTS pipeline.

**Usage:**
```bash
# Test with text input (skip STT)
python test_voice_pipeline.py --text "What is the weather?"

# Test with audio file (full pipeline)
python test_voice_pipeline.py --audio recording.wav

# Play output audio
python test_voice_pipeline.py --text "Hello" --play

# Use different LLM model
python test_voice_pipeline.py --llm-model phi-3-mini --text "Explain AI"
```

**Options:**
| Option | Description |
|--------|-------------|
| `--text` | Text to process (default: "What time is it?") |
| `--audio` | Audio file to transcribe and process |
| `--output` | Output audio path (default: /tmp/pipeline_output.wav) |
| `--llm-model` | LLM model to use (default: qwen2.5-1.5b) |
| `--play` | Play output audio with paplay |

**Example Output:**
```
Pipeline Status:
  STT: Whisper base on NPU
  LLM: qwen2.5-1.5b on NPU
  TTS: MeloTTS (BERT on NPU)

Processing: 'Hello, what can you help me with?'
Response: 'I'm here to assist you with questions or tasks.'
Audio: /tmp/pipeline_output.wav

Timing:
  STT: 0.00s
  LLM: 4.13s
  TTS: 7.17s
  Total: 11.30s
```

---

### `test_audio_devices.py` - Audio Input Device Testing

Lists and tests audio input devices for voice capture.

**Usage:**
```bash
python test_audio_devices.py
```

**Features:**
1. Lists all available audio input devices
2. Interactive device selection
3. Real-time audio level monitoring
4. Tests microphone input with visual feedback

**Example Output:**
```
======================================================================
Audio Device Test
======================================================================
Testing audio device listing...

Found 3 input device(s):
  [0] Built-in Audio Analog Stereo
      Channels: 2, Sample Rate: 44100 Hz
  [1] USB Microphone
      Channels: 1, Sample Rate: 48000 Hz
  [2] Webcam Microphone
      Channels: 2, Sample Rate: 44100 Hz

Select input device [0-2] (Enter for default, 'q' to quit):
```

---

### `test_voice_states.py` - Voice State UI Testing

Tests the plasma widget's visual state transitions during voice interaction.

**Usage:**
```bash
python test_voice_states.py
```

**States Demonstrated:**
| State | Color | Description |
|-------|-------|-------------|
| `idle` | Blue | Waiting for wake word |
| `listening` | Orange (pulsing) | Actively listening for command |
| `processing` | Red/Orange (intense) | Processing voice input |
| `responding` | Purple | Generating/playing response |

**What it Does:**
1. Launches the Xenith plasma widget
2. Cycles through all voice states automatically
3. Demonstrates visual feedback for each state
4. Optionally starts live voice input after demo

---

### `test_plasma.py` - Plasma Widget Testing

Tests the plasma widget's state changes and agent color integration.

**Usage:**
```bash
python test_plasma.py
```

**Features:**
- Tests state changes every 3 seconds
- Tests agent color changes every 4 seconds
- Visual verification of widget animations

---

### `test_simple.py` - Minimal GUI Test

Simple test to verify the plasma widget launches correctly.

**Usage:**
```bash
python test_simple.py
```

**Purpose:**
- Verify GTK4/Adwaita environment is working
- Confirm widget appears in top-right corner
- Basic state change testing

---

## Shell Scripts

### `install-python313.sh` - Python 3.13 Installation

Installs Python 3.13 for the project.

**Usage:**
```bash
./install-python313.sh
# or
make install-python313
```

**Installation Methods (in order of preference):**
1. **Fedora DNF:** `sudo dnf install python3.13`
2. **pyenv:** Downloads and compiles Python 3.13
3. **Fallback:** Installs pyenv first, then Python 3.13

---

## Running the Main Application

### Standard Run
```bash
make run
# or
pipenv run python -m src.main
```

**Default STT Priority (auto mode):**
1. Intel NPU (~1-3W) - ultra-low power
2. Intel Arc GPU (~5-15W) - efficient
3. NVIDIA GPU (~50-100W) - high performance
4. CPU - fallback

### Direct Python Run (in pipenv shell)
```bash
pipenv shell
python -m src.main
```

### With Custom Config
```bash
# Edit config/config.yaml first, then:
make run
```

---

## Environment Setup

### First-Time Setup
```bash
# 1. Install everything
make install

# 2. Activate shell
make shell

# 3. Test audio devices
python test_audio_devices.py

# 4. Test STT backends
python test_stt_backends.py

# 5. Run the app
python -m src.main
```

### For Intel NPU Users
```bash
# Follow the NPU setup guide first
cat docs/INTEL_NPU_SETUP.md

# Then test NPU
python test_stt_backends.py --backend openvino --device NPU
```

---

## Troubleshooting Scripts

### No Display Error
```bash
# If running remotely or headless:
export DISPLAY=:0
# or use X11 forwarding:
ssh -X user@host
```

### Audio Device Issues
```bash
# List ALSA devices
arecord -l

# Test with specific device
python test_audio_devices.py
```

### Permission Issues for NPU
```bash
# Add to render group
sudo usermod -aG render $USER

# Test with group (without logout)
sg render -c 'python test_stt_backends.py'
```

---

## See Also

- [Voice Pipeline Architecture](VOICE_PIPELINE.md) - Complete pipeline overview
- [Intel NPU Setup Guide](INTEL_NPU_SETUP.md) - Full NPU installation instructions
- [STT Backends Reference](STT_BACKENDS.md) - Speech-to-text backend API documentation
- [LLM Backends Reference](LLM_BACKENDS.md) - Language model backend API documentation
- [TTS Backends Reference](TTS_BACKENDS.md) - Text-to-speech backend API documentation

