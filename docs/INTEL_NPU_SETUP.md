# Intel NPU Setup for Ultra-Low Power Voice-to-Text

This guide documents how to configure Xenith to use the Intel Neural Processing Unit (NPU) for ultra-low power speech-to-text processing.

## Overview

Xenith supports multiple Speech-to-Text (STT) backends:

| Backend | Device | Power Draw | Best For |
|---------|--------|------------|----------|
| **OpenVINO** | Intel NPU | ~1-3W | Always-on wake word detection |
| **OpenVINO** | Intel iGPU | ~5-15W | Balanced performance/power |
| **OpenVINO** | CPU | ~15-35W | Fallback |
| **Whisper** | NVIDIA CUDA | ~50-100W | Fast batch processing |
| **Whisper** | CPU | ~15-35W | No GPU fallback |

The Intel NPU (marketed as "Intel AI Boost") is available on Intel Core Ultra processors (Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake). It provides extremely efficient AI inference, making it ideal for always-on voice input.

## Supported Hardware

- **Intel Core Ultra 5/7/9** (Meteor Lake) - 1st gen NPU
- **Intel Core Ultra 200** series (Arrow Lake) - 2nd gen NPU  
- **Intel Core Ultra 200V** series (Lunar Lake) - 3rd gen NPU

Check if your system has an NPU:
```bash
# Check for NPU device
ls /dev/accel/accel0

# Check kernel module
lsmod | grep intel_vpu
```

## Quick Start

If you just want to get NPU working quickly:

```bash
# 1. Install OpenVINO packages
pip install openvino openvino-genai

# 2. Add yourself to render group (then log out/in)
sudo usermod -aG render $USER

# 3. Check if NPU is detected
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

If NPU appears in the list, you're done! If not, follow the full setup below.

---

## Full Setup Guide for Fedora

### Prerequisites

```bash
# Install build tools
sudo dnf install -y git git-lfs cmake ninja-build gcc gcc-c++ make python3

# Install Level Zero runtime (required for NPU)
sudo dnf install -y intel-level-zero intel-compute-runtime oneapi-level-zero
```

### Step 1: Add User to Render Group

The NPU device requires `render` group access:

```bash
sudo usermod -aG render $USER
```

**Important:** Log out and back in for this to take effect. To test immediately without logging out:
```bash
sg render -c 'your_command_here'
```

### Step 2: Build Intel NPU Driver

The NPU user-space driver is not packaged for Fedora, so we build from source:

```bash
# Clone the repository
cd ~/src
git clone https://github.com/intel/linux-npu-driver.git
cd linux-npu-driver

# Initialize submodules
git submodule update --init --recursive

# Configure (without compiler - we'll use pre-built)
cmake -B build -S . -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev

# Build
cmake --build build --parallel $(nproc)

# Install
sudo cmake --install build

# Update library cache
echo "/usr/local/lib64" | sudo tee /etc/ld.so.conf.d/npu-driver.conf
sudo ldconfig

# Reload kernel module
sudo rmmod intel_vpu
sudo modprobe intel_vpu
```

### Step 3: Install NPU Compiler

The NPU compiler is required to compile models for the NPU. Extract it from Intel's Ubuntu package:

```bash
# Download Intel's release package
cd /tmp
wget https://github.com/intel/linux-npu-driver/releases/download/v1.28.0/linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz

# Extract
mkdir -p ~/npu-extract && cd ~/npu-extract
tar -xzf /tmp/linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz

# Extract compiler from deb package
ar x intel-driver-compiler-npu_*.deb
tar -xzf data.tar.gz

# Install compiler library
sudo cp usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so /usr/local/lib64/
sudo ldconfig

# Cleanup
rm -rf ~/npu-extract /tmp/linux-npu-driver*.tar.gz
```

### Step 4: Install Python Packages

```bash
pip install openvino openvino-genai optimum[openvino]
```

### Step 5: Verify NPU Detection

```bash
python -c "
import openvino as ov
core = ov.Core()
print('Available devices:', core.available_devices)
for dev in core.available_devices:
    try:
        name = core.get_property(dev, 'FULL_DEVICE_NAME')
        print(f'  {dev}: {name}')
    except:
        pass
"
```

Expected output:
```
Available devices: ['CPU', 'GPU.0', 'GPU.1', 'NPU']
  CPU: Intel(R) Core(TM) Ultra 7 165H
  GPU.0: Intel(R) Arc(TM) Graphics (iGPU)
  GPU.1: NVIDIA GeForce RTX 4060 Laptop GPU (dGPU)
  NPU: Intel(R) AI Boost
```

### Step 6: Download Whisper Model

```bash
python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.path.expanduser('~/.cache/xenith/openvino_models')
os.makedirs(cache_dir, exist_ok=True)

# Download OpenVINO-optimized Whisper
snapshot_download(
    repo_id='OpenVINO/whisper-base-fp16-ov',
    local_dir=os.path.join(cache_dir, 'whisper-base')
)
print('Model downloaded successfully!')
"
```

### Step 7: Test NPU Inference

```bash
python -c "
import openvino_genai as ov_genai
import numpy as np
import os

model_path = os.path.expanduser('~/.cache/xenith/openvino_models/whisper-base')
pipe = ov_genai.WhisperPipeline(model_path, device='NPU')
print('WhisperPipeline created on NPU!')

# Test with silent audio
audio = (np.random.randn(32000) * 0.001).astype(np.float32).tolist()
result = pipe.generate(audio, max_new_tokens=50)
print('NPU inference successful!')
"
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
audio:
  stt:
    # Backend: "auto", "whisper", "openvino"
    backend: "openvino"
    
    # Device options:
    #   - "NPU": Intel AI Boost (ultra-low power)
    #   - "GPU.0": Intel iGPU (balanced)
    #   - "GPU.1": NVIDIA dGPU (if applicable)
    #   - "CPU": CPU fallback
    #   - "auto": Let OpenVINO choose
    device: "NPU"
    
    # Model size: "tiny", "base", "small", "medium", "large"
    model: "base"
```

### Auto-Selection Behavior

When `backend: "auto"` and `device: "auto"`, Xenith selects the best available option prioritizing power efficiency:

| Priority | Device | Backend | Power Draw | Notes |
|----------|--------|---------|------------|-------|
| 1st | Intel NPU | OpenVINO | ~1-3W | Ultra-low power, ideal for always-on |
| 2nd | Intel GPU (GPU.0) | OpenVINO | ~5-15W | Integrated graphics, efficient |
| 3rd | NVIDIA GPU | Whisper/CUDA | ~50-100W | High performance, high power |
| 4th | CPU | OpenVINO | ~15-35W | Fallback |

This priority ensures minimal power consumption for always-on voice detection.

---

## Troubleshooting

### NPU Not Detected

**Check kernel module:**
```bash
lsmod | grep intel_vpu
# If not loaded:
sudo modprobe intel_vpu
```

**Check device exists:**
```bash
ls -la /dev/accel/accel0
```

**Check permissions:**
```bash
groups | grep render
# If not in group:
sudo usermod -aG render $USER
# Then log out and back in
```

**Check dmesg for errors:**
```bash
sudo dmesg | grep -i "vpu\|npu\|intel_vpu"
```

### Compilation Errors (ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)

This means the NPU compiler is missing:

```bash
# Verify compiler is installed
ls /usr/local/lib64/libnpu_driver_compiler.so

# Check compiler version in OpenVINO
python -c "
import openvino as ov
print(ov.Core().get_property('NPU', 'NPU_COMPILER_VERSION'))
"
# Should return a number > 0
```

### Out of Memory Errors

Try disabling Level0 memory allocation:
```bash
export DISABLE_OPENVINO_GENAI_NPU_L0=1
```

### Model Compatibility

Some models may not be compatible with NPU. Use INT8 quantized models for best results:

```bash
# Download INT8 model
optimum-cli export openvino \
  --model openai/whisper-base \
  --weight-format int8 \
  ~/.cache/xenith/openvino_models/whisper-base-int8
```

---

## Testing

Run the Xenith STT backend test:

```bash
cd ~/src/Xenith
python test_stt_backends.py
```

Test specific backend:
```bash
python test_stt_backends.py --backend openvino --device NPU
```

---

## Files Installed

| File | Location | Description |
|------|----------|-------------|
| `libze_intel_npu.so` | `/usr/local/lib64/` | NPU user-space driver |
| `libnpu_driver_compiler.so` | `/usr/local/lib64/` | NPU model compiler |
| `npu-driver.conf` | `/etc/ld.so.conf.d/` | Library path config |
| `whisper-base/` | `~/.cache/xenith/openvino_models/` | Whisper model for NPU |

---

## Updating

To update the NPU driver:

```bash
cd ~/src/linux-npu-driver
git pull
git submodule update --recursive

cmake --build build --parallel $(nproc)
sudo cmake --install build
sudo ldconfig

sudo rmmod intel_vpu
sudo modprobe intel_vpu
```

---

## References

- [Intel NPU Driver GitHub](https://github.com/intel/linux-npu-driver)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO GenAI](https://openvinotoolkit.github.io/openvino.genai/)
- [Whisper on NPU Guide](https://docs.openvino.ai/nightly/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)

