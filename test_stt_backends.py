#!/usr/bin/env python3
"""Test script for STT backends

Run this to check which STT backends are available and test them.

Usage:
    python test_stt_backends.py
    python test_stt_backends.py --backend openvino --device NPU
    python test_stt_backends.py --test-audio sample.wav
"""

import argparse
import sys
import numpy as np


def check_backends():
    """Check which backends are available"""
    print("\n" + "=" * 60)
    print("Checking STT Backend Availability")
    print("=" * 60)

    # Check OpenAI Whisper
    print("\n1. OpenAI Whisper (PyTorch/CUDA)")
    print("-" * 40)
    try:
        import whisper
        import torch

        print("   ✓ whisper: installed")
        print("   ✓ torch: installed")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ CUDA version: {torch.version.cuda}")
        else:
            print("   ⚠ CUDA: not available (CPU only)")
    except ImportError as e:
        print(f"   ✗ Not installed: {e}")
        print("   Install with: pip install openai-whisper torch")

    # Check OpenVINO
    print("\n2. OpenVINO (Intel NPU/GPU/CPU)")
    print("-" * 40)
    try:
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        print(f"   ✓ openvino: installed (version {ov.__version__})")
        print(f"   ✓ Available devices: {', '.join(devices)}")

        # Check for NPU
        if "NPU" in devices:
            try:
                npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                print(f"   ✓ NPU: {npu_name}")
            except:
                print("   ✓ NPU: available")
        else:
            print("   ⚠ NPU: not available")

        # Check for GPU
        for dev in devices:
            if "GPU" in dev:
                try:
                    gpu_name = core.get_property(dev, "FULL_DEVICE_NAME")
                    print(f"   ✓ GPU: {gpu_name}")
                except:
                    print(f"   ✓ GPU: {dev}")
                break

    except ImportError as e:
        print(f"   ✗ openvino not installed: {e}")
        print("   Install with: pip install openvino")

    # Check OpenVINO GenAI
    try:
        import openvino_genai

        print(f"   ✓ openvino_genai: installed")
    except ImportError as e:
        print(f"   ⚠ openvino_genai not installed: {e}")
        print("   Install with: pip install openvino-genai")

    # Check optimum-intel
    print("\n3. Optimum Intel (Alternative OpenVINO backend)")
    print("-" * 40)
    try:
        from optimum.intel import OVModelForSpeechSeq2Seq

        print("   ✓ optimum-intel: installed")
    except ImportError as e:
        print(f"   ⚠ Not installed: {e}")
        print("   Install with: pip install optimum[openvino]")


def test_backend(backend_name: str, device: str, model: str = "base"):
    """Test a specific backend with sample audio"""
    print("\n" + "=" * 60)
    print(f"Testing Backend: {backend_name} on {device}")
    print("=" * 60)

    try:
        from src.audio.stt_backends import get_stt_backend

        backend = get_stt_backend(backend_name, model_name=model, device=device)

        if backend is None:
            print("✗ Failed to get backend")
            return False

        print(f"\nBackend: {backend}")

        # Load model
        print("\nLoading model...")
        if not backend.load_model():
            print("✗ Failed to load model")
            return False

        print("✓ Model loaded successfully!")

        # Test with synthetic audio (2 seconds of silence with a beep)
        print("\nTesting transcription with synthetic audio...")
        sample_rate = 16000
        duration = 2.0

        # Generate a simple audio signal
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        # Add some noise to simulate audio
        audio = np.random.randn(len(t)).astype(np.float32) * 0.01

        result = backend.transcribe(audio, language="en")
        print(f"✓ Transcription result: '{result.text}'")
        print(f"  (Expected: empty or noise - this is just a test of the pipeline)")

        # Cleanup
        backend.unload_model()
        print("\n✓ Backend test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error testing backend: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_audio_file(audio_path: str, backend_name: str = "auto", device: str = "auto"):
    """Test transcription with an actual audio file"""
    print("\n" + "=" * 60)
    print(f"Testing with audio file: {audio_path}")
    print("=" * 60)

    try:
        import librosa

        # Load audio file
        print(f"\nLoading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"✓ Loaded: {len(audio)/sr:.2f} seconds at {sr}Hz")

        from src.audio.stt_backends import get_stt_backend

        backend = get_stt_backend(backend_name, device=device)
        if backend is None:
            print("✗ Failed to get backend")
            return False

        if not backend.load_model():
            print("✗ Failed to load model")
            return False

        print(f"\nTranscribing with {backend}...")
        result = backend.transcribe(audio, language="en")

        print("\n" + "-" * 40)
        print("TRANSCRIPTION:")
        print("-" * 40)
        print(result.text)
        print("-" * 40)

        backend.unload_model()
        return True

    except ImportError:
        print("✗ librosa not installed. Install with: pip install librosa")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_setup_instructions():
    """Print setup instructions for OpenVINO NPU"""
    print("\n" + "=" * 60)
    print("OpenVINO NPU Setup Instructions")
    print("=" * 60)

    print(
        """
To use Intel NPU for ultra-low power voice-to-text:

1. INSTALL OPENVINO:
   pip install openvino openvino-genai optimum[openvino]

2. CHECK NPU DRIVER (Linux):
   - Intel NPU driver should be included in recent kernels (6.6+)
   - Check: ls /dev/accel*
   - If missing, update your kernel or install Intel NPU driver

3. CHECK NPU DRIVER (Windows):
   - Download Intel NPU Driver from Intel Download Center
   - Version 32.0.100.3104 or newer recommended

4. CONVERT MODEL (first run does this automatically):
   optimum-cli export openvino --model openai/whisper-base whisper-base

5. CONFIGURE XENITH:
   Edit config/config.yaml:
   
   audio:
     stt:
       backend: "openvino"   # Use OpenVINO backend
       device: "NPU"         # Use Intel NPU
       model: "base"         # Model size

6. TROUBLESHOOTING:
   - If you get memory errors, try:
     export DISABLE_OPENVINO_GENAI_NPU_L0=1
   - If NPU isn't detected, fall back to GPU or CPU:
     device: "GPU"  or  device: "CPU"
   - Check device availability:
     python -c "import openvino; print(openvino.Core().available_devices)"

POWER CONSUMPTION COMPARISON:
   - RTX 4060 (CUDA): ~50-100W during inference
   - Intel NPU:       ~1-5W during inference
   
   The NPU is ideal for always-on voice detection!
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Test STT backends")
    parser.add_argument(
        "--backend",
        choices=["whisper", "openvino", "auto"],
        default="auto",
        help="Backend to test",
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto, cpu, cuda, npu, gpu"
    )
    parser.add_argument(
        "--model", default="base", help="Model size: tiny, base, small, medium, large"
    )
    parser.add_argument(
        "--test-audio", metavar="FILE", help="Test transcription with an audio file"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Print OpenVINO NPU setup instructions"
    )

    args = parser.parse_args()

    if args.setup:
        print_setup_instructions()
        return 0

    # Always check backends first
    check_backends()

    # Print backend status from our module
    print("\n")
    try:
        from src.audio.stt_backends import print_backend_status

        print_backend_status()
    except ImportError as e:
        print(f"Could not import stt_backends: {e}")

    # Test specific backend if requested
    if args.backend != "auto" or args.device != "auto":
        test_backend(args.backend, args.device, args.model)

    # Test with audio file if provided
    if args.test_audio:
        test_audio_file(args.test_audio, args.backend, args.device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
