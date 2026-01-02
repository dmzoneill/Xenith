"""OpenVINO Whisper backend for ultra-low power Speech-to-Text

Uses Intel OpenVINO with Whisper models optimized for Intel hardware.
Supports NPU (Neural Processing Unit), GPU, and CPU acceleration.

The NPU in Intel Core Ultra processors provides extremely efficient
inference with minimal power consumption - ideal for always-on voice input.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional

from .base import STTBackend, STTResult


class OpenVINOBackend(STTBackend):
    """OpenVINO Whisper backend for Intel hardware

    This backend uses Intel's OpenVINO toolkit to run Whisper models
    on Intel NPU, GPU, or CPU with optimized performance and power efficiency.

    Advantages:
        - Ultra-low power consumption on Intel NPU
        - Optimized for Intel hardware (Core Ultra, Arc GPUs)
        - Good accuracy with quantized models
        - Ideal for always-on voice detection

    Disadvantages:
        - Requires model conversion to OpenVINO format
        - Intel hardware required for NPU/GPU acceleration

    Device options:
        - "NPU": Intel Neural Processing Unit (lowest power, Core Ultra)
        - "GPU": Intel Arc/Iris GPU (faster than CPU, good efficiency)
        - "GPU.0": Specifically the Intel iGPU (recommended for power efficiency)
        - "GPU.1": Secondary GPU (often NVIDIA dGPU)
        - "CPU": Intel CPU (fallback, still optimized)
        - "AUTO": Let OpenVINO choose the best device
    """

    # Model name mapping to Hugging Face model IDs
    MODEL_MAPPING = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
    }

    def __init__(self, model_name: str = "base", device: str = "NPU"):
        """Initialize OpenVINO Whisper backend

        Args:
            model_name: Model size - "tiny", "base", "small", "medium", "large"
            device: "NPU", "GPU", "CPU", or "AUTO"
        """
        super().__init__(model_name, device.upper())
        self._pipeline = None
        self._model_path = None
        self._ov_genai = None

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
        """Get information about available OpenVINO devices"""
        info = {
            "devices": [],
            "default": "CPU",
            "npu_available": False,
            "gpu_available": False,
        }

        try:
            import openvino as ov

            core = ov.Core()
            available_devices = core.available_devices

            info["devices"] = available_devices

            # Check for NPU
            if "NPU" in available_devices:
                info["npu_available"] = True
                info["default"] = "NPU"
                try:
                    npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                    info["npu_name"] = npu_name
                except:
                    info["npu_name"] = "Intel NPU"

            # Check for GPU (Intel)
            for dev in available_devices:
                if "GPU" in dev:
                    info["gpu_available"] = True
                    if not info["npu_available"]:
                        info["default"] = dev
                    try:
                        gpu_name = core.get_property(dev, "FULL_DEVICE_NAME")
                        info["gpu_name"] = gpu_name
                    except:
                        pass
                    break

        except ImportError:
            info["error"] = "OpenVINO not installed"
        except Exception as e:
            info["error"] = str(e)

        return info

    def _get_model_path(self) -> Path:
        """Get or create the path to the converted OpenVINO model"""
        # Use XDG cache directory or ~/.cache
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        model_dir = (
            cache_dir / "xenith" / "openvino_models" / f"whisper-{self.model_name}"
        )
        return model_dir

    def _convert_model(self) -> bool:
        """Convert Whisper model to OpenVINO format if needed"""
        model_path = self._get_model_path()

        # Check if already converted
        if (model_path / "openvino_decoder_model.xml").exists():
            print(f"[OPENVINO] Found cached model at {model_path}")
            return True

        print(f"[OPENVINO] Converting model to OpenVINO format...")
        print(f"[OPENVINO] This only needs to be done once. Model will be cached at:")
        print(f"[OPENVINO]   {model_path}")

        try:
            # Create directory
            model_path.mkdir(parents=True, exist_ok=True)

            # Get Hugging Face model ID
            hf_model_id = self.MODEL_MAPPING.get(self.model_name)
            if not hf_model_id:
                # Assume it's already a HF model ID
                hf_model_id = self.model_name

            # Use optimum-cli to convert
            import subprocess

            cmd = [
                "optimum-cli",
                "export",
                "openvino",
                "--trust-remote-code",
                "--model",
                hf_model_id,
                str(model_path),
            ]

            print(f"[OPENVINO] Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"[OPENVINO] Conversion failed: {result.stderr}")
                return False

            print(f"[OPENVINO] Model converted successfully!")
            return True

        except FileNotFoundError:
            print("[OPENVINO] optimum-cli not found. Install with:")
            print("[OPENVINO]   pip install optimum[openvino]")
            return False
        except Exception as e:
            print(f"[OPENVINO] Model conversion error: {e}")
            import traceback

            traceback.print_exc()
            return False

    def load_model(self) -> bool:
        """Load the OpenVINO Whisper model"""
        try:
            import openvino_genai as ov_genai

            self._ov_genai = ov_genai

            # Check and convert model if needed
            model_path = self._get_model_path()
            if not self._convert_model():
                return False

            # Check for NPU driver issues
            if self.device == "NPU":
                # Set environment variable to avoid potential L0 memory issues
                if os.environ.get("DISABLE_OPENVINO_GENAI_NPU_L0") is None:
                    # Only set if user hasn't explicitly configured it
                    # This can help with some NPU driver versions
                    pass  # Let OpenVINO handle it by default

            print(f"[OPENVINO] Loading model on {self.device}...")

            # Create the Whisper pipeline
            self._pipeline = ov_genai.WhisperPipeline(
                str(model_path), device=self.device
            )

            self._is_loaded = True
            print(f"[OPENVINO] Model loaded successfully on {self.device}")

            # Print device info
            device_info = self.get_device_info()
            if self.device == "NPU" and device_info.get("npu_name"):
                print(f"[OPENVINO] NPU: {device_info['npu_name']}")
            elif self.device == "GPU" and device_info.get("gpu_name"):
                print(f"[OPENVINO] GPU: {device_info['gpu_name']}")

            return True

        except ImportError as e:
            print(f"[OPENVINO] Import error: {e}")
            print("[OPENVINO] Install with:")
            print("[OPENVINO]   pip install openvino openvino-genai optimum[openvino]")
            return False
        except Exception as e:
            print(f"[OPENVINO] Failed to load model: {e}")
            import traceback

            traceback.print_exc()

            # Provide helpful error messages
            if "NPU" in str(e).upper() or "npu" in str(e).lower():
                print("\n[OPENVINO] NPU Error Troubleshooting:")
                print("[OPENVINO] 1. Ensure Intel NPU driver is installed and updated")
                print(
                    "[OPENVINO] 2. Check driver version (need 32.0.100.3104 or newer)"
                )
                print("[OPENVINO] 3. Try: export DISABLE_OPENVINO_GENAI_NPU_L0=1")
                print("[OPENVINO] 4. Try device='GPU' or device='CPU' as fallback")

            return False

    def transcribe(
        self, audio: np.ndarray, language: str = "en", **kwargs
    ) -> STTResult:
        """Transcribe audio using OpenVINO Whisper

        Args:
            audio: Audio data (float32, mono, 16kHz sample rate)
            language: Language code
            **kwargs: Additional options (max_new_tokens, etc.)
        """
        if not self._is_loaded:
            if not self.load_model():
                return STTResult(text="", language=language)

        try:
            # Ensure audio is the right format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Convert to list (required by OpenVINO GenAI)
            audio_list = audio.tolist()

            # Generate with options
            max_new_tokens = kwargs.get("max_new_tokens", 256)

            result = self._pipeline.generate(
                audio_list,
                max_new_tokens=max_new_tokens,
                # Note: language parameter may vary by openvino-genai version
            )

            # Handle result - could be string or object depending on version
            if isinstance(result, str):
                text = result.strip()
            else:
                text = str(result).strip()

            return STTResult(
                text=text,
                language=language,
                duration=len(audio) / 16000,  # Assuming 16kHz
            )

        except Exception as e:
            print(f"[OPENVINO] Transcription error: {e}")
            import traceback

            traceback.print_exc()
            return STTResult(text="", language=language)

    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        self._is_loaded = False
        print("[OPENVINO] Model unloaded")


class OpenVINOOptimumBackend(STTBackend):
    """Alternative OpenVINO backend using Hugging Face Optimum

    This backend uses optimum-intel for a more Hugging Face-like API.
    Good alternative if openvino-genai has issues.
    """

    def __init__(self, model_name: str = "base", device: str = "NPU"):
        super().__init__(model_name, device.upper())
        self._processor = None
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if optimum-intel is available"""
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq

            return True
        except ImportError:
            return False

    def load_model(self) -> bool:
        """Load model using Optimum Intel"""
        try:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor

            # Get model ID
            hf_model_id = OpenVINOBackend.MODEL_MAPPING.get(
                self.model_name, f"openai/whisper-{self.model_name}"
            )

            print(f"[OPTIMUM] Loading {hf_model_id} for {self.device}...")

            # Load processor
            self._processor = AutoProcessor.from_pretrained(hf_model_id)

            # Load model with OpenVINO backend
            self._model = OVModelForSpeechSeq2Seq.from_pretrained(
                hf_model_id,
                export=True,  # Export to OpenVINO format
                device=self.device,
            )

            self._is_loaded = True
            print(f"[OPTIMUM] Model loaded on {self.device}")
            return True

        except ImportError as e:
            print(f"[OPTIMUM] Import error: {e}")
            print("[OPTIMUM] Install with: pip install optimum[openvino]")
            return False
        except Exception as e:
            print(f"[OPTIMUM] Failed to load model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def transcribe(
        self, audio: np.ndarray, language: str = "en", **kwargs
    ) -> STTResult:
        """Transcribe audio using Optimum Intel"""
        if not self._is_loaded:
            if not self.load_model():
                return STTResult(text="", language=language)

        try:
            # Process audio
            inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")

            # Generate
            generated_ids = self._model.generate(
                inputs["input_features"],
                max_new_tokens=kwargs.get("max_new_tokens", 256),
            )

            # Decode
            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            return STTResult(text=text, language=language, duration=len(audio) / 16000)

        except Exception as e:
            print(f"[OPTIMUM] Transcription error: {e}")
            import traceback

            traceback.print_exc()
            return STTResult(text="", language=language)
