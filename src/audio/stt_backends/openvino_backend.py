"""OpenVINO Whisper backend for ultra-low power Speech-to-Text

Uses Intel OpenVINO with Whisper models optimized for Intel hardware.
Supports NPU (Neural Processing Unit), GPU, and CPU acceleration.

The NPU in Intel Core Ultra processors provides extremely efficient
inference with minimal power consumption - ideal for always-on voice input.
"""

import importlib
import os
import re
import subprocess  # nosec B404 - fixed argv, no shell
import traceback
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from . import _common
from .base import STTBackend, STTResult

# Hugging Face model id ("name" or "org/name"); must not start with "-" so it
# can never be parsed as an optimum-cli option.
HF_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.-]*(/[A-Za-z0-9][\w.-]*)?")


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a CLI tool from an argv list (never a shell), capturing output."""
    return subprocess.run(args, capture_output=True, text=True, check=False)  # nosec


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
    MODEL_MAPPING: ClassVar[dict[str, str]] = {
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
        self._pipeline: Any = None
        self._model_path = None
        self._ov_genai: Any = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenVINO GenAI is available"""
        return _common.module_available("openvino_genai")

    @classmethod
    def get_device_info(cls) -> dict:
        """Get information about available OpenVINO devices"""
        info: dict[str, Any] = {
            "devices": [],
            "default": "CPU",
            "npu_available": False,
            "gpu_available": False,
        }

        try:
            ov = importlib.import_module("openvino")
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
                except _common.RUNTIME_ERRORS:
                    info["npu_name"] = "Intel NPU"

            # Check for GPU (Intel)
            for dev in available_devices:
                if "GPU" in dev:
                    info["gpu_available"] = True
                    if not info["npu_available"]:
                        info["default"] = dev
                    try:
                        info["gpu_name"] = core.get_property(dev, "FULL_DEVICE_NAME")
                    except _common.RUNTIME_ERRORS as e:
                        # The GPU is still usable; only its display name is unknown.
                        info["gpu_name_error"] = str(e)
                    break

        except ImportError:
            info["error"] = "OpenVINO not installed"
        except _common.RUNTIME_ERRORS as e:
            info["error"] = str(e)

        return info

    def _get_model_path(self) -> Path:
        """Get or create the path to the converted OpenVINO model"""
        # Use XDG cache directory or ~/.cache
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        models_root = cache_dir / "xenith" / "openvino_models"
        return models_root / f"whisper-{self.model_name}"

    def _convert_model(self) -> bool:
        """Convert Whisper model to OpenVINO format if needed"""
        model_path = self._get_model_path()

        # Check if already converted
        if (model_path / "openvino_decoder_model.xml").exists():
            print(f"[OPENVINO] Found cached model at {model_path}")
            return True

        print("[OPENVINO] Converting model to OpenVINO format...")
        print("[OPENVINO] This only needs to be done once. Model will be cached at:")
        print(f"[OPENVINO]   {model_path}")

        try:
            # Create directory
            model_path.mkdir(parents=True, exist_ok=True)

            # Get Hugging Face model ID
            hf_model_id = self.MODEL_MAPPING.get(self.model_name)
            if not hf_model_id:
                # Assume it's already a HF model ID
                hf_model_id = self.model_name

            if not HF_MODEL_ID.fullmatch(hf_model_id):
                print(f"[OPENVINO] Refusing invalid model id: {hf_model_id!r}")
                return False

            # Use optimum-cli to convert (argv list, no shell; the model id is
            # validated above and the output path is our own cache directory)
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
            result = _run_cli(cmd)

            if result.returncode != 0:
                print(f"[OPENVINO] Conversion failed: {result.stderr}")
                return False

            print("[OPENVINO] Model converted successfully!")
            return True

        except FileNotFoundError:
            print("[OPENVINO] optimum-cli not found. Install with:")
            print("[OPENVINO]   pip install optimum[openvino]")
            return False
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[OPENVINO] Model conversion error: {e}")
            traceback.print_exc()
            return False

    def load_model(self) -> bool:
        """Load the OpenVINO Whisper model"""
        try:
            ov_genai = importlib.import_module("openvino_genai")
            self._ov_genai = ov_genai

            # Check and convert model if needed
            model_path = self._get_model_path()
            if not self._convert_model():
                return False

            print(f"[OPENVINO] Loading model on {self.device}...")

            # Create the Whisper pipeline
            self._pipeline = ov_genai.WhisperPipeline(
                str(model_path),
                device=self.device,
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
        except _common.RUNTIME_ERRORS as e:
            print(f"[OPENVINO] Failed to load model: {e}")
            traceback.print_exc()

            # Provide helpful error messages
            if "NPU" in str(e).upper() or "npu" in str(e).lower():
                print("\n[OPENVINO] NPU Error Troubleshooting:")
                print("[OPENVINO] 1. Ensure Intel NPU driver is installed and updated")
                print(
                    "[OPENVINO] 2. Check driver version (need 32.0.100.3104 or newer)",
                )
                print("[OPENVINO] 3. Try: export DISABLE_OPENVINO_GENAI_NPU_L0=1")
                print("[OPENVINO] 4. Try device='GPU' or device='CPU' as fallback")

            return False

    def _transcribe(self, audio: np.ndarray, language: str, **kwargs) -> STTResult:
        """Transcribe audio using OpenVINO Whisper

        Args:
            audio: Audio data (float32, mono, 16kHz sample rate)
            language: Language code
            **kwargs: Additional options (max_new_tokens, etc.)
        """
        try:
            audio = _common.normalize_audio(audio)

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

        except _common.RUNTIME_ERRORS as e:
            print(f"[OPENVINO] Transcription error: {e}")
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
        self._processor: Any = None
        self._model: Any = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if optimum-intel is available"""
        return _common.module_available("optimum.intel")

    def load_model(self) -> bool:
        """Load model using Optimum Intel"""
        try:
            optimum_intel = importlib.import_module("optimum.intel")
            ov_model_cls = optimum_intel.OVModelForSpeechSeq2Seq
            auto_processor = importlib.import_module("transformers").AutoProcessor

            # Get model ID
            hf_model_id = OpenVINOBackend.MODEL_MAPPING.get(
                self.model_name,
                f"openai/whisper-{self.model_name}",
            )

            print(f"[OPTIMUM] Loading {hf_model_id} for {self.device}...")

            # Load processor
            self._processor = auto_processor.from_pretrained(hf_model_id)  # nosec B615

            # Load model with OpenVINO backend
            self._model = ov_model_cls.from_pretrained(
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
        except _common.RUNTIME_ERRORS as e:
            return _common.load_failed("OPTIMUM", e)

    def _transcribe(self, audio: np.ndarray, language: str, **kwargs) -> STTResult:
        """Transcribe audio using Optimum Intel"""
        try:
            # Process audio
            inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")

            # Generate
            generated_ids = self._model.generate(
                inputs["input_features"],
                max_new_tokens=kwargs.get("max_new_tokens", 256),
            )

            # Decode
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            text = decoded[0].strip()

            return STTResult(text=text, language=language, duration=len(audio) / 16000)

        except _common.RUNTIME_ERRORS as e:
            print(f"[OPTIMUM] Transcription error: {e}")
            traceback.print_exc()
            return STTResult(text="", language=language)
