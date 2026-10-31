"""MeloTTS backend - C++ OpenVINO-based TTS with NPU support"""

import glob
import importlib
import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import ClassVar

from .base import TTSBackend, TTSResult

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MELOTTS_DIR = PROJECT_ROOT / "vendor" / "MeloTTS.cpp"
MELOTTS_BIN = MELOTTS_DIR / "build" / "meloTTS_ov"
MELOTTS_MODELS = MELOTTS_DIR / "ov_models"


class MeloTTSBackend(TTSBackend):
    """MeloTTS backend using C++ OpenVINO implementation

    Supports running BERT preprocessing on NPU for low-power inference.
    TTS model runs on CPU/GPU (NPU not supported for TTS model).
    """

    # Available English voices
    VOICES: ClassVar[dict[str, str]] = {
        "EN-US": "American English",
        "EN-BR": "British English",
        "EN-INDIA": "Indian English",
        "EN-AU": "Australian English",
        "EN-Default": "Default English",
    }

    def __init__(
        self,
        voice: str = "EN-Default",
        device: str = "auto",
        language: str = "EN",
        use_npu_bert: bool = True,
        use_quantized: bool = True,
    ):
        """Initialize MeloTTS backend

        Args:
            voice: Voice to use (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default)
            device: Device for TTS model ("cpu" or "gpu", NPU not supported for TTS)
            language: Language ("EN" for English, "ZH" for Chinese)
            use_npu_bert: Use NPU for BERT preprocessing (requires static model)
            use_quantized: Use INT8 quantized model (faster, slightly lower quality)
        """
        super().__init__(voice=voice, device=device, language=language)
        self.use_npu_bert = use_npu_bert
        self.use_quantized = use_quantized
        self._temp_dir: str | None = None

    def load(self) -> bool:
        """Verify MeloTTS is available and ready"""
        if not self.is_available():
            print(f"[MeloTTS] Binary not found at {MELOTTS_BIN}")
            return False

        if not MELOTTS_MODELS.exists():
            print(f"[MeloTTS] Models not found at {MELOTTS_MODELS}")
            return False

        # Check for NPU static model if NPU is requested
        if self.use_npu_bert and self.language == "EN":
            static_model = MELOTTS_MODELS / "bert_EN_static_int8.xml"
            if not static_model.exists():
                print(f"[MeloTTS] NPU static model not found at {static_model}, using CPU for BERT")
                self.use_npu_bert = False

        self._is_loaded = True
        bert_device = "NPU" if self.use_npu_bert else "CPU"
        print(f"[MeloTTS] Loaded: voice={self.voice}, bert_device={bert_device}")
        return True

    def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
        speed: float = 1.0,
        **kwargs,
    ) -> TTSResult:
        """Synthesize speech from text using MeloTTS

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio (if None, uses temp file)
            speed: Speech speed multiplier
            **kwargs: Additional options

        Returns:
            TTSResult with audio file path
        """
        if not self._is_loaded and not self.load():
            raise RuntimeError("MeloTTS failed to load")

        # Create temp dir for input/output
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="melotts_")

        # Write input text to file (MeloTTS requires file input)
        input_file = Path(self._temp_dir) / "input.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(text)

        # Determine output path
        if output_path is None:
            output_base = Path(self._temp_dir) / "output"
        else:
            output_base = output_path.with_suffix("")  # Remove extension

        cmd = self._build_command(input_file, output_base, speed)

        # Run synthesis
        try:
            result = subprocess.run(  # nosec B603 # nosemgrep
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                check=False,
            )

            if result.returncode != 0:
                print(f"[MeloTTS] Error: {result.stderr}")
                raise RuntimeError(f"MeloTTS failed: {result.stderr}")

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MeloTTS synthesis timed out") from exc

        expected_file = self._find_output(output_base)

        # Move to requested output path if specified
        final_path = expected_file
        if output_path and output_path != expected_file:
            shutil.copy2(expected_file, output_path)
            final_path = output_path

        return TTSResult(
            audio_path=final_path,
            sample_rate=44100,  # MeloTTS outputs 44100 Hz
            voice=self.voice,
        )

    def _build_command(self, input_file: Path, output_base: Path, speed: float) -> list[str]:
        """Build the meloTTS_ov command line"""
        cmd = [
            str(MELOTTS_BIN),
            "--model_dir",
            str(MELOTTS_MODELS),
            "--input_file",
            str(input_file),
            "--output_filename",
            str(output_base),
            "--language",
            self.language,
            "--speed",
            str(speed),
        ]

        # Device settings
        if self.device in ("gpu", "GPU"):
            cmd.extend(["--tts_device", "GPU"])
        else:
            cmd.extend(["--tts_device", "CPU"])

        # BERT on NPU if available
        if self.use_npu_bert:
            cmd.extend(["--bert_device", "NPU"])

        # Quantization
        if not self.use_quantized:
            cmd.extend(["--quantize", "false"])

        # Specific speaker (much faster than generating all speakers)
        if self.voice:
            cmd.extend(["--speaker", self.voice])

        # Disable noise filter for faster synthesis (~1s savings)
        cmd.extend(["--disable_nf", "true"])

        return cmd

    def _find_output(self, output_base: Path) -> Path:
        """Find the generated audio file for the requested voice"""
        expected_file = Path(f"{output_base}_{self.voice}.wav")
        if expected_file.exists():
            return expected_file
        # Fall back to any generated file
        pattern = f"{output_base}_*.wav"
        files = glob.glob(pattern)
        if not files:
            raise RuntimeError(f"No output audio file found matching {pattern}")
        return Path(files[0])

    def get_available_voices(self) -> list[str]:
        """Get list of available voices"""
        return list(self.VOICES.keys())

    def unload(self) -> None:
        """Clean up temp files"""
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        super().unload()

    @classmethod
    def is_available(cls) -> bool:
        """Check if MeloTTS binary is available"""
        return MELOTTS_BIN.exists() and os.access(MELOTTS_BIN, os.X_OK)

    @classmethod
    def get_device_info(cls) -> dict:
        """Get available devices for MeloTTS"""
        devices = ["cpu"]

        # Check for GPU via OpenVINO
        try:
            available = importlib.import_module("openvino").Core().available_devices
            if any("GPU" in d for d in available):
                devices.append("gpu")
            if "NPU" in available:
                devices.append("npu")  # For BERT only
        except ImportError:
            pass

        return {
            "devices": devices,
            "default": "cpu",
            "notes": {
                "npu": "NPU only used for BERT preprocessing, TTS runs on CPU/GPU",
                "gpu": "Intel Arc or compatible GPU",
            },
        }
