"""Integrated Voice Pipeline: STT → LLM → TTS

This module provides a complete voice processing pipeline that:
1. Converts speech to text (STT)
2. Processes text through an LLM
3. Converts response to speech (TTS)

Designed for low-power operation on Intel NPU.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import numpy as np

from .stt_backends import get_stt_backend, STTBackend
from .llm_backends import get_llm_backend, LLMBackend, LLMConfig
from .tts_backends import get_tts_backend, TTSBackend


@dataclass
class PipelineResult:
    """Result from voice pipeline processing"""

    # Input
    audio_duration: float = 0.0
    transcribed_text: str = ""

    # Processing
    llm_response: str = ""
    llm_tokens: int = 0

    # Output
    audio_path: Optional[Path] = None

    # Timing
    stt_time: float = 0.0
    llm_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0

    # Status
    success: bool = False
    error: Optional[str] = None


class VoicePipeline:
    """Complete voice processing pipeline

    Connects STT, LLM, and TTS backends for end-to-end voice interaction.
    """

    def __init__(
        self,
        stt_backend: str = "auto",
        stt_device: str = "auto",
        stt_model: str = "base",
        llm_backend: str = "auto",
        llm_device: str = "auto",
        llm_model: str = "qwen2.5-1.5b",
        tts_backend: str = "auto",
        tts_device: str = "auto",
        tts_voice: str = "EN-Default",
        system_prompt: Optional[str] = None,
    ):
        """Initialize voice pipeline

        Args:
            stt_backend: STT backend ("auto", "whisper", "openvino")
            stt_device: STT device ("auto", "cpu", "cuda", "npu")
            stt_model: STT model size ("tiny", "base", "small", etc.)
            llm_backend: LLM backend ("auto", "openvino")
            llm_device: LLM device ("auto", "cpu", "gpu", "npu")
            llm_model: LLM model name ("qwen2.5-1.5b", "phi-3-mini", etc.)
            tts_backend: TTS backend ("auto", "melotts")
            tts_device: TTS device ("auto", "cpu", "gpu")
            tts_voice: TTS voice ("EN-US", "EN-Default", etc.)
            system_prompt: System prompt for LLM
        """
        self.stt_config = {
            "backend": stt_backend,
            "device": stt_device,
            "model": stt_model,
        }
        self.llm_config = {
            "backend": llm_backend,
            "device": llm_device,
            "model": llm_model,
        }
        self.tts_config = {
            "backend": tts_backend,
            "device": tts_device,
            "voice": tts_voice,
        }
        self.system_prompt = system_prompt or (
            "You are Xenith, a helpful voice assistant. "
            "Be concise and direct in your responses."
        )

        # Backends (lazy loaded)
        self._stt: Optional[STTBackend] = None
        self._llm: Optional[LLMBackend] = None
        self._tts: Optional[TTSBackend] = None

        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if all backends are loaded"""
        return self._is_loaded

    def load(self) -> bool:
        """Load all backends

        Returns:
            True if all backends loaded successfully
        """
        print("[Pipeline] Loading voice pipeline...")

        # Load STT
        print("[Pipeline] Loading STT backend...")
        self._stt = get_stt_backend(
            backend=self.stt_config["backend"],
            device=self.stt_config["device"],
            model_name=self.stt_config["model"],
        )
        if not self._stt or not self._stt.load_model():
            print("[Pipeline] ✗ Failed to load STT backend")
            return False
        print(f"[Pipeline] ✓ STT: {self._stt}")

        # Load LLM
        print("[Pipeline] Loading LLM backend...")
        self._llm = get_llm_backend(
            backend=self.llm_config["backend"],
            device=self.llm_config["device"],
            model=self.llm_config["model"],
        )
        if not self._llm or not self._llm.load():
            print("[Pipeline] ✗ Failed to load LLM backend")
            return False
        print(f"[Pipeline] ✓ LLM: {self._llm}")

        # Load TTS
        print("[Pipeline] Loading TTS backend...")
        self._tts = get_tts_backend(
            backend=self.tts_config["backend"],
            device=self.tts_config["device"],
            voice=self.tts_config["voice"],
        )
        if not self._tts or not self._tts.load():
            print("[Pipeline] ✗ Failed to load TTS backend")
            return False
        print(f"[Pipeline] ✓ TTS: {self._tts}")

        self._is_loaded = True
        print("[Pipeline] ✓ Voice pipeline loaded successfully!")
        return True

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        output_path: Optional[Path] = None,
        on_transcribed: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ) -> PipelineResult:
        """Process audio through the full pipeline

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate (default 16kHz)
            output_path: Path to save TTS output audio
            on_transcribed: Callback when transcription is complete
            on_response: Callback when LLM response is ready

        Returns:
            PipelineResult with all outputs and timing
        """
        result = PipelineResult()
        start_time = time.time()

        if not self._is_loaded:
            if not self.load():
                result.error = "Failed to load pipeline"
                return result

        # Calculate audio duration
        result.audio_duration = len(audio) / sample_rate

        try:
            # Step 1: Speech-to-Text
            print("[Pipeline] Step 1: Transcribing audio...")
            stt_start = time.time()

            stt_result = self._stt.transcribe(audio)
            result.transcribed_text = stt_result.text
            result.stt_time = time.time() - stt_start

            print(f"[Pipeline] Transcribed: '{result.transcribed_text}'")

            if on_transcribed:
                on_transcribed(result.transcribed_text)

            if not result.transcribed_text.strip():
                result.error = "Empty transcription"
                return result

            # Step 2: LLM Processing
            print("[Pipeline] Step 2: Generating response...")
            llm_start = time.time()

            llm_config = LLMConfig(max_tokens=256, temperature=0.7)
            llm_result = self._llm.generate(
                result.transcribed_text,
                config=llm_config,
                system_prompt=self.system_prompt,
            )
            result.llm_response = llm_result.text
            result.llm_tokens = llm_result.tokens_generated
            result.llm_time = time.time() - llm_start

            print(f"[Pipeline] Response: '{result.llm_response}'")

            if on_response:
                on_response(result.llm_response)

            if not result.llm_response.strip():
                result.error = "Empty LLM response"
                return result

            # Step 3: Text-to-Speech
            print("[Pipeline] Step 3: Synthesizing speech...")
            tts_start = time.time()

            tts_result = self._tts.synthesize(
                result.llm_response,
                output_path=output_path,
            )
            result.audio_path = tts_result.audio_path
            result.tts_time = time.time() - tts_start

            print(f"[Pipeline] Audio: {result.audio_path}")

            result.success = True

        except Exception as e:
            result.error = str(e)
            print(f"[Pipeline] Error: {e}")

        result.total_time = time.time() - start_time
        return result

    def process_text(
        self,
        text: str,
        output_path: Optional[Path] = None,
    ) -> PipelineResult:
        """Process text through LLM → TTS (skip STT)

        Args:
            text: Text input
            output_path: Path to save TTS output audio

        Returns:
            PipelineResult
        """
        result = PipelineResult()
        result.transcribed_text = text
        start_time = time.time()

        if not self._is_loaded:
            if not self.load():
                result.error = "Failed to load pipeline"
                return result

        try:
            # LLM Processing
            print(f"[Pipeline] Processing: '{text}'")
            llm_start = time.time()

            llm_config = LLMConfig(max_tokens=256, temperature=0.7)
            llm_result = self._llm.generate(
                text,
                config=llm_config,
                system_prompt=self.system_prompt,
            )
            result.llm_response = llm_result.text
            result.llm_tokens = llm_result.tokens_generated
            result.llm_time = time.time() - llm_start

            print(f"[Pipeline] Response: '{result.llm_response}'")

            # Text-to-Speech
            print("[Pipeline] Synthesizing speech...")
            tts_start = time.time()

            tts_result = self._tts.synthesize(
                result.llm_response,
                output_path=output_path,
            )
            result.audio_path = tts_result.audio_path
            result.tts_time = time.time() - tts_start

            result.success = True

        except Exception as e:
            result.error = str(e)
            print(f"[Pipeline] Error: {e}")

        result.total_time = time.time() - start_time
        return result

    def unload(self) -> None:
        """Unload all backends"""
        if self._stt:
            self._stt.unload_model()
        if self._llm:
            self._llm.unload()
        if self._tts:
            self._tts.unload()
        self._is_loaded = False
        print("[Pipeline] Unloaded voice pipeline")

    def get_status(self) -> dict:
        """Get pipeline status"""
        return {
            "loaded": self._is_loaded,
            "stt": {
                "backend": self.stt_config["backend"],
                "device": self.stt_config["device"],
                "model": self.stt_config["model"],
                "loaded": self._stt.is_loaded if self._stt else False,
            },
            "llm": {
                "backend": self.llm_config["backend"],
                "device": self.llm_config["device"],
                "model": self.llm_config["model"],
                "loaded": self._llm.is_loaded if self._llm else False,
            },
            "tts": {
                "backend": self.tts_config["backend"],
                "device": self.tts_config["device"],
                "voice": self.tts_config["voice"],
                "loaded": self._tts.is_loaded if self._tts else False,
            },
        }


def create_pipeline_from_config(config: dict) -> VoicePipeline:
    """Create a VoicePipeline from config dictionary

    Args:
        config: Configuration dict (from config.yaml)

    Returns:
        Configured VoicePipeline
    """
    audio_config = config.get("audio", {})
    llm_config = config.get("llm", {})

    stt = audio_config.get("stt", {})
    tts = audio_config.get("tts", {})

    return VoicePipeline(
        stt_backend=stt.get("backend", "auto"),
        stt_device=stt.get("device", "auto"),
        stt_model=stt.get("model", "base"),
        llm_backend=llm_config.get("default_backend", "auto"),
        llm_device=llm_config.get("device", "auto"),
        llm_model=llm_config.get("model", "qwen2.5-1.5b"),
        tts_backend=tts.get("backend", "auto"),
        tts_device=tts.get("device", "auto"),
        tts_voice=tts.get("voice", "EN-Default"),
        system_prompt=llm_config.get("system_prompt"),
    )

