"""Speech-to-text loading and transcription for voice input"""

import importlib
import traceback

# Errors an STT engine may raise while loading a model or transcribing
STT_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
)


class Transcriber:
    """Loads the configured STT backend (or legacy Whisper) and transcribes audio

    Supports multiple backends:
    - "whisper": OpenAI Whisper (CUDA/CPU)
    - "openvino": OpenVINO Whisper (NPU/GPU/CPU) - ultra-low power
    - "auto": Automatically select best available
    """

    def __init__(self, backend: str = "auto", device: str = "auto", model: str = "base"):
        self.backend_name = backend
        self.device = device
        self.model = model
        self.backend = None  # Will be initialized on first use
        # Legacy Whisper support (kept for backwards compatibility)
        self.whisper_model = None
        self.whisper_device = "cpu"  # Device for Whisper (cpu or cuda)
        self.warning_printed = False

    @property
    def available(self) -> bool:
        """True if any engine is loaded"""
        return self.backend is not None or bool(self.whisper_model)

    def load_backend(self) -> bool:
        """Load the configured STT backend, falling back to legacy Whisper on import errors"""
        try:
            stt_backends = importlib.import_module(".stt_backends", package=__package__)
            print("\n[STT] Initializing speech-to-text backend...")
            self.backend = stt_backends.get_stt_backend(
                backend=self.backend_name,
                model_name=self.model,
                device=self.device,
            )
            if self.backend is None:
                print("[STT] No STT backend available!")
                return False
            if not self.backend.load_model():
                print("[STT] Failed to load STT model")
                return False
            print(f"[STT] ✓ Backend ready: {self.backend}")
            return True
        except ImportError as e:
            print(f"[STT] Import error: {e}")
            return self.load_whisper_legacy()
        except STT_ERRORS as e:
            print(f"[STT] Error loading backend: {e}")
            traceback.print_exc()
            return False

    def load_whisper_legacy(self) -> bool:
        """Legacy Whisper loading (fallback if new backend system fails)"""
        try:
            whisper = importlib.import_module("whisper")
            torch = importlib.import_module("torch")
        except ImportError:
            print("Warning: OpenAI Whisper not available. Wake word detection will use simple pattern matching.")
            return False
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[WHISPER] GPU acceleration available: {torch.cuda.get_device_name(0)}")
                print(f"[WHISPER] CUDA version: {torch.version.cuda}")
            else:
                print("[WHISPER] GPU not available, using CPU")
            # Base model (smaller, faster for wake word detection)
            self.whisper_model = whisper.load_model("base", device=device)
            self.whisper_device = device
            if device == "cuda":
                print("[WHISPER] Model loaded on GPU for faster transcription")
            else:
                print("[WHISPER] Model loaded on CPU")
            return True
        except STT_ERRORS as e:
            print(f"Warning: Could not load Whisper model: {e}")
            traceback.print_exc()
            return False

    def load(self) -> bool:
        """Load an engine - uses new backend system or legacy fallback"""
        if self.load_backend():
            return True
        return self.load_whisper_legacy()

    def transcribe(self, audio) -> str | None:
        """Transcribe ``audio`` with the loaded engine; None if no engine is loaded"""
        if self.backend is not None:
            return self.backend.transcribe(audio, language="en").text
        if self.whisper_model:
            return self.whisper_model.transcribe(audio, language="en", fp16=False)["text"].strip()
        return None

    def print_missing_warning(self) -> None:
        """Warn once that no STT engine is available"""
        if self.warning_printed:
            return
        print("[WAKE WORD] ⚠ No STT backend available - wake word detection disabled")
        print("[WAKE WORD] Install one of:")
        print("[WAKE WORD]   pip install openai-whisper torch  # For CUDA/CPU")
        print("[WAKE WORD]   pip install openvino openvino-genai  # For Intel NPU")
        self.warning_printed = True

    def unload(self) -> None:
        """Unload the STT backend"""
        if self.backend is not None:
            try:
                self.backend.unload_model()
            except STT_ERRORS:
                pass
            self.backend = None
