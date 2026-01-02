"""Streaming Voice Pipeline: Real-time STT → LLM → TTS

This module provides a streaming voice processing pipeline with:
- LLM token streaming with sentence buffering
- Parallel TTS synthesis
- Real-time audio playback

Architecture:
    LLM Token Stream → Sentence Buffer → TTS Queue → Audio Player
                                              ↓
                                     Parallel Workers
"""

import queue
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

from .llm_backends import get_llm_backend, LLMBackend, LLMConfig
from .tts_backends import get_tts_backend, TTSBackend
from .tts_backends.fast_tts import FastTTSBackend, ESPEAK_AVAILABLE
from .pipeline_metrics import PipelineMetrics, start_metrics, end_metrics


@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline"""

    # LLM settings
    llm_model: str = "qwen2.5-1.5b"
    llm_device: str = "auto"
    max_tokens: int = 256
    temperature: float = 0.7

    # TTS settings
    tts_voice: str = "EN-Default"
    tts_device: str = "auto"

    # Streaming settings
    use_ram_disk: bool = True  # Use /dev/shm for temp files
    sentence_end_chars: str = ".!?;,:"  # Chars that end sentence (added colon)
    min_sentence_length: int = 3  # Reduced for faster first sentence
    tts_workers: int = 4  # Parallel TTS workers (overlaps model loading)
    fast_first_sentence: bool = False  # Disabled by default
    keep_npu_warm: bool = True  # Background warmup to prevent NPU cold starts

    # Audio settings
    audio_sample_rate: int = 44100  # MeloTTS outputs 44100 Hz
    audio_buffer_size: int = 1024

    # System prompt
    system_prompt: str = "You are Xenith, a helpful voice assistant. Be concise."


class SentenceBuffer:
    """Buffers LLM tokens and yields complete sentences"""

    def __init__(
        self,
        end_chars: str = ".!?;",
        min_length: int = 10,
    ):
        self.end_chars = set(end_chars)
        self.min_length = min_length
        self._buffer = ""

    def add_token(self, token: str) -> Optional[str]:
        """Add a token and return complete sentence if available"""
        self._buffer += token

        # Check for sentence end
        if len(self._buffer) >= self.min_length:
            for i, char in enumerate(self._buffer):
                if char in self.end_chars and i >= self.min_length - 1:
                    # Found sentence end
                    sentence = self._buffer[: i + 1].strip()
                    self._buffer = self._buffer[i + 1 :].lstrip()
                    if sentence:
                        return sentence
        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer"""
        if self._buffer.strip():
            sentence = self._buffer.strip()
            self._buffer = ""
            return sentence
        return None


class AudioPlayer:
    """Real-time audio player using sounddevice"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._playing = False
        self._is_playing = False  # Currently playing audio
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the audio player thread"""
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioPlayer] sounddevice not available, using fallback")
            return

        self._stop_event.clear()
        self._playing = True
        self._thread = threading.Thread(target=self._player_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the audio player"""
        self._stop_event.set()
        self._playing = False
        if self._thread:
            self._thread.join(timeout=2)

    def queue_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """Queue audio data for playback"""
        if sample_rate:
            self._queue.put((audio_data, sample_rate))
        else:
            self._queue.put((audio_data, self.sample_rate))

    def queue_file(self, audio_path: Path):
        """Queue audio file for playback"""
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                file_sample_rate = wf.getframerate()
                audio_data = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                )
                audio_data = audio_data.astype(np.float32) / 32768.0
                # Queue tuple of (audio_data, sample_rate) to handle different rates
                self._queue.put((audio_data, file_sample_rate))
        except Exception as e:
            print(f"[AudioPlayer] Error loading audio: {e}")

    def _player_loop(self):
        """Main playback loop"""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.01)  # 10ms for fast response
                if item is not None:
                    self._is_playing = True
                    # Handle both (audio_data, sample_rate) tuples and raw audio_data
                    if isinstance(item, tuple):
                        audio_data, sample_rate = item
                    else:
                        audio_data = item
                        sample_rate = self.sample_rate
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    self._is_playing = False
            except queue.Empty:
                self._is_playing = False
                continue
            except Exception as e:
                self._is_playing = False
                print(f"[AudioPlayer] Playback error: {e}")

    def wait_until_done(self, timeout: float = 30.0):
        """Wait until all audio has finished playing"""
        import time

        start = time.time()
        while time.time() - start < timeout:
            if self._queue.empty() and not getattr(self, "_is_playing", False):
                # Brief pause to ensure audio fully completes
                time.sleep(0.1)
                return True
            time.sleep(0.05)  # Fast polling (50ms)
        return False

    @property
    def is_idle(self) -> bool:
        """Check if audio player is idle (nothing playing or queued)"""
        return self._queue.empty() and not getattr(self, "_is_playing", False)


class StreamingPipeline:
    """Full streaming voice pipeline

    Provides real-time voice interaction with:
    - LLM token streaming
    - Sentence-buffered TTS
    - Parallel audio playback
    - Background NPU warmup to prevent cold starts
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()

        # Backends
        self._llm: Optional[LLMBackend] = None
        self._tts: Optional[TTSBackend] = None
        self._fast_tts: Optional[FastTTSBackend] = None  # For first sentence

        # Streaming components
        self._sentence_buffer = SentenceBuffer(
            end_chars=self.config.sentence_end_chars,
            min_length=self.config.min_sentence_length,
        )
        self._audio_player = AudioPlayer(self.config.audio_sample_rate)
        self._tts_executor = ThreadPoolExecutor(max_workers=self.config.tts_workers)

        # Temp directory
        if self.config.use_ram_disk:
            self._temp_dir = Path("/dev/shm/xenith_tts")
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="xenith_tts_"))
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        self._is_loaded = False
        self._sentence_counter = 0

        # Background warmup to keep NPU hot
        self._warmup_thread: Optional[threading.Thread] = None
        self._warmup_stop = threading.Event()
        self._warmup_interval = 20  # Warmup every 20 seconds
        self._last_query_time = 0.0  # Track last real query

    def load(self) -> bool:
        """Load all backends"""
        print("[StreamingPipeline] Loading...")

        # Load LLM
        self._llm = get_llm_backend(
            model=self.config.llm_model,
            device=self.config.llm_device,
        )
        if not self._llm or not self._llm.load():
            print("[StreamingPipeline] Failed to load LLM")
            return False
        print(f"[StreamingPipeline] ✓ LLM: {self._llm}")

        # Load TTS
        self._tts = get_tts_backend(
            voice=self.config.tts_voice,
            device=self.config.tts_device,
        )
        if not self._tts or not self._tts.load():
            print("[StreamingPipeline] Failed to load TTS")
            return False
        print(f"[StreamingPipeline] ✓ TTS: {self._tts}")

        # Pre-warm the LLM with a dummy query to compile on NPU
        print("[StreamingPipeline] Warming up LLM...")
        try:
            warmup_config = LLMConfig(max_tokens=5, temperature=0.1)
            self._llm.generate("Hi", warmup_config)
            print("[StreamingPipeline] ✓ LLM warmed up")
        except Exception as e:
            print(f"[StreamingPipeline] LLM warmup failed: {e}")

        # Pre-warm TTS with a short phrase to cache models (in-memory, no file)
        print("[StreamingPipeline] Warming up TTS...")
        try:
            self._tts.synthesize("Ready.")  # No output_path = in-memory only
            print("[StreamingPipeline] ✓ TTS warmed up")
        except Exception as e:
            print(f"[StreamingPipeline] TTS warmup failed: {e}")

        # Load fast TTS for first sentence (optional but faster)
        if self.config.fast_first_sentence and ESPEAK_AVAILABLE:
            self._fast_tts = FastTTSBackend(
                output_dir=str(self._temp_dir),
            )
            if self._fast_tts.load():
                print("[StreamingPipeline] ✓ Fast TTS for first sentence")
            else:
                self._fast_tts = None

        # Start audio player
        self._audio_player.start()
        print("[StreamingPipeline] ✓ Audio player started")

        # Start background warmup thread to keep NPU hot
        self._start_warmup_thread()

        self._is_loaded = True
        return True

    def _start_warmup_thread(self):
        """Start background thread to keep NPU/LLM cache warm"""
        self._warmup_stop.clear()
        self._last_query_time = time.time()

        def warmup_loop():
            while not self._warmup_stop.is_set():
                # Wait for warmup interval
                self._warmup_stop.wait(self._warmup_interval)
                if self._warmup_stop.is_set():
                    break

                # Only warmup if no recent queries (avoid interference)
                time_since_last = time.time() - self._last_query_time
                if time_since_last >= self._warmup_interval - 1:
                    try:
                        # Quick warmup query (minimal tokens)
                        warmup_config = LLMConfig(max_tokens=1, temperature=0.1)
                        self._llm.generate(".", warmup_config)
                        # Don't print to avoid log spam
                    except Exception:
                        pass  # Ignore warmup errors

        self._warmup_thread = threading.Thread(
            target=warmup_loop, daemon=True, name="LLM-Warmup"
        )
        self._warmup_thread.start()
        print("[StreamingPipeline] ✓ Background NPU warmup started")

    def _synthesize_sentence(self, sentence: str, index: int) -> Optional[tuple]:
        """Synthesize a single sentence (runs in thread pool)

        Returns:
            Tuple of (audio_data, sample_rate) for direct playback, or None on error
        """
        try:
            # Use fast TTS for first sentence (much lower latency)
            if index == 1 and self._fast_tts:
                # Fast TTS still uses files, get result and load
                output_path = self._temp_dir / f"sentence_{index}.wav"
                result = self._fast_tts.synthesize(
                    sentence, output_path=str(output_path)
                )
                if result.audio_path:
                    # Load audio from file for playback
                    import wave
                    with wave.open(str(result.audio_path), "rb") as wf:
                        audio_data = np.frombuffer(
                            wf.readframes(wf.getnframes()), dtype=np.int16
                        )
                        audio_data = audio_data.astype(np.float32) / 32768.0
                        return (audio_data, wf.getframerate())
                # Fallback to normal TTS if fast fails

            # Use Piper TTS - returns audio_data directly in memory (no file I/O!)
            result = self._tts.synthesize(sentence)  # No output_path = in-memory only

            if result.audio_data is not None:
                return (result.audio_data, result.sample_rate)

            return None
        except Exception as e:
            print(f"[StreamingPipeline] TTS error: {e}")
            return None

    def process_streaming(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        on_audio_start: Optional[Callable[[], None]] = None,
    ) -> dict:
        """Process prompt with streaming LLM and parallel TTS

        Args:
            prompt: User input
            on_token: Callback for each LLM token
            on_sentence: Callback for each complete sentence
            on_audio_start: Callback when audio starts playing

        Returns:
            Dict with timing and response info
        """
        if not self._is_loaded:
            if not self.load():
                return {"error": "Failed to load pipeline"}

        # Initialize metrics
        metrics = start_metrics(prompt)

        start_time = time.time()
        self._sentence_counter = 0
        sentences = []
        tts_futures = []
        audio_started = False

        # Reset sentence buffer
        self._sentence_buffer = SentenceBuffer(
            end_chars=self.config.sentence_end_chars,
            min_length=self.config.min_sentence_length,
        )

        # Build prompt
        full_prompt = f"{self.config.system_prompt}\n\nUser: {prompt}\nAssistant:"

        # Configure LLM
        llm_config = LLMConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        print(f"[StreamingPipeline] Processing: '{prompt}'")
        metrics.mark("llm_start")
        llm_start = time.time()
        self._last_query_time = llm_start  # Track for warmup thread
        full_response = ""
        first_token_time = None

        # Stream LLM tokens
        try:
            import openvino_genai as ov_genai

            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = self.config.max_tokens
            gen_config.temperature = self.config.temperature

            # Create a custom streamer class with incremental decoding
            class TokenStreamer(ov_genai.StreamerBase):
                def __init__(streamer_self, tokenizer):
                    super().__init__()
                    streamer_self.tokenizer = tokenizer
                    streamer_self.tokens = []
                    streamer_self.prev_text_len = 0

                def put(streamer_self, token_id: int) -> bool:
                    nonlocal full_response, first_token_time, audio_started

                    # Append token
                    streamer_self.tokens.append(token_id)

                    # Incremental decode: only decode new portion
                    # Decode last few tokens for context, extract only new text
                    text = streamer_self.tokenizer.decode(streamer_self.tokens)
                    new_text = text[streamer_self.prev_text_len:]
                    streamer_self.prev_text_len = len(text)

                    if new_text:
                        if first_token_time is None:
                            first_token_time = time.time()
                            metrics.mark("llm_first_token")

                        full_response = text

                        if on_token:
                            on_token(new_text)

                        # Check for complete sentence
                        sentence = self._sentence_buffer.add_token(new_text)
                        if sentence:
                            self._queue_sentence_for_tts(
                                sentence,
                                sentences,
                                tts_futures,
                                on_sentence,
                                on_audio_start if not audio_started else None,
                                metrics,
                            )
                            audio_started = True

                    return False  # False = continue, True = stop

                def end(streamer_self):
                    pass

            # Get tokenizer and create streamer
            pipeline = self._llm._pipeline
            tokenizer = pipeline.get_tokenizer()
            streamer = TokenStreamer(tokenizer)

            # Generate with streaming
            final_result = pipeline.generate(full_prompt, gen_config, streamer)

            # Ensure we have the complete response
            if isinstance(final_result, str) and final_result:
                full_response = final_result

        except Exception as e:
            print(f"[StreamingPipeline] LLM streaming error: {e}")
            # Fallback to non-streaming
            result = self._llm.generate(prompt, llm_config, self.config.system_prompt)
            full_response = result.text

            # Process response as sentences
            for sentence in re.split(r"(?<=[.!?;])\s+", full_response):
                if sentence.strip():
                    self._queue_sentence_for_tts(
                        sentence.strip(),
                        sentences,
                        tts_futures,
                        on_sentence,
                        on_audio_start if not audio_started else None,
                        metrics,
                    )
                    audio_started = True

        llm_time = time.time() - llm_start

        # Flush remaining buffer
        remaining = self._sentence_buffer.flush()
        if remaining:
            self._queue_sentence_for_tts(
                remaining,
                sentences,
                tts_futures,
                on_sentence,
                on_audio_start if not audio_started else None,
                metrics,
            )
            audio_started = True

        # Wait for all TTS to complete (audio is already being played by callbacks)
        tts_start = time.time()
        from concurrent.futures import wait

        # Just wait for completion - audio playback happens in the TTS callbacks
        _, not_done = wait(tts_futures, timeout=60)
        for future in not_done:
            future.cancel()

        if not_done:
            print(f"[StreamingPipeline] Warning: {len(not_done)} TTS tasks timed out")

        tts_time = time.time() - tts_start
        total_time = time.time() - start_time

        # Finalize metrics
        metrics.mark("llm_complete")
        metrics.response = full_response
        metrics.sentence_count = len(sentences)
        metrics.mark("pipeline_complete")

        # Print metrics summary
        metrics.print_summary()
        end_metrics()

        return {
            "response": full_response,
            "sentences": sentences,
            "timing": {
                "total": total_time,
                "llm": llm_time,
                "tts": tts_time,
                "first_token": (
                    (first_token_time - start_time) if first_token_time else None
                ),
            },
            "metrics": metrics.to_dict(),
        }

    def _queue_sentence_for_tts(
        self,
        sentence: str,
        sentences: List[str],
        tts_futures: List,
        on_sentence: Optional[Callable[[str], None]],
        on_first_audio: Optional[Callable[[], None]] = None,
        metrics: Optional[PipelineMetrics] = None,
    ):
        """Queue a sentence for TTS synthesis with immediate audio playback"""
        sentences.append(sentence)
        self._sentence_counter += 1
        idx = self._sentence_counter
        is_first = idx == 1

        if on_sentence:
            on_sentence(sentence)

        # Track first TTS queue time
        if is_first and metrics:
            metrics.mark("tts_first_queued")

        print(f"[StreamingPipeline] Sentence {idx}: '{sentence}'")

        # Define callback that plays audio when TTS completes
        def tts_with_playback():
            tts_start_time = time.time()
            audio_result = self._synthesize_sentence(sentence, idx)
            if audio_result:
                audio_data, sample_rate = audio_result
                # Track first TTS completion
                if is_first and metrics:
                    metrics.mark("tts_first_complete")
                    metrics.sentence_times.append(
                        {
                            "idx": idx,
                            "tts_start": tts_start_time,
                            "tts_complete": time.time(),
                        }
                    )

                # Trigger first audio callback and mark metric
                if is_first:
                    if metrics:
                        metrics.mark("audio_first_playing")
                    if on_first_audio:
                        on_first_audio()
                # Queue audio data directly - no file I/O!
                self._audio_player.queue_audio(audio_data, sample_rate)
            return audio_result

        # Submit to thread pool - audio will play as soon as ready
        future = self._tts_executor.submit(tts_with_playback)
        tts_futures.append(future)

    def process_simple(self, prompt: str) -> dict:
        """Simple non-streaming processing (for comparison)"""
        if not self._is_loaded:
            if not self.load():
                return {"error": "Failed to load pipeline"}

        start_time = time.time()

        # LLM
        llm_start = time.time()
        llm_config = LLMConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        result = self._llm.generate(prompt, llm_config, self.config.system_prompt)
        llm_time = time.time() - llm_start

        # TTS - in-memory, no file I/O
        tts_start = time.time()
        tts_result = self._tts.synthesize(result.text)  # No output_path
        tts_time = time.time() - tts_start

        # Play directly from memory
        if tts_result.audio_data is not None:
            self._audio_player.queue_audio(tts_result.audio_data, tts_result.sample_rate)

        return {
            "response": result.text,
            "timing": {
                "total": time.time() - start_time,
                "llm": llm_time,
                "tts": tts_time,
            },
        }

    def wait_for_audio(self, timeout: float = 30.0) -> bool:
        """Wait for all audio playback to complete"""
        return self._audio_player.wait_until_done(timeout)

    def unload(self):
        """Unload all resources"""
        # Stop background warmup thread
        if self._warmup_thread:
            self._warmup_stop.set()
            self._warmup_thread.join(timeout=1)

        self._audio_player.stop()
        self._tts_executor.shutdown(wait=False)

        if self._llm:
            self._llm.unload()
        if self._tts:
            self._tts.unload()
        if self._fast_tts:
            self._fast_tts.unload()

        # Cleanup temp files
        try:
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

        self._is_loaded = False
        print("[StreamingPipeline] Unloaded")
