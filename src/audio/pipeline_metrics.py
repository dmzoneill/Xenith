"""Pipeline Metrics - Instrumentation for voice pipeline timing

Tracks delays between all stages:
- Wake word detection
- STT processing
- LLM generation (first token, full response)
- TTS synthesis (first sentence, all sentences)
- Audio playback (first audio, completion)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class PipelineMetrics:
    """Collects timing metrics for a single pipeline run"""

    # Timestamps (absolute)
    start_time: float = 0.0
    wake_word_detected: float = 0.0
    stt_start: float = 0.0
    stt_complete: float = 0.0
    llm_start: float = 0.0
    llm_first_token: float = 0.0
    llm_complete: float = 0.0
    tts_first_queued: float = 0.0
    tts_first_complete: float = 0.0
    audio_first_playing: float = 0.0
    audio_complete: float = 0.0
    pipeline_complete: float = 0.0

    # Sentence-level timing
    sentence_times: List[Dict] = field(default_factory=list)

    # Metadata
    prompt: str = ""
    response: str = ""
    sentence_count: int = 0

    def mark(self, event: str):
        """Mark a timestamp for an event"""
        now = time.time()
        if hasattr(self, event):
            setattr(self, event, now)

    def get_latencies(self) -> Dict[str, float]:
        """Calculate latencies between stages"""
        latencies = {}

        # Key user-perceived latencies
        if self.wake_word_detected and self.stt_complete:
            latencies["wake_to_stt"] = self.stt_complete - self.wake_word_detected

        if self.stt_complete and self.llm_first_token:
            latencies["stt_to_first_token"] = self.llm_first_token - self.stt_complete

        if self.llm_first_token and self.audio_first_playing:
            latencies["first_token_to_audio"] = (
                self.audio_first_playing - self.llm_first_token
            )

        if self.wake_word_detected and self.audio_first_playing:
            latencies["wake_to_first_audio"] = (
                self.audio_first_playing - self.wake_word_detected
            )

        # Component latencies
        if self.stt_start and self.stt_complete:
            latencies["stt_processing"] = self.stt_complete - self.stt_start

        if self.llm_start and self.llm_first_token:
            latencies["llm_time_to_first_token"] = self.llm_first_token - self.llm_start

        if self.llm_start and self.llm_complete:
            latencies["llm_total"] = self.llm_complete - self.llm_start

        if self.tts_first_queued and self.tts_first_complete:
            latencies["tts_first_sentence"] = (
                self.tts_first_complete - self.tts_first_queued
            )

        if self.tts_first_complete and self.audio_first_playing:
            latencies["tts_to_audio"] = (
                self.audio_first_playing - self.tts_first_complete
            )

        # Total pipeline time
        if self.start_time and self.pipeline_complete:
            latencies["total_pipeline"] = self.pipeline_complete - self.start_time

        return latencies

    def print_summary(self, detailed: bool = False):
        """Print a summary of timing metrics"""
        latencies = self.get_latencies()

        print("\n" + "=" * 60)
        print("📊 PIPELINE TIMING METRICS")
        print("=" * 60)

        # Key metric: time to first audio
        if "wake_to_first_audio" in latencies:
            print(f"\n⚡ Time to first audio: {latencies['wake_to_first_audio']:.3f}s")

        print("\n--- Stage Latencies ---")

        stage_order = [
            ("wake_to_stt", "Wake → STT complete"),
            ("stt_processing", "  └─ STT processing"),
            ("stt_to_first_token", "STT → First LLM token"),
            ("llm_time_to_first_token", "  └─ LLM warmup"),
            ("first_token_to_audio", "First token → First audio"),
            ("tts_first_sentence", "  └─ TTS first sentence"),
            ("tts_to_audio", "  └─ Audio queue delay"),
            ("llm_total", "LLM total generation"),
            ("total_pipeline", "Total pipeline time"),
        ]

        for key, label in stage_order:
            if key in latencies:
                value = latencies[key]
                # Color code based on speed
                if value < 0.1:
                    indicator = "🟢"
                elif value < 0.5:
                    indicator = "🟡"
                elif value < 1.0:
                    indicator = "🟠"
                else:
                    indicator = "🔴"
                print(f"  {indicator} {label}: {value:.3f}s")

        if detailed and self.sentence_times:
            print("\n--- Per-Sentence Timing ---")
            for i, st in enumerate(self.sentence_times, 1):
                tts_time = st.get("tts_complete", 0) - st.get("tts_start", 0)
                print(f"  Sentence {i}: TTS={tts_time:.3f}s")

        print("=" * 60 + "\n")

    def to_dict(self) -> Dict:
        """Export metrics as dictionary"""
        return {
            "timestamps": {
                "start_time": self.start_time,
                "wake_word_detected": self.wake_word_detected,
                "stt_start": self.stt_start,
                "stt_complete": self.stt_complete,
                "llm_start": self.llm_start,
                "llm_first_token": self.llm_first_token,
                "llm_complete": self.llm_complete,
                "tts_first_queued": self.tts_first_queued,
                "tts_first_complete": self.tts_first_complete,
                "audio_first_playing": self.audio_first_playing,
                "audio_complete": self.audio_complete,
                "pipeline_complete": self.pipeline_complete,
            },
            "latencies": self.get_latencies(),
            "metadata": {
                "prompt": self.prompt,
                "response_length": len(self.response),
                "sentence_count": self.sentence_count,
            },
            "sentence_times": self.sentence_times,
        }


class MetricsCollector:
    """Collects metrics across multiple pipeline runs"""

    def __init__(self):
        self._runs: List[PipelineMetrics] = []
        self._current: Optional[PipelineMetrics] = None

    def start_run(self, prompt: str = "") -> PipelineMetrics:
        """Start a new metrics collection run"""
        self._current = PipelineMetrics()
        self._current.start_time = time.time()
        self._current.prompt = prompt
        return self._current

    @property
    def current(self) -> Optional[PipelineMetrics]:
        """Get current metrics run"""
        return self._current

    def end_run(self):
        """End current run and store it"""
        if self._current:
            self._current.pipeline_complete = time.time()
            self._runs.append(self._current)
            self._current = None

    def get_averages(self) -> Dict[str, float]:
        """Calculate average latencies across all runs"""
        if not self._runs:
            return {}

        all_latencies = [run.get_latencies() for run in self._runs]
        keys = set()
        for lat in all_latencies:
            keys.update(lat.keys())

        averages = {}
        for key in keys:
            values = [lat.get(key) for lat in all_latencies if key in lat]
            if values:
                averages[key] = sum(values) / len(values)

        return averages

    def print_summary(self):
        """Print summary of all runs"""
        if not self._runs:
            print("No metrics collected yet")
            return

        print(f"\n📈 Metrics Summary ({len(self._runs)} runs)")
        print("-" * 40)

        averages = self.get_averages()
        for key, value in sorted(averages.items()):
            print(f"  {key}: {value:.3f}s (avg)")


# Global metrics collector
_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return _collector


def start_metrics(prompt: str = "") -> PipelineMetrics:
    """Start collecting metrics for a pipeline run"""
    return _collector.start_run(prompt)


def current_metrics() -> Optional[PipelineMetrics]:
    """Get current metrics being collected"""
    return _collector.current


def end_metrics():
    """End current metrics collection"""
    _collector.end_run()

