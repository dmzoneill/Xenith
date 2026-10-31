"""State containers and pure audio helpers for voice input"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .audio_deps import np


@dataclass
class AudioConfig:
    """Audio capture and voice activity settings"""

    sample_rate: int = 16000
    chunk_size: int = 1024
    wake_word: str = "hi"
    device: int | None = None  # Audio input device index
    energy_threshold: float = 0.01  # Adjust based on your microphone
    silence_duration: float = 0.3  # Seconds of silence before processing (fast response)


@dataclass
class ListenState:
    """What the voice input is currently doing"""

    wake_word: bool = False  # Listening for the wake word
    command: bool = False  # Listening for a command after the wake word
    processing: bool = False
    muted: bool = False  # Mute during TTS playback
    last_voice_time: float = field(default_factory=time.time)
    command_start_time: float = 0.0  # When we started listening for a command
    ignore_audio_until: float = 0.0  # Ignore audio until this time (after wake word)


@dataclass
class AudioBuffers:
    """Captured command audio and the rolling wake word buffer"""

    max_wake_samples: int
    command: list = field(default_factory=list)
    wake: list = field(default_factory=list)

    def clear(self) -> None:
        """Drop all buffered audio"""
        self.command = []
        self.wake = []

    def add_wake_audio(self, samples: list) -> None:
        """Append to the wake word buffer, keeping only the newest ``max_wake_samples``"""
        self.wake.extend(samples)
        if len(self.wake) > self.max_wake_samples:
            excess = len(self.wake) - self.max_wake_samples
            self.wake = self.wake[excess:]


@dataclass
class WakeWordMonitor:
    """Wake word thread and timers"""

    thread: threading.Thread | None = None
    check_interval: float = 0.5  # Check for wake word every 0.5s
    last_check: float = 0.0
    last_detection: float | None = None
    last_debug_print: float | None = None
    debug_counter: int = 0


@dataclass
class VoiceCallbacks:
    """Callbacks for state changes"""

    on_state_change: Callable[[str], None] | None = None
    on_transcript: Callable[[str], None] | None = None
    on_wake_word_detected: Callable[[], None] | None = None

    def state(self, state: str) -> None:
        """Notify a state change if a listener is set"""
        if self.on_state_change:
            self.on_state_change(state)


def normalize(audio_array):
    """Scale audio so its peak is 1.0 (unchanged if silent)"""
    peak = np.max(np.abs(audio_array))
    return audio_array / peak if peak > 0 else audio_array


def recent_wake_audio(buffer: list, sample_rate: int):
    """Normalized last 2 seconds of the wake buffer, or None if too short or too quiet"""
    if len(buffer) < sample_rate:
        return None
    recent_samples = min(int(sample_rate * 2), len(buffer))
    audio_array = np.array(buffer[-recent_samples:], dtype=np.float32)
    # Only transcribe if audio level is significant
    if np.abs(audio_array).mean() < 0.001:
        return None
    if np.max(np.abs(audio_array)) <= 0:
        return None
    return normalize(audio_array)
