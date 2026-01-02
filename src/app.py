"""Main GTK application"""

import gi
import threading

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw, Gdk, GLib
from typing import Optional
from pathlib import Path
import yaml
from .widgets.plasma_widget import PlasmaWidget
from .audio.voice_input import VoiceInput
from .audio.streaming_pipeline import StreamingPipeline, StreamingConfig
from .audio.pipeline_metrics import start_metrics, current_metrics, end_metrics


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_paths = [
        Path(__file__).parent.parent / "config" / "config.yaml",  # Project config
        Path.home() / ".config" / "xenith" / "config.yaml",  # User config
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")

    return {}


class XenithApp(Adw.Application):
    """Main Xenith application with voice pipeline"""

    def __init__(self):
        super().__init__(application_id="com.xenith.app", flags=0)
        self.connect("activate", self.on_activate)
        self._plasma_widget: Optional[PlasmaWidget] = None
        self._voice_input: Optional[VoiceInput] = None
        self._voice_pipeline: Optional[StreamingPipeline] = None
        self._config = load_config()
        self._is_processing = False  # Prevent overlapping responses

    def on_activate(self, app: Adw.Application):
        """Called when application is activated"""
        # Create and show plasma widget
        self._plasma_widget = PlasmaWidget(app)
        self._plasma_widget.show()

        # Initialize voice input and connect to widget
        try:
            # Allow user to select input device
            print("\n=== Audio Input Device Selection ===")
            selected_device = VoiceInput.select_device_interactive()

            if selected_device is None:
                print("No device selected. Voice input disabled.")
                return

            # Get STT configuration from config file
            audio_config = self._config.get("audio", {})
            stt_config = audio_config.get("stt", {})

            stt_backend = stt_config.get("backend", "auto")
            stt_device = stt_config.get("device", "auto")
            stt_model = stt_config.get("model", "base")

            print(f"\n=== STT Configuration ===")
            print(f"  Backend: {stt_backend}")
            print(f"  Device:  {stt_device}")
            print(f"  Model:   {stt_model}")

            self._voice_input = VoiceInput(
                wake_word="hi",
                device=selected_device,
                stt_backend=stt_backend,
                stt_device=stt_device,
                stt_model=stt_model,
            )
            self._voice_input.on_state_change = self._on_voice_state_change
            self._voice_input.on_transcript = self._on_transcript
            self._voice_input.on_wake_word_detected = self._on_wake_word_detected

            # Start listening for wake word (always-on mode)
            self._voice_input.start_listening()
            print("\nVoice input initialized. Say 'hi' to activate!")
        except Exception as e:
            print(f"Warning: Could not initialize voice input: {e}")
            import traceback

            traceback.print_exc()

        # Initialize streaming voice pipeline (LLM + TTS)
        self._init_voice_pipeline()

    def _init_voice_pipeline(self):
        """Initialize the streaming LLM + TTS pipeline"""
        try:
            # Get LLM/TTS config
            llm_config = self._config.get("llm", {})
            tts_config = self._config.get("audio", {}).get("tts", {})

            config = StreamingConfig(
                llm_model=llm_config.get("model", "qwen2.5-1.5b"),
                llm_device=llm_config.get(
                    "device", "CPU"
                ),  # CPU is faster than NPU for latency
                tts_voice=tts_config.get("voice", "EN-Default"),
                tts_device=tts_config.get("device", "auto"),
                use_ram_disk=True,
                fast_first_sentence=False,  # Disabled: use MeloTTS for consistent voice
            )

            print("\n=== Voice Pipeline ===")
            print(f"  LLM: {config.llm_model} on {config.llm_device}")
            print(f"  TTS: {config.tts_voice}")

            self._voice_pipeline = StreamingPipeline(config)

            # Load in background to not block UI
            def load_pipeline():
                if self._voice_pipeline.load():
                    print("[APP] ✓ Voice pipeline ready!")
                else:
                    print("[APP] ⚠ Voice pipeline failed to load")

            threading.Thread(target=load_pipeline, daemon=True).start()

        except Exception as e:
            print(f"Warning: Could not initialize voice pipeline: {e}")
            import traceback

            traceback.print_exc()

    def _on_voice_state_change(self, state: str):
        """Handle voice state changes and update widget"""
        if self._plasma_widget:
            # Update widget state on main thread
            GLib.idle_add(self._plasma_widget.set_state, state)

    def _on_wake_word_detected(self):
        """Handle wake word detection"""
        print("[APP] Wake word 'hi' detected! Listening for command...")

        # Start metrics collection for this interaction
        metrics = start_metrics()
        metrics.mark("wake_word_detected")
        metrics.mark("stt_start")

        if self._plasma_widget:
            GLib.idle_add(self._plasma_widget.set_state, "listening")

    def _on_transcript(self, text: str):
        """Handle voice transcript - send to LLM and speak response"""
        print(f"[APP] ✓ Command received: '{text}'")

        # Mark STT complete in metrics
        metrics = current_metrics()
        if metrics:
            metrics.mark("stt_complete")
            metrics.prompt = text

        if self._is_processing:
            print("[APP] Already processing a request, ignoring")
            return

        if not self._voice_pipeline or not self._voice_pipeline._is_loaded:
            print("[APP] Voice pipeline not ready")
            return

        # Process in background thread
        def process_command():
            self._is_processing = True

            # Mute microphone to prevent feedback loop during TTS
            if self._voice_input:
                self._voice_input.mute()

            # Update UI to show we're processing
            if self._plasma_widget:
                GLib.idle_add(self._plasma_widget.set_state, "processing")

            def on_audio_start():
                # Update UI to show we're responding
                if self._plasma_widget:
                    GLib.idle_add(self._plasma_widget.set_state, "responding")

            try:
                result = self._voice_pipeline.process_streaming(
                    text,
                    on_token=lambda t: print(t, end="", flush=True),
                    on_audio_start=on_audio_start,
                )
                print(f"\n[APP] Response complete")

                # Wait for audio to finish playing before resuming listening
                print("[APP] Waiting for audio playback to finish...")
                self._voice_pipeline.wait_for_audio(timeout=30)
                print("[APP] Audio playback complete")

            except Exception as e:
                print(f"[APP] Error processing: {e}")
            finally:
                self._is_processing = False
                # Return to idle state
                if self._plasma_widget:
                    GLib.idle_add(self._plasma_widget.set_state, "idle")
                # Unmute and resume listening for wake word
                if self._voice_input:
                    self._voice_input.unmute()
                    print("[APP] Ready - say 'hi' to start")

        threading.Thread(target=process_command, daemon=True).start()

    def start_listening(self):
        """Start voice input"""
        if self._voice_input:
            self._voice_input.start_listening()

    def stop_listening(self):
        """Stop voice input"""
        if self._voice_input:
            self._voice_input.stop_listening()

    def cleanup(self):
        """Cleanup resources"""
        if self._voice_input:
            self._voice_input.cleanup()
        if self._voice_pipeline:
            self._voice_pipeline.unload()
