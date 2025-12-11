"""Main GTK application"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, Gdk, GLib
from typing import Optional
from .widgets.plasma_widget import PlasmaWidget
from .audio.voice_input import VoiceInput


class XenithApp(Adw.Application):
    """Main Xenith application"""
    
    def __init__(self):
        super().__init__(
            application_id='com.xenith.app',
            flags=0
        )
        self.connect('activate', self.on_activate)
        self._plasma_widget: Optional[PlasmaWidget] = None
        self._voice_input: Optional[VoiceInput] = None
    
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
            
            self._voice_input = VoiceInput(wake_word="hi", device=selected_device)
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
    
    def _on_voice_state_change(self, state: str):
        """Handle voice state changes and update widget"""
        if self._plasma_widget:
            # Update widget state on main thread
            GLib.idle_add(self._plasma_widget.set_state, state)
    
    def _on_wake_word_detected(self):
        """Handle wake word detection"""
        print("[APP] Wake word 'hi' detected! Listening for command...")
    
    def _on_transcript(self, text: str):
        """Handle voice transcript"""
        print(f"[APP] ✓ Command received: '{text}'")
        # Here you would process the transcript and generate a response
    
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



