#!/usr/bin/env python3
"""Test voice state changes in plasma widget"""

import sys
import os

# Check for display
if 'DISPLAY' not in os.environ:
    print("Warning: No DISPLAY environment variable set.")
    print("This test requires a graphical environment.")
    sys.exit(1)

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, GLib
from src.app import XenithApp


def simulate_voice_interaction(app):
    """Simulate a voice interaction cycle"""
    widget = app._plasma_widget
    if not widget:
        return
    
    def cycle_states():
        """Cycle through states to demonstrate visual changes"""
        states = [
            ('idle', 2.0),
            ('listening', 3.0),  # Simulate listening
            ('processing', 2.0),  # Simulate processing
            ('responding', 2.0),  # Simulate response
            ('idle', 1.0),
        ]
        
        state_index = [0]
        
        def change_state():
            if state_index[0] < len(states):
                state, duration = states[state_index[0]]
                print(f"  → State: {state}")
                widget.set_state(state)
                state_index[0] += 1
                
                if state_index[0] < len(states):
                    # Schedule next state change
                    next_state, next_duration = states[state_index[0]]
                    GLib.timeout_add(int(duration * 1000), change_state)
                else:
                    print("\nState cycle complete!")
                    print("You can now test actual voice input if available.")
                    # Optionally start voice input
                    try:
                        app.start_listening()
                        print("Voice input started - speak into your microphone!")
                    except Exception as e:
                        print(f"Could not start voice input: {e}")
            
            return False  # Don't repeat
        
        # Start the cycle
        change_state()
    
    # Start state cycle after a short delay
    GLib.timeout_add(1000, cycle_states)


def main():
    """Test voice states"""
    print("Starting Xenith voice state test...")
    print("You should see the plasma widget change states:")
    print("  - idle (blue)")
    print("  - listening (orange, pulsing)")
    print("  - processing (red/orange, intense)")
    print("  - responding (purple)")
    print("  - idle (blue)")
    print("\nPress Ctrl+C to exit.\n")
    
    app = XenithApp()
    
    # Connect to activate to get widget reference
    def on_activate(app_instance):
        widget = app_instance._plasma_widget
        if widget:
            print("Plasma widget created!")
            # Start state simulation
            simulate_voice_interaction(app_instance)
    
    app.connect('activate', on_activate)
    
    try:
        result = app.run(sys.argv)
        return result
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        app.cleanup()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        app.cleanup()
        return 1


if __name__ == '__main__':
    sys.exit(main())

