#!/usr/bin/env python3
"""Simple test for plasma widget - minimal GUI test"""

import sys
import os

# Check for display
if 'DISPLAY' not in os.environ:
    print("Warning: No DISPLAY environment variable set.")
    print("This test requires a graphical environment.")
    print("If running remotely, use X11 forwarding or a virtual display.")
    sys.exit(1)

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, GLib
from src.app import XenithApp

def main():
    """Simple test"""
    print("Starting Xenith application test...")
    print("You should see a plasma widget appear in the top-right corner.")
    print("Press Ctrl+C to exit.")
    
    app = XenithApp()
    
    # Test state changes after a delay
    def test_states():
        widget = app._plasma_widget
        if widget:
            print("Testing state changes...")
            states = ['idle', 'listening', 'processing', 'responding']
            state_idx = [0]
            
            def change_state():
                if widget and widget.get_application():
                    state = states[state_idx[0]]
                    print(f"  → State: {state}")
                    widget.set_state(state)
                    state_idx[0] = (state_idx[0] + 1) % len(states)
                    return True
                return False
            
            GLib.timeout_add(2000, change_state)
    
    # Start state testing after 1 second
    GLib.timeout_add(1000, test_states)
    
    try:
        result = app.run(sys.argv)
        return result
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())



