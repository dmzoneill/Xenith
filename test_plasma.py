#!/usr/bin/env python3
"""Test script for plasma widget"""

import sys
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, GLib
from src.app import XenithApp


def test_state_changes(widget):
    """Test state changes"""
    states = ['idle', 'listening', 'processing', 'responding']
    state_index = 0
    
    def change_state():
        nonlocal state_index
        state = states[state_index]
        print(f"Changing state to: {state}")
        widget.set_state(state)
        state_index = (state_index + 1) % len(states)
        return True
    
    # Change state every 3 seconds
    GLib.timeout_add(3000, change_state)


def test_agent_colors(widget):
    """Test agent color integration"""
    agents = [['general'], ['developer'], ['general', 'developer'], []]
    agent_index = 0
    
    def change_agents():
        nonlocal agent_index
        agent_list = agents[agent_index]
        print(f"Setting active agents: {agent_list}")
        widget.set_active_agents(agent_list)
        agent_index = (agent_index + 1) % len(agents)
        return True
    
    # Change agents every 4 seconds
    GLib.timeout_add(4000, change_agents)


def main():
    """Test the plasma widget"""
    app = XenithApp()
    
    # Store widget reference for testing
    widget_ref = [None]
    
    # Get the widget after activation
    def on_activate(app_instance):
        widget = app_instance._plasma_widget
        widget_ref[0] = widget
        if widget:
            # Start test sequences
            test_state_changes(widget)
            test_agent_colors(widget)
            print("Plasma widget test started!", file=sys.stderr)
            print("Watch the widget change states and agent colors", file=sys.stderr)
            print("Press Ctrl+C to exit", file=sys.stderr)
    
    # Connect before running
    app.connect('activate', on_activate)
    
    # Run the app
    result = app.run(sys.argv)
    return result


if __name__ == '__main__':
    sys.exit(main())

