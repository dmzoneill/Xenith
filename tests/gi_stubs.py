"""Install fake gi / cairo modules so GTK code can be unit tested without a display"""

import sys
import types
from typing import Any
from unittest import mock


class FakeWindow:
    """Minimal stand-in for Gtk.Window"""

    def __init__(self, *_args, **_kwargs):
        self.opacity = 1.0
        self.allocation = types.SimpleNamespace(width=0, height=0)
        self.default_size = None
        self.surface = None
        self.display = None

    def get_opacity(self):
        """Current opacity"""
        return self.opacity

    def set_opacity(self, opacity):
        """Record opacity"""
        self.opacity = opacity

    def get_allocation(self):
        """Current allocation"""
        return self.allocation

    def set_default_size(self, width, height):
        """Record default size"""
        self.default_size = (width, height)

    def get_surface(self):
        """Current surface"""
        return self.surface

    def get_display(self):
        """Current display"""
        return self.display

    def show(self):
        """No-op show"""

    def __getattr__(self, name):
        # Any other GTK method is a no-op mock
        method = mock.MagicMock(name=name)
        setattr(self, name, method)
        return method


class FakeGLibError(Exception):
    """Stand-in for GLib.Error"""


def install() -> dict:
    """Register fake gi and cairo modules; returns the fake namespaces"""
    gi = types.ModuleType("gi")
    vars(gi)["require_version"] = mock.MagicMock()
    repository = types.ModuleType("gi.repository")
    gtk = mock.MagicMock(name="Gtk")
    gtk.Window = FakeWindow
    glib = mock.MagicMock(name="GLib")
    glib.Error = FakeGLibError
    gdk = mock.MagicMock(name="Gdk")
    gdk.Display.get_default.return_value = None
    cairo = mock.MagicMock(name="cairo")
    modules: dict[str, Any] = {
        "gi": gi,
        "gi.repository": repository,
        "gi.repository.Gtk": gtk,
        "gi.repository.GLib": glib,
        "gi.repository.Gdk": gdk,
        "cairo": cairo,
    }
    sys.modules.update(modules)
    return modules
