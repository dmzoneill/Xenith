"""GTK 4 bootstrap: pins the gi versions before loading the namespaces used by the widgets"""

import importlib
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

# gi.require_version() must run before gi.repository is imported
Gdk: Any = importlib.import_module("gi.repository.Gdk")
GLib: Any = importlib.import_module("gi.repository.GLib")
Gtk: Any = importlib.import_module("gi.repository.Gtk")

__all__ = ["Gdk", "GLib", "Gtk"]
