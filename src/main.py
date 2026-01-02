"""Main entry point for Xenith application"""

import sys
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk, Adw
from src.app import XenithApp


def main():
    """Application entry point"""
    app = XenithApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
