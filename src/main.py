"""Main entry point for Xenith application"""

import sys

from src.app import XenithApp


def main():
    """Application entry point"""
    app = XenithApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
