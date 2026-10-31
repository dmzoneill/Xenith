"""X11 window manager helpers (xdotool, xprop, wmctrl) for the plasma widget"""

import importlib
import subprocess  # nosec B404
from typing import Optional

from .gtk4 import GLib
from .plasma_geometry import Area

COMMAND_ERRORS = (OSError, subprocess.SubprocessError)


def get_xid(surface) -> Optional[int]:
    """X11 window id of a GDK surface, or None if it is not an X11 surface"""
    if not surface:
        return None
    try:
        # GdkX11 is only importable on X11 sessions, so resolve it lazily
        gdk_x11 = importlib.import_module("gi.repository.GdkX11")
    except (ImportError, ValueError):
        return None
    if isinstance(surface, gdk_x11.X11Surface):
        return gdk_x11.X11Surface.get_xid(surface)
    return None


def run_quiet(args: list[str], timeout: float = 0.1) -> bool:
    """Run a command with output discarded; returns False if it could not run"""
    try:
        subprocess.run(  # nosec B603 B607
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False
        )
    except COMMAND_ERRORS:
        return False
    return True


def run_capture(args: list[str], timeout: float) -> Optional[str]:
    """Run a command and return its stdout, or None if it failed"""
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    if result.returncode != 0:
        return None
    return result.stdout


def spawn_quiet(args: list[str]) -> bool:
    """Start a command without waiting for it; returns False if it could not start"""
    flags = GLib.SpawnFlags.SEARCH_PATH | GLib.SpawnFlags.STDOUT_TO_DEV_NULL | GLib.SpawnFlags.STDERR_TO_DEV_NULL
    try:
        GLib.spawn_async(args, flags=flags)
    except GLib.Error:
        return False
    return True


def display_geometry() -> Optional[Area]:
    """Screen size from ``xdotool getdisplaygeometry``, or None if unavailable"""
    try:
        output = run_capture(["xdotool", "getdisplaygeometry"], timeout=0.1)
        if output is None:
            return None
        # Parse output: "WIDTH HEIGHT"
        parts = output.strip().split()
        if len(parts) < 2:
            return None
        return Area(0, 0, int(parts[0]), int(parts[1]))
    except COMMAND_ERRORS + (ValueError,):
        return None


def window_position(xid: int) -> Optional[dict[str, int]]:
    """Window X/Y from ``xdotool getwindowgeometry``; {} if xdotool failed, None on error"""
    try:
        output = run_capture(["xdotool", "getwindowgeometry", "--shell", str(xid)], timeout=0.05)
        if output is None:
            return {}
        position = {}
        for line in output.split("\n"):
            if line.startswith("X=") or line.startswith("Y="):
                key, value = line.split("=", 1)
                position[key] = int(value)
        return position
    except COMMAND_ERRORS + (ValueError,):
        return None


def move_window(xid: int, x: int, y: int) -> bool:
    """Move a window and wait for xdotool (short timeout)"""
    return run_quiet(["xdotool", "windowmove", str(xid), str(x), str(y)])


def move_window_async(xid: int, x: int, y: int) -> bool:
    """Move a window without waiting for xdotool"""
    return spawn_quiet(["xdotool", "windowmove", str(xid), str(x), str(y)])


def set_net_wm_state(xid: int, *states: str) -> bool:
    """Set the _NET_WM_STATE atoms of a window"""
    return run_quiet(["xprop", "-id", str(xid), "-f", "_NET_WM_STATE", "32a", "-set", "_NET_WM_STATE", *states])


def set_utility_type(xid: int) -> bool:
    """Set window type to UTILITY (helps skip taskbar/pager)"""
    return run_quiet(
        [
            "xprop",
            "-id",
            str(xid),
            "-f",
            "_NET_WM_WINDOW_TYPE",
            "32a",
            "-set",
            "_NET_WM_WINDOW_TYPE",
            "_NET_WM_WINDOW_TYPE_UTILITY",
        ]
    )


def set_window_above(xid: int) -> None:
    """Set window to stay above using xprop, then wmctrl and xdotool if xprop could run"""
    if not set_net_wm_state(xid, "_NET_WM_STATE_ABOVE"):
        return
    run_quiet(["wmctrl", "-i", "-r", str(xid), "-b", "add,above"])
    run_quiet(["xdotool", "windowraise", str(xid)])
