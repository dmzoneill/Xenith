"""X11 window manager helpers (xdotool, xprop, wmctrl) for the plasma widget"""

import importlib
import subprocess  # nosec B404

from .gtk4 import GLib
from .plasma_geometry import Area

COMMAND_ERRORS = (OSError, subprocess.SubprocessError)


def get_xid(surface) -> int | None:
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


def _completed(run, *args: str, timeout: float = 0.1) -> bool:
    """Call ``run`` (one of the tool runners below); returns False if the tool could not run"""
    try:
        run(*args, timeout=timeout)
    except COMMAND_ERRORS:
        return False
    return True


def _xdotool(*args: str, timeout: float) -> None:
    subprocess.run(  # nosec B603 B607
        ["xdotool", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False
    )


def _xprop(*args: str, timeout: float) -> None:
    subprocess.run(  # nosec B603 B607
        ["xprop", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False
    )


def _wmctrl(*args: str, timeout: float) -> None:
    subprocess.run(  # nosec B603 B607
        ["wmctrl", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False
    )


def run_xdotool(*args: str, timeout: float = 0.1) -> bool:
    """Run xdotool with output discarded; returns False if it could not run"""
    return _completed(_xdotool, *args, timeout=timeout)


def run_xprop(*args: str, timeout: float = 0.1) -> bool:
    """Run xprop with output discarded; returns False if it could not run"""
    return _completed(_xprop, *args, timeout=timeout)


def run_wmctrl(*args: str, timeout: float = 0.1) -> bool:
    """Run wmctrl with output discarded; returns False if it could not run"""
    return _completed(_wmctrl, *args, timeout=timeout)


def xdotool_output(*args: str, timeout: float) -> str | None:
    """Run xdotool and return its stdout, or None if it failed"""
    result = subprocess.run(  # nosec B603 B607
        ["xdotool", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
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


def display_geometry() -> Area | None:
    """Screen size from ``xdotool getdisplaygeometry``, or None if unavailable"""
    try:
        output = xdotool_output("getdisplaygeometry", timeout=0.1)
        if output is None:
            return None
        # Parse output: "WIDTH HEIGHT"
        parts = output.strip().split()
        if len(parts) < 2:
            return None
        return Area(0, 0, int(parts[0]), int(parts[1]))
    except (*COMMAND_ERRORS, ValueError):
        return None


def window_position(xid: int) -> dict[str, int] | None:
    """Window X/Y from ``xdotool getwindowgeometry``; {} if xdotool failed, None on error"""
    try:
        output = xdotool_output("getwindowgeometry", "--shell", str(xid), timeout=0.05)
        if output is None:
            return {}
        position = {}
        for line in output.split("\n"):
            if line.startswith(("X=", "Y=")):
                key, value = line.split("=", 1)
                position[key] = int(value)
        return position
    except (*COMMAND_ERRORS, ValueError):
        return None


def move_window(xid: int, x: int, y: int) -> bool:
    """Move a window and wait for xdotool (short timeout)"""
    return run_xdotool("windowmove", str(xid), str(x), str(y))


def move_window_async(xid: int, x: int, y: int) -> bool:
    """Move a window without waiting for xdotool"""
    return spawn_quiet(["xdotool", "windowmove", str(xid), str(x), str(y)])


def set_net_wm_state(xid: int, *states: str) -> bool:
    """Set the _NET_WM_STATE atoms of a window"""
    return run_xprop("-id", str(xid), "-f", "_NET_WM_STATE", "32a", "-set", "_NET_WM_STATE", *states)


def set_utility_type(xid: int) -> bool:
    """Set window type to UTILITY (helps skip taskbar/pager)"""
    return run_xprop(
        "-id",
        str(xid),
        "-f",
        "_NET_WM_WINDOW_TYPE",
        "32a",
        "-set",
        "_NET_WM_WINDOW_TYPE",
        "_NET_WM_WINDOW_TYPE_UTILITY",
    )


def set_window_above(xid: int) -> None:
    """Set window to stay above using xprop, then wmctrl and xdotool if xprop could run"""
    if not set_net_wm_state(xid, "_NET_WM_STATE_ABOVE"):
        return
    run_wmctrl("-i", "-r", str(xid), "-b", "add,above")
    run_xdotool("windowraise", str(xid))
