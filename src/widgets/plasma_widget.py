"""Plasma widget - floating animated widget"""

import time
from dataclasses import dataclass

from . import x11_window
from .gtk4 import Gdk, GLib, Gtk
from .plasma_geometry import (
    CORNERS,
    Area,
    WindowPlacement,
    corner_position,
    dominant_delta,
    in_resize_corner,
    resized_size,
)
from .plasma_palette import STATE_OPACITIES, VALID_STATES, PlasmaPalette
from .plasma_particles import ParticleSystem
from .plasma_renderer import RenderFrame, draw_frame, frame_geometry

WINDOW_CSS = b"""
        window {
            background-color: transparent;
        }
        """

# Extra retries for always-on-top (some WMs need this), in milliseconds
ALWAYS_ON_TOP_RETRIES_MS = (50, 100, 200, 500)


@dataclass
class DragState:
    """Mouse drag / resize tracking"""

    dragging: bool = False
    resizing: bool = False
    start: tuple[float, float] = (0.0, 0.0)
    resize_start_width: int = 0
    window_start: tuple[int, int] = (0, 0)
    last_move: tuple[int, int] | None = None
    cached_xid: int | None = None


@dataclass
class AnimationClock:
    """Animation timing state"""

    anim_time: float = 0.0
    start_time: float = 0.0
    last_time: float = 0.0
    animation_id: int | None = None
    needs_redraw: bool = True
    last_state_change: float = 0.0

    def advance(self, now: float) -> None:
        """Advance animation time using real elapsed time (clamped to prevent large jumps)"""
        delta = now - self.last_time
        self.last_time = now
        self.anim_time += min(delta, 0.1)  # Max 0.1s per frame
        if self.anim_time > 1000:  # Reset to prevent overflow
            self.anim_time = 0.0


def log_renderer(display) -> None:
    """Report the GTK4 rendering backend (Wayland/GTK4 uses GPU-accelerated compositing)"""
    if not display:
        return
    try:
        # get_renderer() is only available on Wayland, not X11
        renderer = display.get_renderer()
    except AttributeError:
        display_name = display.get_name() if hasattr(display, "get_name") else "X11"
        print(f"[PLASMA] Display: {display_name} (renderer info not available)")
        return
    if renderer:
        renderer_type = renderer.get_type().name
        if "GL" in renderer_type or "Vulkan" in renderer_type:
            print(f"[PLASMA] ✓ GPU-accelerated compositing: {renderer_type}")
        else:
            print(f"[PLASMA] Rendering backend: {renderer_type}")


class PlasmaWidget(Gtk.Window):
    """Floating plasma widget with animations"""

    DEFAULT_SIZE = 120
    MIN_SIZE = 32
    MAX_SIZE = 500
    DEFAULT_OPACITY = 0.75  # Increased from 0.3 for better visibility
    ANIMATION_MARGIN = 10  # Pixels margin from edges for animations

    def __init__(self, app):
        super().__init__(application=app)
        log_renderer(Gdk.Display.get_default())

        # Window properties
        self.set_decorated(False)  # No window borders
        self.set_resizable(True)
        self.set_default_size(self.DEFAULT_SIZE, self.DEFAULT_SIZE)

        # Make window background fully transparent using CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(WINDOW_CSS)
        self.get_style_context().add_provider(css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.set_opacity(self.DEFAULT_OPACITY)

        # Drawing area for the plasma effect (GPU-accelerated compositing on Wayland)
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_draw_func(self.on_draw)
        self.drawing_area.set_can_focus(True)
        self.set_child(self.drawing_area)

        self.connect("notify::default-width", self._on_size_changed)
        self.connect("notify::default-height", self._on_size_changed)
        # Window type hint and skip taskbar/pager are set via X11 after realization
        self.connect("realize", self._on_realize)
        self._add_controllers()

        self._drag = DragState()
        self._clock = AnimationClock()
        self._state = "idle"  # idle, listening, processing, responding
        self._palette = PlasmaPalette()
        self._particles = ParticleSystem(15)
        self._placement = WindowPlacement(width=self.DEFAULT_SIZE, height=self.DEFAULT_SIZE)

        self.start_animation()
        self.update_position()

    @property
    def state(self) -> str:
        """Current widget state (idle, listening, processing, responding)"""
        return self._state

    def _add_controllers(self) -> None:
        """Event controllers for mouse events (GTK4 way)"""
        click_controller = Gtk.GestureClick()
        click_controller.set_button(1)  # Left button
        click_controller.connect("pressed", self.on_button_press)
        click_controller.connect("released", self.on_button_release)
        self.drawing_area.add_controller(click_controller)

        # Drag gesture for window dragging, only on drawing_area to avoid duplicate events
        drag_controller = Gtk.GestureDrag()
        drag_controller.set_button(1)  # Left button
        drag_controller.connect("drag-begin", self.on_drag_begin)
        drag_controller.connect("drag-update", self.on_drag_update)
        drag_controller.connect("drag-end", self.on_drag_end)
        self.drawing_area.add_controller(drag_controller)

        # Motion controller for resizing
        motion_controller = Gtk.EventControllerMotion()
        motion_controller.connect("motion", self.on_motion)
        self.drawing_area.add_controller(motion_controller)

    def _allocated_size(self) -> tuple[int, int]:
        """Current size from the allocation, falling back to the tracked size"""
        alloc = self.get_allocation()
        width = alloc.width if alloc.width > 0 else self._placement.width
        height = alloc.height if alloc.height > 0 else self._placement.height
        return width, height

    def on_draw(self, _area, cr, width: int, height: int):
        """Draw the plasma effect"""
        center, radius, max_radius = frame_geometry(width, height, self.ANIMATION_MARGIN)
        frame = RenderFrame(cr, center, radius, max_radius, self._palette.base_color(), self._clock.anim_time)
        draw_frame(frame, self._state, self._particles.particles)

    def on_button_press(self, _controller, n_press: int, x: float, y: float) -> None:
        """Handle mouse button press"""
        if n_press == 2:
            # Double click - open popup (TODO: implement popup)
            return
        size = self._allocated_size()
        # Store initial click position (relative to widget)
        self._drag.start = (x, y)
        # Check if click is in corner (resize area)
        if in_resize_corner((x, y), size):
            self._drag.resizing = True
            self._drag.resize_start_width = size[0]

    def on_button_release(self, *_args) -> None:
        """Handle mouse button release"""
        self._drag.resizing = False

    def _begin_native_move(self, gesture, start_x: float, start_y: float) -> bool:
        """Delegate the drag to the window manager; returns True if it took over"""
        toplevel = self.get_native()
        surface = toplevel.get_surface() if toplevel else None
        if not surface or not hasattr(surface, "begin_move"):
            return False
        timestamp = Gdk.CURRENT_TIME
        try:
            event = gesture.get_last_event(None)
            if event:
                timestamp = event.get_time()
        except (TypeError, AttributeError):
            pass  # Keep Gdk.CURRENT_TIME
        try:
            surface.begin_move(gesture.get_device(), gesture.get_current_button(), start_x, start_y, timestamp)
        except (TypeError, GLib.Error):
            return False  # Fall through to manual drag
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        return True

    def on_drag_begin(self, gesture, start_x: float, start_y: float) -> None:
        """Handle drag begin - use native window manager drag for smooth movement"""
        if in_resize_corner((start_x, start_y), self._allocated_size()):
            # Don't drag if in resize corner
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return
        if self._begin_native_move(gesture, start_x, start_y):
            return

        # Fallback to manual dragging if native method unavailable
        self._drag.dragging = True
        self._drag.start = (start_x, start_y)
        xid = x11_window.get_xid(self.get_surface())
        if xid is None:
            return
        position = x11_window.window_position(xid)
        if position is None:
            position = {"X": self._placement.window_x, "Y": self._placement.window_y}
        start = self._drag.window_start
        self._drag.window_start = (position.get("X", start[0]), position.get("Y", start[1]))

    def on_drag_update(self, _gesture, offset_x: float, offset_y: float) -> None:
        """Handle drag update - only used for fallback manual dragging"""
        if not self._drag.dragging:
            return
        new_pos = (int(self._drag.window_start[0] + offset_x), int(self._drag.window_start[1] + offset_y))
        # Skip if position unchanged
        if new_pos == self._drag.last_move:
            return
        self._drag.last_move = new_pos
        self._placement.window_x, self._placement.window_y = new_pos

        # Use xdotool for fallback (Xlib has issues with WM interaction)
        if self._drag.cached_xid is None:
            self._drag.cached_xid = x11_window.get_xid(self.get_surface())
        if self._drag.cached_xid:
            x11_window.move_window_async(self._drag.cached_xid, *new_pos)

    def on_drag_end(self, *_args) -> None:
        """Handle drag end"""
        self._drag.dragging = False
        self._drag.last_move = None
        self._drag.cached_xid = None

    def on_motion(self, _controller, x: float, y: float) -> None:
        """Handle mouse motion (for resizing only now)"""
        if not self._drag.resizing:
            return
        # Square resize based on relative mouse movement (positive = grow, negative = shrink)
        delta = dominant_delta(x - self._drag.start[0], y - self._drag.start[1])
        new_size = resized_size(self._drag.resize_start_width, delta, (self.MIN_SIZE, self.MAX_SIZE))
        self._placement.width = new_size
        self._placement.height = new_size
        self.set_default_size(new_size, new_size)
        self.set_size_request(new_size, new_size)

    def start_animation(self):
        """Start the animation loop on main GTK thread"""
        now = time.time()
        self._clock.start_time = now
        self._clock.last_time = now
        self._clock.needs_redraw = True
        # Start with 30fps for idle state (still animated, saves CPU/GPU)
        self._clock.animation_id = GLib.timeout_add(33, self._on_animation_tick)

    def _animate(self) -> bool:
        """Advance one animation frame and queue a redraw"""
        now = time.time()
        self._clock.advance(now)
        self._palette.transition.update(now)

        # Update state-based opacity - only if changed significantly
        opacity = STATE_OPACITIES.get(self._state, self.DEFAULT_OPACITY)
        if abs(opacity - self.get_opacity()) > 0.01:
            Gtk.Window.set_opacity(self, opacity)

        # Always queue redraw to keep animations (particles, waves) moving
        self.drawing_area.queue_draw()
        self._clock.needs_redraw = False
        return True

    def _on_animation_tick(self) -> bool:
        """Animate, then schedule the next frame: 30fps when idle, 60fps during states/transitions"""
        self._animate()
        idle = self._state == "idle" and not self._palette.transition.active
        self._clock.animation_id = GLib.timeout_add(33 if idle else 16, self._on_animation_tick)
        return False  # Don't repeat (we schedule manually)

    def set_state(self, state: str):
        """Set the widget state (idle, listening, processing, responding)"""
        if state not in VALID_STATES or self._state == state:
            return
        # Don't allow immediate return to idle within 2 seconds of listening; this prevents
        # the flash when wake word detection triggers multiple times
        if state == "idle" and self._state == "listening" and time.time() - self._clock.last_state_change < 2.0:
            return

        # Set up the transition before updating the state (prevents race conditions)
        self._palette.transition_to_state(state)
        self._state = state
        self._clock.last_state_change = time.time()
        super().set_opacity(STATE_OPACITIES.get(state, self.DEFAULT_OPACITY))
        self.drawing_area.queue_draw()

    def set_agent_color(self, color: tuple[float, float, float] | None, intensity: float = 0.5):
        """Set agent color to blend with plasma (RGB 0-1, intensity 0-1)"""
        self._palette.tint.set_color(color, intensity)
        self.drawing_area.queue_draw()

    def set_active_agents(self, agent_names: list[str]):
        """Set active agents and update plasma color"""
        self._palette.tint.set_agents(agent_names)
        self.drawing_area.queue_draw()

    def set_particle_count(self, count: int):
        """Set number of particles (0 to disable)"""
        self._particles.set_count(count)

    def set_opacity(self, opacity: float):
        """Set widget opacity (0.0 to 1.0)"""
        Gtk.Window.set_opacity(self, max(0.0, min(1.0, opacity)))

    def set_corner(self, corner: str):
        """Set widget corner position"""
        if corner in CORNERS:
            self._placement.corner = corner
            self.update_position()

    def _monitor_area(self) -> Area | None:
        """Primary monitor geometry from GTK, if available"""
        display = self.get_display()
        try:
            monitor = display.get_primary_monitor() if display else None
        except AttributeError:
            return None  # Position will be managed by window manager
        if not monitor:
            return None
        geometry = monitor.get_geometry()
        return Area(geometry.x, geometry.y, geometry.width, geometry.height)

    def update_position(self):
        """Update widget position based on corner setting"""
        size = self._allocated_size()
        # Screen dimensions from xdotool (more reliable), falling back to GTK monitor geometry
        area = x11_window.display_geometry() or self._monitor_area()
        if area is None:
            return
        position = corner_position(self._placement.corner, area, size, self._placement.offset)
        if position is None:
            return
        self._placement.window_x, self._placement.window_y = position
        xid = x11_window.get_xid(self.get_surface())
        if xid:
            x11_window.move_window(xid, *position)

    def _on_realize(self, _widget):
        """Called when window is realized - set window properties via X11 window manager hints"""
        xid = x11_window.get_xid(self.get_surface())
        if xid:
            x11_window.set_utility_type(xid)
            x11_window.set_window_above(xid)
            GLib.idle_add(self._apply_position)
        for delay in ALWAYS_ON_TOP_RETRIES_MS:
            GLib.timeout_add(delay, self._set_always_on_top)

    def _apply_position(self):
        """Apply the window position using xdotool"""
        xid = x11_window.get_xid(self.get_surface())
        if xid:
            # Recalculate position to ensure we have correct size
            self.update_position()
            x11_window.move_window_async(xid, self._placement.window_x, self._placement.window_y)
        return False

    def _set_always_on_top(self):
        """Set always-on-top property via window manager (retry)"""
        xid = x11_window.get_xid(self.get_surface())
        if xid:
            x11_window.set_window_above(xid)
            # Also set skip taskbar and pager
            x11_window.set_net_wm_state(
                xid, "_NET_WM_STATE_ABOVE", "_NET_WM_STATE_SKIP_TASKBAR", "_NET_WM_STATE_SKIP_PAGER"
            )
            self._apply_position()
        return False  # Don't repeat

    def _on_size_changed(self, *_args):
        """Handle size changes"""
        alloc = self.get_allocation()
        if alloc.width > 0:
            self._placement.width = alloc.width
        if alloc.height > 0:
            self._placement.height = alloc.height

    def show(self):
        """Show the widget"""
        super().show()
        GLib.idle_add(self.update_position)
