"""Plasma widget - floating animated widget"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from typing import Optional, List, Tuple
import math
import random
import sys
import os
import time
import cairo
import subprocess
from .color_manager import ColorManager


class PlasmaWidget(Gtk.Window):
    """Floating plasma widget with animations"""

    DEFAULT_SIZE = 120
    MIN_SIZE = 32
    MAX_SIZE = 500
    DEFAULT_OPACITY = 0.75  # Increased from 0.3 for better visibility
    ANIMATION_MARGIN = 10  # Pixels margin from edges for animations

    def __init__(self, app: Gtk.Application):
        super().__init__(application=app)

        # Check GTK4 rendering backend (Wayland/GTK4 uses GPU-accelerated compositing)
        display = Gdk.Display.get_default()
        if display:
            # get_renderer() is only available on Wayland, not X11
            try:
                renderer = display.get_renderer()
                if renderer:
                    renderer_type = renderer.get_type().name
                    gpu_available = "GL" in renderer_type or "Vulkan" in renderer_type
                    if gpu_available:
                        print(
                            f"[PLASMA] ✓ GPU-accelerated compositing: {renderer_type}"
                        )
                    else:
                        print(f"[PLASMA] Rendering backend: {renderer_type}")
            except AttributeError:
                # X11 display doesn't have get_renderer()
                display_name = (
                    display.get_name() if hasattr(display, "get_name") else "X11"
                )
                print(f"[PLASMA] Display: {display_name} (renderer info not available)")

        # Window properties
        self.set_decorated(False)  # No window borders
        self.set_resizable(True)
        self.set_default_size(self.DEFAULT_SIZE, self.DEFAULT_SIZE)

        # Make window background fully transparent using CSS
        # In GTK4, we handle transparency via CSS and draw handlers
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
        window {
            background-color: transparent;
        }
        """
        )
        style_context = self.get_style_context()
        style_context.add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Set opacity
        self.set_opacity(self.DEFAULT_OPACITY)

        # Note: Window type hint and skip taskbar/pager will be set via X11
        # in _on_realize() since GTK4 removed these methods

        # Create drawing area for plasma effect
        # GTK4's DrawingArea uses GPU-accelerated compositing on Wayland
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_draw_func(self.on_draw)

        # In GTK4, we need to make the widget receive events
        # Use add_controller for event handling
        self.drawing_area.set_can_focus(True)

        self.set_child(self.drawing_area)

        # Track size changes
        self.connect("notify::default-width", self._on_size_changed)
        self.connect("notify::default-height", self._on_size_changed)

        # Set window properties after realization
        self.connect("realize", self._on_realize)

        # Event controllers for mouse events (GTK4 way)
        click_controller = Gtk.GestureClick()
        click_controller.set_button(1)  # Left button
        click_controller.connect("pressed", self.on_button_press)
        click_controller.connect("released", self.on_button_release)
        self.drawing_area.add_controller(click_controller)

        # Drag gesture for window dragging (GTK4 native)
        # Only add to drawing_area (not window) to avoid duplicate events
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

        # Dragging state
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_root_x = 0
        self._drag_start_root_y = 0
        self._window_start_x = 0
        self._window_start_y = 0
        self._resizing = False
        self._window_x = 0
        self._window_y = 0
        self._current_width = self.DEFAULT_SIZE

        # Rate limiting for drag updates (prevent too many subprocess calls)
        self._last_drag_update_time = 0
        self._drag_update_interval = (
            0.005  # ~200fps max update rate for very smooth dragging
        )
        self._pending_drag_position = None
        self._drag_process = None  # Track the last subprocess to avoid queue buildup
        self._current_height = self.DEFAULT_SIZE

        # Animation state
        self._animation_time = 0.0
        self._state = "idle"  # idle, listening, processing, responding

        # Color transition state
        self._current_base_color: Tuple[float, float, float] = (
            0.2,
            0.4,
            0.8,
        )  # Start with idle color
        self._target_base_color: Tuple[float, float, float] = (
            0.2,
            0.4,
            0.8,
        )  # Target color
        self._color_transition_progress: float = 1.0  # 0.0 = start, 1.0 = complete
        self._color_transition_duration: float = (
            2.0  # Duration in seconds for color transition (smooth, gradual transitions)
        )
        self._color_transition_start_time: float = 0.0
        self._last_state_change_time: float = (
            0.0  # Track when state last changed for debouncing
        )
        self._needs_redraw: bool = (
            True  # Track if redraw is needed (performance optimization)
        )

        # Particle system
        self._particles: List[dict] = []
        self._particle_count = 15
        self._init_particles()

        # Agent color integration
        self._color_manager = ColorManager()
        self._agent_color: Optional[Tuple[float, float, float]] = None  # RGB 0-1
        self._agent_color_intensity = 0.0  # 0.0 to 1.0
        self._active_agents: List[str] = []

        # Wave/ripple effects
        self._waves: List[dict] = []

        # State-based opacity (increased for better visibility)
        self._state_opacities = {
            "idle": 0.7,
            "listening": 0.8,
            "processing": 0.85,
            "responding": 0.8,
        }

        # Start animation timer
        self._animation_id = None
        self.start_animation()

        # Position in corner (default: bottom-left with 20px offset)
        self._corner = "bottom-left"
        self._position_offset = 20  # Offset from edges in pixels
        self.update_position()

    def _init_particles(self):
        """Initialize particle system"""
        self._particles = []
        for _ in range(self._particle_count):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.3, 0.9)
            speed = random.uniform(0.5, 1.5)
            size = random.uniform(1, 3)
            self._particles.append(
                {
                    "angle": angle,
                    "distance": distance,
                    "speed": speed,
                    "size": size,
                    "phase": random.uniform(0, 2 * math.pi),
                }
            )

    def on_draw(self, area: Gtk.DrawingArea, cr, width: int, height: int):
        """Draw the plasma effect"""
        # Clear background (fully transparent)
        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        # Get center
        center_x = width / 2
        center_y = height / 2
        # Calculate max radius with 10px margin from all edges
        max_radius = (min(width, height) / 2) - self.ANIMATION_MARGIN
        # Core radius is smaller, but glow extends to margin edge
        radius = max_radius * 0.85

        # Draw base plasma effect based on state
        if self._state == "idle":
            self._draw_idle_plasma(cr, center_x, center_y, radius, width, height)
        elif self._state == "listening":
            self._draw_listening_plasma(cr, center_x, center_y, radius, width, height)
        elif self._state == "processing":
            self._draw_processing_plasma(cr, center_x, center_y, radius, width, height)
        elif self._state == "responding":
            self._draw_responding_plasma(cr, center_x, center_y, radius, width, height)

        # Draw particles
        self._draw_particles(cr, center_x, center_y, radius)

        # Draw waves/ripples (pass max_radius to constrain them)
        self._draw_waves(cr, center_x, center_y, radius, max_radius)

    def _ease_in_out_cubic(self, t: float) -> float:
        """Easing function for smooth transitions (ease-in-out cubic)"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def _ease_in_out_quintic(self, t: float) -> float:
        """Smoother easing function (ease-in-out quintic) for very smooth transitions"""
        if t < 0.5:
            return 16 * t * t * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 5) / 2

    def _ease_in_out_logarithmic(self, t: float) -> float:
        """Logarithmic easing for very smooth fade in/out transitions"""
        # Logarithmic ease-in-out: smooth acceleration and deceleration
        # Uses a logarithmic curve for natural-feeling transitions
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0

        # Logarithmic ease-in-out approximation
        # This creates a very smooth S-curve
        if t < 0.5:
            # Ease in (logarithmic growth)
            return 0.5 * math.log(2 * t + 1) / math.log(2)
        else:
            # Ease out (logarithmic decay)
            return 1.0 - 0.5 * math.log(2 * (1 - t) + 1) / math.log(2)

    def _get_base_color(self, state: str) -> Tuple[float, float, float]:
        """Get base color for state, blended with agent color"""
        # During a transition, always interpolate from _current_base_color to _target_base_color
        # This ensures smooth transitions regardless of which state's drawing function is called
        if self._color_transition_progress < 1.0:
            # We're transitioning - interpolate from current to target using logarithmic easing
            eased_t = self._ease_in_out_logarithmic(self._color_transition_progress)
            base_r = (
                self._current_base_color[0]
                + (self._target_base_color[0] - self._current_base_color[0]) * eased_t
            )
            base_g = (
                self._current_base_color[1]
                + (self._target_base_color[1] - self._current_base_color[1]) * eased_t
            )
            base_b = (
                self._current_base_color[2]
                + (self._target_base_color[2] - self._current_base_color[2]) * eased_t
            )
        else:
            # Transition complete - use the target color
            base_r, base_g, base_b = self._target_base_color

        # Blend with agent color if set
        if self._agent_color and self._agent_color_intensity > 0:
            agent_r, agent_g, agent_b = self._agent_color
            blend = self._agent_color_intensity
            r = base_r * (1 - blend) + agent_r * blend
            g = base_g * (1 - blend) + agent_g * blend
            b = base_b * (1 - blend) + agent_b * blend
            return (r, g, b)

        return (base_r, base_g, base_b)

    def _draw_idle_plasma(
        self,
        cr,
        center_x: float,
        center_y: float,
        radius: float,
        width: int,
        height: int,
    ):
        """Draw idle state - cool blue tones"""
        r, g, b = self._get_base_color("idle")
        # Max radius with margin from edges
        max_radius = (min(width, height) / 2) - self.ANIMATION_MARGIN

        # Outer glow - extends to edges with smooth fade
        pattern1 = cairo.RadialGradient(
            center_x, center_y, radius * 0.6, center_x, center_y, max_radius
        )
        pattern1.add_color_stop_rgba(0.0, r, g, b, 0.2)
        pattern1.add_color_stop_rgba(0.7, r, g, b, 0.05)
        pattern1.add_color_stop_rgba(1.0, r, g, b, 0.0)  # Fully transparent at edge
        cr.set_source(pattern1)
        cr.arc(center_x, center_y, max_radius, 0, 2 * math.pi)
        cr.fill()

        # Main plasma body - fades smoothly to transparent
        pattern2 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius
        )
        pattern2.add_color_stop_rgba(0.0, r, g, b, 0.5)
        pattern2.add_color_stop_rgba(0.4, r * 0.7, g * 0.7, b * 0.7, 0.3)
        pattern2.add_color_stop_rgba(0.7, r * 0.4, g * 0.4, b * 0.4, 0.15)
        pattern2.add_color_stop_rgba(
            1.0, r * 0.2, g * 0.2, b * 0.2, 0.0
        )  # Fade to transparent

        cr.set_source(pattern2)
        cr.arc(center_x, center_y, radius, 0, 2 * math.pi)
        cr.fill()

        # Inner core
        pattern3 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * 0.35
        )
        pattern3.add_color_stop_rgba(0.0, r * 1.3, g * 1.3, b * 1.3, 0.6)
        pattern3.add_color_stop_rgba(1.0, r, g, b, 0.3)
        cr.set_source(pattern3)
        cr.arc(center_x, center_y, radius * 0.35, 0, 2 * math.pi)
        cr.fill()

    def _draw_listening_plasma(
        self,
        cr,
        center_x: float,
        center_y: float,
        radius: float,
        width: int,
        height: int,
    ):
        """Draw listening state - warm orange tones with pulsing"""
        r, g, b = self._get_base_color("listening")
        # Max radius with margin from edges
        max_radius = (min(width, height) / 2) - self.ANIMATION_MARGIN

        # Pulsing effect
        pulse = 1.0 + 0.1 * math.sin(self._animation_time * 2)
        pulse2 = 1.0 + 0.05 * math.sin(self._animation_time * 3)

        # Outer pulsing glow - extends to edges
        glow_radius = min(max_radius, radius * pulse * 1.15)
        pattern1 = cairo.RadialGradient(
            center_x, center_y, radius * 0.7, center_x, center_y, glow_radius
        )
        pattern1.add_color_stop_rgba(0.0, r, g, b, 0.3)
        pattern1.add_color_stop_rgba(0.6, r, g, b, 0.1)
        pattern1.add_color_stop_rgba(1.0, r, g, b, 0.0)  # Fully transparent at edge
        cr.set_source(pattern1)
        cr.arc(center_x, center_y, glow_radius, 0, 2 * math.pi)
        cr.fill()

        # Main body with pulse - fades smoothly
        pattern2 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * pulse
        )
        pattern2.add_color_stop_rgba(0.0, r, g, b, 0.6)
        pattern2.add_color_stop_rgba(0.4, r * 0.8, g * 0.8, b * 0.8, 0.4)
        pattern2.add_color_stop_rgba(0.7, r * 0.5, g * 0.5, b * 0.5, 0.2)
        pattern2.add_color_stop_rgba(
            1.0, r * 0.3, g * 0.3, b * 0.3, 0.0
        )  # Fade to transparent

        cr.set_source(pattern2)
        cr.arc(center_x, center_y, radius * pulse, 0, 2 * math.pi)
        cr.fill()

        # Inner core with secondary pulse
        pattern3 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * 0.4 * pulse2
        )
        pattern3.add_color_stop_rgba(0.0, r * 1.3, g * 1.3, b * 1.3, 0.7)
        pattern3.add_color_stop_rgba(1.0, r, g, b, 0.4)
        cr.set_source(pattern3)
        cr.arc(center_x, center_y, radius * 0.4 * pulse2, 0, 2 * math.pi)
        cr.fill()

    def _draw_processing_plasma(
        self,
        cr,
        center_x: float,
        center_y: float,
        radius: float,
        width: int,
        height: int,
    ):
        """Draw processing state - hot red/orange tones with intense animation"""
        r, g, b = self._get_base_color("processing")

        # Intense pulsing
        pulse = 1.0 + 0.2 * math.sin(self._animation_time * 4)
        pulse2 = 1.0 + 0.15 * math.sin(self._animation_time * 6)
        pulse3 = 1.0 + 0.1 * math.sin(self._animation_time * 8)

        # Multiple pulsing layers for intensity
        for i, p in enumerate([pulse, pulse2, pulse3]):
            alpha = 0.3 / (i + 1)
            pattern = cairo.RadialGradient(
                center_x, center_y, 0, center_x, center_y, radius * p
            )
            pattern.add_color_stop_rgba(0.0, r, g, b, alpha)
            pattern.add_color_stop_rgba(0.5, r * 0.9, g * 0.9, b * 0.9, alpha * 0.8)
            pattern.add_color_stop_rgba(1.0, r * 0.7, g * 0.7, b * 0.7, alpha * 0.6)

            cr.set_source(pattern)
            cr.arc(center_x, center_y, radius * p, 0, 2 * math.pi)
            cr.fill()

        # Hot core
        pattern_core = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * 0.3
        )
        pattern_core.add_color_stop_rgba(0.0, 1.0, 0.5, 0.2, 0.6)
        pattern_core.add_color_stop_rgba(1.0, r, g, b, 0.4)
        cr.set_source(pattern_core)
        cr.arc(center_x, center_y, radius * 0.3, 0, 2 * math.pi)
        cr.fill()

    def _draw_responding_plasma(
        self,
        cr,
        center_x: float,
        center_y: float,
        radius: float,
        width: int,
        height: int,
    ):
        """Draw responding state - purple/violet tones"""
        r, g, b = self._get_base_color("responding")
        # Max radius with margin from edges
        max_radius = (min(width, height) / 2) - self.ANIMATION_MARGIN

        # Gentle wave effect
        wave = 1.0 + 0.05 * math.sin(self._animation_time * 1.5)

        # Outer glow - constrain to max_radius
        glow_radius = min(radius * 1.2, max_radius)
        pattern1 = cairo.RadialGradient(
            center_x, center_y, radius * 0.8, center_x, center_y, glow_radius
        )
        pattern1.add_color_stop_rgba(0.0, r, g, b, 0.2)
        pattern1.add_color_stop_rgba(1.0, r, g, b, 0.0)
        cr.set_source(pattern1)
        cr.arc(center_x, center_y, glow_radius, 0, 2 * math.pi)
        cr.fill()

        # Main body
        pattern2 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * wave
        )
        pattern2.add_color_stop_rgba(0.0, r, g, b, 0.4)
        pattern2.add_color_stop_rgba(0.5, r * 0.8, g * 0.8, b * 0.8, 0.3)
        pattern2.add_color_stop_rgba(1.0, r * 0.6, g * 0.6, b * 0.6, 0.2)

        cr.set_source(pattern2)
        cr.arc(center_x, center_y, radius * wave, 0, 2 * math.pi)
        cr.fill()

        # Inner core
        pattern3 = cairo.RadialGradient(
            center_x, center_y, 0, center_x, center_y, radius * 0.35
        )
        pattern3.add_color_stop_rgba(0.0, r * 1.1, g * 1.1, b * 1.1, 0.5)
        pattern3.add_color_stop_rgba(1.0, r, g, b, 0.3)
        cr.set_source(pattern3)
        cr.arc(center_x, center_y, radius * 0.35, 0, 2 * math.pi)
        cr.fill()

    def _draw_particles(self, cr, center_x: float, center_y: float, radius: float):
        """Draw floating particles"""
        if not self._particles:
            return

        # Update and draw particles
        for particle in self._particles:
            # Update particle position
            particle["angle"] += particle["speed"] * 0.01
            if particle["angle"] > 2 * math.pi:
                particle["angle"] -= 2 * math.pi

            # Calculate position - ensure particles stay within margin
            # Use slightly smaller radius to account for particle size
            max_particle_dist = radius * 0.9  # Leave room for particle size
            dist = max_particle_dist * particle["distance"]
            x = center_x + dist * math.cos(
                particle["angle"] + self._animation_time * particle["speed"]
            )
            y = center_y + dist * math.sin(
                particle["angle"] + self._animation_time * particle["speed"]
            )

            # Get color based on state
            r, g, b = self._get_base_color(self._state)

            # Particle size varies with animation
            size = particle["size"] * (
                1.0 + 0.3 * math.sin(self._animation_time * 2 + particle["phase"])
            )

            # Draw particle
            cr.set_source_rgba(r, g, b, 0.6)
            cr.arc(x, y, size, 0, 2 * math.pi)
            cr.fill()

    def _draw_waves(
        self, cr, center_x: float, center_y: float, radius: float, max_radius: float
    ):
        """Draw wave/ripple effects"""
        # Add waves based on state activity
        if self._state == "processing":
            # Intense waves for processing - constrain to max_radius
            for i in range(3):
                wave_radius = radius * (
                    1.2 + i * 0.3 + 0.2 * math.sin(self._animation_time * 4 + i)
                )
                # Ensure waves don't exceed the margin
                wave_radius = min(wave_radius, max_radius)
                alpha = (
                    0.15
                    / (i + 1)
                    * (1.0 - (wave_radius - radius * 1.2) / (radius * 0.6))
                )
                if alpha > 0:
                    r, g, b = self._get_base_color(self._state)
                    cr.set_source_rgba(r, g, b, alpha)
                    cr.set_line_width(2)
                    cr.arc(center_x, center_y, wave_radius, 0, 2 * math.pi)
                    cr.stroke()

        elif self._state == "listening":
            # Gentle waves for listening - constrain to max_radius
            wave_radius = radius * (1.1 + 0.1 * math.sin(self._animation_time * 2))
            # Ensure waves don't exceed the margin
            wave_radius = min(wave_radius, max_radius)
            r, g, b = self._get_base_color(self._state)
            cr.set_source_rgba(r, g, b, 0.1)
            cr.set_line_width(1.5)
            cr.arc(center_x, center_y, wave_radius, 0, 2 * math.pi)
            cr.stroke()

    def on_button_press(
        self, controller: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """Handle mouse button press"""
        if n_press == 2:
            # Double click - open popup (TODO: implement popup)
            return

        # Get current size from allocation
        alloc = self.get_allocation()
        width = alloc.width if alloc.width > 0 else self._current_width
        height = alloc.height if alloc.height > 0 else self._current_height
        corner_size = 20

        # Store initial click position (relative to widget)
        self._drag_start_x = x
        self._drag_start_y = y

        # Check if click is in corner (resize area)
        if x > width - corner_size and y > height - corner_size:
            # Start resizing
            self._resizing = True
            self._resize_start_width = width
            self._resize_start_height = height

    def on_button_release(
        self, controller: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """Handle mouse button release"""
        self._resizing = False

    def on_drag_begin(
        self, gesture: Gtk.GestureDrag, start_x: float, start_y: float
    ) -> None:
        """Handle drag begin - use native window manager drag for smooth movement"""
        # Check if we're in resize area (corner)
        alloc = self.get_allocation()
        width = alloc.width if alloc.width > 0 else self._current_width
        height = alloc.height if alloc.height > 0 else self._current_height
        corner_size = 20

        if start_x > width - corner_size and start_y > height - corner_size:
            # Don't drag if in resize corner
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return

        # Use GTK4's native toplevel drag - this delegates to the window manager
        # which handles the drag smoothly without coordinate desync issues
        toplevel = self.get_native()
        if toplevel:
            surface = toplevel.get_surface()
            if surface:
                # Get the device (mouse/touch) from the gesture
                device = gesture.get_device()

                # Get the button that triggered the drag
                button = gesture.get_current_button()

                # Get timestamp - try multiple methods
                timestamp = Gdk.CURRENT_TIME
                try:
                    event = gesture.get_last_event(None)
                    if event:
                        timestamp = event.get_time()
                except:
                    pass

                # Initiate native window move - window manager handles tracking
                try:
                    # GTK4 ToplevelSurface.begin_move() for native WM drag
                    if hasattr(surface, "begin_move"):
                        surface.begin_move(device, button, start_x, start_y, timestamp)
                        # Don't set _dragging - WM handles it
                        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
                        return
                except Exception:
                    pass  # Fall through to manual drag

        # Fallback to manual dragging if native method unavailable
        self._dragging = True
        self._drag_start_x = start_x
        self._drag_start_y = start_y

        # Get window position for manual drag
        try:
            from gi.repository import GdkX11

            surface = self.get_surface()
            if surface and isinstance(surface, GdkX11.X11Surface):
                xid = GdkX11.X11Surface.get_xid(surface)
                result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", str(xid)],
                    capture_output=True,
                    text=True,
                    timeout=0.05,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("X="):
                            self._window_start_x = int(line.split("=")[1])
                        elif line.startswith("Y="):
                            self._window_start_y = int(line.split("=")[1])
        except Exception:
            self._window_start_x = getattr(self, "_window_x", 0)
            self._window_start_y = getattr(self, "_window_y", 0)

    def on_drag_update(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        """Handle drag update - only used for fallback manual dragging"""
        # If native WM drag is active, this won't be called (WM handles movement)
        if not self._dragging:
            return

        # Manual fallback drag - only used if begin_move wasn't available
        new_x = int(self._window_start_x + offset_x)
        new_y = int(self._window_start_y + offset_y)

        # Skip if position unchanged
        if hasattr(self, "_last_move_x") and hasattr(self, "_last_move_y"):
            if new_x == self._last_move_x and new_y == self._last_move_y:
                return

        self._last_move_x = new_x
        self._last_move_y = new_y
        self._window_x = new_x
        self._window_y = new_y

        # Use xdotool for fallback (Xlib has issues with WM interaction)
        if not hasattr(self, "_cached_xid"):
            try:
                from gi.repository import GdkX11

                surface = self.get_surface()
                if surface and isinstance(surface, GdkX11.X11Surface):
                    self._cached_xid = GdkX11.X11Surface.get_xid(surface)
            except:
                self._cached_xid = None

        if self._cached_xid:
            try:
                subprocess.Popen(
                    [
                        "xdotool",
                        "windowmove",
                        str(self._cached_xid),
                        str(new_x),
                        str(new_y),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except:
                pass

    def on_drag_end(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        """Handle drag end"""
        self._dragging = False

        # Reset tracking state
        for attr in ["_last_move_x", "_last_move_y", "_cached_xid"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def on_motion(
        self, controller: Gtk.EventControllerMotion, x: float, y: float
    ) -> None:
        """Handle mouse motion (for resizing only now)"""
        if self._resizing:
            # Resize based on relative mouse movement
            dx = x - self._drag_start_x
            dy = y - self._drag_start_y
            # Use the larger of dx or dy for square resize
            delta = max(abs(dx), abs(dy))
            # Determine direction (positive = grow, negative = shrink)
            if abs(dx) > abs(dy):
                delta = dx if dx > 0 else dx
            else:
                delta = dy if dy > 0 else dy

            new_size = max(
                self.MIN_SIZE, min(self.MAX_SIZE, self._resize_start_width + delta)
            )
            self._current_width = new_size
            self._current_height = new_size
            self.set_default_size(new_size, new_size)
            self.set_size_request(new_size, new_size)

    def start_animation(self):
        """Start the animation loop on main GTK thread"""
        self._animation_start_time = time.time()
        self._last_animation_time = time.time()
        self._needs_redraw = True  # Track if redraw is needed

        def animate():
            # Use actual elapsed time for smoother animation
            current_time = time.time()
            delta = current_time - self._last_animation_time
            self._last_animation_time = current_time

            needs_update = False

            # Update animation time (clamp delta to prevent large jumps)
            self._animation_time += min(delta, 0.1)  # Max 0.1s per frame
            if self._animation_time > 1000:  # Reset to prevent overflow
                self._animation_time = 0.0

            # Update color transition (lightweight operation)
            if self._color_transition_progress < 1.0:
                elapsed = current_time - self._color_transition_start_time
                old_progress = self._color_transition_progress
                self._color_transition_progress = min(
                    1.0, elapsed / self._color_transition_duration
                )

                # When transition completes, update current to target
                if self._color_transition_progress >= 1.0:
                    self._current_base_color = self._target_base_color

                # Only redraw if progress actually changed (skip redundant frames)
                if (
                    abs(self._color_transition_progress - old_progress) > 0.01
                ):  # 1% change threshold
                    needs_update = True

            # Update state-based opacity (lightweight) - only if changed
            opacity = self._state_opacities.get(self._state, self.DEFAULT_OPACITY)
            current_opacity = self.get_opacity()
            if (
                abs(opacity - current_opacity) > 0.01
            ):  # Only update if significant change
                Gtk.Window.set_opacity(self, opacity)

            # Always queue redraw to keep animations (particles, waves) moving
            # The conditional check was preventing animations in idle state
            self.drawing_area.queue_draw()
            self._needs_redraw = False

            return True  # Continue animation

        # Use adaptive frame rate: 30fps when idle (still animated), 60fps during transitions
        # This saves CPU/GPU while keeping animations alive
        # Simple approach: always call animate, but adjust interval based on state
        def animate_with_adaptive_rate():
            animate()  # Always animate to keep particles/waves moving
            # Schedule next frame with appropriate rate
            if self._state == "idle" and self._color_transition_progress >= 1.0:
                interval = (
                    33  # 30fps for idle state (still animated, but saves resources)
                )
            else:
                interval = 16  # 60fps during active states/transitions
            self._animation_id = GLib.timeout_add(interval, animate_with_adaptive_rate)
            return False  # Don't repeat (we schedule manually)

        # Start with 30fps for idle state (still animated, saves CPU/GPU)
        self._animation_id = GLib.timeout_add(
            33, animate_with_adaptive_rate
        )  # 30fps initially

    def set_state(self, state: str):
        """Set the widget state (idle, listening, processing, responding)"""
        if state in ["idle", "listening", "processing", "responding"]:
            # Only start transition if state actually changed
            if self._state == state:
                return

            # Prevent rapid state changes (time-based debounce)
            # Only prevent rapid toggling between listening and idle (within 2 seconds)
            current_time = time.time()
            if state == "idle" and self._state == "listening":
                # Don't allow immediate return to idle if we just went to listening (< 2 seconds ago)
                # This prevents the flash when wake word detection triggers multiple times
                # But allow it after 2 seconds (legitimate timeout)
                if current_time - self._last_state_change_time < 2.0:
                    return

            # Base colors for each state
            base_colors = {
                "idle": (0.2, 0.4, 0.8),  # Cool blue
                "listening": (1.0, 0.5, 0.2),  # Warm orange
                "processing": (1.0, 0.3, 0.1),  # Hot red/orange
                "responding": (0.6, 0.3, 0.8),  # Purple/violet
            }

            # Capture the actual current displayed color BEFORE changing anything
            # This is critical to prevent flashes - we need the exact color being displayed right now
            if self._color_transition_progress < 1.0:
                # We're in the middle of a transition, so calculate current visual color using logarithmic easing
                eased_t = self._ease_in_out_logarithmic(self._color_transition_progress)
                current_r = (
                    self._current_base_color[0]
                    + (self._target_base_color[0] - self._current_base_color[0])
                    * eased_t
                )
                current_g = (
                    self._current_base_color[1]
                    + (self._target_base_color[1] - self._current_base_color[1])
                    * eased_t
                )
                current_b = (
                    self._current_base_color[2]
                    + (self._target_base_color[2] - self._current_base_color[2])
                    * eased_t
                )
                captured_color = (current_r, current_g, current_b)
            else:
                # Transition complete, current color is the target
                captured_color = self._target_base_color

            # Now atomically set up the new transition
            # Set current to captured color FIRST, then target, then reset progress
            # This ensures _get_base_color always has consistent values
            self._current_base_color = captured_color
            self._target_base_color = base_colors.get(state, (0.5, 0.5, 0.5))
            self._color_transition_progress = 0.0
            self._color_transition_start_time = time.time()

            # Update state AFTER transition is set up (prevents race conditions)
            self._state = state
            self._last_state_change_time = time.time()  # Track when state changed
            # Update opacity based on state
            opacity = self._state_opacities.get(state, self.DEFAULT_OPACITY)
            super().set_opacity(opacity)
            self.drawing_area.queue_draw()

    def set_agent_color(
        self, color: Optional[Tuple[float, float, float]], intensity: float = 0.5
    ):
        """Set agent color to blend with plasma (RGB 0-1, intensity 0-1)"""
        self._agent_color = color
        self._agent_color_intensity = max(0.0, min(1.0, intensity))
        self.drawing_area.queue_draw()

    def set_active_agents(self, agent_names: List[str]):
        """Set active agents and update plasma color"""
        self._active_agents = agent_names
        if agent_names:
            r, g, b, intensity = self._color_manager.blend_agent_colors(agent_names)
            self.set_agent_color((r, g, b), intensity)
        else:
            self.set_agent_color(None, 0.0)
        self.drawing_area.queue_draw()

    def set_particle_count(self, count: int):
        """Set number of particles (0 to disable)"""
        self._particle_count = max(0, count)
        if count > len(self._particles):
            # Add more particles
            for _ in range(count - len(self._particles)):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0.3, 0.9)
                speed = random.uniform(0.5, 1.5)
                size = random.uniform(1, 3)
                self._particles.append(
                    {
                        "angle": angle,
                        "distance": distance,
                        "speed": speed,
                        "size": size,
                        "phase": random.uniform(0, 2 * math.pi),
                    }
                )
        else:
            # Remove excess particles
            self._particles = self._particles[:count]

    def set_opacity(self, opacity: float):
        """Set widget opacity (0.0 to 1.0)"""
        opacity = max(0.0, min(1.0, opacity))
        Gtk.Window.set_opacity(self, opacity)

    def set_corner(self, corner: str):
        """Set widget corner position"""
        if corner in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            self._corner = corner
            self.update_position()

    def update_position(self):
        """Update widget position based on corner setting"""
        try:
            # Get actual window size from allocation or use current size
            alloc = self.get_allocation()
            width = alloc.width if alloc.width > 0 else self._current_width
            height = alloc.height if alloc.height > 0 else self._current_height
            offset = getattr(self, "_position_offset", 20)

            # Try to get screen dimensions using xdotool (more reliable)
            try:
                result = subprocess.run(
                    ["xdotool", "getdisplaygeometry"],
                    capture_output=True,
                    text=True,
                    timeout=0.1,
                )
                if result.returncode == 0:
                    # Parse output: "WIDTH HEIGHT"
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        screen_width = int(parts[0])
                        screen_height = int(parts[1])

                        if self._corner == "top-left":
                            x = offset
                            y = offset
                        elif self._corner == "top-right":
                            x = screen_width - width - offset
                            y = offset
                        elif self._corner == "bottom-left":
                            x = offset
                            # Calculate Y from bottom of screen
                            y = screen_height - height - offset
                        elif self._corner == "bottom-right":
                            x = screen_width - width - offset
                            y = screen_height - height - offset
                        else:
                            return

                        self._window_x = x
                        self._window_y = y

                        # Move window to position using xdotool
                        try:
                            surface = self.get_surface()
                            if surface:
                                from gi.repository import GdkX11

                                if isinstance(surface, GdkX11.X11Surface):
                                    xid = GdkX11.X11Surface.get_xid(surface)
                                    subprocess.run(
                                        [
                                            "xdotool",
                                            "windowmove",
                                            str(xid),
                                            str(x),
                                            str(y),
                                        ],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        timeout=0.1,
                                    )
                        except:
                            pass
                        return
            except:
                pass

            # Fallback: use GTK4 monitor geometry
            display = self.get_display()
            if display:
                monitor = display.get_primary_monitor()
                if monitor:
                    geometry = monitor.get_geometry()

                    if self._corner == "top-left":
                        x = geometry.x + offset
                        y = geometry.y + offset
                    elif self._corner == "top-right":
                        x = geometry.x + geometry.width - width - offset
                        y = geometry.y + offset
                    elif self._corner == "bottom-left":
                        x = geometry.x + offset
                        # Calculate Y from bottom: monitor bottom - window height - offset from bottom
                        y = geometry.y + geometry.height - height - offset
                    elif self._corner == "bottom-right":
                        x = geometry.x + geometry.width - width - offset
                        y = geometry.y + geometry.height - height - offset
                    else:
                        return

                    self._window_x = x
                    self._window_y = y

                    # Move window to position using xdotool if available
                    try:
                        surface = self.get_surface()
                        if surface:
                            from gi.repository import GdkX11

                            if isinstance(surface, GdkX11.X11Surface):
                                xid = GdkX11.X11Surface.get_xid(surface)
                                subprocess.run(
                                    ["xdotool", "windowmove", str(xid), str(x), str(y)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    timeout=0.1,
                                )
                    except:
                        pass
        except Exception as e:
            # Fallback if display/monitor access fails
            pass  # Silently fail - position will be managed by window manager

    def _on_realize(self, widget):
        """Called when window is realized - set window properties"""
        # Set window properties using X11 window manager hints
        try:
            from gi.repository import GdkX11

            surface = self.get_surface()
            if surface and isinstance(surface, GdkX11.X11Surface):
                # Get the X11 window ID
                xid = GdkX11.X11Surface.get_xid(surface)

                # Set window type to UTILITY (helps skip taskbar/pager)
                subprocess.run(
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
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.1,
                )

                # Set always-on-top using multiple methods for reliability
                self._set_window_above(xid)

                # Set position immediately after realization
                GLib.idle_add(self._apply_position)
        except Exception:
            # Fallback: window type hint should help
            pass

        # Retry setting always-on-top multiple times (some WMs need this)
        GLib.timeout_add(50, self._set_always_on_top)
        GLib.timeout_add(100, self._set_always_on_top)
        GLib.timeout_add(200, self._set_always_on_top)
        GLib.timeout_add(500, self._set_always_on_top)  # Extra retry

    def _apply_position(self):
        """Apply the window position using xdotool"""
        try:
            from gi.repository import GdkX11

            surface = self.get_surface()
            if surface and isinstance(surface, GdkX11.X11Surface):
                xid = GdkX11.X11Surface.get_xid(surface)
                # Recalculate position to ensure we have correct size
                self.update_position()
                # Apply position
                if hasattr(self, "_window_x") and hasattr(self, "_window_y"):
                    subprocess.Popen(
                        [
                            "xdotool",
                            "windowmove",
                            str(xid),
                            str(self._window_x),
                            str(self._window_y),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception:
            pass
        return False

    def _set_window_above(self, xid):
        """Set window to stay above using multiple methods"""
        try:
            # Method 1: Use xprop to set _NET_WM_STATE_ABOVE
            subprocess.run(
                [
                    "xprop",
                    "-id",
                    str(xid),
                    "-f",
                    "_NET_WM_STATE",
                    "32a",
                    "-set",
                    "_NET_WM_STATE",
                    "_NET_WM_STATE_ABOVE",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.1,
            )

            # Method 2: Also try wmctrl if available
            try:
                subprocess.run(
                    ["wmctrl", "-i", "-r", str(xid), "-b", "add,above"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.1,
                )
            except:
                pass

            # Method 3: Use xdotool to set window above
            try:
                subprocess.run(
                    ["xdotool", "windowraise", str(xid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.1,
                )
            except:
                pass
        except:
            pass

    def _set_always_on_top(self):
        """Set always-on-top property via window manager (retry)"""
        try:
            from gi.repository import GdkX11

            surface = self.get_surface()
            if surface and isinstance(surface, GdkX11.X11Surface):
                xid = GdkX11.X11Surface.get_xid(surface)

                # Use multiple methods to ensure it stays on top
                self._set_window_above(xid)

                # Also set skip taskbar and pager
                subprocess.run(
                    [
                        "xprop",
                        "-id",
                        str(xid),
                        "-f",
                        "_NET_WM_STATE",
                        "32a",
                        "-set",
                        "_NET_WM_STATE",
                        "_NET_WM_STATE_ABOVE",
                        "_NET_WM_STATE_SKIP_TASKBAR",
                        "_NET_WM_STATE_SKIP_PAGER",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.1,
                )

                # Also retry position
                self._apply_position()
        except Exception:
            # Silently fail - not all systems support this
            pass
        return False  # Don't repeat

    def _on_size_changed(self, widget, param):
        """Handle size changes"""
        alloc = self.get_allocation()
        if alloc.width > 0:
            self._current_width = alloc.width
        if alloc.height > 0:
            self._current_height = alloc.height

    def show(self):
        """Show the widget"""
        super().show()
        # Update position after showing
        GLib.idle_add(self.update_position)
