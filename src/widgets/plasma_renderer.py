"""Cairo drawing routines for the plasma widget"""

import math
from dataclasses import dataclass
from typing import Any

import cairo

from .plasma_easing import RGB
from .plasma_particles import advance_particle, particle_position, particle_size

# (offset, red factor, green factor, blue factor, alpha)
ColorStop = tuple[float, float, float, float, float]


@dataclass
class RenderFrame:
    """Everything needed to draw one frame of the plasma"""

    cr: Any  # cairo.Context
    center: tuple[float, float]
    radius: float
    max_radius: float
    color: RGB
    anim_time: float


def frame_geometry(width: int, height: int, margin: int) -> tuple[tuple[float, float], float, float]:
    """Center, core radius and max radius for a drawing area of ``width`` x ``height``"""
    # Calculate max radius with margin from all edges
    max_radius = (min(width, height) / 2) - margin
    # Core radius is smaller, but glow extends to margin edge
    return (width / 2, height / 2), max_radius * 0.85, max_radius


def fill_radial(frame: RenderFrame, inner: float, outer: float, stops: list[ColorStop]) -> None:
    """Fill a circle of radius ``outer`` with a radial gradient tinted by the frame color"""
    cx, cy = frame.center
    r, g, b = frame.color
    pattern = cairo.RadialGradient(cx, cy, inner, cx, cy, outer)
    for offset, rf, gf, bf, alpha in stops:
        pattern.add_color_stop_rgba(offset, r * rf, g * gf, b * bf, alpha)
    frame.cr.set_source(pattern)
    frame.cr.arc(cx, cy, outer, 0, 2 * math.pi)
    frame.cr.fill()


def _stop(offset: float, factor: float, alpha: float) -> ColorStop:
    return (offset, factor, factor, factor, alpha)


def draw_idle(frame: RenderFrame) -> None:
    """Draw idle state - cool blue tones"""
    radius = frame.radius
    # Outer glow - extends to edges with smooth fade
    fill_radial(frame, radius * 0.6, frame.max_radius, [_stop(0.0, 1, 0.2), _stop(0.7, 1, 0.05), _stop(1.0, 1, 0.0)])
    # Main plasma body - fades smoothly to transparent
    fill_radial(
        frame, 0, radius, [_stop(0.0, 1, 0.5), _stop(0.4, 0.7, 0.3), _stop(0.7, 0.4, 0.15), _stop(1.0, 0.2, 0.0)]
    )
    # Inner core
    fill_radial(frame, 0, radius * 0.35, [_stop(0.0, 1.3, 0.6), _stop(1.0, 1, 0.3)])


def draw_listening(frame: RenderFrame) -> None:
    """Draw listening state - warm orange tones with pulsing"""
    radius = frame.radius
    pulse = 1.0 + 0.1 * math.sin(frame.anim_time * 2)
    pulse2 = 1.0 + 0.05 * math.sin(frame.anim_time * 3)
    # Outer pulsing glow - extends to edges
    glow_radius = min(frame.max_radius, radius * pulse * 1.15)
    fill_radial(frame, radius * 0.7, glow_radius, [_stop(0.0, 1, 0.3), _stop(0.6, 1, 0.1), _stop(1.0, 1, 0.0)])
    # Main body with pulse - fades smoothly
    fill_radial(
        frame,
        0,
        radius * pulse,
        [_stop(0.0, 1, 0.6), _stop(0.4, 0.8, 0.4), _stop(0.7, 0.5, 0.2), _stop(1.0, 0.3, 0.0)],
    )
    # Inner core with secondary pulse
    fill_radial(frame, 0, radius * 0.4 * pulse2, [_stop(0.0, 1.3, 0.7), _stop(1.0, 1, 0.4)])


def draw_processing(frame: RenderFrame) -> None:
    """Draw processing state - hot red/orange tones with intense animation"""
    radius = frame.radius
    pulses = [
        1.0 + 0.2 * math.sin(frame.anim_time * 4),
        1.0 + 0.15 * math.sin(frame.anim_time * 6),
        1.0 + 0.1 * math.sin(frame.anim_time * 8),
    ]
    # Multiple pulsing layers for intensity
    for i, pulse in enumerate(pulses):
        alpha = 0.3 / (i + 1)
        fill_radial(
            frame, 0, radius * pulse, [_stop(0.0, 1, alpha), _stop(0.5, 0.9, alpha * 0.8), _stop(1.0, 0.7, alpha * 0.6)]
        )
    # Hot core: the first stop is a fixed color, not tinted by the state color
    cx, cy = frame.center
    r, g, b = frame.color
    pattern_core = cairo.RadialGradient(cx, cy, 0, cx, cy, radius * 0.3)
    pattern_core.add_color_stop_rgba(0.0, 1.0, 0.5, 0.2, 0.6)
    pattern_core.add_color_stop_rgba(1.0, r, g, b, 0.4)
    frame.cr.set_source(pattern_core)
    frame.cr.arc(cx, cy, radius * 0.3, 0, 2 * math.pi)
    frame.cr.fill()


def draw_responding(frame: RenderFrame) -> None:
    """Draw responding state - purple/violet tones"""
    radius = frame.radius
    wave = 1.0 + 0.05 * math.sin(frame.anim_time * 1.5)
    # Outer glow - constrain to max_radius
    fill_radial(frame, radius * 0.8, min(radius * 1.2, frame.max_radius), [_stop(0.0, 1, 0.2), _stop(1.0, 1, 0.0)])
    # Main body
    fill_radial(frame, 0, radius * wave, [_stop(0.0, 1, 0.4), _stop(0.5, 0.8, 0.3), _stop(1.0, 0.6, 0.2)])
    # Inner core
    fill_radial(frame, 0, radius * 0.35, [_stop(0.0, 1.1, 0.5), _stop(1.0, 1, 0.3)])


STATE_DRAWERS = {
    "idle": draw_idle,
    "listening": draw_listening,
    "processing": draw_processing,
    "responding": draw_responding,
}


def draw_particles(frame: RenderFrame, particles: list[dict]) -> None:
    """Advance and draw floating particles"""
    r, g, b = frame.color
    for particle in particles:
        advance_particle(particle)
        x, y = particle_position(particle, frame.center, frame.radius, frame.anim_time)
        frame.cr.set_source_rgba(r, g, b, 0.6)
        frame.cr.arc(x, y, particle_size(particle, frame.anim_time), 0, 2 * math.pi)
        frame.cr.fill()


def _stroke_ring(frame: RenderFrame, ring_radius: float, alpha: float, line_width: float) -> None:
    r, g, b = frame.color
    frame.cr.set_source_rgba(r, g, b, alpha)
    frame.cr.set_line_width(line_width)
    frame.cr.arc(frame.center[0], frame.center[1], ring_radius, 0, 2 * math.pi)
    frame.cr.stroke()


def draw_waves(frame: RenderFrame, state: str) -> None:
    """Draw wave/ripple effects, constrained to the frame's max radius"""
    radius = frame.radius
    if state == "processing":
        # Intense waves for processing
        for i in range(3):
            wave_radius = radius * (1.2 + i * 0.3 + 0.2 * math.sin(frame.anim_time * 4 + i))
            wave_radius = min(wave_radius, frame.max_radius)
            alpha = 0.15 / (i + 1) * (1.0 - (wave_radius - radius * 1.2) / (radius * 0.6))
            if alpha > 0:
                _stroke_ring(frame, wave_radius, alpha, 2)
    elif state == "listening":
        # Gentle waves for listening
        wave_radius = min(radius * (1.1 + 0.1 * math.sin(frame.anim_time * 2)), frame.max_radius)
        _stroke_ring(frame, wave_radius, 0.1, 1.5)


def draw_frame(frame: RenderFrame, state: str, particles: list[dict]) -> None:
    """Clear the surface and draw the full plasma for ``state``"""
    # Clear background (fully transparent)
    frame.cr.set_operator(cairo.OPERATOR_CLEAR)
    frame.cr.paint()
    frame.cr.set_operator(cairo.OPERATOR_OVER)

    drawer = STATE_DRAWERS.get(state)
    if drawer:
        drawer(frame)
    draw_particles(frame, particles)
    draw_waves(frame, state)
