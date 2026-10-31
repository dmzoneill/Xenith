"""Easing and interpolation helpers for the plasma widget"""

import math

RGB = tuple[float, float, float]


def ease_in_out_cubic(t: float) -> float:
    """Easing function for smooth transitions (ease-in-out cubic)"""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_out_quintic(t: float) -> float:
    """Smoother easing function (ease-in-out quintic) for very smooth transitions"""
    if t < 0.5:
        return 16 * t * t * t * t * t
    return 1 - pow(-2 * t + 2, 5) / 2


def ease_in_out_logarithmic(t: float) -> float:
    """Logarithmic easing for very smooth fade in/out transitions"""
    t = min(max(t, 0.0), 1.0)
    if t < 0.5:
        # Ease in (logarithmic growth)
        return 0.5 * math.log(2 * t + 1) / math.log(2)
    # Ease out (logarithmic decay)
    return 1.0 - 0.5 * math.log(2 * (1 - t) + 1) / math.log(2)


def lerp_color(start: RGB, end: RGB, t: float) -> RGB:
    """Linearly interpolate between two RGB colors"""
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
    )
