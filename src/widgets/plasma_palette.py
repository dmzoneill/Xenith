"""Color state for the plasma widget: state colors, transitions and agent tint"""

import time

from .color_manager import ColorManager
from .plasma_easing import RGB, ease_in_out_logarithmic, lerp_color

VALID_STATES = ("idle", "listening", "processing", "responding")

# Base colors for each state
STATE_BASE_COLORS: dict[str, RGB] = {
    "idle": (0.2, 0.4, 0.8),  # Cool blue
    "listening": (1.0, 0.5, 0.2),  # Warm orange
    "processing": (1.0, 0.3, 0.1),  # Hot red/orange
    "responding": (0.6, 0.3, 0.8),  # Purple/violet
}

# State-based opacity (increased for better visibility)
STATE_OPACITIES: dict[str, float] = {
    "idle": 0.7,
    "listening": 0.8,
    "processing": 0.85,
    "responding": 0.8,
}

FALLBACK_COLOR: RGB = (0.5, 0.5, 0.5)


class ColorTransition:
    """Smooth transition between two base colors"""

    def __init__(self, initial: RGB = STATE_BASE_COLORS["idle"], duration: float = 2.0):
        self.current: RGB = initial
        self.target: RGB = initial
        self.progress: float = 1.0  # 0.0 = start, 1.0 = complete
        self.duration: float = duration
        self.start_time: float = 0.0

    @property
    def active(self) -> bool:
        """True while a transition is still in progress"""
        return self.progress < 1.0

    def displayed_color(self) -> RGB:
        """Color currently displayed, interpolated with logarithmic easing"""
        if self.active:
            return lerp_color(self.current, self.target, ease_in_out_logarithmic(self.progress))
        return self.target

    def start(self, target: RGB, now: float | None = None) -> None:
        """Begin a transition from the displayed color to ``target``"""
        # Capture the displayed color BEFORE changing anything to prevent flashes
        captured = self.displayed_color()
        self.current = captured
        self.target = target
        self.progress = 0.0
        self.start_time = time.time() if now is None else now

    def update(self, now: float) -> None:
        """Advance the transition to time ``now``"""
        if not self.active:
            return
        elapsed = now - self.start_time
        self.progress = min(1.0, elapsed / self.duration)
        if self.progress >= 1.0:
            self.current = self.target


class AgentTint:
    """Agent color blended on top of the state color"""

    def __init__(self, color_manager: ColorManager | None = None):
        self.color_manager = color_manager or ColorManager()
        self.color: RGB | None = None  # RGB 0-1
        self.intensity: float = 0.0  # 0.0 to 1.0
        self.active_agents: list[str] = []

    def set_color(self, color: RGB | None, intensity: float) -> None:
        """Set the tint color and clamp its intensity to 0-1"""
        self.color = color
        self.intensity = max(0.0, min(1.0, intensity))

    def set_agents(self, agent_names: list[str]) -> None:
        """Set active agents and derive the tint from their colors"""
        self.active_agents = agent_names
        if agent_names:
            r, g, b, intensity = self.color_manager.blend_agent_colors(agent_names)
            self.set_color((r, g, b), intensity)
        else:
            self.set_color(None, 0.0)

    def apply(self, base: RGB) -> RGB:
        """Blend ``base`` with the agent color if one is set"""
        if self.color and self.intensity > 0:
            blend = self.intensity
            return (
                base[0] * (1 - blend) + self.color[0] * blend,
                base[1] * (1 - blend) + self.color[1] * blend,
                base[2] * (1 - blend) + self.color[2] * blend,
            )
        return base


class PlasmaPalette:
    """Combines the state color transition with the agent tint"""

    def __init__(self):
        self.transition = ColorTransition()
        self.tint = AgentTint()

    def base_color(self) -> RGB:
        """Current base color, blended with the agent color"""
        return self.tint.apply(self.transition.displayed_color())

    def transition_to_state(self, state: str, now: float | None = None) -> None:
        """Start a color transition towards ``state``'s base color"""
        self.transition.start(STATE_BASE_COLORS.get(state, FALLBACK_COLOR), now)
