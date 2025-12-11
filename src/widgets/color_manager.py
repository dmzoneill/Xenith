"""Color management for agents and plasma integration"""

from typing import Dict, Tuple, Optional
import colorsys


class ColorManager:
    """Manages agent colors and color blending"""
    
    # Predefined agent colors (HSV for easy manipulation)
    AGENT_COLORS: Dict[str, Tuple[float, float, float]] = {
        'general': (0.55, 0.7, 0.9),      # Blue
        'developer': (0.1, 0.8, 0.9),     # Green
        'assistant': (0.0, 0.7, 0.9),     # Red
        'creative': (0.8, 0.7, 0.9),      # Magenta
        'analytical': (0.6, 0.7, 0.9),   # Cyan
        'mcp': (0.3, 0.7, 0.9),           # Yellow
    }
    
    def __init__(self):
        self._agent_colors: Dict[str, Tuple[float, float, float]] = {}
        self._active_agents: list[str] = []
    
    def get_agent_color(self, agent_name: str) -> Tuple[float, float, float]:
        """Get RGB color (0-1) for an agent"""
        if agent_name in self._agent_colors:
            hsv = self._agent_colors[agent_name]
        elif agent_name in self.AGENT_COLORS:
            hsv = self.AGENT_COLORS[agent_name]
        else:
            # Generate color based on agent name hash
            hsv = self._generate_color_from_name(agent_name)
            self._agent_colors[agent_name] = hsv
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
        return rgb
    
    def _generate_color_from_name(self, name: str) -> Tuple[float, float, float]:
        """Generate a color from agent name (deterministic)"""
        # Simple hash-based color generation
        hash_val = hash(name) % 360
        hue = (hash_val / 360.0) % 1.0
        saturation = 0.7
        value = 0.9
        return (hue, saturation, value)
    
    def set_active_agents(self, agent_names: list[str]):
        """Set list of active agents for color blending"""
        self._active_agents = agent_names
    
    def blend_agent_colors(self, agent_names: list[str]) -> Tuple[float, float, float, float]:
        """Blend multiple agent colors and return (R, G, B, intensity)"""
        if not agent_names:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Get colors for all agents
        colors = [self.get_agent_color(name) for name in agent_names]
        
        # Average the colors
        r = sum(c[0] for c in colors) / len(colors)
        g = sum(c[1] for c in colors) / len(colors)
        b = sum(c[2] for c in colors) / len(colors)
        
        # Intensity based on number of agents (more agents = more intense)
        intensity = min(0.7, 0.3 + len(agent_names) * 0.1)
        
        return (r, g, b, intensity)
    
    def get_state_color(self, state: str) -> Tuple[float, float, float]:
        """Get base color for a state"""
        state_colors = {
            'idle': (0.2, 0.4, 0.8),      # Cool blue
            'listening': (1.0, 0.5, 0.2),  # Warm orange
            'processing': (1.0, 0.3, 0.1), # Hot red/orange
            'responding': (0.6, 0.3, 0.8)  # Purple/violet
        }
        return state_colors.get(state, (0.5, 0.5, 0.5))
    
    def register_agent_color(self, agent_name: str, color: Tuple[float, float, float]):
        """Register a custom color for an agent (HSV)"""
        self._agent_colors[agent_name] = color



