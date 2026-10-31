"""Particle system for the plasma widget"""

import math
import random


def new_particle() -> dict:
    """Create a particle with a random orbit"""
    angle = random.uniform(0, 2 * math.pi)  # nosec B311
    distance = random.uniform(0.3, 0.9)  # nosec B311
    speed = random.uniform(0.5, 1.5)  # nosec B311
    size = random.uniform(1, 3)  # nosec B311
    return {
        "angle": angle,
        "distance": distance,
        "speed": speed,
        "size": size,
        "phase": random.uniform(0, 2 * math.pi),  # nosec B311
    }


def advance_particle(particle: dict) -> None:
    """Advance a particle's orbit angle by one frame"""
    particle["angle"] += particle["speed"] * 0.01
    if particle["angle"] > 2 * math.pi:
        particle["angle"] -= 2 * math.pi


def particle_position(particle: dict, center: tuple[float, float], radius: float, anim_time: float):
    """Position of a particle, kept within 90% of ``radius`` to leave room for its size"""
    dist = radius * 0.9 * particle["distance"]
    phase = particle["angle"] + anim_time * particle["speed"]
    return (center[0] + dist * math.cos(phase), center[1] + dist * math.sin(phase))


def particle_size(particle: dict, anim_time: float) -> float:
    """Particle size, varying with the animation"""
    return particle["size"] * (1.0 + 0.3 * math.sin(anim_time * 2 + particle["phase"]))


class ParticleSystem:
    """Holds the plasma particles"""

    def __init__(self, count: int = 15):
        self.count = count
        self.particles: list[dict] = []
        self.waves: list[dict] = []
        self.reset()

    def reset(self) -> None:
        """Recreate all particles"""
        self.particles = [new_particle() for _ in range(self.count)]

    def set_count(self, count: int) -> None:
        """Set number of particles (0 to disable)"""
        self.count = max(0, count)
        if count > len(self.particles):
            self.particles.extend(new_particle() for _ in range(count - len(self.particles)))
        else:
            self.particles = self.particles[:count]
