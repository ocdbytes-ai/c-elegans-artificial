"""The body.

For world v1 the worm is a single point with a heading — no segments, no
physics. Everything about *how it moves* is here so that swapping in an N-link
body later (PROJECT_PLAN.md §2.4) touches this file and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry
from .config import WorldConfig, WormConfig


@dataclass
class MoveResult:
    """What a single kinematic step actually did — handy for logging."""

    turn_rate: float  # radians / second actually applied
    speed: float  # world units / second actually travelled
    distance: float  # world units covered this step


class Worm:
    """A point body with a heading, driven by ``[turn, throttle]`` in [-1, 1]."""

    def __init__(self, config: WormConfig, world: WorldConfig):
        self.config = config
        self.world = world
        self.position = np.zeros(2, dtype=np.float64)
        self.heading = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        """Drop the worm somewhere uniformly random, facing a random direction."""
        self.position = rng.uniform(0.0, np.array(self.world.size), size=2)
        self.heading = float(rng.uniform(-np.pi, np.pi))

    def step(self, action: np.ndarray, speed_factor: float = 1.0) -> MoveResult:
        """Turn, then translate along the new heading.

        ``speed_factor`` is the metabolic throttle: a starving worm moves
        slower, but never stops (see :class:`envs.metabolism.Metabolism`).
        """
        turn, throttle = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if not self.config.allow_reverse:
            throttle = max(throttle, 0.0)

        dt = self.world.dt
        turn_rate = float(turn) * self.config.max_turn_rate
        self.heading = float(geometry.wrap_angle(self.heading + turn_rate * dt))

        speed = float(throttle) * self.config.max_speed * speed_factor
        step_vector = geometry.heading_vector(self.heading) * speed * dt
        self.position = geometry.apply_boundary(
            self.position + step_vector, self.world.size, self.world.boundary
        )
        return MoveResult(turn_rate=turn_rate, speed=speed, distance=abs(speed) * dt)

    @property
    def heading_sin_cos(self) -> tuple[float, float]:
        """Heading as ``(sin, cos)`` — no 0/2*pi discontinuity for the policy."""
        return float(np.sin(self.heading)), float(np.cos(self.heading))
