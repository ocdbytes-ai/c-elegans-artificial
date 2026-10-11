"""The worm's body.

For world v1 the body is a single point with a heading: no segments, no physics
beyond kinematics. Everything about how the worm moves lives here, so replacing
it with an N-link chain later touches this module alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry
from .config import RandomizationConfig, WorldConfig, WormConfig


@dataclass
class MoveResult:
    """Outcome of a single kinematic step.

    Attributes:
        turn_rate: Angular velocity applied, in radians per second.
        speed: Linear velocity commanded, in world units per second.
        distance: World units actually covered, after the boundary.
        blocked: Share of the commanded motion the boundary discarded, in
            ``[0, 1]``.
    """

    turn_rate: float
    speed: float
    distance: float
    blocked: float


class Worm:
    """A point body with a heading, driven by ``[turn, throttle]`` in ``[-1, 1]``.

    Attributes:
        config: Nominal body parameters.
        world: Arena dimensions and boundary convention.
        randomization: Per-episode actuation spread.
        position: Current position in world coordinates.
        heading: Current heading in radians.
        max_speed: This episode's speed ceiling. Redrawn each reset when
            randomisation is enabled, so the body is one sample from a band
            rather than a fixed calibration.
        max_turn_rate: This episode's turn-rate ceiling.
        blocked: Share of the last step's commanded motion the boundary
            refused, which the ``touch`` observation channel reports.
        phase: Head sweep phase in radians. The head is on the worm's left
            while ``sin(phase)`` is positive. Stays 0 at reset when the worm
            does not sweep.
    """

    def __init__(
        self, config: WormConfig, world: WorldConfig, randomization: RandomizationConfig
    ):
        """Initialises a body at the origin with nominal actuation.

        Args:
            config: Nominal body parameters.
            world: Arena dimensions and boundary convention.
            randomization: Per-episode actuation spread.
        """
        self.config = config
        self.world = world
        self.randomization = randomization
        self.position = np.zeros(2, dtype=np.float64)
        self.heading = 0.0
        self.max_speed = config.max_speed
        self.max_turn_rate = config.max_turn_rate
        self.blocked = 0.0
        self.phase = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        """Places the worm randomly and draws a fresh body for the episode.

        Args:
            rng: Source of randomness, owned by the environment.
        """
        self.position = rng.uniform(0.0, np.array(self.world.size), size=2)
        self.heading = float(rng.uniform(-np.pi, np.pi))
        self.max_speed, self.max_turn_rate = self._sample_actuation(rng)
        self.blocked = 0.0
        # Drawn only when sweeping. Food and toxin are scattered from the same
        # RNG right after this, so an unconditional draw would move every
        # pellet and make old seeds give different worlds.
        sweeping = self.config.head_sweep_amplitude > 0
        self.phase = float(rng.uniform(-np.pi, np.pi)) if sweeping else 0.0

    def _sample_actuation(self, rng: np.random.Generator) -> tuple[float, float]:
        """Draws this episode's speed and turn-rate ceilings.

        Args:
            rng: Source of randomness.

        Returns:
            ``(max_speed, max_turn_rate)`` for the coming episode.
        """
        if not self.randomization.enabled:
            return self.config.max_speed, self.config.max_turn_rate
        return (
            float(self.config.max_speed * rng.uniform(*self.randomization.speed_scale)),
            float(self.config.max_turn_rate * rng.uniform(*self.randomization.turn_rate_scale)),
        )

    def step(self, action: np.ndarray, speed_factor: float = 1.0) -> MoveResult:
        """Turns, then translates along the new heading.

        Args:
            action: ``[turn, throttle]``, each a fraction of the corresponding
                ceiling. Values outside ``[-1, 1]`` are clipped.
            speed_factor: Metabolic multiplier on speed. A starving worm moves
                slower but never stops; see :class:`envs.metabolism.Metabolism`.

        Returns:
            What the step achieved, including how much the boundary blocked.
        """
        turn, throttle = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if not self.config.allow_reverse:
            throttle = max(throttle, 0.0)

        dt = self.world.dt
        turn_rate = float(turn) * self.max_turn_rate
        self.heading = float(geometry.wrap_angle(self.heading + turn_rate * dt))

        speed = float(throttle) * self.max_speed * speed_factor
        commanded = abs(speed) * dt
        step_vector = geometry.heading_vector(self.heading) * speed * dt

        start = self.position
        self.position = geometry.apply_boundary(
            start + step_vector, self.world.size, self.world.boundary
        )
        # Measured under the arena's own convention, so a wrap-mode crossing
        # reads as the short hop it physically is rather than as an obstruction.
        travelled = float(
            geometry.distance(start, self.position, self.world.size, self.world.boundary)
        )
        blocked = (
            0.0 if commanded < 1e-12 else float(np.clip(1.0 - travelled / commanded, 0.0, 1.0))
        )
        # Snap float residue so unobstructed motion rests at exactly zero.
        self.blocked = 0.0 if blocked < 1e-9 else blocked
        # Advanced by time, not distance: a fixed rhythm the policy steers on
        # top of, so a worm that has stopped still samples both sides.
        self.phase = float(
            geometry.wrap_angle(self.phase + 2.0 * np.pi * dt / self.config.head_sweep_period)
        )
        return MoveResult(
            turn_rate=turn_rate, speed=speed, distance=travelled, blocked=self.blocked
        )

    @property
    def heading_sin_cos(self) -> tuple[float, float]:
        """Heading as ``(sin, cos)``, avoiding the 0/2pi discontinuity."""
        return float(np.sin(self.heading)), float(np.cos(self.heading))

    @property
    def head_position(self) -> np.ndarray:
        """Where the chemosensors are: the body point, swung sideways by the sweep.

        The swing runs along the left-hand normal ``(-sin(heading),
        cos(heading))`` with length ``head_sweep_amplitude * sin(phase)``. The
        arena boundary applies to the head as it does to the body, so under
        ``clamp`` the head cannot sense through a wall. Equal to ``position``
        when the worm does not sweep.
        """
        sin_heading, cos_heading = self.heading_sin_cos
        swing = self.config.head_sweep_amplitude * np.sin(self.phase)
        head = self.position + swing * np.array([-sin_heading, cos_heading])
        return geometry.apply_boundary(head, self.world.size, self.world.boundary)

    @property
    def head_phase_sin_cos(self) -> tuple[float, float]:
        """Sweep phase as ``(sin, cos)``, for the proprioception channels."""
        return float(np.sin(self.phase)), float(np.cos(self.phase))
