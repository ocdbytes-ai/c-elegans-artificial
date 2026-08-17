"""Energy: the reason the worm has to do anything at all.

Per step::

    energy -= basal_cost                  # existing is not free
    energy -= move_cost * |action|^2      # moving is a little more expensive
    energy += food_value                  # on eat
    dead    = energy <= 0

Two rules are enforced by the config validators and implemented here:

1. ``basal_cost`` dominates ``move_cost``. If idling were cheap enough,
   standing still would be the optimal policy and training would stop there.
2. ``speed_factor`` is a smooth ramp, never a cliff. A hard threshold makes
   low-energy states unrecoverable — the worm slows below the speed it needs to
   reach food and every state under the threshold becomes a guaranteed death.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MetabolismConfig


@dataclass
class EnergyLedger:
    """Energy in and out for one step. The agent never sees this; you do."""

    basal_cost: float = 0.0
    move_cost: float = 0.0
    intake: float = 0.0

    @property
    def net(self) -> float:
        return self.intake - self.basal_cost - self.move_cost


class Metabolism:
    def __init__(self, config: MetabolismConfig):
        self.config = config
        self.energy = config.initial_energy
        self.ledger = EnergyLedger()

    def reset(self) -> None:
        self.energy = self.config.initial_energy
        self.ledger = EnergyLedger()

    def spend(self, action: np.ndarray) -> EnergyLedger:
        """Charge one step of living and moving. Call before applying the move."""
        effort = float(np.sum(np.square(np.clip(action, -1.0, 1.0))))
        self.ledger = EnergyLedger(
            basal_cost=self.config.basal_cost,
            move_cost=self.config.move_cost * effort,
        )
        self.energy -= self.ledger.basal_cost + self.ledger.move_cost
        return self.ledger

    def eat(self, pellets: int) -> float:
        """Absorb ``pellets`` worth of food, capped at ``max_energy``."""
        if pellets <= 0:
            return 0.0
        before = self.energy
        self.energy = min(self.energy + pellets * self.config.food_value, self.config.max_energy)
        gained = self.energy - before
        self.ledger.intake = gained
        return gained

    @property
    def energy_fraction(self) -> float:
        """Energy in [0, 1] — what the worm actually senses (interoception)."""
        return float(np.clip(self.energy / self.config.max_energy, 0.0, 1.0))

    @property
    def is_dead(self) -> bool:
        return self.energy <= 0.0

    @property
    def speed_factor(self) -> float:
        """Smooth movement multiplier in ``[min_speed_factor, 1]``.

        Uses a smoothstep ramp over ``[0, speed_knee]`` so both the value and
        its slope are continuous — a starving worm slows down gradually and can
        still reach food.
        """
        knee = self.config.speed_knee
        t = float(np.clip(self.energy_fraction / knee, 0.0, 1.0))
        smooth = t * t * (3.0 - 2.0 * t)
        floor = self.config.min_speed_factor
        return floor + (1.0 - floor) * smooth
