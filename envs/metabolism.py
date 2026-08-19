"""Energy bookkeeping, and the death condition that gives the worm a problem.

Per step::

    energy -= basal_cost
    energy -= move_cost * |action|^2
    energy += food_value                  # on eat
    dead    = energy <= 0

Two invariants, enforced by the config validators and relied on here:

1. ``basal_cost`` dominates ``move_cost``. If idling were meaningfully cheaper
   than moving, standing still would maximise lifespan and training would
   converge there.
2. ``speed_factor`` is a smooth ramp rather than a threshold. A cliff makes
   low-energy states unrecoverable, since the worm slows below the speed it
   needs to reach food and every state under the threshold becomes a certain
   death.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MetabolismConfig


@dataclass
class EnergyLedger:
    """Energy in and out over a single step, for diagnostics only.

    Recording intake and outgo separately is what distinguishes a worm that
    starved from one that exhausted itself moving; the two are identical in the
    reward.

    Attributes:
        basal_cost: Unconditional cost of existing this step.
        move_cost: Cost attributable to the action taken.
        intake: Energy absorbed from food this step.
    """

    basal_cost: float = 0.0
    move_cost: float = 0.0
    intake: float = 0.0

    @property
    def net(self) -> float:
        """Net energy change over the step."""
        return self.intake - self.basal_cost - self.move_cost


class Metabolism:
    """Tracks the worm's energy and the movement penalty that low energy brings.

    Attributes:
        config: Metabolic rates and limits.
        energy: Current energy, in the same units as ``food_value``.
        ledger: Energy flows from the most recent step.
    """

    def __init__(self, config: MetabolismConfig):
        """Initialises the metabolism at ``config.initial_energy``.

        Args:
            config: Metabolic rates and limits.
        """
        self.config = config
        self.energy = config.initial_energy
        self.ledger = EnergyLedger()

    def reset(self) -> None:
        """Restores the starting energy and clears the ledger."""
        self.energy = self.config.initial_energy
        self.ledger = EnergyLedger()

    def spend(self, action: np.ndarray) -> EnergyLedger:
        """Charges one step of living and moving.

        Call before applying the move, so the cost of an action is paid whether
        or not the worm has the energy left to complete it.

        Args:
            action: The action about to be taken, in ``[-1, 1]``.

        Returns:
            The ledger for this step, with intake still zero.
        """
        effort = float(np.sum(np.square(np.clip(action, -1.0, 1.0))))
        self.ledger = EnergyLedger(
            basal_cost=self.config.basal_cost,
            move_cost=self.config.move_cost * effort,
        )
        self.energy -= self.ledger.basal_cost + self.ledger.move_cost
        return self.ledger

    def eat(self, pellets: int) -> float:
        """Absorbs food, capped at ``max_energy``.

        Args:
            pellets: Number of pellets eaten this step.

        Returns:
            Energy actually gained, which is less than the nominal value when
            the cap is reached.
        """
        if pellets <= 0:
            return 0.0
        before = self.energy
        self.energy = min(self.energy + pellets * self.config.food_value, self.config.max_energy)
        gained = self.energy - before
        self.ledger.intake = gained
        return gained

    @property
    def energy_fraction(self) -> float:
        """Energy as a fraction of the maximum, clipped to ``[0, 1]``.

        This is the interoceptive signal the worm senses, not the raw energy.
        """
        return float(np.clip(self.energy / self.config.max_energy, 0.0, 1.0))

    @property
    def is_dead(self) -> bool:
        """Whether energy has run out."""
        return self.energy <= 0.0

    @property
    def speed_factor(self) -> float:
        """Movement multiplier in ``[min_speed_factor, 1]``.

        Uses a smoothstep ramp over ``[0, speed_knee]``, so both the value and
        its slope are continuous. A starving worm slows gradually and can still
        reach food.
        """
        knee = self.config.speed_knee
        t = float(np.clip(self.energy_fraction / knee, 0.0, 1.0))
        smooth = t * t * (3.0 - 2.0 * t)
        floor = self.config.min_speed_factor
        return floor + (1.0 - floor) * smooth
