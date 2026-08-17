"""Food pellets and the scent field they generate.

A pellet (bacteria as per biology) has two radii:

- ``eat_radius``  — contact. Touch it and it is eaten.
- ``scent_radius`` — sensation. The concentration peaks at the pellet's centre
  and falls off outwards over this scale, so the worm can smell a pellet long
  before it can eat one.

The field is the sum over all pellets, which means overlapping pellets build a
richer landscape than any single one so this will lead the worm towards the food
space where most of it is located instead of attraction to one single pellet
as per chemical response.
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .config import FoodConfig, WorldConfig
from .scent import make_scent_profile


class FoodField:
    """Pellet positions plus the scalar scent field sampled anywhere in the world."""

    def __init__(self, config: FoodConfig, world: WorldConfig):
        self.config = config
        self.world = world
        self.profile = make_scent_profile(config)
        self.positions = np.zeros((config.count, 2), dtype=np.float64)
        self.eaten_total = 0

    def reset(self, rng: np.random.Generator, worm_position: np.ndarray) -> None:
        """Scatter every pellet at random, keeping clear of the worm's head."""
        self.positions = geometry.sample_positions(
            rng,
            self.config.count,
            self.world.size,
            self.world.boundary,
            exclude=worm_position,
            min_distance=self.config.min_spawn_distance,
        )
        self.eaten_total = 0

    def scent_at(self, points: np.ndarray) -> np.ndarray:
        """Total concentration at ``points`` — shape ``(2,)`` or ``(..., 2)``.

        Returns a scalar array for a single point, otherwise one value per point.
        """
        query = np.asarray(points, dtype=np.float64)
        flat = query.reshape(-1, 2)
        # (n_points, n_pellets) distances -> profile -> sum over pellets.
        distances = geometry.distance(
            flat[:, None, :], self.positions[None, :, :], self.world.size, self.world.boundary
        )
        totals = self.profile(distances).sum(axis=-1)
        return totals.reshape(query.shape[:-1])

    def nearest(self, position: np.ndarray) -> tuple[int, np.ndarray, float]:
        """Index, displacement vector and distance to the closest pellet."""
        deltas = geometry.displacement(
            position, self.positions, self.world.size, self.world.boundary
        )
        distances = np.linalg.norm(deltas, axis=-1)
        index = int(np.argmin(distances))
        return index, deltas[index], float(distances[index])

    def consume(self, position: np.ndarray, rng: np.random.Generator) -> int:
        """Eat every pellet within ``eat_radius`` and respawn it elsewhere.

        Returns how many were eaten this step (usually 0 or 1).
        """
        distances = geometry.distance(
            position, self.positions, self.world.size, self.world.boundary
        )
        eaten = distances <= self.config.eat_radius
        count = int(eaten.sum())
        if count == 0:
            return 0

        self.eaten_total += count
        if self.config.respawn_on_eat:
            self.positions[eaten] = geometry.sample_positions(
                rng,
                count,
                self.world.size,
                self.world.boundary,
                exclude=position,
                min_distance=self.config.min_spawn_distance,
            )
        else:
            self.positions = self.positions[~eaten]
        return count

    @property
    def max_possible_scent(self) -> float:
        """Upper bound for the observation space: every pellet stacked at the head."""
        return float(self.config.count * self.config.scent_peak)
