"""Food pellets and the scent field they generate.

A pellet (bacteria, per the biology notes) has two radii:

- ``eat_radius``: contact. Touching a pellet eats it.
- ``scent_radius``: sensation. Concentration peaks at the pellet's centre and
  falls off over this scale, so a pellet can be smelled long before it can be
  eaten.

The field is the sum over all pellets rather than the maximum, so overlapping
pellets build a richer landscape than any single one. That draws the worm toward
regions of high food density rather than toward one pellet, which matches how a
chemical gradient behaves. It also means the field's local maxima do not
necessarily sit on pellets; see ``notes/training.md``.
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .config import FoodConfig, RandomizationConfig, WorldConfig
from .scent import make_scent_profile


class FoodField:
    """Pellet positions and the scalar scent field they produce.

    Attributes:
        config: Pellet and scent parameters.
        world: Arena dimensions and boundary convention.
        randomization: Per-episode spread on the pellet count.
        profile: Distance-to-concentration function for one pellet.
        positions: Live pellet positions, shape ``(count, 2)``.
        initial_count: Pellets drawn at the last reset, before any were eaten.
        eaten_total: Pellets eaten so far this episode.
    """

    def __init__(
        self, config: FoodConfig, world: WorldConfig, randomization: RandomizationConfig
    ):
        """Initialises an empty field sized for the nominal pellet count.

        Args:
            config: Pellet and scent parameters.
            world: Arena dimensions and boundary convention.
            randomization: Per-episode spread on the pellet count.
        """
        self.config = config
        self.world = world
        self.randomization = randomization
        self.profile = make_scent_profile(config)
        self.positions = np.zeros((config.count, 2), dtype=np.float64)
        self.initial_count = config.count
        self.eaten_total = 0

    @property
    def count(self) -> int:
        """Pellets currently in the world.

        Falls below ``initial_count`` as food is eaten when
        ``respawn_on_eat`` is disabled.
        """
        return len(self.positions)

    def reset(self, rng: np.random.Generator, worm_position: np.ndarray) -> None:
        """Draws this episode's pellet count and scatters them clear of the worm.

        Args:
            rng: Source of randomness, owned by the environment.
            worm_position: Position to keep pellets away from.
        """
        low, high = self.randomization.food_count_bounds(self.config.count)
        self.initial_count = int(rng.integers(low, high + 1))
        self.positions = geometry.sample_positions(
            rng,
            self.initial_count,
            self.world.size,
            self.world.boundary,
            exclude=worm_position,
            min_distance=self.config.min_spawn_distance,
        )
        self.eaten_total = 0

    def scent_at(self, points: np.ndarray) -> np.ndarray:
        """Evaluates the total concentration at one or many points.

        Args:
            points: Query positions of shape ``(2,)`` or ``(..., 2)``.

        Returns:
            Concentration summed over all pellets, with the query's shape minus
            its last axis. A single point yields a scalar array.
        """
        query = np.asarray(points, dtype=np.float64)
        flat = query.reshape(-1, 2)
        distances = geometry.distance(
            flat[:, None, :], self.positions[None, :, :], self.world.size, self.world.boundary
        )
        totals = self.profile(distances).sum(axis=-1)
        return totals.reshape(query.shape[:-1])

    def nearest(self, position: np.ndarray) -> tuple[int, np.ndarray, float]:
        """Finds the closest pellet.

        Args:
            position: Position to measure from.

        Returns:
            ``(index, displacement, distance)`` for the closest pellet, or
            ``(-1, zeros, inf)`` when the world has been emptied. Emptying is
            only reachable under ``respawn_on_eat: false``; reporting an
            infinite distance keeps the environment steppable while the worm
            starves out the remainder of its episode.
        """
        if self.count == 0:
            return -1, np.zeros(2, dtype=np.float64), float("inf")

        deltas = geometry.displacement(
            position, self.positions, self.world.size, self.world.boundary
        )
        distances = np.linalg.norm(deltas, axis=-1)
        index = int(np.argmin(distances))
        return index, deltas[index], float(distances[index])

    def consume(self, position: np.ndarray, rng: np.random.Generator) -> int:
        """Eats every pellet within ``eat_radius`` of a position.

        Eaten pellets either respawn elsewhere or are removed permanently,
        depending on ``respawn_on_eat``.

        Args:
            position: The worm's position.
            rng: Source of randomness, used when respawning.

        Returns:
            How many pellets were eaten this step, usually 0 or 1.
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


def max_possible_scent(config: FoodConfig, randomization: RandomizationConfig) -> float:
    """Computes the ceiling for the ``food_smell`` observation channel.

    Sized from the largest pellet count an episode can draw rather than the
    nominal one, because the observation space is fixed for the environment's
    lifetime and cannot be resized when a rich episode comes up.

    Args:
        config: Pellet and scent parameters.
        randomization: Per-episode spread on the pellet count.

    Returns:
        Concentration with every pellet of the richest possible episode stacked
        on the worm's head.
    """
    _, max_count = randomization.food_count_bounds(config.count)
    return float(max_count * config.scent_peak)
