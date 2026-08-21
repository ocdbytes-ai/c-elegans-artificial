"""Scented sources: food pellets and toxin hazards.

Both are point sources emitting a scalar field, so the positions, the field and
the queries against it live in :class:`ScentField`. What differs is what the
source does to a worm that reaches it, and only that is subclassed.

Food has two radii:

- ``eat_radius``: contact. Touching a pellet eats it.
- ``scent_radius``: sensation. Concentration peaks at the pellet's centre and
  falls off over this scale, so a pellet can be smelled long before it can be
  eaten.

A toxin has no contact radius at all. Harm is a dose proportional to the
concentration at the worm's head, which makes it graded rather than binary; see
:class:`~envs.config.ToxinConfig`.

Every field is the sum over its sources rather than the maximum, so overlapping
sources build a richer landscape than any single one. That draws the worm toward
regions of high density rather than toward one pellet, which matches how a
chemical gradient behaves. It also means the field's local maxima do not
necessarily sit on sources; see ``notes/training.md``.
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .config import FoodConfig, RandomizationConfig, ToxinConfig, WorldConfig
from .scent import make_scent_profile


class ScentField:
    """Point sources and the scalar concentration field they produce.

    Attributes:
        config: Source count and scent parameters.
        world: Arena dimensions and boundary convention.
        profile: Distance-to-concentration function for one source.
        positions: Live source positions, shape ``(count, 2)``.
    """

    def __init__(self, config: FoodConfig | ToxinConfig, world: WorldConfig):
        """Initialises an empty field sized for the nominal source count.

        Args:
            config: Source count and scent parameters.
            world: Arena dimensions and boundary convention.
        """
        self.config = config
        self.world = world
        self.profile = make_scent_profile(config)
        self.positions = np.zeros((config.count, 2), dtype=np.float64)

    @property
    def count(self) -> int:
        """Sources currently in the world."""
        return len(self.positions)

    def _scatter(
        self, rng: np.random.Generator, count: int, exclude: np.ndarray
    ) -> np.ndarray:
        """Draws ``count`` positions clear of a point.

        Args:
            rng: Source of randomness, owned by the environment.
            count: How many positions to draw.
            exclude: Position to keep clear of.

        Returns:
            Positions of shape ``(count, 2)``.
        """
        return geometry.sample_positions(
            rng,
            count,
            self.world.size,
            self.world.boundary,
            exclude=exclude,
            min_distance=self.config.min_spawn_distance,
        )

    def reset(self, rng: np.random.Generator, worm_position: np.ndarray) -> None:
        """Scatters the configured number of sources clear of the worm.

        Args:
            rng: Source of randomness, owned by the environment.
            worm_position: Position to keep sources away from.
        """
        self.positions = self._scatter(rng, self.config.count, worm_position)

    def scent_at(self, points: np.ndarray) -> np.ndarray:
        """Evaluates the total concentration at one or many points.

        Args:
            points: Query positions of shape ``(2,)`` or ``(..., 2)``.

        Returns:
            Concentration summed over all sources, with the query's shape minus
            its last axis. A single point yields a scalar array.
        """
        query = np.asarray(points, dtype=np.float64)
        flat = query.reshape(-1, 2)
        if self.count == 0:
            return np.zeros(flat.shape[0]).reshape(query.shape[:-1])
        distances = geometry.distance(
            flat[:, None, :], self.positions[None, :, :], self.world.size, self.world.boundary
        )
        totals = self.profile(distances).sum(axis=-1)
        return totals.reshape(query.shape[:-1])

    def nearest(self, position: np.ndarray) -> tuple[int, np.ndarray, float]:
        """Finds the closest source.

        Args:
            position: Position to measure from.

        Returns:
            ``(index, displacement, distance)`` for the closest source, or
            ``(-1, zeros, inf)`` when there are none. An empty food world is
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


class FoodField(ScentField):
    """Pellets that can be eaten, restoring energy.

    Attributes:
        randomization: Per-episode spread on the pellet count.
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
        super().__init__(config, world)
        self.config: FoodConfig = config
        self.randomization = randomization
        self.initial_count = config.count
        self.eaten_total = 0

    def reset(self, rng: np.random.Generator, worm_position: np.ndarray) -> None:
        """Draws this episode's pellet count and scatters them clear of the worm.

        Args:
            rng: Source of randomness, owned by the environment.
            worm_position: Position to keep pellets away from.
        """
        low, high = self.randomization.food_count_bounds(self.config.count)
        self.initial_count = int(rng.integers(low, high + 1))
        self.positions = self._scatter(rng, self.initial_count, worm_position)
        self.eaten_total = 0

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
            self.positions[eaten] = self._scatter(rng, count, position)
        else:
            self.positions = self.positions[~eaten]
        return count


class ToxinField(ScentField):
    """Hazards that cost energy to be near and are never consumed.

    Adds no behaviour to :class:`ScentField`: a toxin only has to be smelled and
    to be somewhere. The dose it inflicts is applied by
    :meth:`envs.metabolism.Metabolism.poison`, which keeps every energy
    transaction in one module.
    """

    def __init__(self, config: ToxinConfig, world: WorldConfig):
        """Initialises an empty field sized for the configured source count.

        Args:
            config: Hazard and scent parameters.
            world: Arena dimensions and boundary convention.
        """
        super().__init__(config, world)
        self.config: ToxinConfig = config


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


def max_possible_toxin(config: ToxinConfig) -> float:
    """Computes the ceiling for the ``toxin_smell`` observation channel.

    Args:
        config: Hazard and scent parameters.

    Returns:
        Concentration with every source stacked on the worm's head. Toxin count
        is not randomised, so this is exact rather than an upper bound.
    """
    return float(max(config.count, 1) * config.scent_peak)
