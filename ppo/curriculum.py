"""Tightens the world from forgiving to real over a single run.

The env config handed to the trainer is the *target*: the world the policy must
eventually handle. This module produces the world it trains in today,
interpolating from :class:`~ppo.config.CurriculumConfig`'s starting values to
that target over the first ``anneal_fraction`` of training, then holding.

It mutates the live :class:`~envs.config.EnvConfig` in place. That is deliberate:
the environment holds a reference to the same ``FoodConfig`` object, and
``FoodField`` re-reads ``count``, ``eat_radius`` and ``scent_radius`` on every
reset, so a change lands on the next episode with no env rebuild and no loss of
the worm's learned state.
"""

from __future__ import annotations

from envs.config import EnvConfig

from .config import CurriculumConfig


class Curriculum:
    """Schedules the world from its starting values to the target env config.

    Attributes:
        config: The schedule.
        target: The world the schedule ends at.
    """

    def __init__(self, config: CurriculumConfig, target: EnvConfig):
        """Binds a schedule to its target world.

        Args:
            config: The schedule.
            target: The world to anneal toward.
        """
        self.config = config
        self.target = target

    def apply(self, live: EnvConfig, epoch: int, total_epochs: int) -> dict[str, float]:
        """Sets ``live`` to the world for a given epoch.

        Args:
            live: The config the environment reads, mutated in place.
            epoch: One-based epoch number.
            total_epochs: Total epochs in the run, which sets the ramp length.

        Returns:
            The values chosen, for logging.

        Raises:
            ValueError: If the schedule produces an invalid world, for instance
                walking ``eat_radius`` past ``scent_radius``.
        """
        fraction = self.config.progress(epoch, total_epochs)
        food, target = live.food, self.target.food

        if not self.config.enabled:
            # Report, never touch: a disabled curriculum must leave the config
            # byte-identical to what was passed, not merely close to it.
            return _describe(fraction, live)

        if self.config.food_count_start is not None:
            food.count = max(1, round(_lerp(self.config.food_count_start, target.count, fraction)))
        if self.config.eat_radius_start is not None:
            food.eat_radius = _lerp(self.config.eat_radius_start, target.eat_radius, fraction)
        if self.config.scent_radius_start is not None:
            food.scent_radius = _lerp(
                self.config.scent_radius_start, target.scent_radius, fraction
            )

        # Re-run the validators so an impossible schedule fails loudly instead
        # of quietly producing a world where food is eaten before it is smelled.
        food.__post_init__()

        if self.config.metabolism_scale_start is not None:
            # Both rates divided by the same factor: a pure time dilation that
            # preserves basal >> move. Scaling only basal would make full-effort
            # movement cost more than idling and hand PPO the freeze trap.
            scale = _lerp(self.config.metabolism_scale_start, 1.0, fraction)
            live.metabolism.basal_cost = self.target.metabolism.basal_cost / scale
            live.metabolism.move_cost = self.target.metabolism.move_cost / scale
            live.metabolism.__post_init__()

        return _describe(fraction, live)

    def starting_world(self) -> EnvConfig:
        """Builds a copy of the target set to its epoch-1 values.

        The environment must be constructed from this rather than from the
        target, because the ``food_smell`` observation ceiling is fixed at
        construction and has to cover the richest stage the run will ever see.

        Returns:
            A fresh config at the start of the ramp.
        """
        live = EnvConfig.from_dict(self.target.to_dict())
        self.apply(live, epoch=1, total_epochs=1)
        return live


def _describe(fraction: float, live: EnvConfig) -> dict[str, float]:
    """Summarises the current world for the progress log.

    Args:
        fraction: Curriculum progress in ``[0, 1]``.
        live: The world as it currently stands.

    Returns:
        The logged curriculum columns.
    """
    return {
        "curriculum": fraction,
        "food_count": live.food.count,
        "eat_radius": live.food.eat_radius,
        "scent_radius": live.food.scent_radius,
        "basal_cost": live.metabolism.basal_cost,
    }


def _lerp(start: float, end: float, fraction: float) -> float:
    """Interpolates, landing exactly on the endpoints.

    ``start + (end - start) * 1.0`` is off by an ulp for most values, which
    would leave the final world a hair from the target it is meant to have
    reached and make a saved config compare unequal to its own source.

    Args:
        start: Value at ``fraction == 0``.
        end: Value at ``fraction == 1``.
        fraction: Interpolation position, clamped to ``[0, 1]``.

    Returns:
        The interpolated value.
    """
    if fraction >= 1.0:
        return end
    if fraction <= 0.0:
        return start
    return start + (end - start) * fraction
