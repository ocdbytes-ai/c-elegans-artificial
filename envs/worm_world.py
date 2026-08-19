"""The Gymnasium environment: a worm that must eat to keep existing.

A point body on a continuous plane, pellets that emit a scent field, and a
metabolism. The only reward is ``+1`` per step survived, so foraging has to
emerge rather than being rewarded directly.

The environment is deliberately thin. It owns the RNG and the step ordering and
delegates the rest: :mod:`envs.worm` moves the body, :mod:`envs.food` owns
pellets and scent, :mod:`envs.metabolism` owns energy, and
:mod:`envs.observations` decides what any of it looks like to the policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .food import FoodField
from .metabolism import Metabolism
from .observations import ObservationBuilder, Sensors
from .step_info import StepInfo
from .worm import Worm


class WormWorldEnv(gym.Env):
    """A worm that must find food to survive.

    The action is ``Box(-1, 1, (2,))`` holding ``[turn, throttle]``, each a
    fraction of the corresponding configured ceiling. Positive turn is
    anticlockwise, zero holds the heading; negative throttle reverses, which
    real *C. elegans* does as part of the pirouette.

    The observation is set by ``config.observation``; read
    ``env.unwrapped.observation_labels`` for the live layout.

    Episodes end on death via ``terminated``. There is no built-in step cap:
    truncation comes from ``max_episode_steps`` in the registration, so death
    and timeout stay distinguishable. PPO must bootstrap differently for each.

    Attributes:
        config: The resolved environment config.
        worm: The body.
        food: Pellets and the scent field.
        metabolism: Energy state.
        steps: Steps taken this episode.
        render_mode: One of ``metadata["render_modes"]``, or None.
    """

    # Class-level deliberately: gym.make() reads this off the class before any
    # instance exists, to decide whether to auto-apply HumanRendering. __init__
    # replaces it per instance so render_fps reflects the actual config.
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: EnvConfig | dict[str, Any] | str | Path | None = None,
        render_mode: str | None = None,
    ):
        """Builds the world.

        Args:
            config: An :class:`~envs.config.EnvConfig`, a nested dict, a path to
                a YAML file, or None for the defaults.
            render_mode: ``"human"``, ``"rgb_array"``, or None to disable.

        Raises:
            ValueError: If ``render_mode`` is not supported.
        """
        super().__init__()
        self.config = EnvConfig.resolve(config)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": self.config.render.fps}

        self.worm = Worm(self.config.worm, self.config.world, self.config.randomization)
        self.food = FoodField(self.config.food, self.config.world, self.config.randomization)
        self.metabolism = Metabolism(self.config.metabolism)
        self._observation = ObservationBuilder(self.config)
        self._renderer: Any | None = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = self._observation.space
        self.steps = 0

    @property
    def observation_labels(self) -> list[str]:
        """Names of the observation entries, in order."""
        return self._observation.labels

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, StepInfo]:
        """Starts a new episode with a fresh body, world and energy.

        Args:
            seed: Seeds the environment's RNG when given.
            options: Unused; present for the Gymnasium API.

        Returns:
            The first observation and its diagnostics.
        """
        super().reset(seed=seed)
        self.worm.reset(self.np_random)
        self.food.reset(self.np_random, self.worm.position)
        self.metabolism.reset()
        self.steps = 0

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, self._info(eaten=0, moved=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, StepInfo]:
        """Advances the simulation by one step.

        Args:
            action: ``[turn, throttle]``; values outside ``[-1, 1]`` are clipped.

        Returns:
            The Gymnasium 5-tuple. Truncation is always False here and is
            supplied by the ``TimeLimit`` wrapper from the registration.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Pay first, move second: an action costs energy whether or not the worm
        # has enough left to complete it.
        self.metabolism.spend(action)
        move = self.worm.step(action, speed_factor=self.metabolism.speed_factor)
        eaten = self.food.consume(self.worm.position, self.np_random)
        self.metabolism.eat(eaten)
        self.steps += 1

        terminated = self.metabolism.is_dead
        reward = 0.0 if terminated else self.config.reward.survival
        if terminated:
            reward -= self.config.reward.death_penalty

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, False, self._info(eaten, move.distance)

    def render(self) -> np.ndarray | None:
        """Draws the current frame.

        Returns:
            An RGB array under ``"rgb_array"``, otherwise None.
        """
        if self.render_mode is None:
            gym.logger.warn("render() called with render_mode=None; nothing to draw")
            return None
        if self._renderer is None:
            from .rendering import PygameRenderer  # imported lazily: pygame is optional

            self._renderer = PygameRenderer(self.config, self.render_mode)
        return self._renderer.draw(self.worm, self.food, self.metabolism, self.steps)

    @property
    def window_closed(self) -> bool:
        """True once the user has closed the render window.

        The simulation carries on regardless, since nothing in it depends on
        being drawn. An open-ended viewer should poll this and stop.
        """
        return self._renderer is not None and self._renderer.closed

    def close(self) -> None:
        """Releases the renderer and any window it owns."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observe(self) -> np.ndarray:
        """Reads the current observation."""
        return self._observation(Sensors(self.worm, self.food, self.metabolism))

    def _info(self, eaten: int, moved: float) -> StepInfo:
        """Assembles the per-step diagnostics.

        Diagnostics only; the agent optimises the scalar reward alone. See
        :mod:`envs.step_info` for the field contract.

        Args:
            eaten: Pellets eaten this step.
            moved: World units actually travelled this step.

        Returns:
            The populated info dict.
        """
        ledger = self.metabolism.ledger
        _, _, food_distance = self.food.nearest(self.worm.position)
        return {
            "steps": self.steps,
            "energy": float(self.metabolism.energy),
            "energy_fraction": self.metabolism.energy_fraction,
            "speed_factor": self.metabolism.speed_factor,
            "basal_cost": ledger.basal_cost,
            "move_cost": ledger.move_cost,
            "energy_intake": ledger.intake,
            "food_eaten": eaten,
            "food_eaten_total": self.food.eaten_total,
            "food_smell": float(self.food.scent_at(self.worm.position)),
            "nearest_food_distance": food_distance,
            "distance_moved": moved,
            "touch": self.worm.blocked,
            "position": self.worm.position.copy(),
            "heading": self.worm.heading,
            "max_speed": self.worm.max_speed,
            "max_turn_rate": self.worm.max_turn_rate,
            "food_count": self.food.count,
        }
