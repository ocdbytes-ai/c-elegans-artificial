"""The Gymnasium environment: a worm that has to eat to keep existing.

A point body on a continuous plane, pellets that emit a scent field, and a
metabolism. The only reward is ``+1`` for each step survived; foraging has to
emerge from that.
The env itself is deliberately thin. It owns the RNG and the step ordering and
delegates everything else: :mod:`envs.worm` moves the body, :mod:`envs.food`
owns pellets and scent, :mod:`envs.metabolism` owns energy, and
:mod:`envs.observations` decides what any of that looks like to the policy.
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
from .worm import Worm


class WormWorldEnv(gym.Env):
    """A worm that must find food.

    Action space: ``Box(-1, 1, (2,))`` — ``[turn, throttle]``. Both are
    fractions of the configured maxima; throttle may be negative (worms reverse).
    
    Box(-1, 1, (2,)) means our policy gives two outputs :
    - Turn (+1 -> left, -1 -> right, 0 -> hold heading) 
    - Throttle (+1 -> forward, -1 -> backward)
    Each value can be in range [-1, 1]

    Observation space: ``[energy, food_smell, heading_sin, heading_cos]`` —
    see :mod:`envs.observations`, and ``env.unwrapped.observation_labels`` for
    the live layout.

    Episodes end on death (``terminated``). There is no built-in step cap
    truncation comes from the ``max_episode_steps`` in the registration, i.e.
    from Gymnasium's ``TimeLimit`` wrapper, so that death and timeout stay
    distinguishable (PPO needs to bootstrap differently for each).
    """

    # Class-level on purpose: gym.make() reads this off the class, before any
    # instance exists, to decide whether to auto-apply HumanRendering. __init__
    # then replaces it per instance so render_fps reflects the actual config.
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: EnvConfig | dict[str, Any] | str | Path | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = EnvConfig.resolve(config)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": self.config.render.fps}

        self.worm = Worm(self.config.worm, self.config.world)
        self.food = FoodField(self.config.food, self.config.world)
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.worm.reset(self.np_random)
        self.food.reset(self.np_random, self.worm.position)
        self.metabolism.reset()
        self.steps = 0

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, self._info(eaten=0, moved=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Pay first, move second: the cost of an action is charged whether or
        # not the worm has the energy left to complete it.
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
        if self.render_mode is None:
            gym.logger.warn("render() called with render_mode=None; nothing to draw")
            return None
        if self._renderer is None:
            from .rendering import PygameRenderer  # imported lazily: pygame is optional

            self._renderer = PygameRenderer(self.config, self.render_mode)
        return self._renderer.draw(self.worm, self.food, self.metabolism, self.steps)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observe(self) -> np.ndarray:
        return self._observation(Sensors(self.worm, self.food, self.metabolism))

    def _info(self, eaten: int, moved: float) -> dict[str, Any]:
        """Diagnostics only — the agent optimises the scalar reward alone.

        Energy in vs. out is logged separately here so that a worm that is
        starving and a worm that is over-exerting look different in the logs.
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
            "position": self.worm.position.copy(),
            "heading": self.worm.heading,
        }
