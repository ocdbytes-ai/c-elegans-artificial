"""Worm world environment.

Importing this package registers the id with Gymnasium::

    import envs                       # noqa: F401  (registers the id)
    import gymnasium as gym

    env = gym.make("WormWorld-v1", render_mode="human")

A dot on a continuous plane that has to eat to keep existing. It senses food as
a single scalar scent concentration at its head, so chemotaxis is a learning
problem, not a lookup — see :mod:`envs.observations`.
"""

from gymnasium.envs.registration import register

from .config import EnvConfig
from .episodes import EpisodeSummary, EpochStats, run_episodes
from .food import FoodField
from .metabolism import Metabolism
from .step_info import StepInfo
from .worm import Worm
from .worm_world import WormWorldEnv

__all__ = [
    "ENV_ID",
    "EnvConfig",
    "EpisodeSummary",
    "EpochStats",
    "FoodField",
    "Metabolism",
    "StepInfo",
    "Worm",
    "WormWorldEnv",
    "run_episodes",
]

# The registered id, exported so callers reference it rather than re-spelling
# the string. Renaming the env then touches exactly one line.
ENV_ID = "WormWorld-v1"
MAX_EPISODE_STEPS = 2000

register(
    id=ENV_ID,
    entry_point="envs.worm_world:WormWorldEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
