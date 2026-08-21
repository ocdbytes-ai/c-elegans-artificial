"""Worm world environment.

Importing this package registers the id with Gymnasium::

    import envs                       # noqa: F401  (registers the id)
    import gymnasium as gym

    env = gym.make("WormWorld-v1", render_mode="human")

A dot on a continuous plane that has to eat to keep existing. It senses food as
a single scalar scent concentration at its head, so chemotaxis is a learning
problem, not a lookup — see :mod:`envs.observations`.
"""

import sys

from gymnasium.envs.registration import register

from .config import EnvConfig
from .episodes import EpisodeSummary, EpochStats, run_episodes
from .metabolism import Metabolism
from .sources import FoodField, ScentField, ToxinField
from .step_info import StepInfo
from .worm import Worm
from .worm_world import WormWorldEnv

__all__ = [
    "ENV_ID",
    "UNLIMITED_EPISODE_STEPS",
    "EnvConfig",
    "EpisodeSummary",
    "EpochStats",
    "FoodField",
    "Metabolism",
    "ScentField",
    "StepInfo",
    "ToxinField",
    "Worm",
    "WormWorldEnv",
    "run_episodes",
]

# The registered id, exported so callers reference it rather than re-spelling
# the string. Renaming the env then touches exactly one line.
ENV_ID = "WormWorld-v1"
MAX_EPISODE_STEPS = 4000

# Pass as ``gym.make(..., max_episode_steps=UNLIMITED_EPISODE_STEPS)`` to let an
# episode run until the worm starves. Gymnasium has no way to *remove* a
# registered limit: passing None falls back to the spec's value, and it asserts
# the limit is positive so math.inf is rejected. A limit no run can reach is the
# available spelling.
#
# For watching only. In training, an immortal policy would never finish an
# episode, so every per-episode statistic would be NaN and the freeze-trap
# detector would go blind.
UNLIMITED_EPISODE_STEPS = sys.maxsize

register(
    id=ENV_ID,
    entry_point="envs.worm_world:WormWorldEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
