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
from .food import FoodField
from .metabolism import Metabolism
from .worm import Worm
from .worm_world import WormWorldEnv

__all__ = ["EnvConfig", "FoodField", "Metabolism", "Worm", "WormWorldEnv"]

MAX_EPISODE_STEPS = 2000

register(
    id="WormWorld-v1",
    entry_point="envs.worm_world:WormWorldEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
