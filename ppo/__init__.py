"""PPO for the worm world.

A standalone implementation following OpenAI Spinning Up's PPO, with the pieces
separated so each can be read, tested and swapped on its own:

    config.py      hyperparameters (dataclasses + YAML, same contract as envs)
    networks.py    MLP actor-critic with a Gaussian policy
    buffer.py      on-policy rollout storage and GAE-Lambda
    curriculum.py  anneals the world from forgiving to real during a run
    metrics.py     episode statistics and CSV/console logging
    trainer.py     the loop: collect -> advantages -> clipped update

Typical use::

    from ppo import PPOTrainer
    PPOTrainer(env_config="configs/world_v1.yaml", ppo_config="configs/ppo.yaml").run()

or from the command line::

    uv run python train.py --epochs 300
"""

from .buffer import RolloutBuffer
from .config import (
    CurriculumConfig,
    NetworkConfig,
    OptimConfig,
    PPOConfig,
    RolloutConfig,
    RunConfig,
)
from .curriculum import Curriculum
from .metrics import EpisodeSummary, EpochStats, RunLogger
from .networks import ActorCritic
from .trainer import PPOTrainer, load_policy, make_env

__all__ = [
    "ActorCritic",
    "Curriculum",
    "CurriculumConfig",
    "EpisodeSummary",
    "EpochStats",
    "NetworkConfig",
    "OptimConfig",
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "RolloutConfig",
    "RunConfig",
    "RunLogger",
    "load_policy",
    "make_env",
]
