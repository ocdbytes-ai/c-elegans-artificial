"""Episode statistics and rollouts, shared by the trainer and the scripts.

These live in :mod:`envs` rather than :mod:`ppo` because they describe the
worm's life rather than the algorithm: a hand-coded baseline and a trained
policy must be measured by the same code or the comparison is not like for
like. They also read the :class:`~envs.step_info.StepInfo` contract directly.

The diagnostics are the ones the project plan calls for:

- **mean lifespan**: under ``+1``/step this is the return, and the headline.
- **food eaten per life**: separates foraging from luck.
- **mean |action|**: the freeze-trap detector. Decaying toward zero while
  lifespan plateaus means the worm has learned to stand still, which is a
  metabolism-config problem rather than an algorithm one.
- **energy in vs. out**: starvation and over-exertion are identical in the
  reward and obvious here.
- **deaths vs. timeouts**: episodes all ending in truncation means the run has
  outgrown its ``max_episode_steps``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .step_info import StepInfo


@dataclass
class EpisodeSummary:
    """One completed life.

    Attributes:
        length: Steps survived. Equals the return under a ``+1``/step reward.
        ret: Total reward accumulated.
        eaten: Pellets eaten over the episode.
        mean_abs_action: Mean absolute action, averaged over steps.
        energy_in: Total energy absorbed from food.
        energy_out: Total energy spent on basal and movement costs.
        distance: World units travelled.
        died: True if energy ran out, False if the episode was truncated.
    """

    length: int
    ret: float
    eaten: int
    mean_abs_action: float
    energy_in: float
    energy_out: float
    distance: float
    died: bool

    @property
    def ending(self) -> str:
        """How the episode finished, for display."""
        return "starved" if self.died else "timeout"


@dataclass
class EpisodeAccumulator:
    """Running totals for the episode currently in progress.

    Attributes:
        length: Steps taken so far.
        ret: Reward accumulated so far.
        abs_action: Sum of per-step mean absolute actions.
        energy_in: Energy absorbed so far.
        energy_out: Energy spent so far.
        distance: World units travelled so far.
        eaten: Pellets eaten so far.
    """

    length: int = 0
    ret: float = 0.0
    abs_action: float = 0.0
    energy_in: float = 0.0
    energy_out: float = 0.0
    distance: float = 0.0
    eaten: int = 0

    def add(self, action: np.ndarray, reward: float, info: StepInfo) -> None:
        """Folds one step into the running totals.

        Args:
            action: The action taken.
            reward: The reward received.
            info: The environment's diagnostics for the step.
        """
        self.length += 1
        self.ret += reward
        self.abs_action += float(np.abs(action).mean())
        self.energy_in += info["energy_intake"]
        self.energy_out += info["basal_cost"] + info["move_cost"]
        self.distance += info["distance_moved"]
        self.eaten = info["food_eaten_total"]

    def close(self, died: bool) -> EpisodeSummary:
        """Finalises the episode.

        Args:
            died: True if energy ran out, False if the episode was truncated.

        Returns:
            The completed summary.
        """
        return EpisodeSummary(
            length=self.length,
            ret=self.ret,
            eaten=self.eaten,
            mean_abs_action=self.abs_action / max(self.length, 1),
            energy_in=self.energy_in,
            energy_out=self.energy_out,
            distance=self.distance,
            died=died,
        )


@dataclass
class EpochStats:
    """Every episode that finished within one epoch or measurement run.

    Attributes:
        episodes: The completed episodes, in order.
    """

    episodes: list[EpisodeSummary] = field(default_factory=list)

    def add(self, episode: EpisodeSummary) -> None:
        """Records a completed episode.

        Args:
            episode: The episode to record.
        """
        self.episodes.append(episode)

    def summary(self) -> dict[str, float]:
        """Aggregates the episodes into a single loggable row.

        Returns:
            One value per statistic. Every field is NaN when no episode
            finished, which happens when episodes outlast the epoch.
        """
        if not self.episodes:
            return dict.fromkeys(_SUMMARY_KEYS, float("nan")) | {"episodes": 0}

        lengths = np.array([e.length for e in self.episodes], dtype=np.float64)
        energy_out = np.array([e.energy_out for e in self.episodes], dtype=np.float64)
        distance = np.array([e.distance for e in self.episodes], dtype=np.float64)
        return {
            "episodes": len(self.episodes),
            "lifespan_mean": float(lengths.mean()),
            "lifespan_std": float(lengths.std()),
            "lifespan_min": float(lengths.min()),
            "lifespan_max": float(lengths.max()),
            "return_mean": float(np.mean([e.ret for e in self.episodes])),
            "eaten_mean": float(np.mean([e.eaten for e in self.episodes])),
            "mean_abs_action": float(np.mean([e.mean_abs_action for e in self.episodes])),
            "energy_in": float(np.mean([e.energy_in for e in self.episodes])),
            "energy_out": float(energy_out.mean()),
            "distance_per_energy": float(distance.sum() / max(energy_out.sum(), 1e-9)),
            "death_rate": float(np.mean([e.died for e in self.episodes])),
        }


_SUMMARY_KEYS = (
    "episodes",
    "lifespan_mean",
    "lifespan_std",
    "lifespan_min",
    "lifespan_max",
    "return_mean",
    "eaten_mean",
    "mean_abs_action",
    "energy_in",
    "energy_out",
    "distance_per_energy",
    "death_rate",
)


class Policy(Protocol):
    """Anything that turns an observation into an action."""

    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


def run_episodes(
    env: Any,
    policy: Policy,
    episodes: int | None = 10,
    seed: int = 0,
    on_episode: Callable[[int, EpisodeSummary], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> EpochStats:
    """Rolls out complete episodes and gathers their statistics.

    Each episode is seeded ``seed + index``, so a measurement is reproducible
    and two policies can be compared on identical worlds.

    Args:
        env: A Gymnasium environment, wrapped or otherwise.
        policy: Maps an observation to an action.
        episodes: How many episodes to run, or None to go on until ``stop``
            says otherwise. Unbounded rollouts are for watching, not measuring.
        seed: Seed of the first episode.
        on_episode: Called as each life ends, for progressive reporting. An
            unbounded caller should accumulate here, since a run that is
            interrupted never reaches the return.
        stop: Polled every step. The rollout ends as soon as it returns True,
            which is what lets a closed window take effect mid-episode rather
            than at the next death. The life in progress is discarded: a
            partial episode is not a lifespan.

    Returns:
        Statistics over every completed episode.
    """
    stats = EpochStats()
    index = 0
    while episodes is None or index < episodes:
        observation, _ = env.reset(seed=seed + index)
        accumulator = EpisodeAccumulator()
        while True:
            if stop is not None and stop():
                return stats
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            accumulator.add(action, reward, info)
            if terminated or truncated:
                summary = accumulator.close(died=terminated)
                stats.add(summary)
                if on_episode is not None:
                    on_episode(index, summary)
                break
        index += 1
    return stats


def episode_line(index: int, episode: EpisodeSummary) -> str:
    """Formats one episode as a console line.

    Args:
        index: Zero-based episode number.
        episode: The completed episode.

    Returns:
        A single line of text.
    """
    return (
        f"  episode {index:>3}  steps {episode.length:>5}  "
        f"eaten {episode.eaten:>3}  ({episode.ending})"
    )


def summary_line(stats: EpochStats) -> str:
    """Formats the aggregate result as a console line.

    Args:
        stats: Statistics over the completed episodes.

    Returns:
        A leading blank line followed by the summary.
    """
    values = stats.summary()
    return (
        f"\nmean lifespan {values['lifespan_mean']:.1f} +/- {values['lifespan_std']:.1f} steps"
        f"   mean eaten {values['eaten_mean']:.2f}"
    )
