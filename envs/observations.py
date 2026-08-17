"""What the worm senses.

One scalar for food: *how strongly it smells right here*. No direction, no
distance, no pellet count which the same thing a real worm gets from a chemosensory
neuron. To find food it has to move and compare, which is why this pairs with
frame stacking::

    from gymnasium.wrappers import FrameStackObservation
    env = FrameStackObservation(gym.make("WormWorld-v1"), stack_size=4)

Stacking turns "how much" into "more or less than k steps ago", which is all
klinokinesis needs

Observations are assembled from :class:`Channel` entries a name, its bounds,
and how to read it off the simulation. Adding a sense later (the unlabeled
adversary channel in §2.3, temperature, proprioception) means appending a
channel in :func:`build_channels`, not touching the environment.

The guiding constraint: Anything the worm must learn to react to has to be present in here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .food import FoodField
from .metabolism import Metabolism
from .worm import Worm


@dataclass(frozen=True)
class Sensors:
    """Everything a channel is allowed to look at."""

    worm: Worm
    food: FoodField
    metabolism: Metabolism


@dataclass(frozen=True)
class Channel:
    """One scalar entry of the observation vector."""

    name: str
    low: float
    high: float
    read: Callable[[Sensors], float]


def build_channels(config: EnvConfig) -> list[Channel]:
    """The worm's full sensory complement, in observation order."""
    channels: list[Channel] = []

    if config.observation.include_energy:
        # Interoception. Required for any hunger-modulated behaviour.
        channels.append(
            Channel("energy", 0.0, 1.0, lambda s: s.metabolism.energy_fraction)
        )

    # Total food scent at the head. The ceiling is every pellet stacked on the
    # worm at once — unreachable in practice, but it bounds the space honestly.
    ceiling = float(config.food.count * config.food.scent_peak)
    channels.append(
        Channel(
            "food_smell",
            0.0,
            ceiling,
            lambda s: float(s.food.scent_at(s.worm.position)),
        )
    )

    # sin/cos rather than raw theta
    # this is better to represent the direction in terms of a range
    channels.append(
        Channel("heading_sin", -1.0, 1.0, lambda s: s.worm.heading_sin_cos[0])
    )
    channels.append(
        Channel("heading_cos", -1.0, 1.0, lambda s: s.worm.heading_sin_cos[1])
    )

    return channels


class ObservationBuilder:
    """Turns the channel list into a bounded, float32 observation vector."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.channels = build_channels(config)
        self._low = np.array([c.low for c in self.channels], dtype=np.float32)
        self._high = np.array([c.high for c in self.channels], dtype=np.float32)

    @property
    def labels(self) -> list[str]:
        return [c.name for c in self.channels]

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    def __call__(self, sensors: Sensors) -> np.ndarray:
        values = np.array([c.read(sensors) for c in self.channels], dtype=np.float32)
        # Clip rather than trust the channels: an out-of-bounds observation is a
        # silent contract violation that only surfaces as a confusing crash later.
        return np.clip(values, self._low, self._high)
