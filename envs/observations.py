"""Assembly of the worm's observation vector.

Food is sensed as a single scalar: how strongly it smells at the head. There is
no direction, no distance and no pellet count, matching what a real chemosensory
neuron reports. Finding food therefore requires moving and comparing, which is
why this pairs with frame stacking::

    from gymnasium.wrappers import FrameStackObservation
    env = FrameStackObservation(gym.make("WormWorld-v1"), stack_size=4)

Stacking turns "how much" into "more or less than k steps ago", which is what
klinokinesis needs.

Observations are assembled from :class:`Channel` entries, each naming a signal,
its bounds and how to read it off the simulation. Adding a sense later — the
unlabelled adversary channel, temperature, proprioception — means appending a
channel in :func:`build_channels` rather than touching the environment.

The governing constraint: anything the worm must learn to react to has to be
present here. A feature of the world absent from the observation cannot be
learned about, however obvious it is to an observer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .food import FoodField, max_possible_scent
from .metabolism import Metabolism
from .worm import Worm


@dataclass(frozen=True)
class Sensors:
    """The simulation state a channel is permitted to read.

    Attributes:
        worm: The body, for position, heading and mechanosensation.
        food: The pellet field, for scent.
        metabolism: Energy state, for interoception.
    """

    worm: Worm
    food: FoodField
    metabolism: Metabolism


@dataclass(frozen=True)
class Channel:
    """One scalar entry of the observation vector.

    Attributes:
        name: Identifier reported by ``observation_labels``.
        low: Lower bound declared to the observation space.
        high: Upper bound declared to the observation space.
        read: Extracts the channel's value from the simulation state.
    """

    name: str
    low: float
    high: float
    read: Callable[[Sensors], float]


def build_channels(config: EnvConfig) -> list[Channel]:
    """Builds the worm's sensory complement, in observation order.

    Args:
        config: Environment config, whose ``observation`` section decides which
            optional channels are present.

    Returns:
        The channels making up one observation frame.
    """
    channels: list[Channel] = []

    if config.observation.include_energy:
        # Interoception, required for any hunger-modulated behaviour.
        channels.append(Channel("energy", 0.0, 1.0, lambda s: s.metabolism.energy_fraction))

    # The ceiling covers the richest episode the world can draw. Unreachable in
    # practice, but it bounds the space honestly.
    ceiling = max_possible_scent(config.food, config.randomization)
    channels.append(
        Channel("food_smell", 0.0, ceiling, lambda s: float(s.food.scent_at(s.worm.position)))
    )

    if config.observation.include_touch:
        # Mechanosensation: 0 in open water, approaching 1 pushing head-on into
        # a wall, always 0 under `boundary: wrap`. Without it a wall cannot be
        # perceived, so leaving one cannot be learned — a policy lacking this
        # channel spent 26.7% of its life pinned against an edge.
        channels.append(Channel("touch", 0.0, 1.0, lambda s: s.worm.blocked))

    # sin/cos rather than raw theta, so nearly identical headings are nearly
    # identical inputs instead of straddling the 0/2pi discontinuity.
    channels.append(Channel("heading_sin", -1.0, 1.0, lambda s: s.worm.heading_sin_cos[0]))
    channels.append(Channel("heading_cos", -1.0, 1.0, lambda s: s.worm.heading_sin_cos[1]))

    return channels


class ObservationBuilder:
    """Turns the channel list into a bounded float32 observation vector.

    Attributes:
        config: Environment config the channels were built from.
        channels: The channels making up one observation frame.
    """

    def __init__(self, config: EnvConfig):
        """Builds the channel list and caches its bounds.

        Args:
            config: Environment config.
        """
        self.config = config
        self.channels = build_channels(config)
        self._low = np.array([c.low for c in self.channels], dtype=np.float32)
        self._high = np.array([c.high for c in self.channels], dtype=np.float32)

    @property
    def labels(self) -> list[str]:
        """Channel names, in observation order."""
        return [c.name for c in self.channels]

    @property
    def space(self) -> spaces.Box:
        """The Gymnasium space these observations inhabit."""
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    def __call__(self, sensors: Sensors) -> np.ndarray:
        """Reads every channel into a single observation.

        Args:
            sensors: Current simulation state.

        Returns:
            The observation, clipped into the declared bounds. Clipping rather
            than trusting the channels keeps an out-of-bounds value from
            becoming a confusing crash somewhere downstream.
        """
        values = np.array([c.read(sensors) for c in self.channels], dtype=np.float32)
        return np.clip(values, self._low, self._high)
