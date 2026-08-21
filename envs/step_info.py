"""Contract for the ``info`` dict returned by ``reset()`` and ``step()``.

The agent optimises the scalar reward alone, so everything here is diagnostics.
The contract gets its own module because the dict is *produced* in
:mod:`envs.worm_world` and *consumed* in :mod:`envs.episodes`. Written as bare
string literals in both, renaming a field would break the consumer with a
``KeyError`` partway through a rollout and nothing would catch it beforehand.

``TypedDict`` is not enforced at runtime, so
``tests/test_env.py::test_info_dict_matches_the_declared_contract`` asserts the
environment's actual keys against these annotations. That is what turns a rename
into a test failure rather than a surprise during training.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class StepInfo(TypedDict):
    """Per-step diagnostics from :class:`envs.worm_world.WormWorldEnv`."""

    steps: int

    # Interoception, and the energy ledger behind it. Logging intake and outgo
    # separately is what distinguishes a worm that starved from one that
    # exhausted itself moving — identical in the reward, obvious here.
    energy: float
    energy_fraction: float
    speed_factor: float
    basal_cost: float
    move_cost: float
    energy_intake: float
    toxin_damage: float

    # Foraging.
    food_eaten: int
    food_eaten_total: int
    food_smell: float
    toxin_smell: float
    nearest_food_distance: float

    # Kinematics. `touch` is the share of commanded motion the boundary ate,
    # and is what the mechanosensory observation channel reports.
    distance_moved: float
    touch: float
    position: np.ndarray
    heading: float

    # This episode's domain-randomisation draw. Not observable by the worm —
    # logged so lifespan can be checked against the body it happened to get.
    max_speed: float
    max_turn_rate: float
    food_count: int
