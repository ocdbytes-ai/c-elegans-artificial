"""Scent falloff profiles.

A pellet's scent is strongest at its centre and fades outwards; ``scent_radius``
sets the scale of that fade. A profile maps distance -> concentration and is
the single knob that decides *what shape of gradient the worm has to climb*, so
they are kept apart from the food logic and swapped by name from the config.

All profiles are peak-normalised: ``f(0) == scent_peak``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import FoodConfig

ScentProfile = Callable[[np.ndarray], np.ndarray]


def gaussian(distance: np.ndarray, radius: float, peak: float, sigma_scale: float) -> np.ndarray:
    """Smooth bump, never exactly zero.

    Infinite support is deliberate: a truncated profile creates dead zones where
    the gradient is exactly 0 and a smell-only worm has nothing to follow. With
    the default ``sigma_scale`` the concentration is ~1.7% of peak at
    ``scent_radius``.
    """
    sigma = max(radius * sigma_scale, 1e-9)
    return peak * np.exp(-0.5 * (distance / sigma) ** 2)


def linear(distance: np.ndarray, radius: float, peak: float) -> np.ndarray:
    """Cone: peak at the centre, exactly zero at ``scent_radius`` and beyond."""
    return peak * np.clip(1.0 - distance / radius, 0.0, 1.0)


def inverse_square(distance: np.ndarray, radius: float, peak: float) -> np.ndarray:
    """Softened 1/r^2 — a long tail, so distant pellets are still faintly smelled."""
    return peak / (1.0 + (distance / (0.5 * radius)) ** 2)


def make_scent_profile(config: FoodConfig) -> ScentProfile:
    """Bind the configured profile's parameters into a distance -> scent function."""
    radius, peak = config.scent_radius, config.scent_peak

    if config.scent_profile == "gaussian":
        scale = config.gaussian_sigma_scale
        return lambda d: gaussian(d, radius, peak, scale)
    if config.scent_profile == "linear":
        return lambda d: linear(d, radius, peak)
    if config.scent_profile == "inverse_square":
        return lambda d: inverse_square(d, radius, peak)
    raise ValueError(f"unknown scent profile {config.scent_profile!r}")
