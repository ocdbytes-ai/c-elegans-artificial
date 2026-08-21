"""Scent falloff profiles.

A profile maps distance from a pellet to concentration, and so determines the
shape of gradient the worm has to climb. Profiles are kept separate from the
food logic and selected by name from the config. All are peak-normalised:
``f(0) == scent_peak``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import FoodConfig, ToxinConfig

ScentProfile = Callable[[np.ndarray], np.ndarray]


def gaussian(distance: np.ndarray, radius: float, peak: float, sigma_scale: float) -> np.ndarray:
    """Evaluates a Gaussian bump with infinite support.

    Support is deliberately unbounded: a truncated profile creates regions where
    the gradient is exactly zero and a smell-only worm has nothing to follow. At
    the default ``sigma_scale`` the concentration is ~1.7% of peak at ``radius``.

    Args:
        distance: Distance from the pellet centre.
        radius: Nominal scent radius.
        peak: Concentration at the centre.
        sigma_scale: Standard deviation as a fraction of ``radius``.

    Returns:
        Concentration at each distance.
    """
    sigma = max(radius * sigma_scale, 1e-9)
    return peak * np.exp(-0.5 * (distance / sigma) ** 2)


def linear(distance: np.ndarray, radius: float, peak: float) -> np.ndarray:
    """Evaluates a cone reaching exactly zero at ``radius``.

    Args:
        distance: Distance from the pellet centre.
        radius: Distance at which the concentration reaches zero.
        peak: Concentration at the centre.

    Returns:
        Concentration at each distance.
    """
    return peak * np.clip(1.0 - distance / radius, 0.0, 1.0)


def inverse_square(distance: np.ndarray, radius: float, peak: float) -> np.ndarray:
    """Evaluates a softened inverse-square law with a long tail.

    Args:
        distance: Distance from the pellet centre.
        radius: Sets the falloff scale; concentration is peak/2 at ``radius/2``.
        peak: Concentration at the centre.

    Returns:
        Concentration at each distance.
    """
    return peak / (1.0 + (distance / (0.5 * radius)) ** 2)


def make_scent_profile(config: FoodConfig | ToxinConfig) -> ScentProfile:
    """Binds the configured profile's parameters into a callable.

    Args:
        config: Food or toxin config naming the profile and its parameters.
            Both emit scent the same way; only what contact with the source
            does to the worm differs.

    Returns:
        A function mapping distance to concentration.

    Raises:
        ValueError: If ``config.scent_profile`` names an unknown profile.
    """
    radius, peak = config.scent_radius, config.scent_peak

    if config.scent_profile == "gaussian":
        scale = config.gaussian_sigma_scale
        return lambda d: gaussian(d, radius, peak, scale)
    if config.scent_profile == "linear":
        return lambda d: linear(d, radius, peak)
    if config.scent_profile == "inverse_square":
        return lambda d: inverse_square(d, radius, peak)
    raise ValueError(f"unknown scent profile {config.scent_profile!r}")
