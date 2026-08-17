"""Positions, distances and angles on a bounded 2D plane.

The world is either a torus (``boundary="wrap"``) or a box with hard edges
(``boundary="clamp"``). Everything that measures a distance has to agree on
which one is in force, so all of it lives here — the worm, the scent field and
the renderer all call into these helpers.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Fold an angle into [-pi, pi)."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def apply_boundary(
    position: np.ndarray, size: tuple[float, float], boundary: str
) -> np.ndarray:
    """Keep a position inside the arena, wrapping or clamping as configured."""
    extent = np.asarray(size, dtype=np.float64)
    if boundary == "wrap":
        return np.mod(position, extent)
    if boundary == "clamp":
        return np.clip(position, 0.0, extent)
    raise ValueError(f"unknown boundary {boundary!r}")


def displacement(
    origin: np.ndarray,
    target: np.ndarray,
    size: tuple[float, float],
    boundary: str,
) -> np.ndarray:
    """Vector from ``origin`` to ``target``, broadcasting over leading axes.

    On a torus this is the *minimum image* convention: the shortest of the nine
    equivalent copies of ``target``. Without it a pellet just across the seam
    would read as maximally far away.
    """
    delta = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    if boundary == "wrap":
        extent = np.asarray(size, dtype=np.float64)
        delta = (delta + extent / 2.0) % extent - extent / 2.0
    return delta


def distance(
    origin: np.ndarray,
    target: np.ndarray,
    size: tuple[float, float],
    boundary: str,
) -> np.ndarray:
    """Euclidean distance under the same boundary convention."""
    return np.linalg.norm(displacement(origin, target, size, boundary), axis=-1)


def max_distance(size: tuple[float, float], boundary: str) -> float:
    """Largest distance two points can be apart — used to normalise observations."""
    width, height = size
    if boundary == "wrap":
        return float(np.hypot(width / 2.0, height / 2.0))
    return float(np.hypot(width, height))


def heading_vector(theta: float) -> np.ndarray:
    """Unit vector pointing along a heading."""
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


def sample_positions(
    rng: np.random.Generator,
    count: int,
    size: tuple[float, float],
    boundary: str,
    exclude: np.ndarray | None = None,
    min_distance: float = 0.0,
    max_attempts: int = 32,
) -> np.ndarray:
    """Uniformly sample ``count`` positions, optionally away from ``exclude``.

    Rejection sampling with a bounded attempt count: a config asking for an
    impossible clearance degrades to "as far as we got" rather than hanging.
    """
    extent = np.asarray(size, dtype=np.float64)
    positions = rng.uniform(0.0, extent, size=(count, 2))
    if exclude is None or min_distance <= 0.0:
        return positions

    for _ in range(max_attempts):
        too_close = distance(exclude, positions, size, boundary) < min_distance
        if not np.any(too_close):
            break
        positions[too_close] = rng.uniform(0.0, extent, size=(int(too_close.sum()), 2))
    return positions
