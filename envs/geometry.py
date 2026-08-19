"""Positions, distances and angles on a bounded 2D plane.

The arena is either a torus (``boundary="wrap"``) or a box with hard edges
(``boundary="clamp"``). Every distance measured anywhere in the project goes
through these helpers so that the worm, the scent field and the renderer cannot
disagree about which convention is in force.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Folds an angle into [-pi, pi).

    Args:
        theta: Angle in radians, scalar or array.

    Returns:
        The equivalent angle in [-pi, pi).
    """
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def apply_boundary(position: np.ndarray, size: tuple[float, float], boundary: str) -> np.ndarray:
    """Keeps a position inside the arena.

    Args:
        position: Position to constrain, shape ``(2,)`` or ``(..., 2)``.
        size: Arena ``(width, height)``.
        boundary: Either ``"wrap"`` or ``"clamp"``.

    Returns:
        The constrained position, same shape as the input.

    Raises:
        ValueError: If ``boundary`` is not a known convention.
    """
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
    """Computes the vector from origin to target, broadcasting over leading axes.

    Under ``"wrap"`` this uses the minimum-image convention: the shortest of the
    nine equivalent copies of ``target``. Without it, a pellet just across the
    seam would read as maximally distant.

    Args:
        origin: Start position, shape ``(2,)`` or ``(..., 2)``.
        target: End position, broadcastable against ``origin``.
        size: Arena ``(width, height)``.
        boundary: Either ``"wrap"`` or ``"clamp"``.

    Returns:
        Displacement vectors with the broadcast shape of the inputs.
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
    """Computes Euclidean distance under the arena's boundary convention.

    Args:
        origin: Start position, shape ``(2,)`` or ``(..., 2)``.
        target: End position, broadcastable against ``origin``.
        size: Arena ``(width, height)``.
        boundary: Either ``"wrap"`` or ``"clamp"``.

    Returns:
        Distances with the broadcast shape of the inputs, minus the last axis.
    """
    return np.linalg.norm(displacement(origin, target, size, boundary), axis=-1)


def max_distance(size: tuple[float, float], boundary: str) -> float:
    """Returns the largest distance two points in the arena can be apart.

    Args:
        size: Arena ``(width, height)``.
        boundary: Either ``"wrap"`` or ``"clamp"``.

    Returns:
        Half the diagonal on a torus, the full diagonal in a box.
    """
    width, height = size
    if boundary == "wrap":
        return float(np.hypot(width / 2.0, height / 2.0))
    return float(np.hypot(width, height))


def heading_vector(theta: float) -> np.ndarray:
    """Returns the unit vector pointing along a heading.

    Args:
        theta: Heading in radians.

    Returns:
        ``[cos(theta), sin(theta)]``.
    """
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
    """Samples positions uniformly, optionally keeping clear of a point.

    Uses rejection sampling with a bounded attempt count, so a config demanding
    an impossible clearance degrades to a best effort rather than hanging.

    Args:
        rng: Source of randomness.
        count: Number of positions to draw.
        size: Arena ``(width, height)``.
        boundary: Either ``"wrap"`` or ``"clamp"``.
        exclude: Position to stay away from, or None for no exclusion.
        min_distance: Required clearance from ``exclude``.
        max_attempts: Rejection-sampling rounds before giving up.

    Returns:
        Positions of shape ``(count, 2)``.
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
