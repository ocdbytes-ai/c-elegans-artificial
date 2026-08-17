"""Configuration schema for the worm world.

Every tunable constant lives here (and in ``configs/*.yaml``)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

BOUNDARIES = ("wrap", "clamp")
SCENT_PROFILES = ("gaussian", "linear", "inverse_square")


@dataclass
class WorldConfig:
    """The arena itself. Units are arbitrary ("body lengths"-ish)."""

    width: float = 20.0
    height: float = 20.0
    dt: float = 0.1  # seconds of sim time per step
    boundary: str = "wrap"  # "wrap" (torus, no walls) or "clamp" (hard edges)

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}, got {self.boundary!r}")
        if min(self.width, self.height, self.dt) <= 0:
            raise ValueError("width, height and dt must be positive")

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)

    @property
    def diagonal(self) -> float:
        return float((self.width**2 + self.height**2) ** 0.5)


@dataclass
class WormConfig:
    """The body. For v1 it is a single point with a heading."""

    max_speed: float = 2.0  # world units / second at full throttle
    max_turn_rate: float = 3.0  # radians / second at full turn
    allow_reverse: bool = True  # C. elegans reverses (part of the pirouette)
    radius: float = 0.25  # cosmetic only; eating uses food.eat_radius

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_turn_rate, self.radius) <= 0:
            raise ValueError("max_speed, max_turn_rate and radius must be positive")


@dataclass
class FoodConfig:
    """Pellets and the scent they emit.

    Scent is a scalar field: it peaks at the centre of a pellet and falls off
    with distance, so ``scent_radius`` is the *sensory* footprint while
    ``eat_radius`` is the much smaller *contact* footprint.
    """

    count: int = 8
    eat_radius: float = 0.4
    scent_radius: float = 4.0
    scent_peak: float = 1.0  # concentration at the exact centre of a pellet
    scent_profile: str = "gaussian"  # see envs/scent.py
    gaussian_sigma_scale: float = 0.35  # sigma = scale * scent_radius
    respawn_on_eat: bool = True
    min_spawn_distance: float = 2.0  # keep new pellets off the worm's head

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("food.count must be >= 1")
        if self.scent_profile not in SCENT_PROFILES:
            raise ValueError(
                f"scent_profile must be one of {SCENT_PROFILES}, got {self.scent_profile!r}"
            )
        if self.eat_radius <= 0 or self.scent_radius <= 0:
            raise ValueError("eat_radius and scent_radius must be positive")
        if self.eat_radius > self.scent_radius:
            raise ValueError("eat_radius must not exceed scent_radius (nothing to follow)")


@dataclass
class MetabolismConfig:
    """Energy bookkeeping — the only thing standing between the worm and death.

    ``basal_cost`` must dominate ``move_cost``: if moving costs meaningfully
    more than idling, freezing maximises lifespan and PPO will happily find
    that local optimum instead of learning to forage (PROJECT_PLAN.md §2.0).
    """

    max_energy: float = 100.0
    initial_energy: float = 50.0
    basal_cost: float = 0.25  # per step, unconditional
    move_cost: float = 0.02  # per step, times |action|^2 (max 2.0 -> 0.04)
    food_value: float = 25.0  # energy per pellet eaten
    min_speed_factor: float = 0.35  # starving worms are slow, never frozen
    speed_knee: float = 0.25  # energy fraction above which speed is unimpaired

    def __post_init__(self) -> None:
        if self.max_energy <= 0:
            raise ValueError("max_energy must be positive")
        if not 0 < self.initial_energy <= self.max_energy:
            raise ValueError("initial_energy must be in (0, max_energy]")
        if not 0 < self.min_speed_factor <= 1:
            raise ValueError("min_speed_factor must be in (0, 1] — 0 is a death spiral")
        if not 0 < self.speed_knee <= 1:
            raise ValueError("speed_knee must be in (0, 1]")
        if self.basal_cost <= 0:
            raise ValueError("basal_cost must be positive — existing is not free")


@dataclass
class RewardConfig:
    """Survival is the only reward. Resist adding sub-rewards here."""

    survival: float = 1.0  # per surviving step
    death_penalty: float = 0.0  # optional sharpening, not required

    def __post_init__(self) -> None:
        if self.death_penalty < 0:
            raise ValueError("death_penalty is subtracted; give it a positive magnitude")


@dataclass
class ObservationConfig:
    """What the worm can sense.

    Food is a single scalar concentration sampled at the head — chemotaxis has
    to be learned, never handed over. See :mod:`envs.observations` for the
    channel list.
    """

    include_energy: bool = True  # interoception; needed for hunger-modulated behaviour


@dataclass
class RenderConfig:
    window_size: int = 700  # pixels along the world's longer axis
    fps: int = 30
    show_scent_field: bool = True
    scent_grid: int = 96  # resolution of the scent heat-map, before upscaling
    show_scent_rings: bool = True  # outline each pellet's scent_radius

    def __post_init__(self) -> None:
        if self.window_size < 64 or self.scent_grid < 8 or self.fps < 1:
            raise ValueError("window_size >= 64, scent_grid >= 8 and fps >= 1 required")


@dataclass
class EnvConfig:
    """Root config object handed to :class:`envs.worm_world.WormWorldEnv`."""

    world: WorldConfig = field(default_factory=WorldConfig)
    worm: WormConfig = field(default_factory=WormConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EnvConfig:
        """Build a config from (partial) nested dicts; unknown keys are errors."""
        return _from_dict(cls, data or {}, path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def resolve(cls, source: EnvConfig | dict[str, Any] | str | Path | None) -> EnvConfig:
        """Accept whatever the caller has: object, dict, YAML path, or nothing."""
        if source is None:
            return cls()
        if isinstance(source, cls):
            return source
        if isinstance(source, (str, Path)):
            return cls.from_yaml(source)
        if isinstance(source, dict):
            return cls.from_dict(source)
        raise TypeError(f"cannot build an EnvConfig from {type(source).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _from_dict(cls: type, data: dict[str, Any], path: str) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"config section {path or '<root>'!r} must be a mapping, got {data!r}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = f"{path}." if path else ""
        raise ValueError(
            f"unknown config key(s) {sorted(where + key for key in unknown)}; "
            f"valid keys here: {sorted(known)}"
        )

    # `from __future__ import annotations` leaves field types as strings, so
    # resolve them properly rather than pattern-matching on the text.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        child = f"{path}.{name}" if path else name
        field_type = hints[name]
        # Nested sections recurse; leaves are plain scalars.
        kwargs[name] = _from_dict(field_type, value, child) if is_dataclass(field_type) else value
    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj
