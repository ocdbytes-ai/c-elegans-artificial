"""YAML to nested-dataclass loading, shared by the environment and the trainer.

Both :mod:`envs.config` and :mod:`ppo.config` describe themselves as trees of
dataclasses tuned from YAML, so the loading and round-tripping lives here once.

Two rules it enforces:

- Unknown keys are errors. A typo in a config file should surface at startup
  rather than silently keeping a default and changing what the run measured.
- YAML may be partial. Anything omitted keeps its dataclass default, so a config
  file can be a short list of deltas.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Self, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


def build(cls: type[T], data: dict[str, Any] | None, path: str = "") -> T:
    """Instantiates a possibly nested dataclass from partial dict data.

    Args:
        cls: The dataclass to build.
        data: Values to apply. Omitted fields keep their defaults.
        path: Dotted prefix used in error messages during recursion.

    Returns:
        The constructed dataclass.

    Raises:
        TypeError: If a section is not a mapping.
        ValueError: If ``data`` contains a key the dataclass does not define.
    """
    data = data or {}
    if not isinstance(data, dict):
        raise TypeError(f"config section {path or '<root>'!r} must be a mapping, got {data!r}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = f"{path}." if path else ""
        raise ValueError(
            f"unknown config key(s) {sorted(where + key for key in unknown)}; "
            f"valid keys here: {sorted(known)}"
        )

    # `from __future__ import annotations` leaves field types as strings, so
    # resolve them rather than pattern-matching on the text.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        child = f"{path}.{name}" if path else name
        field_type = hints[name]
        kwargs[name] = build(field_type, value, child) if is_dataclass(field_type) else value
    return cls(**kwargs)


def as_dict(obj: Any) -> Any:
    """Converts a dataclass tree to plain dicts, ready for YAML.

    Args:
        obj: A dataclass, or any leaf value.

    Returns:
        Nested dicts, with tuples flattened to lists since YAML has no tuple.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def load_yaml(cls: type[T], path: str | Path) -> T:
    """Builds a dataclass from a YAML file.

    Args:
        cls: The dataclass to build.
        path: File to read.

    Returns:
        The constructed dataclass.
    """
    with open(path) as handle:
        return build(cls, yaml.safe_load(handle))


def save_yaml(obj: Any, path: str | Path) -> None:
    """Writes a dataclass tree to a YAML file, preserving field order.

    Args:
        obj: The dataclass to write.
        path: File to write.
    """
    with open(path, "w") as handle:
        yaml.safe_dump(as_dict(obj), handle, sort_keys=False)


def coerce(cls: type[T], source: T | dict[str, Any] | str | Path | None) -> T:
    """Builds a config from whatever form the caller has.

    Args:
        cls: The dataclass to build.
        source: An instance, a nested dict, a path to YAML, or None for the
            defaults.

    Returns:
        The constructed dataclass. An instance is passed through unchanged, so
        callers can share and mutate one.

    Raises:
        TypeError: If ``source`` is none of the accepted forms.
    """
    if source is None:
        return cls()
    if isinstance(source, cls):
        return source
    if isinstance(source, (str, Path)):
        return load_yaml(cls, source)
    if isinstance(source, dict):
        return build(cls, source)
    raise TypeError(f"cannot build a {cls.__name__} from {type(source).__name__}")


class ConfigRoot:
    """Mixin supplying the loader entry points to a root config dataclass.

    :class:`envs.config.EnvConfig` and :class:`ppo.config.PPOConfig` are trees
    of dataclasses with identical loading semantics, so their constructors live
    here once.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        """Builds from partial nested dicts.

        Args:
            data: Values to apply, or None for the defaults.

        Returns:
            The constructed config.
        """
        return build(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Builds from a YAML file.

        Args:
            path: File to read.

        Returns:
            The constructed config.
        """
        return load_yaml(cls, path)

    @classmethod
    def resolve(cls, source: Self | dict[str, Any] | str | Path | None) -> Self:
        """Builds from an instance, a dict, a YAML path, or nothing.

        Args:
            source: Whatever form the caller has.

        Returns:
            The constructed config.
        """
        return coerce(cls, source)

    def to_dict(self) -> dict[str, Any]:
        """Returns the config as nested plain dicts."""
        return as_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Writes the config to a YAML file.

        Args:
            path: File to write.
        """
        save_yaml(self, path)


def validate_range(name: str, value: Any) -> tuple[float, float]:
    """Normalises a YAML list into a checked ``(low, high)`` tuple.

    Args:
        name: Field name, used in error messages.
        value: A two-element sequence.

    Returns:
        The validated pair.

    Raises:
        ValueError: If ``value`` is not a pair, ``low`` is not positive, or
            ``low`` exceeds ``high``.
    """
    try:
        low, high = (float(entry) for entry in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a [low, high] pair, got {value!r}") from None
    if low <= 0:
        raise ValueError(f"{name} low must be positive, got {low}")
    if low > high:
        raise ValueError(f"{name} low must not exceed high, got [{low}, {high}]")
    return (low, high)
