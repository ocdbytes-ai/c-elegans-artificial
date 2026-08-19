"""Shared test setup: import path, and helpers used by more than one module."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def assert_yaml_covers_config():
    """Provides an assertion that a YAML file names every config field.

    The YAML is the tuning surface, so its values are expected to drift from the
    dataclass defaults. What must not drift is coverage: a newly added constant
    that never reaches the config file is invisible to whoever is tuning it.
    Unknown keys are already rejected by the loader.

    Returns:
        A callable taking a YAML path and a config class, which asserts that
        every field of the class tree appears in the file.
    """

    def check(yaml_path: str, config_cls: type) -> None:
        """Asserts the file covers every field of the config tree.

        Args:
            yaml_path: File to check.
            config_cls: Root config dataclass.
        """
        with open(yaml_path) as handle:
            raw = yaml.safe_load(handle)

        def walk(actual, expected, path):
            """Recurses both trees, asserting the file covers every field."""
            missing = set(expected) - set(actual or {})
            assert not missing, (
                f"{yaml_path} is missing {sorted(f'{path}{key}' for key in missing)}"
            )
            for key, value in expected.items():
                if isinstance(value, dict):
                    walk(actual[key], value, f"{path}{key}.")

        walk(raw, config_cls().to_dict(), "")

    return check
