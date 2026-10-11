"""Argument types shared by the command-line scripts."""

from __future__ import annotations

import argparse


def positive_int(text: str) -> int:
    """Parses a count that must be at least 1.

    The scripts never give 0 a special meaning such as "forever" or "no
    limit"; those modes have flags of their own. A 0 or a negative here is a
    typo, so it is rejected rather than quietly running nothing.

    Args:
        text: The raw command-line value.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If ``text`` is not an integer of at least 1.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value
