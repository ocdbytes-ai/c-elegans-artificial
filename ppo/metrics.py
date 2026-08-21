"""Run logging.

Episode statistics themselves live in :mod:`envs.episodes`, so a hand-coded
baseline and a trained policy are measured by the same code. This module is
only the training-run bookkeeping on top of them.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from envs.episodes import EpisodeAccumulator, EpisodeSummary, EpochStats

__all__ = ["EpisodeAccumulator", "EpisodeSummary", "EpochStats", "RunLogger"]


class RunLogger:
    """Appends one row per epoch to ``progress.csv`` and prints a short table.

    Reopens the file per row rather than holding a handle. At once per epoch
    that costs nothing, and a run killed mid-training still leaves a complete
    CSV behind.

    Attributes:
        run_dir: Directory the run writes to.
        path: The CSV being appended to.
        start: Wall-clock time the logger was created.
    """

    # eval_lifespan is the only column here scored on a fixed world, so it is
    # the only one that can be compared across epochs. It is NaN on epochs that
    # were not evaluated, and those are skipped rather than printed as "nan".
    HEADLINE = (
        "food_count",
        "lifespan_mean",
        "eaten_mean",
        "mean_abs_action",
        "entropy",
        "death_rate",
        "eval_lifespan",
    )

    def __init__(self, run_dir: Path):
        """Prepares to log into a run directory.

        Args:
            run_dir: Directory to write ``progress.csv`` into.
        """
        self.run_dir = run_dir
        self.path = run_dir / "progress.csv"
        self.start = time.time()
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        """Writes one epoch's row and prints its headline columns.

        The first row fixes the column set, so every later row must carry the
        same keys.

        Args:
            row: Statistics for the epoch. Must include ``epoch`` and
                ``total_steps``.
        """
        row = {"elapsed_s": round(time.time() - self.start, 1), **row}
        first_row = self._fieldnames is None
        if first_row:
            self._fieldnames = list(row)

        with open(self.path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if first_row:
                writer.writeheader()
            writer.writerow(row)

        headline = "  ".join(
            f"{key.replace('_mean', '')} {row[key]:>8.3f}"
            for key in self.HEADLINE
            if key in row and row[key] == row[key]  # skip NaN
        )
        print(f"epoch {row['epoch']:>4}  steps {row['total_steps']:>9,}  {headline}")
