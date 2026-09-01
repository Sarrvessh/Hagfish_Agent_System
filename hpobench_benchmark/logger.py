"""Unified anytime logger for benchmark runners.

This module standardizes logging across Optuna and Ray Tune execution engines.
The output schema is intentionally fixed to:

    [algorithm, seed, cumulative_simulated_cost, best_validation_error]
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


STANDARD_COLUMNS = [
    "algorithm",
    "seed",
    "cumulative_simulated_cost",
    "best_validation_error",
]


@dataclass
class _RunState:
    """Internal running state for one (algorithm, seed) stream."""

    cumulative_cost: float = 0.0
    best_error: float = float("inf")


class UnifiedResultLogger:
    """Write standardized anytime benchmark trajectories to CSV."""

    def __init__(self, output_csv: str) -> None:
        self.output_path = Path(output_csv).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[Tuple[str, int], _RunState] = {}

        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(STANDARD_COLUMNS)

    def log_observation(
        self,
        algorithm: str,
        seed: int,
        incremental_simulated_cost: float,
        validation_error: float,
    ) -> None:
        """Append one standardized observation row."""

        key = (str(algorithm), int(seed))
        state = self._state.setdefault(key, _RunState())

        state.cumulative_cost += float(max(0.0, incremental_simulated_cost))
        state.best_error = min(state.best_error, float(validation_error))

        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    key[0],
                    key[1],
                    state.cumulative_cost,
                    state.best_error,
                ]
            )

    def log_preaggregated(
        self,
        algorithm: str,
        seed: int,
        costs_and_errors: Iterable[Tuple[float, float]],
    ) -> None:
        """Log an iterable of (incremental_cost, validation_error) tuples."""

        for incremental_cost, validation_error in costs_and_errors:
            self.log_observation(
                algorithm=algorithm,
                seed=seed,
                incremental_simulated_cost=float(incremental_cost),
                validation_error=float(validation_error),
            )

    def reset_stream(self, algorithm: str, seed: int) -> None:
        """Reset one algorithm/seed stream state."""

        key = (str(algorithm), int(seed))
        if key in self._state:
            del self._state[key]

    @property
    def output_csv(self) -> str:
        """Return output CSV path as a string."""

        return str(self.output_path)
