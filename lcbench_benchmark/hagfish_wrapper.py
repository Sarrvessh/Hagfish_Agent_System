"""Legacy LCBench integration wrapper retained only for provenance.

This module does not execute AdaptiveTrainer, PlannerAgent, CriticAgent, or
AgentMemory and must not be interpreted as the Full HAT controller evaluated
in the paper. Current HAT experiments use ``camera_ready_studies.py`` and
``run_real_hat_full.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class HagfishObservation:
    """A single observation emitted by the benchmark runner to Hagfish."""

    trial_id: int
    fidelity: float
    validation_error: float
    simulated_cost: float
    config: Dict[str, Any]


@dataclass
class HagfishWrapper:
    """Legacy random-search/geometric-fidelity integration wrapper.

    This class is retained to document the superseded integration path.
    It receives exactly the same search-space dictionary and fidelity bounds that
    the baseline methods use.

    Parameters
    ----------
    search_space:
        Hyperparameter search-space as a dict of ConfigSpace hyperparameters
        (or equivalent objects) keyed by name.
    fidelity_name:
        Name of the fidelity dimension in YAHPO (e.g., "epoch").
    min_fidelity:
        Lowest fidelity level available in the scenario.
    max_fidelity:
        Highest fidelity level available in the scenario.
    seed:
        Random seed for deterministic behavior.
    eta:
        Multiplicative factor used by the default geometric fidelity schedule.
    """

    search_space: Dict[str, Any]
    fidelity_name: str
    min_fidelity: float
    max_fidelity: float
    seed: int
    eta: float = 3.0
    rng: np.random.Generator = field(init=False)
    history: List[HagfishObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        """Suggest the next hyperparameter configuration.

        Replace this implementation with your Hagfish decision rule.
        The default implementation is random sampling for a safe baseline.
        """

        config: Dict[str, Any] = {}
        for name, hp in self.search_space.items():
            config[name] = self._sample_hyperparameter(name=name, hp=hp)
        return config

    def _sample_hyperparameter(self, name: str, hp: Any) -> Any:
        """Sample from common ConfigSpace hyperparameter variants.

        This fallback keeps the template self-contained even if specific
        ConfigSpace utility methods are unavailable.
        """

        if hasattr(hp, "choices"):
            choices = list(hp.choices)
            return choices[int(self.rng.integers(0, len(choices)))]

        if hasattr(hp, "sequence"):
            seq = list(hp.sequence)
            return seq[int(self.rng.integers(0, len(seq)))]

        if hasattr(hp, "value"):
            return hp.value

        if hasattr(hp, "lower") and hasattr(hp, "upper"):
            lower = float(hp.lower)
            upper = float(hp.upper)
            is_log = bool(getattr(hp, "log", False))
            is_int = isinstance(hp.lower, int) and isinstance(hp.upper, int)

            if is_log:
                sample = float(
                    np.exp(
                        self.rng.uniform(np.log(lower), np.log(upper))
                    )
                )
            else:
                sample = float(self.rng.uniform(lower, upper))

            if is_int:
                return int(round(sample))
            return sample

        raise TypeError(
            f"Unsupported hyperparameter object for '{name}': "
            f"{type(hp).__name__}"
        )

    def fidelity_schedule(self, trial_id: int) -> Sequence[float]:
        """Return the fidelity ladder for a trial.

        Replace this with Hagfish's adaptive budget policy if desired.
        The default schedule is geometric: min -> ... -> max.
        """

        levels = [float(self.min_fidelity)]
        current = float(self.min_fidelity)

        while current < self.max_fidelity:
            proposed = min(self.max_fidelity, max(current + 1, current * self.eta))
            if int(proposed) == int(current):
                proposed = current + 1
            current = float(min(self.max_fidelity, proposed))
            if levels[-1] != current:
                levels.append(current)

        return levels

    def observe(self, observation: HagfishObservation) -> None:
        """Receive an observation after each evaluation.

        Add your online learning / memory update code here.
        """

        self.history.append(observation)

    def on_trial_complete(self, trial_id: Any) -> None:
        """Optional callback after a trial has fully completed."""

        # This hook is intentionally left minimal for easy extension.
        _ = trial_id
