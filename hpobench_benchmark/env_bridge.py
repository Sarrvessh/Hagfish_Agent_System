"""Bridge utilities between simple-hpo-bench and optimization engines.

The bridge exposes a common multi-fidelity objective over
`hpo_benchmarks.HPOBench(dataset_name=...)` so Optuna and Ray Tune can share
one evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from hpo_benchmarks import HPOBench


@dataclass
class _SearchDimension:
    """Simple search-space dimension wrapper used by Ray searcher logic."""

    name: str
    values: List[Any]


@dataclass
class EvaluationResult:
    """Normalized result of one benchmark query."""

    validation_error: float
    simulated_cost: float


class HPOBenchObjectiveFunction:
    """Adapter that exposes one common multi-fidelity objective.

    Args:
        dataset_name: simple-hpo-bench dataset (e.g., "australian").
        benchmark_seed: Optional seed for benchmark stochasticity.
        fidelity_name: Name of the conceptual fidelity key.
        min_budget: Minimum fidelity value.
        max_budget: Maximum fidelity value.
        num_fidelity_steps: Number of stepped evaluations per trial.
        metric_key: Optional metric key to optimize.
        cost_key: Optional result key interpreted as simulated cost.
    """

    def __init__(
        self,
        dataset_name: str,
        benchmark_seed: Optional[int] = None,
        fidelity_name: str = "budget",
        min_budget: float = 0.1,
        max_budget: float = 1.0,
        num_fidelity_steps: int = 5,
        metric_key: Optional[str] = None,
        cost_key: Optional[str] = None,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.benchmark_seed = benchmark_seed
        self.fidelity_name = fidelity_name
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.num_fidelity_steps = int(max(1, num_fidelity_steps))
        self.metric_key = metric_key
        self.cost_key = cost_key

        if self.dataset_name not in HPOBench.available_dataset_names:
            available = ", ".join(HPOBench.available_dataset_names)
            raise ValueError(
                f"Unknown simple-hpo-bench dataset '{self.dataset_name}'. "
                f"Available datasets: {available}"
            )

        self._benchmark = HPOBench(dataset_name=self.dataset_name, seed=self.benchmark_seed)
        self._search_space = self._benchmark.search_space

    def fidelity_schedule(self) -> List[float]:
        """Generate a monotonically increasing fidelity schedule."""

        if self.num_fidelity_steps == 1:
            return [self.max_budget]
        return list(
            np.linspace(
                self.min_budget,
                self.max_budget,
                self.num_fidelity_steps,
                dtype=float,
            )
        )

    def sample_random_configuration(self, seed: int) -> Dict[str, Any]:
        """Sample one configuration from simple-hpo-bench search space."""

        rng = np.random.RandomState(int(seed))
        cfg: Dict[str, Any] = {}
        for name, values in self._search_space.items():
            idx = int(rng.randint(0, len(values)))
            cfg[name] = values[idx]
        return cfg

    def get_hyperparameters(self) -> List[_SearchDimension]:
        """Return dimension wrappers matching runner expectations."""

        return [
            _SearchDimension(name=name, values=list(values))
            for name, values in self._search_space.items()
        ]

    def optuna_sample_from_trial(self, trial: Any) -> Dict[str, Any]:
        """Sample one Optuna config from simple-hpo-bench discrete values."""

        config: Dict[str, Any] = {}
        for name, values in self._search_space.items():
            config[name] = trial.suggest_categorical(name, list(values))
        return config

    def ray_param_space(self) -> Dict[str, Any]:
        """Build Ray Tune discrete parameter space."""

        from ray import tune

        return {name: tune.choice(list(values)) for name, values in self._search_space.items()}

    def _extract_metric_value(self, payload: Dict[str, Any]) -> float:
        """Get objective metric from benchmark payload."""

        if self.metric_key and self.metric_key in payload:
            return float(payload[self.metric_key])

        preferred_keys: List[str] = []
        if getattr(self._benchmark, "metric_names", None):
            preferred_keys.extend(self._benchmark.metric_names)
        preferred_keys.extend(["val_acc", "accuracy", "score", "loss"])

        for key in preferred_keys:
            if key in payload:
                return float(payload[key])

        return float(next(iter(payload.values())))

    def _metric_to_error(self, metric_value: float, payload: Dict[str, Any]) -> float:
        """Convert metric to minimization error used by all runners."""

        metric_name = self.metric_key
        if metric_name is None and getattr(self._benchmark, "metric_names", None):
            metric_name = self._benchmark.metric_names[0]

        directions = getattr(self._benchmark, "directions", {})
        direction = directions.get(metric_name, "minimize") if isinstance(directions, dict) else "minimize"

        if direction == "maximize":
            if -1e-12 <= metric_value <= 1.0 + 1e-12:
                return 1.0 - metric_value
            return -metric_value
        return metric_value

    def _extract_simulated_cost(self, payload: Dict[str, Any], fidelity: float) -> float:
        """Extract or synthesize simulated cost."""

        if self.cost_key and self.cost_key in payload:
            return float(payload[self.cost_key])

        for key in ["cost", "runtime", "time", "train_time"]:
            if key in payload:
                return float(payload[key])

        # simple-hpo-bench metrics can omit explicit runtime cost; use fidelity-scaled surrogate cost.
        return max(1e-9, float(fidelity) ** 2)

    def evaluate(self, config: Dict[str, Any], fidelity: float, seed: int) -> EvaluationResult:
        """Evaluate one (config, fidelity) query on simple-hpo-bench."""

        if self.benchmark_seed is None:
            self._benchmark.reseed(int(seed))

        # simple-hpo-bench expects fidelity (e.g. budget) inside the config dict.
        eval_config = dict(config)
        eval_config[self.fidelity_name] = float(fidelity)

        payload = self._benchmark(eval_config)
        if not isinstance(payload, dict):
            raise TypeError("simple-hpo-bench must return a dictionary payload")

        metric_value = self._extract_metric_value(payload)
        validation_error = self._metric_to_error(metric_value, payload)
        simulated_cost = self._extract_simulated_cost(payload, fidelity=fidelity)

        return EvaluationResult(
            validation_error=float(validation_error),
            simulated_cost=float(simulated_cost),
        )
