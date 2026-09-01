"""Main runner for reproducible tabular HPO benchmarks using YAHPO + Syne Tune.

This script evaluates baseline schedulers and retains an optional legacy
integration wrapper for provenance:
- Random Search
- ASHA
- BOHB
- DEHB-style (lightweight provenance baseline, not official DEHB)
- Legacy wrapper (not the Full HAT controller)

The benchmark is fully tabular: no real model training is performed. All metrics
are queried from YAHPO Gym surrogate tables.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from lcbench.hagfish_wrapper import HagfishObservation, HagfishWrapper
except ModuleNotFoundError:
    # Workspace-local fallback when the lcbench package namespace is unavailable.
    from hagfish_wrapper import HagfishObservation, HagfishWrapper


@dataclass
class EvalObservation:
    """Normalized output from one YAHPO query."""

    validation_error: float
    cost: float
    raw_result: Dict[str, Any]


@dataclass
class TrialRecord:
    """State tracked for one logical trial."""

    trial_id: int
    config: Dict[str, Any]


class YAHPOScenario:
    """Thin adapter around YAHPO Gym benchmark scenarios."""

    def __init__(self, scenario_name: str, instance: str, seed: int) -> None:
        try:
            from yahpo_gym import benchmark_set
        except ImportError as exc:
            raise ImportError(
                "yahpo-gym is required. Install dependencies via requirements.txt"
            ) from exc

        self.scenario_name = scenario_name
        self.instance = instance
        self.seed = seed
        self.bench = benchmark_set.BenchmarkSet(scenario_name)
        self.bench.set_instance(str(instance))

        self.opt_space = self.bench.get_opt_space(drop_fidelity_params=True)
        self.fidelity_space = self.bench.get_fidelity_space()
        self.fidelity_name, self.min_fidelity, self.max_fidelity = (
            self._extract_fidelity_info()
        )

        self._validation_key: Optional[str] = None
        self._accuracy_key: Optional[str] = None
        self._cost_key: Optional[str] = None

        if hasattr(self.opt_space, "seed"):
            self.opt_space.seed(seed)

    def sample_random_config(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample one configuration from the extracted YAHPO search space."""

        if hasattr(self.opt_space, "sample_configuration"):
            cfg = self.opt_space.sample_configuration()
            if hasattr(cfg, "get_dictionary"):
                cfg = cfg.get_dictionary()
            return normalize_jsonable(dict(cfg))

        raise RuntimeError("Unsupported optimization space type from YAHPO")

    def evaluate(self, config: Dict[str, Any], fidelity: float) -> EvalObservation:
        """Query surrogate output for one (config, fidelity) point."""

        query = dict(config)
        query[self.fidelity_name] = coerce_fidelity_value(fidelity)

        # Some YAHPO scenarios allow an explicit seed argument.
        try:
            raw = self.bench.objective_function(query, seed=self.seed)
        except TypeError:
            raw = self.bench.objective_function(query)

        if isinstance(raw, list):
            if not raw:
                raise RuntimeError("YAHPO returned an empty result list")
            raw = raw[0]

        if not isinstance(raw, dict):
            raise RuntimeError(
                f"Unexpected YAHPO result type: {type(raw).__name__}. "
                "Expected dict-like result."
            )

        raw = normalize_jsonable(raw)
        validation_error, cost = self._extract_metrics(raw)

        return EvalObservation(
            validation_error=float(validation_error),
            cost=float(cost),
            raw_result=raw,
        )

    def _extract_fidelity_info(self) -> Tuple[str, float, float]:
        """Infer fidelity dimension name and bounds from scenario metadata."""

        # Preferred path: a ConfigSpace with one fidelity hyperparameter.
        if hasattr(self.fidelity_space, "get_hyperparameters"):
            hps = self.fidelity_space.get_hyperparameters()
            if len(hps) != 1:
                raise RuntimeError(
                    "Expected exactly one fidelity dimension, got "
                    f"{len(hps)}"
                )
            hp = hps[0]
            name = hp.name
            lower = float(getattr(hp, "lower", 1))
            upper = float(getattr(hp, "upper", lower))
            return name, lower, upper

        # Fallback path: dict-like fidelity info.
        if isinstance(self.fidelity_space, dict):
            if len(self.fidelity_space) != 1:
                raise RuntimeError(
                    "Expected exactly one fidelity dimension in dict metadata"
                )
            name, bounds = next(iter(self.fidelity_space.items()))
            if isinstance(bounds, dict):
                return (
                    str(name),
                    float(bounds.get("lower", bounds.get("min", 1))),
                    float(bounds.get("upper", bounds.get("max", 1))),
                )

        raise RuntimeError(
            "Could not infer fidelity information from YAHPO scenario"
        )

    def _extract_metrics(self, result: Dict[str, Any]) -> Tuple[float, float]:
        """Map scenario-specific output keys to (validation_error, cost)."""

        numeric_items = {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }

        if not numeric_items:
            raise RuntimeError("YAHPO result did not contain numeric metrics")

        if self._cost_key is None:
            self._cost_key = choose_key(
                list(numeric_items.keys()),
                [
                    "runtime",
                    "time",
                    "cost",
                    "timetrain",
                    "train_time",
                    "eval_time",
                ],
            )

        if self._validation_key is None and self._accuracy_key is None:
            self._validation_key = choose_key(
                list(numeric_items.keys()),
                [
                    "val_error",
                    "validation_error",
                    "valid_error",
                    "val_loss",
                    "valid_loss",
                    "error",
                ],
            )
            if self._validation_key is None:
                self._accuracy_key = choose_key(
                    list(numeric_items.keys()),
                    [
                        "val_accuracy",
                        "valid_accuracy",
                        "val_acc",
                        "accuracy",
                        "auc",
                    ],
                )

        if self._validation_key is not None:
            validation_error = numeric_items[self._validation_key]
        elif self._accuracy_key is not None:
            validation_error = 1.0 - numeric_items[self._accuracy_key]
        else:
            # Last-resort fallback: pick first numeric key not recognized as cost.
            remaining = [
                key for key in numeric_items if key != self._cost_key
            ]
            if not remaining:
                raise RuntimeError(
                    "Could not infer a validation metric from YAHPO outputs"
                )
            validation_error = numeric_items[remaining[0]]

        if self._cost_key is not None:
            cost = numeric_items[self._cost_key]
        else:
            # If scenario has no explicit runtime metric, count each query as unit cost.
            cost = 1.0

        if cost < 0:
            raise RuntimeError(f"Negative simulated cost detected: {cost}")

        return validation_error, cost


def get_available_instances(scenario_name: str) -> List[str]:
    """Return valid instance IDs declared by a YAHPO scenario.

    This prevents full benchmark sweeps from failing when users pass task IDs
    that are not available in the local YAHPO data snapshot.
    """
    try:
        from yahpo_gym import benchmark_set
    except ImportError as exc:
        raise ImportError(
            "yahpo-gym is required. Install dependencies via requirements.txt"
        ) from exc

    bench = benchmark_set.BenchmarkSet(scenario_name)

    candidates: List[str] = []

    # Preferred source: materialized list of allowed instance IDs.
    if hasattr(bench, "instances"):
        try:
            instances_obj = getattr(bench, "instances")
            if isinstance(instances_obj, (list, tuple, set)):
                candidates = [str(x) for x in instances_obj]
            elif instances_obj is not None and not isinstance(instances_obj, str):
                candidates = [str(x) for x in list(instances_obj)]
        except Exception:
            candidates = []

    # Fallback for API variations where instances may be attached to config.
    if not candidates:
        config = getattr(bench, "config", None)
        instance_names = getattr(config, "instance_names", None)
        if instance_names is not None and not isinstance(instance_names, str):
            try:
                candidates = [str(x) for x in list(instance_names)]
            except Exception:
                candidates = []

    if not candidates:
        raise RuntimeError(
            f"Could not determine valid instances for scenario '{scenario_name}'"
        )

    # Preserve order while removing duplicates.
    seen = set()
    ordered = []
    for inst in candidates:
        if inst not in seen:
            seen.add(inst)
            ordered.append(inst)
    return ordered


def resolve_instances(
    scenario_name: str,
    requested_instances: Sequence[str],
) -> List[str]:
    """Resolve requested IDs to the subset valid for the scenario.

    Special values: "auto" or "*" select all available instances.
    """
    available = get_available_instances(scenario_name)
    available_set = set(available)

    requested = [str(x) for x in requested_instances]
    use_all = (not requested) or any(x.lower() in {"auto", "*"} for x in requested)
    if use_all:
        print(
            f"[INFO] Auto-selected {len(available)} valid instances for scenario "
            f"'{scenario_name}'."
        )
        return available

    resolved = [inst for inst in requested if inst in available_set]
    dropped = [inst for inst in requested if inst not in available_set]
    if dropped:
        preview = ", ".join(dropped[:8])
        suffix = "..." if len(dropped) > 8 else ""
        print(
            f"[WARN] Skipping {len(dropped)} invalid instance(s) for scenario "
            f"'{scenario_name}': {preview}{suffix}"
        )

    if not resolved:
        raise ValueError(
            f"No valid instances remain for scenario '{scenario_name}'. "
            f"Available examples: {', '.join(available[:10])}"
        )

    return resolved


class RandomSearchStrategy:
    """Simple random search over the extracted scenario space."""

    def __init__(self, scenario: YAHPOScenario, seed: int) -> None:
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        _ = trial_id
        return self.scenario.sample_random_config(self.rng)

    def fidelity_schedule(self, trial_id: int) -> Sequence[float]:
        _ = trial_id
        return geometric_fidelity_levels(
            self.scenario.min_fidelity,
            self.scenario.max_fidelity,
        )

    def observe(self, **_: Any) -> None:
        return

    def on_trial_complete(self, trial: TrialRecord) -> None:
        _ = trial


class SyneTuneBaselineStrategy:
    """Adapter for Syne Tune ASHA and BOHB inside the tabular loop.

    Scheduler failures are raised instead of being replaced by random search;
    this prevents failed integrations from being reported under baseline names.
    """

    def __init__(
        self,
        algorithm: str,
        scenario: YAHPOScenario,
        seed: int,
    ) -> None:
        self.algorithm = algorithm
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scheduler = self._build_scheduler()
        self._active_trials: Dict[int, Any] = {}
        self._last_results: Dict[int, Dict[str, Any]] = {}

    def _build_scheduler(self) -> Any:
        try:
            from syne_tune.optimizer.baselines import ASHA, BOHB
        except ImportError as exc:
            raise ImportError(
                "syne-tune is required. Install dependencies via requirements.txt"
            ) from exc

        factory = {
            "asha": ASHA,
            "bohb": BOHB,
        }.get(self.algorithm)

        if factory is None:
            raise ValueError(f"Unsupported Syne Tune baseline: {self.algorithm}")

        config_space = build_syne_tune_config_space(self.scenario.opt_space)

        candidate_kwargs: Dict[str, Any] = {
            "config_space": config_space,
            "metric": "validation_error",
            "mode": "min",
            "random_seed": self.seed,
            "time_attr": self.scenario.fidelity_name,
            "max_t": int(self.scenario.max_fidelity),
            "resource_attr": self.scenario.fidelity_name,
            "max_resource_attr": self.scenario.fidelity_name,
            "max_resource_level": int(self.scenario.max_fidelity),
            "grace_period": int(self.scenario.min_fidelity),
        }

        sig = inspect.signature(factory)
        valid_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in sig.parameters
        }

        return factory(**valid_kwargs)

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        from syne_tune.backend.trial_status import Trial

        suggestion = self.scheduler.suggest()
        if suggestion is None:
            raise RuntimeError(
                f"{self.algorithm.upper()} returned no suggestion for trial {trial_id}"
            )
        config = extract_config_from_suggestion(suggestion)
        if config is None:
            raise RuntimeError(
                f"Could not extract a configuration from {self.algorithm.upper()} "
                f"suggestion type {type(suggestion).__name__}"
            )

        config = normalize_jsonable(config)
        scheduler_trial = Trial(
            trial_id=trial_id,
            config=config,
            creation_time=datetime.now(timezone.utc),
        )
        self.scheduler.on_trial_add(scheduler_trial)
        self._active_trials[trial_id] = scheduler_trial

        return config

    def fidelity_schedule(self, trial_id: int) -> Sequence[float]:
        _ = trial_id
        return geometric_fidelity_levels(
            self.scenario.min_fidelity,
            self.scenario.max_fidelity,
        )

    def observe(
        self,
        trial: TrialRecord,
        fidelity: float,
        validation_error: float,
        cost: float,
    ) -> bool:
        """Push observations into scheduler, return False if scheduler stops trial."""

        result = {
            "validation_error": float(validation_error),
            self.scenario.fidelity_name: coerce_fidelity_value(fidelity),
            "simulated_cost": float(cost),
        }

        scheduler_trial = self._active_trials[trial.trial_id]
        decision = self.scheduler.on_trial_result(
            trial=scheduler_trial,
            result=result,
        )
        self._last_results[trial.trial_id] = result
        return "STOP" not in str(decision).upper()

    def on_trial_complete(self, trial: TrialRecord) -> None:
        scheduler_trial = self._active_trials.pop(trial.trial_id)
        result = self._last_results.pop(trial.trial_id)
        if float(result[self.scenario.fidelity_name]) >= self.scenario.max_fidelity:
            self.scheduler.on_trial_complete(trial=scheduler_trial, result=result)
        else:
            self.scheduler.on_trial_remove(trial=scheduler_trial)


class DEHBFallbackStrategy:
    """Lightweight DEHB-style fallback when syne-tune DEHB is unavailable.

    This keeps the benchmark executable across syne-tune versions while
    preserving the same search space and fidelity bounds.
    """

    def __init__(self, scenario: YAHPOScenario, seed: int) -> None:
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.population: List[Dict[str, Any]] = []
        self.scores: Dict[int, float] = {}

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        if len(self.population) < 6:
            cfg = self.scenario.sample_random_config(self.rng)
            self.population.append(cfg)
            return cfg

        # Use best-so-far config as target and create a differential-style mutant.
        best_trial = min(self.scores, key=self.scores.get) if self.scores else None
        if best_trial is None:
            cfg = self.scenario.sample_random_config(self.rng)
            self.population.append(cfg)
            return cfg

        target = self.population[(best_trial - 1) % len(self.population)]
        donors = self.rng.choice(len(self.population), size=3, replace=False)
        a = self.population[int(donors[0])]
        b = self.population[int(donors[1])]
        c = self.population[int(donors[2])]

        mutant = self._mutate_config(target=target, a=a, b=b, c=c)
        self.population.append(mutant)
        return mutant

    def _mutate_config(
        self,
        target: Dict[str, Any],
        a: Dict[str, Any],
        b: Dict[str, Any],
        c: Dict[str, Any],
        F: float = 0.5,
        CR: float = 0.5,
    ) -> Dict[str, Any]:
        """Differential mutation with clipping for common ConfigSpace types."""

        try:
            from ConfigSpace.hyperparameters import (
                CategoricalHyperparameter,
                Constant,
                OrdinalHyperparameter,
                UniformFloatHyperparameter,
                UniformIntegerHyperparameter,
            )
        except ImportError:
            # Safe fallback if ConfigSpace is not importable for any reason.
            return self.scenario.sample_random_config(self.rng)

        hp_map = {
            hp.name: hp for hp in self.scenario.opt_space.get_hyperparameters()
        }
        child = dict(target)

        for name, hp in hp_map.items():
            if self.rng.random() > CR:
                continue

            if isinstance(hp, UniformFloatHyperparameter):
                proposal = float(a[name]) + F * (float(b[name]) - float(c[name]))
                child[name] = float(np.clip(proposal, hp.lower, hp.upper))
            elif isinstance(hp, UniformIntegerHyperparameter):
                proposal = float(a[name]) + F * (float(b[name]) - float(c[name]))
                clipped = int(round(np.clip(proposal, hp.lower, hp.upper)))
                child[name] = clipped
            elif isinstance(hp, CategoricalHyperparameter):
                # With categories, use donor crossover rather than arithmetic mutation.
                child[name] = a[name] if self.rng.random() < 0.5 else target[name]
            elif isinstance(hp, OrdinalHyperparameter):
                seq = list(hp.sequence)
                idx = int(self.rng.integers(0, len(seq)))
                child[name] = seq[idx]
            elif isinstance(hp, Constant):
                child[name] = hp.value

        return normalize_jsonable(child)

    def fidelity_schedule(self, trial_id: int) -> Sequence[float]:
        _ = trial_id
        return geometric_fidelity_levels(
            self.scenario.min_fidelity,
            self.scenario.max_fidelity,
        )

    def observe(
        self,
        trial: TrialRecord,
        fidelity: float,
        validation_error: float,
        cost: float,
    ) -> bool:
        _ = fidelity, cost
        # Track best observed loss per trial; used by suggest for evolutionary bias.
        prev = self.scores.get(trial.trial_id, math.inf)
        self.scores[trial.trial_id] = min(prev, float(validation_error))
        return True

    def on_trial_complete(self, trial: TrialRecord) -> None:
        _ = trial


def choose_key(keys: List[str], priority_substrings: List[str]) -> Optional[str]:
    """Return the first key matching a preferred substring list."""

    lowered = [(key, key.lower()) for key in keys]
    for needle in priority_substrings:
        for original, low in lowered:
            if needle in low:
                return original
    return None


def geometric_fidelity_levels(
    min_fidelity: float,
    max_fidelity: float,
    eta: float = 3.0,
) -> List[float]:
    """Generate a geometric rung schedule between min and max fidelities."""

    levels = [float(min_fidelity)]
    current = float(min_fidelity)

    while current < max_fidelity:
        proposal = min(max_fidelity, max(current + 1.0, current * eta))
        if int(proposal) == int(current):
            proposal = current + 1.0
        current = float(min(max_fidelity, proposal))
        if levels[-1] != current:
            levels.append(current)

    return levels


def coerce_fidelity_value(value: float) -> Any:
    """Use int when fidelity is effectively integral, else keep float."""

    if abs(value - round(value)) < 1e-10:
        return int(round(value))
    return float(value)


def normalize_jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy/scalar values to plain Python types for serialization."""

    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        elif isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        else:
            normalized[key] = value
    return normalized


def build_syne_tune_config_space(opt_space: Any) -> Dict[str, Any]:
    """Map ConfigSpace hyperparameters to Syne Tune search-space definitions."""

    try:
        from ConfigSpace.hyperparameters import (
            CategoricalHyperparameter,
            Constant,
            OrdinalHyperparameter,
            UniformFloatHyperparameter,
            UniformIntegerHyperparameter,
        )
        from syne_tune.config_space import (
            choice,
            lograndint,
            loguniform,
            randint,
            uniform,
        )
    except ImportError as exc:
        raise ImportError(
            "ConfigSpace and syne-tune are required for baseline construction"
        ) from exc

    if not hasattr(opt_space, "get_hyperparameters"):
        raise RuntimeError("Unsupported optimization space type for conversion")

    config_space: Dict[str, Any] = {}
    for hp in opt_space.get_hyperparameters():
        if isinstance(hp, UniformFloatHyperparameter):
            if bool(getattr(hp, "log", False)):
                config_space[hp.name] = loguniform(hp.lower, hp.upper)
            else:
                config_space[hp.name] = uniform(hp.lower, hp.upper)
        elif isinstance(hp, UniformIntegerHyperparameter):
            if bool(getattr(hp, "log", False)):
                config_space[hp.name] = lograndint(hp.lower, hp.upper)
            else:
                config_space[hp.name] = randint(hp.lower, hp.upper)
        elif isinstance(hp, CategoricalHyperparameter):
            config_space[hp.name] = choice(list(hp.choices))
        elif isinstance(hp, OrdinalHyperparameter):
            config_space[hp.name] = choice(list(hp.sequence))
        elif isinstance(hp, Constant):
            config_space[hp.name] = choice([hp.value])
        else:
            raise RuntimeError(
                f"Unsupported hyperparameter type: {type(hp).__name__}"
            )

    return config_space


def extract_config_from_suggestion(suggestion: Any) -> Optional[Dict[str, Any]]:
    """Normalize configuration returned by syne-tune scheduler suggestion."""

    if suggestion is None:
        return None

    if isinstance(suggestion, dict):
        return suggestion

    if hasattr(suggestion, "config") and isinstance(suggestion.config, dict):
        return suggestion.config

    if hasattr(suggestion, "trial_config") and isinstance(
        suggestion.trial_config, dict
    ):
        return suggestion.trial_config

    return None


def build_strategy(
    algorithm: str,
    scenario: YAHPOScenario,
    seed: int,
) -> Any:
    """Factory for all algorithms."""

    if algorithm == "random_search":
        return RandomSearchStrategy(scenario=scenario, seed=seed)

    if algorithm in {"asha", "bohb"}:
        return SyneTuneBaselineStrategy(
            algorithm=algorithm,
            scenario=scenario,
            seed=seed,
        )

    if algorithm == "dehb_style":
        return DEHBFallbackStrategy(scenario=scenario, seed=seed)

    if algorithm == "legacy_wrapper":
        search_space = {
            hp.name: hp for hp in scenario.opt_space.get_hyperparameters()
        }
        return HagfishWrapper(
            search_space=search_space,
            fidelity_name=scenario.fidelity_name,
            min_fidelity=scenario.min_fidelity,
            max_fidelity=scenario.max_fidelity,
            seed=seed,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_single_experiment(
    algorithm: str,
    scenario_name: str,
    instance: str,
    seed: int,
    max_trials: int,
) -> pd.DataFrame:
    """Run one algorithm on one (scenario, instance, seed)."""

    scenario = YAHPOScenario(scenario_name=scenario_name, instance=instance, seed=seed)
    strategy = build_strategy(algorithm=algorithm, scenario=scenario, seed=seed)

    cumulative_cost = 0.0
    best_error = math.inf
    rows: List[Dict[str, Any]] = []

    for trial_id in range(1, max_trials + 1):
        config = strategy.suggest(trial_id)
        trial = TrialRecord(trial_id=trial_id, config=config)

        fidelity_schedule = strategy.fidelity_schedule(trial_id)

        for rung_idx, fidelity in enumerate(fidelity_schedule, start=1):
            obs = scenario.evaluate(config=config, fidelity=fidelity)
            cumulative_cost += obs.cost
            best_error = min(best_error, obs.validation_error)

            rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario_name,
                    "instance": str(instance),
                    "seed": seed,
                    "trial_id": trial_id,
                    "rung": rung_idx,
                    "fidelity": fidelity,
                    "validation_error": obs.validation_error,
                    "simulated_cost": obs.cost,
                    "cumulative_cost": cumulative_cost,
                    "best_validation_error": best_error,
                    "config_json": json.dumps(normalize_jsonable(config)),
                }
            )

            keep_running = True
            if algorithm == "legacy_wrapper":
                strategy.observe(
                    HagfishObservation(
                        trial_id=trial_id,
                        fidelity=fidelity,
                        validation_error=obs.validation_error,
                        simulated_cost=obs.cost,
                        config=config,
                    )
                )
            else:
                keep_running = strategy.observe(
                    trial=trial,
                    fidelity=fidelity,
                    validation_error=obs.validation_error,
                    cost=obs.cost,
                )

            if not keep_running:
                break

        if hasattr(strategy, "on_trial_complete"):
            if algorithm == "legacy_wrapper":
                strategy.on_trial_complete(trial_id)
            else:
                strategy.on_trial_complete(trial)

    return pd.DataFrame(rows)


def run_benchmark(
    algorithms: Sequence[str],
    scenario_name: str,
    instances: Sequence[str],
    seeds: Sequence[int],
    max_trials: int,
    output_dir: Path,
) -> None:
    """Execute full benchmark matrix and persist logs for reproducibility."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []

    total_jobs = len(algorithms) * len(instances) * len(seeds)

    with tqdm(total=total_jobs, desc="Benchmark jobs") as pbar:
        for algorithm in algorithms:
            for instance in instances:
                for seed in seeds:
                    try:
                        run_df = run_single_experiment(
                            algorithm=algorithm,
                            scenario_name=scenario_name,
                            instance=instance,
                            seed=seed,
                            max_trials=max_trials,
                        )

                        per_run_path = raw_dir / (
                            f"{scenario_name}__{instance}__{algorithm}__seed{seed}.csv"
                        )
                        run_df.to_csv(per_run_path, index=False)
                        all_runs.append(run_df)
                    except Exception as exc:
                        failures.append(
                            {
                                "algorithm": algorithm,
                                "scenario": scenario_name,
                                "instance": str(instance),
                                "seed": seed,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    finally:
                        pbar.update(1)

    if all_runs:
        full_df = pd.concat(all_runs, ignore_index=True)
        full_df.to_csv(output_dir / "all_results.csv", index=False)

        summary = (
            full_df.sort_values("cumulative_cost")
            .groupby(["algorithm", "scenario", "instance", "seed"], as_index=False)
            .tail(1)
            .groupby(["algorithm", "scenario", "instance"], as_index=False)
            .agg(
                mean_final_best_error=("best_validation_error", "mean"),
                std_final_best_error=("best_validation_error", "std"),
                mean_total_cost=("cumulative_cost", "mean"),
                std_total_cost=("cumulative_cost", "std"),
                n_seeds=("seed", "nunique"),
            )
        )
        summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "scenario": scenario_name,
        "instances": list(map(str, instances)),
        "algorithms": list(algorithms),
        "seeds": list(map(int, seeds)),
        "max_trials": int(max_trials),
        "n_successful_runs": len(all_runs),
        "n_failed_runs": len(failures),
        "failures": failures,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YAHPO tabular baselines; Full HAT has a separate runner"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["lcbench", "rbv2"],
        default="lcbench",
        help="YAHPO scenario name",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["auto"],
        help=(
            "One or more scenario instance IDs. Use 'auto' to run all valid "
            "instances for the selected scenario. Example: --instances 3945"
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["random_search", "asha", "bohb", "dehb_style"],
        help="Subset to run; legacy_wrapper is provenance-only and is not Full HAT",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Random seeds (default: 0..9)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=60,
        help="Maximum number of new trial suggestions per run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs",
        help="Directory where logs and summaries are written",
    )
    return parser.parse_args()


def validate_algorithms(algorithms: Iterable[str]) -> List[str]:
    supported = {"random_search", "asha", "bohb", "dehb_style", "legacy_wrapper"}
    normalized = [algo.strip().lower() for algo in algorithms]
    unknown = sorted(set(normalized) - supported)
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}")
    return normalized


def main() -> None:
    args = parse_args()

    algorithms = validate_algorithms(args.algorithms)
    output_dir = Path(args.output_dir)
    instances = resolve_instances(args.scenario, args.instances)

    run_benchmark(
        algorithms=algorithms,
        scenario_name=args.scenario,
        instances=instances,
        seeds=args.seeds,
        max_trials=args.max_trials,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
