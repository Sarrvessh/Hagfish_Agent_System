"""Run native Ray Tune baselines on simple-hpo-bench datasets.

Baselines implemented with official Ray Tune APIs:
- PopulationBasedTraining scheduler (PBT)
- Epsilon-greedy searcher (minimal custom Ray Searcher)

Both modes query simple-hpo-bench via env_bridge.py only (no model training).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import ray
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import Searcher
from ray.tune.search.basic_variant import BasicVariantGenerator

from env_bridge import HPOBenchObjectiveFunction
from logger import UnifiedResultLogger


class EpsilonGreedySearcher(Searcher):
    """Minimal epsilon-greedy searcher for Ray Tune.

    This implementation is intentionally lightweight and statistically sound:
    - Explore with probability epsilon by uniform random sampling.
    - Exploit otherwise by mutating the current incumbent configuration.

    It is used only for the user-requested epsilon-greedy baseline.
    """

    def __init__(self, bridge: HPOBenchObjectiveFunction, epsilon: float, seed: int) -> None:
        super().__init__(metric="best_validation_error", mode="min")
        self.bridge = bridge
        self.epsilon = float(np.clip(epsilon, 0.0, 1.0))
        self.rng = random.Random(seed)
        self.best_config: Optional[Dict[str, object]] = None
        self.best_metric = float("inf")
        self._counter = 0

    def suggest(self, trial_id: str) -> Optional[Dict[str, object]]:
        """Return a configuration for a new trial."""

        self._counter += 1
        should_explore = self.best_config is None or self.rng.random() < self.epsilon

        if should_explore:
            return self.bridge.sample_random_configuration(seed=self.rng.randint(0, 10_000_000))

        # Exploit incumbent with small random mutation on one dimension.
        candidate = dict(self.best_config)
        hps = self.bridge.get_hyperparameters()
        hp = self.rng.choice(hps)

        # Resample one dimension from ConfigSpace to keep candidate valid.
        sampled = self.bridge.sample_random_configuration(seed=self.rng.randint(0, 10_000_000))
        candidate[hp.name] = sampled[hp.name]
        return candidate

    def on_trial_complete(
        self,
        trial_id: str,
        result: Optional[Dict[str, object]] = None,
        error: bool = False,
    ) -> None:
        """Update incumbent on successful trial completion."""

        if error or not result:
            return

        metric = result.get("best_validation_error")
        config = result.get("config")
        if metric is None or config is None:
            return

        metric = float(metric)
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_config = dict(config)


def build_trainable(
    bridge: HPOBenchObjectiveFunction,
    algorithm: str,
    seed: int,
):
    """Create Ray Tune function trainable for surrogate benchmark querying."""

    schedule = bridge.fidelity_schedule()

    def trainable(config: Dict[str, object]) -> None:
        best_error = float("inf")

        for iteration, budget in enumerate(schedule, start=1):
            result = bridge.evaluate(config=config, fidelity=budget, seed=seed)
            best_error = min(best_error, result.validation_error)

            # We report both incremental and cumulative-friendly fields.
            tune.report(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "config": config,
                    "training_iteration": iteration,
                    "budget": float(budget),
                    "incremental_simulated_cost": float(result.simulated_cost),
                    "validation_error": float(result.validation_error),
                    "best_validation_error": float(best_error),
                }
            )

    return trainable


def pbt_hyperparam_mutations(bridge: HPOBenchObjectiveFunction) -> Dict[str, object]:
    """Build PBT mutation space matching the benchmark ConfigSpace."""

    return bridge.ray_param_space()


def _collect_trial_events(result_grid: "tune.ResultGrid") -> List[Dict[str, float]]:
    """Collect per-iteration event rows from all Ray trials."""

    events: List[Dict[str, float]] = []

    for result in result_grid:
        df = result.metrics_dataframe
        if df is None or df.empty:
            continue

        columns = set(df.columns)
        required = {
            "incremental_simulated_cost",
            "best_validation_error",
        }
        if not required.issubset(columns):
            continue

        for _, row in df.iterrows():
            timestamp = float(row.get("timestamp", 0.0))
            events.append(
                {
                    "timestamp": timestamp,
                    "incremental_simulated_cost": float(row["incremental_simulated_cost"]),
                    "best_validation_error": float(row["best_validation_error"]),
                }
            )

    events.sort(key=lambda x: x["timestamp"])
    return events


def run_one_ray_algorithm(
    algorithm: str,
    seed: int,
    n_samples: int,
    bridge: HPOBenchObjectiveFunction,
    results_logger: UnifiedResultLogger,
    pbt_population_size: int,
    epsilon: float,
) -> None:
    """Run one Ray baseline and write standardized anytime trajectory."""

    trainable = build_trainable(bridge=bridge, algorithm=algorithm, seed=seed)
    param_space: Dict[str, object]

    if algorithm == "pbt":
        param_space = bridge.ray_param_space()
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="best_validation_error",
            mode="min",
            perturbation_interval=max(1, bridge.num_fidelity_steps // 2),
            hyperparam_mutations=pbt_hyperparam_mutations(bridge),
            quantile_fraction=0.25,
            resample_probability=0.25,
        )
        search_alg = BasicVariantGenerator(random_state=seed)
        tune_config = tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=max(n_samples, pbt_population_size),
            max_concurrent_trials=pbt_population_size,
        )
    elif algorithm == "epsilon_greedy":
        # Searcher proposes fully concrete configs, so param_space must stay empty.
        param_space = {}
        scheduler = None
        search_alg = EpsilonGreedySearcher(bridge=bridge, epsilon=epsilon, seed=seed)
        tune_config = tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=n_samples,
        )
    else:
        raise ValueError(f"Unsupported Ray algorithm: {algorithm}")

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune_config,
    )
    result_grid = tuner.fit()

    # Convert trial-local reports into one global anytime stream per algorithm/seed.
    events = _collect_trial_events(result_grid)
    results_logger.reset_stream(algorithm=algorithm, seed=seed)
    for row in events:
        results_logger.log_observation(
            algorithm=algorithm,
            seed=seed,
            incremental_simulated_cost=row["incremental_simulated_cost"],
            validation_error=row["best_validation_error"],
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run native Ray Tune baselines on simple-hpo-bench datasets."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="simple-hpo-bench dataset name, e.g. 'australian'",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=None,
        help="Optional seed passed to hpo_benchmarks.HPOBench.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["pbt", "epsilon_greedy"],
        choices=["pbt", "epsilon_greedy"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--pbt-population-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--fidelity-name", type=str, default="budget")
    parser.add_argument("--min-budget", type=float, default=0.1)
    parser.add_argument("--max-budget", type=float, default=1.0)
    parser.add_argument("--num-fidelity-steps", type=int, default=5)
    parser.add_argument("--metric-key", type=str, default=None)
    parser.add_argument("--cost-key", type=str, default=None)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiments/output/simple_hpo_bench_native/results.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for Ray baseline execution."""

    args = parse_args()
    # local_mode=True keeps execution deterministic and easy to inspect.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False)

    bridge = HPOBenchObjectiveFunction(
        dataset_name=args.dataset_name,
        benchmark_seed=args.benchmark_seed,
        fidelity_name=args.fidelity_name,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        num_fidelity_steps=args.num_fidelity_steps,
        metric_key=args.metric_key,
        cost_key=args.cost_key,
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_logger = UnifiedResultLogger(output_csv=args.output_csv)

    for algorithm in args.algorithms:
        for seed in args.seeds:
            run_one_ray_algorithm(
                algorithm=algorithm,
                seed=seed,
                n_samples=args.n_samples,
                bridge=bridge,
                results_logger=results_logger,
                pbt_population_size=args.pbt_population_size,
                epsilon=args.epsilon,
            )

    print(f"Saved standardized results to: {results_logger.output_csv}")
    ray.shutdown()


if __name__ == "__main__":
    main()
