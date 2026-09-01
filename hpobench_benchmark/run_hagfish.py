"""Run Hagfish algorithm via native Optuna sampler API.

This implementation integrates the project-native `AdaptiveTrainer` planner/
critic/memory loop into Optuna's sampler lifecycle. Trial outcomes are fed back
to the trainer so future suggestions adapt based on observed metric-cost tradeoffs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import optuna
from optuna.samplers import BaseSampler

from env_bridge import HPOBenchObjectiveFunction
from logger import UnifiedResultLogger

try:
    from adaptive_trainer import AdaptiveTrainer
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from adaptive_trainer import AdaptiveTrainer


class HagfishOptunaSampler(BaseSampler):
    """AdaptiveTrainer-driven sampler that hooks Hagfish logic into Optuna."""

    def __init__(
        self,
        epsilon: float = 0.20,
        alpha: float = 5e-4,
        n_trials_hint: int = 30,
        seed: int = 0,
    ) -> None:
        self.epsilon = float(np.clip(epsilon, 0.0, 1.0))
        self.alpha = float(alpha)
        self.n_trials_hint = int(max(1, n_trials_hint))
        self.rng = random.Random(seed)
        self.random_sampler = optuna.samplers.RandomSampler(seed=seed)
        self.trainer = AdaptiveTrainer(alpha=self.alpha)
        self._trial_plan: Dict[int, Dict[str, int]] = {}
        self._trial_params: Dict[int, Dict[str, object]] = {}
        self._episode = 0

    def infer_relative_search_space(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        """Return relative search space (empty for independent sampling)."""

        del study, trial
        return {}

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, object]:
        """No relative sampling in this template; delegate to independent calls."""

        del study, trial, search_space
        return {}

    def _completed_trials(self, study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
        return [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]

    def _build_trial_plan(self, study: optuna.study.Study) -> Dict[str, int]:
        completed = self._completed_trials(study)
        context = {
            "problem_size": max(1, len(study.best_params)) if completed else 8,
            "dataset_size": max(1, len(study.best_params)) if completed else 8,
            "iteration": len(study.trials),
            "current_pop_size": 64,
            "max_iterations": self.n_trials_hint,
        }
        raw_plan = self.trainer.plan(context)
        pop_size = int(np.clip(raw_plan.get("pop_size", 64), 16, 200))
        max_iter = int(np.clip(raw_plan.get("max_iter", 150), 50, 1000))
        elite_size = int(np.clip(raw_plan.get("elite_size", max(2, pop_size // 8)), 1, max(2, pop_size // 2)))
        return {
            "pop_size": pop_size,
            "max_iter": max_iter,
            "elite_size": elite_size,
        }

    def _sample_categorical(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        distribution: optuna.distributions.CategoricalDistribution,
    ) -> object:
        choices = list(distribution.choices)
        if not choices:
            raise ValueError(f"Categorical distribution for '{param_name}' has no choices")

        plan = self._trial_plan.setdefault(trial.number, self._build_trial_plan(study))
        trial_params = self._trial_params.setdefault(trial.number, {})
        completed = self._completed_trials(study)
        incumbent = min(completed, key=lambda t: float(t.value)) if completed else None
        incumbent_value = incumbent.params.get(param_name) if incumbent else None

        exploration_boost = 0.02 * max(0, self.trainer.memory.stagnation_count)
        effective_epsilon = float(np.clip(self.epsilon + exploration_boost, 0.05, 0.95))

        # Explore if there is no incumbent signal yet or if random exploration is selected.
        if incumbent_value not in choices or self.rng.random() < effective_epsilon:
            sampled = self.random_sampler.sample_independent(
                study=study,
                trial=trial,
                param_name=param_name,
                param_distribution=distribution,
            )
            trial_params[param_name] = sampled
            return sampled

        # Exploit around incumbent choice with mutation radius controlled by planner output.
        incumbent_idx = choices.index(incumbent_value)
        mutation_scale = max(1, int(round(plan["elite_size"] / 3)))
        window = max(1, min(len(choices) - 1, mutation_scale))
        low = max(0, incumbent_idx - window)
        high = min(len(choices) - 1, incumbent_idx + window)
        sampled_idx = self.rng.randint(low, high)
        sampled = choices[sampled_idx]
        trial_params[param_name] = sampled
        return sampled

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> object:
        """Sample one parameter using AdaptiveTrainer-guided exploration/exploitation."""

        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            return self._sample_categorical(study, trial, param_name, param_distribution)

        sampled = self.random_sampler.sample_independent(
            study=study,
            trial=trial,
            param_name=param_name,
            param_distribution=param_distribution,
        )
        self._trial_params.setdefault(trial.number, {})[param_name] = sampled
        return sampled

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[list[float]],
    ) -> None:
        """Feed trial outcome back to AdaptiveTrainer memory/critic loop."""

        del study
        plan = self._trial_plan.get(trial.number, {"pop_size": 64, "max_iter": 150, "elite_size": 8})
        trial_cost = float(trial.user_attrs.get("hagfish_trial_cost", 0.0))

        observed_error: Optional[float] = None
        if values:
            observed_error = float(values[0])
        elif trial.intermediate_values:
            observed_error = float(min(trial.intermediate_values.values()))

        if observed_error is not None and np.isfinite(observed_error):
            metric = 1.0 / (1.0 + max(0.0, observed_error))
            self._episode += 1
            self.trainer.observe(
                metric=metric,
                cost=trial_cost,
                params=plan,
                episode=self._episode,
            )

        self._trial_plan.pop(trial.number, None)
        self._trial_params.pop(trial.number, None)

    def reseed_rng(self) -> None:
        """Reseed internal RNG for reproducibility hooks."""

        self.rng.seed(self.rng.randint(0, 2**31 - 1))


def run_hagfish(
    seed: int,
    n_trials: int,
    bridge: HPOBenchObjectiveFunction,
    results_logger: UnifiedResultLogger,
    epsilon: float,
    pruner_name: str,
    alpha: float,
) -> None:
    """Run one Hagfish experiment with stepped fidelities."""

    sampler = HagfishOptunaSampler(
        epsilon=epsilon,
        alpha=alpha,
        n_trials_hint=n_trials,
        seed=seed,
    )
    lname = pruner_name.lower()
    if lname == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    elif lname == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, int(bridge.num_fidelity_steps)),
            reduction_factor=3,
        )
    elif lname in {"none", "nop"}:
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(
            f"Unknown pruner '{pruner_name}'. Choose one of: sha, hyperband, none"
        )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"hagfish_seed_{seed}",
    )

    schedule = bridge.fidelity_schedule()

    def objective(trial: optuna.trial.Trial) -> float:
        config = bridge.optuna_sample_from_trial(trial)
        best_error = float("inf")
        trial_cost = 0.0

        for step_idx, budget in enumerate(schedule):
            result = bridge.evaluate(config=config, fidelity=budget, seed=seed)
            best_error = min(best_error, result.validation_error)
            trial_cost += float(result.simulated_cost)

            results_logger.log_observation(
                algorithm="hagfish",
                seed=seed,
                incremental_simulated_cost=result.simulated_cost,
                validation_error=best_error,
            )

            trial.report(best_error, step=step_idx + 1)
            if trial.should_prune():
                trial.set_user_attr("hagfish_trial_cost", trial_cost)
                raise optuna.TrialPruned()

        trial.set_user_attr("hagfish_trial_cost", trial_cost)

        return best_error

    results_logger.reset_stream(algorithm="hagfish", seed=seed)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Hagfish adaptive algorithm on simple-hpo-bench dataset."
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument(
        "--alpha",
        type=float,
        default=5e-4,
        help="Cost-sensitivity used by AdaptiveTrainer reward shaping.",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="sha",
        choices=["sha", "hyperband", "none"],
        help="Optuna pruner used by Hagfish trials for fair cost-aware comparison.",
    )
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
    """Entrypoint for Hagfish runner."""

    args = parse_args()
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

    for seed in args.seeds:
        run_hagfish(
            seed=seed,
            n_trials=args.n_trials,
            bridge=bridge,
            results_logger=results_logger,
            epsilon=args.epsilon,
            pruner_name=args.pruner,
            alpha=args.alpha,
        )

    print(f"Saved standardized results to: {results_logger.output_csv}")


if __name__ == "__main__":
    main()
