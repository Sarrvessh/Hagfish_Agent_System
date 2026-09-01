"""Baseline optimizer adapters used by the pathfinding benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import cma
import numpy as np
from mealpy import FloatVar
from mealpy.evolutionary_based import DE, GA
from mealpy.swarm_based import GWO, PSO
from scipy.optimize import dual_annealing


Objective = Callable[[np.ndarray], float]


def _bounds(bounds: Sequence[object], dim: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if lower.ndim == 0:
        lower = np.full(dim, float(lower))
    if upper.ndim == 0:
        upper = np.full(dim, float(upper))
    return lower, upper


@dataclass
class MealpyAdapter:
    algorithm: str
    pop_size: int = 40

    def optimize(
        self,
        objective_fn: Objective,
        bounds: Sequence[object],
        dim: int,
        max_iterations: int = 150,
    ) -> tuple[np.ndarray, float, list[float]]:
        lower, upper = _bounds(bounds, dim)
        problem = {
            "bounds": FloatVar(lb=lower, ub=upper, name="position"),
            "obj_func": objective_fn,
            "minmax": "min",
            "log_to": None,
        }
        models = {
            "PSO": PSO.OriginalPSO,
            "GWO": GWO.OriginalGWO,
            "GA": GA.BaseGA,
            "DE": DE.OriginalDE,
        }
        model = models[self.algorithm](epoch=max_iterations, pop_size=self.pop_size)
        best = model.solve(problem)
        convergence = [
            float(agent.target.fitness) for agent in model.history.list_global_best
        ]
        return np.asarray(best.solution), float(best.target.fitness), convergence


@dataclass
class CMAESAdapter:
    pop_size: int = 40

    def optimize(
        self,
        objective_fn: Objective,
        bounds: Sequence[object],
        dim: int,
        max_iterations: int = 150,
    ) -> tuple[np.ndarray, float, list[float]]:
        lower, upper = _bounds(bounds, dim)
        initial = (lower + upper) / 2.0
        sigma = max(float(np.mean(upper - lower)) / 4.0, 1e-8)
        result, strategy = cma.fmin2(
            objective_fn,
            initial,
            sigma,
            options={
                "bounds": [lower.tolist(), upper.tolist()],
                "maxiter": max_iterations,
                "popsize": self.pop_size,
                "verbose": -9,
            },
        )
        fitness = float(objective_fn(np.asarray(result)))
        convergence = [float(fitness)] * max(1, int(strategy.countiter))
        return np.asarray(result), fitness, convergence


@dataclass
class SimulatedAnnealingAdapter:
    pop_size: int = 40

    def optimize(
        self,
        objective_fn: Objective,
        bounds: Sequence[object],
        dim: int,
        max_iterations: int = 150,
    ) -> tuple[np.ndarray, float, list[float]]:
        lower, upper = _bounds(bounds, dim)
        result = dual_annealing(
            objective_fn,
            bounds=list(zip(lower, upper)),
            maxiter=max_iterations,
            no_local_search=True,
        )
        return np.asarray(result.x), float(result.fun), [float(result.fun)]


def get_available_baseline_optimizers(pop_size: int = 40) -> Dict[str, object]:
    """Return fresh optimizer adapters under the names used in paper outputs."""
    return {
        "PSO": MealpyAdapter("PSO", pop_size),
        "GWO": MealpyAdapter("GWO", pop_size),
        "GA": MealpyAdapter("GA", pop_size),
        "DE": MealpyAdapter("DE", pop_size),
        "CMAES": CMAESAdapter(pop_size),
        "SimulatedAnnealing": SimulatedAnnealingAdapter(pop_size),
    }
