import time
from typing import Dict, Optional

import numpy as np

from .memory import AgentMemory
from .critic import CriticAgent
from .planner import PlannerAgent


class AgenticLoop:
    """Episodic ML experiment loop for adaptive training budget optimization.

    This orchestrator runs an episodic feedback loop over training budget
    allocations:
      PlannerAgent -> (training budget) -> SolverAgent (trains model) -> CriticAgent -> AgentMemory

    Each episode is one training job where the planner selects a budget
    (batch size, epochs, reserved capacity); the solver (trainer) consumes that
    budget and reports validation performance. The critic compares validation
    metrics and the memory records outcomes to enable bandit-style analysis.

    The loop is alpha-aware: it uses the alpha parameter to compute rewards and
    to pass cost-sensitivity into the planner so that decisions prioritize the
    overall Reward = metric - alpha * cost instead of pure accuracy.
    """

    def __init__(self, dist_matrix: np.ndarray, alpha: float = 0.0005):
        self.D = dist_matrix
        self.n = dist_matrix.shape[0]
        self.memory = AgentMemory()
        self.planner = PlannerAgent()
        self.critic = CriticAgent()
        self.alpha = float(alpha)
        # SolverAgent is part of the top-level code and is instantiated
        # at runtime by the loop to preserve backward compatibility with
        # existing experiment code.

    def run(self, episodes: int = 5, base_seed: int = 42, verbose: bool = True) -> Dict[str, Optional[object]]:
        """Run the agentic loop for a number of episodes.

        Returns a dictionary with final best result and memory.
        """
        from solver_agent import SolverAgent  # local import to avoid package dependency

        for ep in range(1, episodes + 1):
            # Planner returns a training budget: batch (pop_size), epochs (max_iter), reserved (elite_size)
            # Maintain backward compatibility with user-supplied planner functions that
            # may not accept the new `alpha` parameter by falling back to the
            # legacy two-argument call if needed.
            try:
                params = self.planner.choose(self.n, self.memory, alpha=self.alpha)
            except TypeError:
                params = self.planner.choose(self.n, self.memory)

            # Deterministic seed schedule
            seed = int(base_seed) + ep

            if verbose:
                # Human-readable label for the chosen training budget
                print(f"Episode {ep:2d}: Planner -> training_budget={params}, seed={seed}")

            start = time.time()
            # SolverAgent consumes allocated resources: pop_size = batch_size, max_iter = n_epochs
            res = SolverAgent(self.D).run(pop_size=params["pop_size"],
                                          max_iter=params["max_iter"],
                                          elite_size=params["elite_size"],
                                          random_seed=seed)
            elapsed = time.time() - start

            # Compute explicit training budget cost for this workload (simple proxy)
            resource_cost = int(params["pop_size"]) * int(params["max_iter"])  # batch * epochs

            # current metric: prefer 'best_metric' (ML trainer) otherwise fall back to 'best_distance'
            current_metric = res.get("best_metric", res.get("best_distance"))

            previous_best = self.memory.best_distance
            current_best = current_metric

            outcome = self.critic.assess(previous_best, current_best, self.memory)
            # Compute ML reward and record episode (reward = metric - alpha * cost)
            reward = float(current_metric) - self.alpha * resource_cost

            # Memory records the episode and updates canonical best; include reward and cost
            # Keep storing metric in 'distance' field for backward compatibility
            self.memory.record_episode(ep, res["params"], current_metric, res.get("best_tour", []), elapsed, outcome, reward=reward, cost=resource_cost)

            if verbose:
                print(
                    f"  -> Outcome: {outcome.upper():9s} | Episode validation_metric: {current_metric:.4f} | Global best: {self.memory.best_distance:.4f} | Time: {elapsed:.2f}s | Training budget: batch={params['pop_size']}, epochs={params['max_iter']}, reserved={params['elite_size']} | Cost={resource_cost} | Reward={reward:.4f}"
                )

        if verbose:
            print("\nFinal Best Validation Metric:", self.memory.best_distance)
        # Keep original return keys for backward compatibility while the values
        # reflect ML metrics and model info kept in memory
        return {"best_distance": self.memory.best_distance, "best_tour": self.memory.best_tour, "memory": self.memory}


class AdaptiveTrainer:
    """High-level wrapper exposing a simple API for external use.

    This class provides a compact interface for integrating the existing rule-
    based planner, critic, and memory into other systems. It delegates to the
    PlannerAgent, CriticAgent and AgentMemory and preserves existing logic.

    Methods
    -------
    plan(context) -> dict
        Produce a training budget given a context dict. Expected key: 'dataset_size'.

    observe(metric, cost)
        Notify the adapter of an observed validation metric and cost; updates
        memory and critic state accordingly.
    """

    def __init__(self, alpha: float = 1e-4):
        self.memory = AgentMemory()
        self.planner = PlannerAgent()
        self.critic = CriticAgent()
        self.alpha = float(alpha)

    def plan(self, context: Dict) -> Dict:
        """Return a training budget given a context dictionary.

        Context should contain at least 'dataset_size' (int). If missing,
        a default size of 40 is used. The planner is passed the `alpha` value
        so that planning decisions can account for cost-sensitivity.
        """
        size = int(context.get("dataset_size", context.get("problem_size", 40)))
        return self.planner.choose(size, self.memory, alpha=self.alpha)
    def observe(self, metric: float, cost: float, params: Dict = None, episode: int = None, elapsed_time: float = 0.0):
        """Record an observed validation metric and resource cost.

        This method updates the critic and memory to keep internal state in
        sync with observed outcomes. It does not modify the core reward logic.
        """
        previous_best = self.memory.best_distance
        current_best = float(metric)
        outcome = self.critic.assess(previous_best, current_best, self.memory)
        reward = float(metric) - float(self.alpha) * float(cost)

        # Use 0 / placeholder values if params or episode are not given
        if params is None:
            params = {}
        if episode is None:
            episode = len(self.memory.episode_history) + 1

        self.memory.record_episode(episode, params, current_best, [], float(elapsed_time), outcome, reward=reward, cost=int(cost))
