from typing import Dict, Optional

from .memory import AgentMemory
from .critic import CriticAgent
from .planner import PlannerAgent


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

    def __init__(
        self,
        alpha: float = 1e-4,
        *,
        use_memory: bool = True,
        use_elite: bool = True,
        use_burst: bool = True,
        use_cost_control: bool = True,
        seed: Optional[int] = None,
    ):
        self.memory = AgentMemory()
        self.planner = PlannerAgent(
            use_memory=use_memory,
            use_elite=use_elite,
            use_burst=use_burst,
            seed=seed,
        )
        self.critic = CriticAgent()
        self.alpha = float(alpha)
        self.use_memory = bool(use_memory)
        self.use_elite = bool(use_elite)
        self.use_burst = bool(use_burst)
        self.use_cost_control = bool(use_cost_control)

    def plan(self, context: Dict) -> Dict:
        """Return a training budget given a context dictionary.

        Context should contain at least 'dataset_size' (int). If missing,
        a default size of 40 is used. The planner is passed the `alpha` value
        so that planning decisions can account for cost-sensitivity.
        """
        size = int(context.get("dataset_size", context.get("problem_size", 40)))
        effective_alpha = self.alpha if self.use_cost_control else 0.0
        return self.planner.choose(size, self.memory, alpha=effective_alpha)
    def observe(self, metric: float, cost: float, params: Dict = None, episode: int = None, elapsed_time: float = 0.0):
        """Record an observed validation metric and resource cost.

        This method updates the critic and memory to keep internal state in
        sync with observed outcomes. It does not modify the core reward logic.
        """
        previous_best = self.memory.best_distance
        current_best = float(metric)
        outcome = self.critic.assess(previous_best, current_best, self.memory)
        effective_alpha = self.alpha if self.use_cost_control else 0.0
        reward = float(metric) - float(effective_alpha) * float(cost)

        # Use 0 / placeholder values if params or episode are not given
        if params is None:
            params = {}
        if episode is None:
            episode = len(self.memory.episode_history) + 1

        self.memory.record_episode(
            episode,
            params,
            current_best,
            [],
            float(elapsed_time),
            outcome,
            reward=reward,
            cost=float(cost),
            use_memory=self.use_memory,
            use_elite=self.use_elite,
        )
