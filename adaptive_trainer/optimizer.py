import time
from typing import Dict, Optional

import numpy as np

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
