from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class AgentMemory:
    """Stores run history and current best validation metric and feedback signals.

    In the ML training budget framing, the memory holds performance and
    efficiency indicators that guide future allocations:
      - `best_distance` stores the primary validation metric (higher is better).
      - `stagnation_count` is an efficiency signal: many stagnant episodes
         imply the current training budget is not yielding improvements and
         the planner should consider escalating resources.
      - `episode_history` stores past allocations and outcomes for auditability
         and for bandit-style analysis.

    Notes
    -----
    - For backward compatibility the field names `best_distance` and the
      history key `distance` are preserved, but they represent validation
      metrics (e.g., validation accuracy) in this ML-focused system.

    Attributes
    ----------
    best_distance: float
        The best validation metric observed so far (higher is better).
    best_tour: np.ndarray
        Placeholder for model info or metadata corresponding to the best run.
    stagnation_count: int
        How many consecutive episodes have shown no improvement.
    episode_history: List[Dict[str, Any]]
        Chronological record of episodes. Each entry contains keys:
        - episode: int
        - params: dict
        - distance: float (validation metric)
        - elapsed_time: float
        - outcome: str ("improved" or "stagnated")
    """
    # 'best_distance' stores the best observed metric (validation accuracy); initialize to -inf
    best_distance: float = float("-inf")
    best_tour: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    stagnation_count: int = 0
    episode_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_episode(self, episode: int, params: Dict[str, Any], distance: float,
                       tour: np.ndarray, elapsed_time: float, outcome: str,
                       reward: float = None, cost: int = None) -> None:
        """Record a single episode and update best if necessary.

        Parameters
        ----------
        reward: float, optional
            Optional ML reward signal (higher is better). If None, omitted.
        cost: int, optional
            Optional resource cost (workers * budget). If None, omitted.

        This method does not change the stagnation_count logic; the CriticAgent
        is responsible for updating stagnation_count. The memory keeps the
        canonical best_distance and best_tour and stores optional ML signals
        for later analysis.
        """
        record = {
            "episode": int(episode),
            "params": dict(params),
            "distance": float(distance),
            "elapsed_time": float(elapsed_time),
            "outcome": str(outcome),
        }

        # Conditionally include ML-specific fields to preserve backward compatibility
        if reward is not None:
            record["reward"] = float(reward)
        if cost is not None:
            record["resource_cost"] = int(cost)

        self.episode_history.append(record)

        # Update best if this episode produced a new best metric (higher is better)
        if distance > self.best_distance:
            self.best_distance = float(distance)
            self.best_tour = np.array(tour, dtype=int)

    def last_outcome(self) -> str:
        """Return the outcome label of the most recent episode, or empty string."""
        if not self.episode_history:
            return ""
        return str(self.episode_history[-1].get("outcome", ""))
