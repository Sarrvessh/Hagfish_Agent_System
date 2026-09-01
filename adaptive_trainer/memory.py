from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np


@dataclass
class AgentMemory:
    """Hagfish-inspired memory with slime trail tracking and elite metrics.

    Bio-inspired mechanisms:
      - `slime_trails`: Set of (pop_size, max_iter) budget configs marked as "bad paths"
         that weak agents (poor performers) have traveled. Slime repels exploration.
      - `elite_path`: Best (pop_size, max_iter) config that elite agents favor.
      - `slime_intensity`: How aggressively to penalize slimed configs (0.0 to 1.0).
      - `stagnation_count`: Triggers elite slime burst to force escape from local optima.

    In the ML training budget framing:
      - `best_distance` stores the primary validation metric (higher is better).
      - Weak agents (bad metrics) deposit slime on their budget allocations.
      - Elite agents can trigger slime bursts to force population-wide exploration.
      - Episode history records which budgets led to improvements vs stagnation.

    Attributes
    ----------
    best_distance: float
        The best validation metric observed so far (higher is better).
    best_tour: np.ndarray
        Elite path (best budget configuration) for slime burst reference.
    stagnation_count: int
        How many consecutive episodes have shown no improvement.
    slime_trails: Set[Tuple[int, int]]
        Budget configurations (pop_size, max_iter) marked as "bad paths".
    slime_intensity: float
        Penalty multiplier for slimed configs (0.0 = no penalty, 1.0 = full repulsion).
    episode_history: List[Dict[str, Any]]
        Chronological record of episodes. Each entry contains keys:
        - episode: int
        - params: dict (budget config)
        - distance: float (validation metric)
        - elapsed_time: float
        - outcome: str ("improved", "saturated", or "stagnated")
        - is_slimed: bool (whether this config was penalized by slime)
    """
    # Core metrics
    best_distance: float = float("-inf")
    best_tour: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    stagnation_count: int = 0
    
    # Hagfish slime mechanics
    slime_trails: Set[Tuple[int, int]] = field(default_factory=set)
    elite_path: Tuple[int, int] = (64, 150)  # (pop_size, max_iter)
    slime_intensity: float = 0.3  # Strength of slime repulsion (0-1)
    
    episode_history: List[Dict[str, Any]] = field(default_factory=list)

    def last_outcome(self) -> str:
        """Return the outcome label of the most recent episode, or empty string."""
        if not self.episode_history:
            return ""
        return str(self.episode_history[-1].get("outcome", ""))

    def get_efficiency_trend(self, n: int = 3) -> float:
        """Compute accuracy_gain / cost_spent over the last `n` episodes.

        The method computes the improvement in validation metric (distance)
        over the window and divides it by the total resource cost observed in
        that window. This is a simple proxy for the marginal efficiency of
        recent budget allocations. If insufficient data exists or costs are
        zero, the method returns 0.0.

        Returns
        -------
        float
            Ratio of accuracy_gain to total cost over the last `n` episodes.
        """
        if len(self.episode_history) < 2:
            return 0.0

        window = self.episode_history[-n:]
        # Require at least two points to measure gain
        if len(window) < 2:
            return 0.0

        start_distance = float(window[0].get("distance", 0.0))
        end_distance = float(window[-1].get("distance", 0.0))
        accuracy_gain = max(0.0, end_distance - start_distance)

        total_cost = sum([float(r.get("resource_cost", 0.0)) for r in window])
        if total_cost <= 0:
            return 0.0

        return float(accuracy_gain) / float(total_cost)

    def get_last_reward_slope(self) -> float:
        """Return the slope of the reward between the last two episodes.

        This is computed as (last_reward - previous_reward). If fewer than two
        episodes exist, returns 0.0. The planner can use a negative slope to
        avoid escalating resources when reward is decreasing.
        """
        if len(self.episode_history) < 2:
            return 0.0
        last = self.episode_history[-1].get("reward", None)
        prev = self.episode_history[-2].get("reward", None)
        if last is None or prev is None:
            return 0.0
        return float(last) - float(prev)
    # ===================== Hagfish Slime Mechanics =====================

    def deposit_slime(self, params: Dict[str, int]) -> None:
        """Weak agents deposit slime on bad budget paths.
        
        When a budget allocation results in stagnation or saturation,
        mark it as a "bad path" so other agents avoid it.
        """
        key = (int(params.get("pop_size", 0)), int(params.get("max_iter", 0)))
        self.slime_trails.add(key)

    def get_slime_penalty(self, params: Dict[str, int]) -> float:
        """Compute slime repulsion penalty for a budget config (0.0 to 1.0).
        
        Agents moving on slimed paths suffer reduced effective distance.
        The penalty scales with slime_intensity and presence in slime_trails.
        
        Returns
        -------
        float
            Penalty multiplier (0.0 = full repulsion, 1.0 = no penalty)
        """
        key = (int(params.get("pop_size", 0)), int(params.get("max_iter", 0)))
        if key in self.slime_trails:
            return 1.0 - self.slime_intensity
        return 1.0

    def elite_slime_burst(self, problem_size: int) -> Dict[str, int]:
        """Elite agents release slime burst to force population escape.
        
        When stagnation is severe (stagnation_count >= 3), the elite agent
        "explodes" with slime, creating a repulsive field that forces the
        population away from the elite path. This drives exploration and
        breaks local optima.
        
        Returns
        -------
        Dict[str, int]
            Burst configuration: scaled elite path + random perturbation
        """
        elite_pop, elite_iter = self.elite_path
        
        # Explosive scaling: multiply by random factor to force exploration
        burst_pop = int(elite_pop * np.random.uniform(1.5, 2.5))
        burst_iter = int(elite_iter * np.random.uniform(1.5, 2.5))
        
        # Clamp to reasonable bounds
        burst_pop = min(200, max(16, burst_pop))
        burst_iter = min(1000, max(50, burst_iter))
        
        return {"pop_size": burst_pop, "max_iter": burst_iter, "elite_size": max(2, burst_pop // 8)}

    def update_elite_path(self, params: Dict[str, int]) -> None:
        """Update the elite path when a new best metric is found.
        
        The elite path becomes the reference point for slime bursts.
        """
        self.elite_path = (int(params.get("pop_size", 64)), int(params.get("max_iter", 150)))

    def record_episode(self, episode: int, params: Dict[str, Any], distance: float,
                       tour: np.ndarray, elapsed_time: float, outcome: str,
                       reward: float = None, cost: float = None,
                       use_memory: bool = True, use_elite: bool = True) -> None:
        """Record a single episode and update best if necessary.

        Parameters
        ----------
        reward: float, optional
            Optional ML reward signal (higher is better). If None, omitted.
        cost: int, optional
            Optional resource cost (workers * budget). If None, omitted.

        This method updates slime trails based on outcome:
          - 'stagnated' / 'saturated' outcomes deposit slime on the config
          - 'improved' outcomes update the elite path
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
            record["resource_cost"] = float(cost)

        self.episode_history.append(record)

        # Update best if this episode produced a new best metric (higher is better)
        if distance > self.best_distance:
            self.best_distance = float(distance)
            self.best_tour = np.array(tour, dtype=int)
            # Elite path updates to the winning configuration
            if use_elite:
                self.update_elite_path(params)

        # Hagfish slime mechanics: weak agents deposit slime on bad paths
        if use_memory and outcome in ["stagnated", "saturated"]:
            self.deposit_slime(params)
