from typing import Dict

from .memory import AgentMemory


class PlannerAgent:
    """Training budget planner (rule-based) with resource-efficiency heuristics.

    This planner allocates training budgets while explicitly accounting for
    the resource cost vs. improvement trade-off (reward = accuracy - alpha*cost).
    The planner is designed to help find the "knee of the curve" where added
    cost produces diminishing returns.

    New behaviors:
      - Accept an `alpha` parameter from the optimizer to compute a conservative
        adjustment when risk (cost-sensitivity) is high.
      - Replace fixed additive adjustments with percentage scaling.
      - Respond to a 'saturated' critic outcome by de-escalating budgets.
      - Implement a 'random jump' (for stagnation_count == 3) and a 'cool down'
        policy for prolonged stagnation to avoid runaway cost escalation.
    """

    def __init__(self) -> None:
        # Minimal constructor; no external services or state required
        pass

    def choose(self, problem_size: int, memory: AgentMemory, alpha: float = 1e-4) -> Dict[str, int]:
        """Return solver parameters using deterministic heuristics.

        Parameters
        ----------
        alpha : float
            Cost-sensitivity parameter; higher alpha makes the planner more
            conservative when escalating budgets.
        """
        return self._rule_choose(problem_size, memory, float(alpha))

    def _rule_choose(self, problem_size: int, memory: AgentMemory, alpha: float) -> Dict[str, int]:
        """Deterministic heuristic planner with percentage-based scaling.

        Overview:
          - Base sizes are computed from `problem_size`.
          - If recent outcome is 'saturated' -> aggressively de-escalate (we're
            spending with minimal gains).
          - If `stagnation_count == 3` -> perform a larger 'random jump' to
            escape local optima, but scale the jump by risk derived from alpha.
          - If `stagnation_count > 3` -> enter a 'cool down' phase to reduce
            budget and avoid unbounded cost growth.
          - If last outcome was 'improved' -> reduce effort by a percentage.
        """
        # New conservative budget floors so the trainer starts cheap and only
        # increases if necessary. Base settings intentionally close to a
        # 'Standard' fixed policy to avoid overspending early on.
        base_pop = min(64, max(16, problem_size // 20))
        base_iter = min(150, max(50, problem_size // 5))
        base_elite = min(12, max(2, problem_size // 10))

        pop = int(base_pop)
        maxi = int(base_iter)
        elite = int(base_elite)

        last = memory.last_outcome()

        # If the critic says 'saturated' we should back off (~20% cheaper)
        if last == "saturated":
            pop = max(16, int(pop * 0.8))
            maxi = max(10, int(maxi * 0.8))
            elite = max(2, elite - 1)
            # Enforce solver safety
            pop = min(pop, problem_size)
            maxi = max(maxi, 10)
            return {"pop_size": pop, "max_iter": maxi, "elite_size": elite}

        # If the recent reward slope is negative, stop escalating and cool down
        slope = 0.0
        if hasattr(memory, "get_last_reward_slope"):
            slope = memory.get_last_reward_slope()
        if slope < 0.0:
            pop = max(16, int(pop * 0.9))
            maxi = max(10, int(maxi * 0.9))
            pop = min(pop, problem_size)
            maxi = max(maxi, 10)
            return {"pop_size": pop, "max_iter": maxi, "elite_size": elite}

        # Handle stagnation with conservative alpha-aware escalation
        if memory.stagnation_count >= 3:
            # Use smaller escalation multipliers when alpha (cost-sensitivity) is high
            if alpha > 1e-5:
                factor_pop = 1.1
                factor_iter = 1.1
            else:
                factor_pop = 1.25
                factor_iter = 1.25

            if memory.stagnation_count == 3:
                pop = min(150, int(pop * factor_pop))
                maxi = min(1000, int(maxi * factor_iter))
                elite = min(12, elite + 2)
            else:
                # If still stagnating beyond 3, perform a controlled cool down
                pop = max(16, int(pop * 0.85))
                maxi = max(10, int(maxi * 0.85))
                elite = max(2, elite - 1)

        # If recent improvement, slightly reduce budgets (percentage reduction)
        if last == "improved":
            pop = max(16, int(pop * 0.9))
            maxi = max(10, int(maxi * 0.9))

        # Solver safety constraints
        pop = min(pop, problem_size)  # do not allocate batch size larger than problem size
        maxi = max(maxi, 10)  # ensure at least 10 epochs

        return {"pop_size": pop, "max_iter": maxi, "elite_size": elite}
