from typing import Dict

from .memory import AgentMemory


class PlannerAgent:
    """Training budget planner (rule-based).

    This planner performs deterministic allocation of training budgets to the
    downstream trainer. Semantics (conceptual mapping):
      - `pop_size`  -> batch size (mini-batch samples)
      - `max_iter`  -> training epochs
      - `elite_size`-> reserved capacity / regularization

    The allocation adapts based on a feedback signal (`stagnation_count`) in
    `AgentMemory`: when validation performance stalls, the planner escalates the
    training budget (to encourage exploration); when an improvement is
    observed, the planner slightly reduces budget to consolidate gains. The
    interface remains deterministic and backward-compatible:
    `choose(problem_size, memory) -> dict`.
    """

    def __init__(self) -> None:
        # Minimal constructor; no external services or state required
        pass

    def choose(self, problem_size: int, memory: AgentMemory) -> Dict[str, int]:
        """Return solver parameters using deterministic heuristics."""
        return self._rule_choose(problem_size, memory)

    def _rule_choose(self, problem_size: int, memory: AgentMemory) -> Dict[str, int]:
        """Deterministic heuristic planner (baseline).

        Preserves previous behavior exactly:
        - base_pop = min(150, max(60, problem_size // 2))
        - base_iter = min(1000, max(300, problem_size * 4))
        - base_elite = min(12, max(2, problem_size // 10))

        - If memory.stagnation_count >= 3: escalate (pop*1.2, max_iter*1.25, elite+1)
        - If last outcome was 'improved': reduce effort slightly
        """
        base_pop = min(150, max(60, problem_size // 2))
        base_iter = min(1000, max(300, problem_size * 4))
        base_elite = min(12, max(2, problem_size // 10))

        pop = int(base_pop)
        maxi = int(base_iter)
        elite = int(base_elite)

        if memory.stagnation_count >= 3:
            pop = int(pop * 1.2)
            maxi = int(maxi * 1.25)
            elite = min(12, elite + 1)

        if memory.last_outcome() == "improved":
            pop = max(60, pop - 10)
            maxi = max(200, maxi - 50)

        return {"pop_size": pop, "max_iter": maxi, "elite_size": elite}
