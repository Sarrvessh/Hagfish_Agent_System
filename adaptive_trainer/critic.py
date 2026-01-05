from typing import Literal

from .memory import AgentMemory


class CriticAgent:
    """Performance critic that evaluates validation metric improvement.

    The critic compares previous and current validation metrics and emits a
    small set of outcome labels used as feedback to the planner:
      - 'improved' : validation metric increased (training budget was effective)
      - 'saturated' : tiny positive improvement (we're spending more but gaining almost nothing)
      - 'stagnated': no improvement observed; planner should consider escalation

    The critic also updates `memory.stagnation_count`, which signals prolonged
    lack of meaningful improvement and triggers resource adaptations in the planner.

    The goal is to help the planner find the "knee of the curve" where accuracy
    stops improving significantly while costs keep growing.
    """

    def assess(self, previous_best: float, current_best: float, memory: AgentMemory) -> Literal["improved", "stagnated", "saturated"]:
        """Assess improvement between previous and current validation metrics.

        The method distinguishes three outcomes:
        - 'improved' when the metric increases beyond a tiny numerical tolerance
        - 'saturated' when the metric increases only marginally (small but > 0)
        - 'stagnated' when there is no improvement or the metric decreases

        The `saturated` label is intended to signal that more resources are
        being spent with practically no gain and the planner should back off.
        """
        # New tolerances: be harder to please to avoid tiny gains that cost a lot
        # `tol` now represents the minimal relative improvement considered worthwhile
        tol = 0.005  # 0.5% relative improvement required to be considered a true 'improved'
        # Very small improvements considered noise/stagnation threshold
        tiny_improvement = 1e-4

        # For validation metrics (higher is better)
        improvement = current_best - previous_best
        if improvement > tol:
            # Clearly improved -> reset stagnation counter
            memory.stagnation_count = 0
            return "improved"

        # Small but non-negligible improvement -> saturated (not worth the cost)
        if tiny_improvement < improvement <= tol:
            memory.stagnation_count += 1
            return "saturated"

        # Otherwise no meaningful improvement
        memory.stagnation_count += 1
        return "stagnated"