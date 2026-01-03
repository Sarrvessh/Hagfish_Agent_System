from typing import Literal

from .memory import AgentMemory


class CriticAgent:
    """Performance critic that evaluates validation metric improvement.

    The critic compares previous and current validation metrics and emits a
    simple outcome label used as feedback to the planner:
      - 'improved' : validation metric increased (training budget was effective)
      - 'stagnated': no improvement observed; planner should consider escalation

    The critic also updates `memory.stagnation_count`, which signals prolonged
    lack of improvement and triggers resource escalation in the planner.
    """

    def assess(self, previous_best: float, current_best: float, memory: AgentMemory) -> Literal["improved", "stagnated"]:
        """Assess improvement between previous and current validation metrics.

        Comparison uses a small numerical tolerance and treats larger values as
        better (higher validation accuracy indicates improvement).
        """
        # Small numerical tolerance
        tol = 1e-12
        # For validation metrics (higher is better) consider improvement when strictly larger
        if current_best > previous_best + tol:
            # Improvement -> reset stagnation counter (resource effective)
            memory.stagnation_count = 0
            return "improved"
        else:
            # No improvement -> increment stagnation (resource may be insufficient)
            memory.stagnation_count += 1
            return "stagnated"