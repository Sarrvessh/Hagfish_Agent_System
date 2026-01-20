from typing import Literal

from .memory import AgentMemory


class CriticAgent:
    """Hagfish critic: evaluates training budget effectiveness and deposits slime.
    
    The critic acts like the "sensory system" of the hagfish school. When agents
    (budget configurations) return from exploring the ocean (training), the critic
    evaluates whether the journey was worthwhile:
    
      - 'improved': The budget led to better validation metrics (good path - no slime)
      - 'saturated': Tiny gains for high cost (weak path - deposit slime)
      - 'stagnated': No improvement (bad path - deposit slime heavily)
    
    The critic also updates the stagnation counter, which triggers elite slime
    bursts when the population is trapped in local optima.
    """

    def assess(self, previous_best: float, current_best: float, memory: AgentMemory) -> Literal["improved", "stagnated", "saturated"]:
        """Evaluate budget effectiveness and update memory slime mechanics.

        The critic distinguishes three outcomes:
          - 'improved': Metric increases beyond meaningful threshold (good path)
          - 'saturated': Marginal gain with high cost (weak path - mark with slime)
          - 'stagnated': No improvement or regression (bad path - mark with slime)

        Parameters
        ----------
        previous_best : float
            Best validation metric from prior episodes
        current_best : float
            Current episode's validation metric
        memory : AgentMemory
            Shared memory; slime will be deposited on weak outcomes

        Returns
        -------
        Literal["improved", "stagnated", "saturated"]
            Outcome label for the planner and memory
        """
        # Thresholds for meaningful improvement
        tol = 0.005  # 0.5% relative improvement required to be considered 'improved'
        tiny_improvement = 1e-4  # Very small gains are treated as stagnation

        # For validation metrics (higher is better)
        improvement = current_best - previous_best
        
        if improvement > tol:
            # Clear improvement - good path, no slime
            memory.stagnation_count = 0
            return "improved"

        # Small but non-negligible improvement - mark as saturated (not worth the cost)
        if tiny_improvement < improvement <= tol:
            memory.stagnation_count += 1
            # Don't deposit slime yet - need to see pattern
            return "saturated"

        # Otherwise no meaningful improvement - bad path, mark with slime
        memory.stagnation_count += 1
        return "stagnated"
