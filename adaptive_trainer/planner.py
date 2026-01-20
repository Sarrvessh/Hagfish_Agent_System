from typing import Dict
import numpy as np

from .memory import AgentMemory


class PlannerAgent:
    """Hagfish bio-inspired training budget planner.
    
    This planner maintains a population of "Hagfish agents" that explore
    the training budget allocation space. Each agent navigates through
    budget configurations while respecting slime trails (repulsive pheromones)
    deposited by weak agents that explored bad paths.
    
    Bio-inspired mechanisms:
      1. Population-based search: Multiple budget configurations explored in parallel
      2. Slime trails: Weak agents mark bad budget paths to guide population away
      3. Elite slime burst: When stagnated, elite agents "explode" slime to force escape
      4. Pheromone dynamics: Slime intensity decays over time (forgetting mechanism)
      5. Adaptive exploration: Balance between exploitation (best paths) and exploration
    
    The framework maps biological hagfish behavior to ML training budget allocation:
      - Hagfish schools -> Population of candidate budgets
      - Ocean/search space -> Budget configuration space (pop_size × max_iter)
      - Physical distance -> Training cost
      - Slime pheromone -> Penalty for explored but ineffective configs
    """

    def __init__(self, population_size: int = 10, decay_rate: float = 0.05) -> None:
        """Initialize the Hagfish planner.
        
        Parameters
        ----------
        population_size : int
            Number of budget agents in the population (default 10)
        decay_rate : float
            How fast slime pheromone decays (0-1, default 0.05 = 5% per episode)
        """
        self.population_size = int(population_size)
        self.decay_rate = float(decay_rate)
        self.population = []  # Will be initialized in choose()

    def choose(self, problem_size: int, memory: AgentMemory, alpha: float = 1e-4) -> Dict[str, int]:
        """Select a training budget using Hagfish swarming with slime defense.

        Parameters
        ----------
        problem_size : int
            Dataset size or problem complexity
        memory : AgentMemory
            Shared memory with slime trails and elite path
        alpha : float
            Cost-sensitivity parameter (higher = more conservative)
            
        Returns
        -------
        Dict[str, int]
            Selected budget {"pop_size": int, "max_iter": int, "elite_size": int}
        """
        # Initialize population on first call
        if not self.population:
            self._initialize_population(problem_size)
        
        # Decay slime intensity over time (forgetting mechanism)
        memory.slime_intensity = max(0.0, memory.slime_intensity * (1.0 - self.decay_rate))
        
        # Handle stagnation with elite slime burst
        if memory.stagnation_count >= 3:
            return self._elite_slime_burst(memory, problem_size, alpha)
        
        # Standard Hagfish swarming: select best non-slimed agent
        return self._swarm_choose(problem_size, memory, alpha)

    def _initialize_population(self, problem_size: int) -> None:
        """Initialize population of budget agents with diverse configurations."""
        base_pop = max(16, min(64, problem_size // 10))
        base_iter = max(50, min(150, problem_size // 5))
        
        # Create diverse population around base config
        for i in range(self.population_size):
            # Random variation in both pop_size and max_iter
            var_pop = base_pop * np.random.uniform(0.5, 2.0)
            var_iter = base_iter * np.random.uniform(0.5, 2.0)
            
            config = {
                "pop_size": int(max(16, min(200, var_pop))),
                "max_iter": int(max(50, min(1000, var_iter))),
                "elite_size": int(max(2, var_pop / 8))
            }
            self.population.append(config)

    def _swarm_choose(self, problem_size: int, memory: AgentMemory, alpha: float) -> Dict[str, int]:
        """Hagfish agents navigate away from slimed paths toward elite."""
        # Evaluate each agent in population, applying slime penalties
        scores = []
        for config in self.population:
            slime_penalty = memory.get_slime_penalty(config)
            # Score = proximity to elite path × (1 - slime penalty)
            # Elite path = (elite_pop, elite_iter)
            elite_pop, elite_iter = memory.elite_path
            
            dist_to_elite = (
                (config["pop_size"] - elite_pop)**2 + 
                (config["max_iter"] - elite_iter)**2
            ) ** 0.5
            
            # Closer to elite = higher score, slime reduces score
            score = (1.0 / (1.0 + dist_to_elite)) * slime_penalty
            scores.append(score)
        
        # Select best non-slimed agent
        best_idx = int(np.argmax(scores))
        best_agent = dict(self.population[best_idx])
        
        # Apply cost-sensitivity: alpha-aware adaptation
        if alpha > 1e-3:  # High cost-sensitivity: be more conservative
            best_agent["pop_size"] = max(16, int(best_agent["pop_size"] * 0.9))
            best_agent["max_iter"] = max(10, int(best_agent["max_iter"] * 0.9))
        
        # Update one random agent toward elite (exploitation-exploration balance)
        idx = np.random.randint(0, self.population_size)
        self.population[idx]["pop_size"] = int(
            0.7 * self.population[idx]["pop_size"] + 0.3 * memory.elite_path[0]
        )
        self.population[idx]["max_iter"] = int(
            0.7 * self.population[idx]["max_iter"] + 0.3 * memory.elite_path[1]
        )
        
        return best_agent

    def _elite_slime_burst(self, memory: AgentMemory, problem_size: int, alpha: float) -> Dict[str, int]:
        """Elite agents release slime burst to force population escape from stagnation.
        
        This mimics the hagfish's defensive behavior: when trapped (local optima),
        the elite agent "explodes" with slime, forcing the entire school to
        disperse and explore new areas.
        """
        # Increase slime intensity dramatically during burst
        memory.slime_intensity = min(1.0, memory.slime_intensity + 0.4)
        
        # Get elite burst configuration
        burst_config = memory.elite_slime_burst(problem_size)
        
        # Reset population to explore new areas (dispersion)
        self._initialize_population(problem_size)
        
        # Encourage large jumps in population
        for config in self.population:
            config["pop_size"] = int(config["pop_size"] * np.random.uniform(1.2, 1.8))
            config["max_iter"] = int(config["max_iter"] * np.random.uniform(1.2, 1.8))
        
        # Clamp to reasonable bounds
        for config in self.population:
            config["pop_size"] = max(16, min(200, config["pop_size"]))
            config["max_iter"] = max(50, min(1000, config["max_iter"]))
            config["elite_size"] = max(2, config["pop_size"] // 8)
        
        return burst_config
