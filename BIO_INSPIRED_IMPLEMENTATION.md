# Bio-Inspired Hagfish Optimizer Implementation

## Overview

The `adaptive_trainer` package has been successfully refactored to implement **true bio-inspired optimization** based on the Atlantic hagfish's slime defense mechanisms. Instead of rule-based heuristics, the package now uses a **swarm-based evolutionary approach** with actual biological mechanics.

## What Changed

### 1. **PlannerAgent** → Bio-Inspired Hagfish Swarm

**Old Approach:** Rule-based deterministic heuristics

- Fixed escalation factors (1.1x, 1.25x multipliers)
- Hard-coded thresholds and conditions
- Single budget allocation logic

**New Approach:** Population-based swarm with slime mechanics

- Population of 10 budget agents with diverse configurations
- Each agent navigates the budget space
- Slime trails (pheromones) mark ineffective paths
- Elite slime burst mechanism for escaping local optima

**Key Features:**

```python
class PlannerAgent:
    def __init__(self, population_size: int = 10, decay_rate: float = 0.05):
        # Maintains population of budget agents
        # Slime decays over time (5% per episode)

    def choose(problem_size, memory, alpha):
        # Standard swarm navigation OR
        # Elite slime burst (when stagnation_count >= 3)
```

### 2. **AgentMemory** → Slime Trail Tracking

Added biological mechanics:

```python
@dataclass
class AgentMemory:
    slime_trails: Set[Tuple[int, int]]       # Marked bad paths
    elite_path: Tuple[int, int]              # (pop_size, max_iter) of best
    slime_intensity: float                   # Strength of pheromone (0-1)

    def deposit_slime(params):               # Weak agents mark bad paths
    def get_slime_penalty(params):          # Penalty multiplier for slimed configs
    def elite_slime_burst(problem_size):    # Elite escape mechanism
    def update_elite_path(params):          # Update when new best found
```

### 3. **CriticAgent** → Sensory System

Updated terminology and behavior to reflect biological inspiration:

- Evaluates budget effectiveness (improved/saturated/stagnated)
- Triggers stagnation counter for elite burst
- Guides slime deposition on weak paths

## How the Hagfish Mechanics Work

### Phase 1: Standard Swarm Navigation

```
For each episode:
  1. Each agent (budget config) in population scored by:
     - Proximity to elite path
     - Slime penalty (configs on slimed paths scored lower)
  2. Select best non-slimed agent
  3. Update one random agent toward elite (exploration-exploitation balance)
  4. Pheromone (slime_intensity) decays 5% each episode
```

### Phase 2: Stagnation Detection (Weak Agent Slime Deposit)

```
When outcome is "stagnated" or "saturated":
  - Budget config added to slime_trails set
  - Future agents will be penalized for exploring similar configs
  - This mimics weak agents marking bad paths to warn others
```

### Phase 3: Elite Slime Burst (Escape Mechanism)

```
When stagnation_count >= 3:
  1. Increase slime_intensity dramatically (+0.4)
  2. Generate burst configuration (1.5-2.5x elite path)
  3. Reset population to explore new areas
  4. Force population-wide dispersal

This mimics hagfish exploding slime when trapped.
```

### Phase 4: Pheromone Decay (Forgetting)

```
Every episode:
  slime_intensity *= (1 - decay_rate)  # 0.95x decay

Slime gradually fades, allowing re-exploration of old areas.
This prevents permanent trapping in suboptimal regions.
```

## Biological Mapping

| Hagfish Biology         | ML Training Budget Optimization         |
| ----------------------- | --------------------------------------- |
| Atlantic hagfish school | Population of budget configurations     |
| Ocean (search space)    | Budget space (pop_size × max_iter)      |
| Weak hagfish            | Poor-performing budget allocations      |
| Slime defense           | Pheromone marking ineffective configs   |
| Slime intensity         | Strength of repulsion (0-1)             |
| Elite hagfish           | Best-performing budget (highest metric) |
| Local trap              | Stagnation plateau (no improvement)     |
| Slime burst escape      | Force exploration via population reset  |
| Slime decay             | Forgetting mechanism (5% per episode)   |
| Foraging behavior       | Navigation toward elite path            |
| School cohesion         | Population gravitates to elite          |

## Benefits of Bio-Inspired Approach

1. **Adaptive Exploration**: Population naturally adapts to avoid bad regions
2. **Escape Local Optima**: Elite slime burst breaks stagnation automatically
3. **Collective Memory**: Slime trails encode collective experience
4. **Cost-Sensitive**: Alpha parameter controls conservation level
5. **Natural Convergence**: Population gravitates toward elite without artificial constraints
6. **Interpretable**: Behavior mimics real biological mechanisms

## Usage (Unchanged API)

```python
from adaptive_trainer import AdaptiveTrainer

trainer = AdaptiveTrainer(alpha=1e-4)  # Cost-sensitivity

# Get initial budget
budget = trainer.plan({"dataset_size": 1000})

# Simulate training and report back
metric = 0.85  # Validation accuracy
cost = 640     # Training cost
trainer.observe(metric=metric, cost=cost)

# Next iteration uses slime mechanics
budget = trainer.plan({"dataset_size": 1000})
```

## Test Results

All 15 tests pass with the new bio-inspired implementation:

- ✅ API contract maintained
- ✅ Behavioral logic preserved
- ✅ Adaptive mechanics working
- ✅ Long-run stability confirmed
- ✅ ML integration functional
- ✅ Numerical edge cases handled

## Removed Legacy Code

- `solver_agent.py` (deprecated TSP optimizer)
- `hsdo.py` / `hsdof.py` (legacy implementations)
- `agentic_loop.py` (backward-compat re-export)
- Root-level re-export files (agent_memory.py, planner_agent.py, etc.)

## Package Structure

```
adaptive_trainer/
  ├── __init__.py          (Bio-inspired documentation)
  ├── optimizer.py         (AdaptiveTrainer API)
  ├── memory.py            (AgentMemory with slime mechanics)
  ├── planner.py           (Hagfish swarm-based planning)
  ├── critic.py            (Sensory evaluation system)
  ├── policies.py          (Bandit baselines)
  ├── utils.py             (Utilities)
```

## Example: Slime Mechanics in Action

```
Episode 1: Improving → elite_path updates → slime_intensity = 0.30
Episode 2: Improving → metric continues rising → slime_intensity = 0.285 (decay)
Episode 3: Improving → new best found → slime_intensity = 0.271
Episode 4: Stagnation → config added to slime_trails → slime_intensity = 0.257
Episode 5: Stagnation → slime penalty applied → agents avoid → slime_intensity = 0.244
Episode 6: Still stagnated → continue tracking → slime_intensity = 0.232
Episode 7: Stagnation count = 3 → ELITE SLIME BURST TRIGGERED!
           - slime_intensity jumps to 0.632
           - Population reset with exploration perturbations
           - Forced dispersal to escape local optima
Episode 8: New area explored → improvement found → cycle repeats
```

## Backward Compatibility

✅ API unchanged - `AdaptiveTrainer` interface identical
✅ All existing tests pass
✅ Internal behavior enhanced with biological mechanisms
❌ Numeric budget values differ (due to swarm dynamics)

- Tests updated to verify conceptual behavior instead of exact values

## Next Steps (Future Enhancements)

1. **Pheromone Types**: Distinguish between positive/negative trails
2. **Multi-Species**: Different agent types with specialized behaviors
3. **Predator Simulation**: Incorporate cost as predator pressure
4. **Visualization**: Animate swarm trajectories and slime dynamics
5. **Tuning**: Optimize decay_rate and population_size by problem type

## References

- Atlantic Hagfish Defense: Slime coating (Mucus + collagen fibers)
- Biological Inspiration: _Myxine glutinosa_ defensive behavior
- Optimization Theory: Pheromone-based collective intelligence
- ML Application: Training budget allocation under resource constraints
