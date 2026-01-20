"""Adaptive Trainer package — bio-inspired Hagfish slime defense optimizer.

Bio-Inspired Framework
======================
This package implements a machine learning training budget optimizer inspired
by the defensive mechanisms of Atlantic hagfish (Myxine glutinosa). The hagfish
releases defensive slime when threatened, clogging predators' gills and forcing
them to escape. We apply this behavior to optimize ML training:

  - Hagfish school → Population of training budget configurations
  - Weak agents → Poor-performing budget allocations  
  - Slime trails → Pheromones marking ineffective budget paths
  - Elite slime burst → Forced exploration when trapped in local optima
  - Foraging behavior → Adaptive search for optimal accuracy-cost tradeoff

Key Components
==============

**AdaptiveTrainer** (Main API)
  High-level interface for adaptive training budget allocation:
    - plan(context) → budget configuration
    - observe(metric, cost) → feedback and slime deposition
  
**PlannerAgent** (Hagfish Swarming)
  Population-based planner using slime defense mechanics:
    - Maintains population of budget agents
    - Respects slime trails (avoids bad paths)
    - Elite slime burst escape mechanism for stagnation
    - Pheromone decay over time (forgetting)

**CriticAgent** (Sensory System)
  Evaluates budget effectiveness and guides slime deposition:
    - Assesses improvement ('improved', 'saturated', 'stagnated')
    - Triggers stagnation counter for elite burst
  
**AgentMemory** (Collective Memory)
  Shared state and slime mechanics:
    - Tracks episode history
    - Manages slime trails and elite path
    - Implements slime penalties and burst logic

Usage Example
=============

    from adaptive_trainer import AdaptiveTrainer
    
    trainer = AdaptiveTrainer(alpha=1e-4)  # Cost-sensitivity
    
    # Get initial budget
    budget = trainer.plan({"dataset_size": 1000})
    # → {"pop_size": 64, "max_iter": 150, "elite_size": 8}
    
    # Train your model with this budget...
    # Then report back the result
    trainer.observe(metric=0.85, cost=640)
    
    # Next iteration automatically uses slime mechanics
    # to avoid bad paths and escape stagnation
    budget = trainer.plan({"dataset_size": 1000})

Correct import path for the primary adapter is::

    from adaptive_trainer import AdaptiveTrainer

This module exposes `AdaptiveTrainer` and a package `__version__`.
"""

from .optimizer import AdaptiveTrainer

__version__ = "0.2.0"

__all__ = ["AdaptiveTrainer", "__version__"]
