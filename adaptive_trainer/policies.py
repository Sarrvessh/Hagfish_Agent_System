from typing import Dict, List, Optional, Sequence, Tuple
import random


class FixedPolicy:
    """Bandit baseline: always returns the same training budget configuration.

    Interpreted fields:
      - pop_size -> batch_size
      - max_iter -> n_epochs
      - elite_size -> reserved capacity

    This policy is used as a simple baseline when comparing adaptive bandit
    strategies for training budget allocation.
    """

    def __init__(self, config: Dict[str, int]):
        self._config = dict(config)

    def choose(self) -> Dict[str, int]:
        return dict(self._config)

    # No update method — stateless


class RandomPolicy:
    """Uniformly samples a training budget configuration from arms."""

    def __init__(self, arms: Sequence[Dict[str, int]], seed: Optional[int] = None):
        self.arms = list(arms)
        self.random = random.Random(seed)

    def choose(self) -> Dict[str, int]:
        return dict(self.random.choice(self.arms))

    # No update method — stateless


class GreedyPolicy:
    """Keeps track of best_config (training budget) by observed reward and selects it.

    If a higher reward is observed via update(), best_config is replaced.
    """

    def __init__(self, initial_config: Dict[str, int]):
        self.best_config = dict(initial_config)
        self.best_reward: float = float("-inf")

    def choose(self) -> Dict[str, int]:
        return dict(self.best_config)

    def update(self, config: Dict[str, int], reward: float) -> None:
        if reward is None:
            return
        if reward > self.best_reward:
            self.best_reward = float(reward)
            self.best_config = dict(config)
