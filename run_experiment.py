"""
Adaptive Training Budget Optimization demo (Agentic bandit framework).

Top-level note: the supervised dataset (breast_cancer) is loaded inside
`SolverAgent` and the training/validation split is prepared there. This
script orchestrates episodic experiments and compares bandit baselines.

This script demonstrates an episodic adaptive resource allocation loop for
supervised ML training workloads. Each round represents a training job where
the planner allocates a training budget (batch size, epochs, reserved
capacity). The trainer executes the workload and returns validation
performance; the critic evaluates outcomes and the memory records performance
and resource usage to enable adaptive budget allocation.
"""

import argparse
import numpy as np

from adaptive_trainer.optimizer import AgenticLoop
from solver_agent import SolverAgent
from adaptive_trainer.policies import FixedPolicy, RandomPolicy, GreedyPolicy

# Bandit arms: predefined resource configurations
RESOURCE_ARMS = [
    {"pop_size": 60, "max_iter": 250, "elite_size": 2},
    {"pop_size": 60, "max_iter": 300, "elite_size": 2},
    {"pop_size": 72, "max_iter": 375, "elite_size": 3},
    {"pop_size": 80, "max_iter": 500, "elite_size": 3},
]


def build_placeholder_matrix(dataset_size: int, seed: int = 123) -> np.ndarray:
    """Builds a lightweight placeholder matrix used as a shape proxy.

    The active ML dataset is loaded inside `SolverAgent`; this function only
    creates a reproducible matrix whose shape is used by the AgenticLoop for
    experiment sizing and deterministic behavior.
    """
    rs = np.random.RandomState(seed)
    coords = rs.rand(dataset_size, 2) * 100.0
    D = np.zeros((dataset_size, dataset_size), dtype=float)
    for i in range(dataset_size):
        for j in range(i + 1, dataset_size):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d
    return D


def run_bandit(policy, solver_agent: SolverAgent, rounds: int = 3, base_seed: int = 42):
    """Run a simple bandit-style evaluation loop for a given policy.

    For each round:
      - policy.choose() returns a training budget configuration
      - solver_agent.run(...) executes the training job
      - training budget cost and reward are computed and returned
      - if policy supports .update(config, reward) it will be called

    Returns list of rewards observed.
    """
    rewards = []
    for ep in range(1, rounds + 1):
        params = policy.choose()
        seed = int(base_seed) + ep
        res = solver_agent.run(pop_size=params["pop_size"], max_iter=params["max_iter"], elite_size=params["elite_size"], random_seed=seed)
        resource_cost = int(params["pop_size"]) * int(params["max_iter"])  # batch_size * epochs
        alpha = 0.0001
        metric = res.get("best_metric", res.get("best_distance"))
        # Reward = metric - alpha * cost (higher metric is better)
        reward = float(metric) - alpha * resource_cost
        rewards.append(reward)
        # If policy provides update, call it
        if hasattr(policy, "update"):
            try:
                policy.update(params, reward)
            except Exception:
                pass
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Run adaptive ML training budget experiment")

    # New flags (preferred)
    parser.add_argument("--dataset-size", type=int, default=40, help="Dataset size proxy (ignored; SolverAgent loads real supervised dataset)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of agent rounds (episodes)")

    # Deprecated flags (kept for backward compatibility)
    parser.add_argument("--cities", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, help=argparse.SUPPRESS)

    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")

    args = parser.parse_args()

    # Backward compatibility: if old flags are provided use them unless new flags are set
    dataset_size = args.dataset_size if args.dataset_size is not None else (args.cities if args.cities is not None else 40)
    rounds = args.rounds if args.rounds is not None else (args.episodes if args.episodes is not None else 3)

    # The SolverAgent loads the supervised dataset internally; the placeholder
    # matrix below is only used to set the experiment size/shape for the loop.
    D = build_placeholder_matrix(dataset_size, seed=args.seed)

    loop = AgenticLoop(D)
    result = loop.run(episodes=rounds, base_seed=args.seed, verbose=True)

    print("\nEpisode history summary:")
    for rec in result["memory"].episode_history:
        cost = rec.get("resource_cost", "N/A")
        # `distance` key stores validation metric for backward compatibility
        print(
            f" Episode {rec['episode']:2d} | outcome: {rec['outcome']:9s} | validation_metric: {rec['distance']:.4f} | time: {rec['elapsed_time']:.2f}s | training_budget: {rec['params']} | cost: {cost} | reward: {rec.get('reward', 'N/A')}"
        )

    print("\nFinal result:")
    print(f"  Final Best Validation Metric: {result['best_distance']:.4f}")
    print(f"  Best model info length: {len(result['best_tour']) if len(result['best_tour']) else 'N/A'}")

    # ------------------------
    # Bandit baselines
    # ------------------------
    print("\nRunning bandit baselines...")

    solver = SolverAgent(D)
    episodes = args.episodes

    fixed = FixedPolicy(RESOURCE_ARMS[1])
    random_p = RandomPolicy(RESOURCE_ARMS, seed=args.seed)
    greedy = GreedyPolicy(RESOURCE_ARMS[0])

    r_fixed = run_bandit(fixed, solver, rounds=rounds, base_seed=args.seed)
    r_random = run_bandit(random_p, solver, rounds=rounds, base_seed=args.seed)
    r_greedy = run_bandit(greedy, solver, rounds=rounds, base_seed=args.seed)

    cum_fixed = sum(r_fixed)
    cum_random = sum(r_random)
    cum_greedy = sum(r_greedy)

    # Cumulative reward from adaptive AgenticLoop (if rewards recorded)
    adaptive_rewards = [rec.get("reward") for rec in loop.memory.episode_history if rec.get("reward") is not None]
    cum_adaptive = sum(adaptive_rewards) if adaptive_rewards else None

    print("\nBandit baseline cumulative rewards:")
    print(f" Fixed:  {cum_fixed:.4f}")
    print(f" Random: {cum_random:.4f}")
    print(f" Greedy: {cum_greedy:.4f}")
    print(f" Adaptive system (AgenticLoop): {cum_adaptive if cum_adaptive is not None else 'N/A'}")


if __name__ == "__main__":
    main()
