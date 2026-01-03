import unittest
import numpy as np

from solver_agent import SolverAgent


def make_grid_distance():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d
    return D


class TestSolverAgent(unittest.TestCase):
    def test_deterministic_runs(self):
        D = make_grid_distance()
        agent = SolverAgent(D)

        r1 = agent.run(pop_size=10, max_iter=5, elite_size=1, random_seed=1)
        r2 = agent.run(pop_size=10, max_iter=5, elite_size=1, random_seed=1)

        # Deterministic runs should produce the same best metric and params
        self.assertAlmostEqual(r1["best_metric"], r2["best_metric"], places=12)
        self.assertEqual(r1["params"], r2["params"])
        self.assertIn("training_time", r1)
        # History contains validation metrics per epoch and best_metric is at least as large
        self.assertIn("history", r1)
        self.assertTrue(min(r1["history"]) <= r1["best_metric"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
