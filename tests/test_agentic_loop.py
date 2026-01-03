import unittest
import numpy as np

from agentic_loop import AgenticLoop


def make_grid_distance():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d
    return D


class TestAgenticLoop(unittest.TestCase):
    def test_loop_updates_memory(self):
        D = make_grid_distance()
        loop = AgenticLoop(D)
        # Use a small deterministic planner to avoid large pop_size that causes solver init errors on tiny problems
        loop.planner.choose = lambda problem_size, memory: {"pop_size": 8, "max_iter": 10, "elite_size": 2}
        res = loop.run(episodes=2, base_seed=7, verbose=False)

        self.assertEqual(len(res["memory"].episode_history), 2)
        for rec in res["memory"].episode_history:
            self.assertIn(rec["outcome"], ("improved", "stagnated"))
            self.assertTrue(rec["distance"] >= 0.0)
        self.assertTrue(res["best_distance"] < float("inf"))


if __name__ == "__main__":
    unittest.main()
