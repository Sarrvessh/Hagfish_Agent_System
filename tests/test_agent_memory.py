import unittest
import numpy as np

from agent_memory import AgentMemory


class TestAgentMemory(unittest.TestCase):
    def test_record_and_update_best(self):
        m = AgentMemory()
        # Initialized to -inf for metric maximization
        self.assertEqual(m.best_distance, float("-inf"))
        tour = np.array([0, 1, 2], dtype=int)
        # Use a metric value (e.g., accuracy) where higher is better
        m.record_episode(episode=1, params={"p": 1}, distance=0.5, tour=tour, elapsed_time=0.1, outcome="improved")
        self.assertEqual(len(m.episode_history), 1)
        self.assertAlmostEqual(m.best_distance, 0.5)
        np.testing.assert_array_equal(m.best_tour, tour)


if __name__ == "__main__":
    unittest.main()
