import unittest

from adaptive_trainer.memory import AgentMemory


class TestAgentMemory(unittest.TestCase):
    def test_record_and_update_best(self):
        mem = AgentMemory()
        self.assertEqual(mem.best_distance, float("-inf"))

        mem.record_episode(1, {"pop_size": 10}, 0.50, [], 0.1, "improved", reward=0.45, cost=100)
        self.assertAlmostEqual(mem.best_distance, 0.50)
        self.assertEqual(len(mem.episode_history), 1)
        rec = mem.episode_history[0]
        self.assertIn("reward", rec)
        self.assertIn("resource_cost", rec)
        self.assertEqual(rec["episode"], 1)
        self.assertEqual(rec["outcome"], "improved")

        # New better metric updates best
        mem.record_episode(2, {"pop_size": 20}, 0.60, [], 0.2, "improved", reward=0.58, cost=200)
        self.assertAlmostEqual(mem.best_distance, 0.60)
        self.assertEqual(mem.last_outcome(), "improved")

        # Non-improving metric should not lower best
        mem.record_episode(3, {"pop_size": 5}, 0.55, [], 0.15, "stagnated", reward=0.53, cost=50)
        self.assertAlmostEqual(mem.best_distance, 0.60)
        self.assertEqual(mem.last_outcome(), "stagnated")


if __name__ == "__main__":
    unittest.main()
