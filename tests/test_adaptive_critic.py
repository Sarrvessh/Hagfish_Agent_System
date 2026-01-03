import unittest

from adaptive_trainer.critic import CriticAgent
from adaptive_trainer.memory import AgentMemory


class TestCriticAgent(unittest.TestCase):
    def test_improvement_and_stagnation(self):
        critic = CriticAgent()
        mem = AgentMemory()

        prev = float("-inf")
        cur = 0.5
        outcome = critic.assess(prev, cur, mem)
        self.assertEqual(outcome, "improved")
        self.assertEqual(mem.stagnation_count, 0)

        # No improvement
        prev2 = mem.best_distance
        cur2 = prev2
        outcome2 = critic.assess(prev2, cur2, mem)
        self.assertEqual(outcome2, "stagnated")
        self.assertGreaterEqual(mem.stagnation_count, 1)


if __name__ == "__main__":
    unittest.main()
