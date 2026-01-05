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

    def test_saturated_detection(self):
        critic = CriticAgent()
        mem = AgentMemory()

        prev = 0.5
        # Small improvement within saturated band (0.0001 < imp <= 0.005)
        cur_saturated = prev + 0.003
        outcome_s = critic.assess(prev, cur_saturated, mem)
        self.assertEqual(outcome_s, "saturated")

        # Clear improvement above tol
        cur_improved = prev + 0.01
        outcome_i = critic.assess(prev, cur_improved, mem)
        self.assertEqual(outcome_i, "improved")

        # Tiny improvement below tiny_improvement -> stagnated
        cur_tiny = prev + 0.00005
        outcome_t = critic.assess(prev, cur_tiny, mem)
        self.assertEqual(outcome_t, "stagnated")



if __name__ == "__main__":
    unittest.main()
