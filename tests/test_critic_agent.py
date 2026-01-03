import unittest

from critic_agent import CriticAgent
from agent_memory import AgentMemory


class TestCriticAgent(unittest.TestCase):
    def test_improvement_and_stagnation(self):
        critic = CriticAgent()
        m = AgentMemory()

        # Improvement (higher is better metric)
        outcome = critic.assess(previous_best=0.60, current_best=0.65, memory=m)
        self.assertEqual(outcome, "improved")
        self.assertEqual(m.stagnation_count, 0)

        # Stagnation
        outcome2 = critic.assess(previous_best=0.65, current_best=0.63, memory=m)
        self.assertEqual(outcome2, "stagnated")
        self.assertEqual(m.stagnation_count, 1)


if __name__ == "__main__":
    unittest.main()
