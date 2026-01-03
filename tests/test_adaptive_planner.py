import unittest

from adaptive_trainer.planner import PlannerAgent
from adaptive_trainer.memory import AgentMemory


class TestPlannerAgent(unittest.TestCase):
    def test_choose_base_and_escalation(self):
        planner = PlannerAgent()
        mem = AgentMemory()

        # Base choose for a small dataset
        params = planner.choose(40, mem)
        self.assertIsInstance(params, dict)
        self.assertIn("pop_size", params)
        self.assertIn("max_iter", params)
        self.assertIn("elite_size", params)

        # Simulate stagnation to force escalation
        mem.stagnation_count = 3
        params2 = planner.choose(40, mem)
        self.assertGreaterEqual(params2["pop_size"], params["pop_size"])
        self.assertGreaterEqual(params2["max_iter"], params["max_iter"])

        # Simulate recent improvement to reduce budget
        mem.stagnation_count = 0
        mem.record_episode(1, params2, 0.9, [], 0.1, "improved", reward=0.9, cost=100)
        params3 = planner.choose(40, mem)
        self.assertLessEqual(params3["pop_size"], params2["pop_size"])


if __name__ == "__main__":
    unittest.main()
