import unittest

from adaptive_trainer.planner import PlannerAgent
from adaptive_trainer.memory import AgentMemory


class TestPlannerAgent(unittest.TestCase):
    def test_choose_base_and_escalation(self):
        planner = PlannerAgent()
        mem = AgentMemory()

        # Base choose for a small dataset should start cheap (floors lowered)
        params = planner.choose(40, mem)
        self.assertEqual(params["pop_size"], 16)   # min(64, max(16, 40//20)) -> 16
        self.assertEqual(params["max_iter"], 50)  # min(150, max(50, 40//5)) -> 50
        self.assertEqual(params["elite_size"], 4)

        # Simulate stagnation to force escalation; check alpha-aware conservative scaling
        mem.stagnation_count = 3
        params_low_alpha = planner.choose(40, mem, alpha=1e-6)
        params_high_alpha = planner.choose(40, mem, alpha=1e-4)
        # High alpha -> smaller escalation (1.1x), low alpha -> larger (1.25x)
        self.assertGreaterEqual(params_low_alpha["pop_size"], params["pop_size"])
        self.assertLessEqual(params_high_alpha["pop_size"], params_low_alpha["pop_size"])

        # If recent improvement, reduce budget
        mem.stagnation_count = 0
        mem.record_episode(1, params_low_alpha, 0.9, [], 0.1, "improved", reward=0.9, cost=100)
        params3 = planner.choose(40, mem)
        self.assertLessEqual(params3["pop_size"], params_low_alpha["pop_size"])

        # Saturated should cause de-escalation ~20%
        mem.record_episode(2, params3, 0.9005, [], 0.1, "saturated", reward=0.9005, cost=100)
        params4 = planner.choose(40, mem, 1e-4)
        self.assertLessEqual(params4["pop_size"], params3["pop_size"])
        self.assertLessEqual(params4["max_iter"], params3["max_iter"])

        # Solver safety: pop_size <= problem_size and max_iter >= 10
        params_safe = planner.choose(5, mem)  # tiny problem size
        self.assertLessEqual(params_safe["pop_size"], 5)
        self.assertGreaterEqual(params_safe["max_iter"], 10)


if __name__ == "__main__":
    unittest.main()
