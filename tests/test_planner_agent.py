import unittest

from planner_agent import PlannerAgent
from agent_memory import AgentMemory


class TestPlannerAgent(unittest.TestCase):
    def test_choose_base_and_escalation(self):
        planner = PlannerAgent()
        m = AgentMemory()

        params = planner.choose(problem_size=50, memory=m)
        # New conservative floors
        self.assertEqual(params["pop_size"], 16)
        self.assertEqual(params["max_iter"], 50)
        self.assertEqual(params["elite_size"], 5)

        # Escalation on stagnation (alpha default is conservative -> 1.1x)
        m.stagnation_count = 3
        params2 = planner.choose(problem_size=50, memory=m)
        self.assertEqual(params2["pop_size"], int(16 * 1.1))
        self.assertEqual(params2["max_iter"], int(50 * 1.1))
        self.assertEqual(params2["elite_size"], 7)

        # Improvement reduces effort (percentage-based reduction now)
        m = AgentMemory()
        m.episode_history.append({"outcome": "improved"})
        params3 = planner.choose(problem_size=50, memory=m)
        self.assertLessEqual(params3["pop_size"], params["pop_size"])


if __name__ == "__main__":
    unittest.main()
