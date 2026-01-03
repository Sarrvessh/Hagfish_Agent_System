import unittest

from planner_agent import PlannerAgent
from agent_memory import AgentMemory


class TestPlannerAgent(unittest.TestCase):
    def test_choose_base_and_escalation(self):
        planner = PlannerAgent()
        m = AgentMemory()

        params = planner.choose(problem_size=50, memory=m)
        self.assertEqual(params["pop_size"], 60)
        self.assertEqual(params["max_iter"], 300)
        self.assertEqual(params["elite_size"], 5)

        # Escalation on stagnation
        m.stagnation_count = 3
        params2 = planner.choose(problem_size=50, memory=m)
        self.assertEqual(params2["pop_size"], int(60 * 1.2))
        self.assertEqual(params2["max_iter"], int(300 * 1.25))
        self.assertEqual(params2["elite_size"], 6)

        # Improvement reduces effort
        m = AgentMemory()
        m.episode_history.append({"outcome": "improved"})
        params3 = planner.choose(problem_size=50, memory=m)
        self.assertEqual(params3["pop_size"], max(60, 60 - 10))


if __name__ == "__main__":
    unittest.main()
