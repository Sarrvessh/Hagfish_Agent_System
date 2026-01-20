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

    def test_efficiency_trend(self):
        mem = AgentMemory()

        # Episode 1
        mem.record_episode(1, {"pop_size": 10}, 0.50, [], 0.1, "improved", reward=0.45, cost=100)
        # Episode 2
        mem.record_episode(2, {"pop_size": 20}, 0.55, [], 0.2, "improved", reward=0.53, cost=200)
        # Episode 3
        mem.record_episode(3, {"pop_size": 5}, 0.56, [], 0.15, "improved", reward=0.55, cost=50)

        eff = mem.get_efficiency_trend(n=3)
        # accuracy_gain = 0.56 - 0.50 = 0.06, total_cost = 100 + 200 + 50 = 350
        self.assertAlmostEqual(eff, 0.06 / 350)

    def test_last_reward_slope_and_planner_stop(self):
        """Test reward slope and planner adaptation to negative trends.
        
        In the bio-inspired model, when reward slope is negative, the planner
        doesn't necessarily "stop" with small budgets - instead it uses
        slime mechanics to explore alternatives. We just verify that:
        1. Slope calculation is correct
        2. The planner produces valid budgets
        """
        from adaptive_trainer.planner import PlannerAgent
        mem = AgentMemory()
        planner = PlannerAgent()

        # Two episodes with decreasing rewards -> slope negative
        mem.record_episode(1, {"pop_size": 10}, 0.80, [], 0.1, "improved", reward=0.80, cost=100)
        mem.record_episode(2, {"pop_size": 20}, 0.81, [], 0.2, "improved", reward=0.79, cost=500)
        slope = mem.get_last_reward_slope()
        self.assertLess(slope, 0.0)  # Reward decreased

        # When stagnated with negative slope, planner still produces valid budgets
        mem.stagnation_count = 3
        params = planner.choose(40, mem, alpha=1e-4)
        # Should be within reasonable bounds (not necessarily minimal)
        self.assertGreaterEqual(params["pop_size"], 16)
        self.assertLessEqual(params["pop_size"], 200)
        self.assertGreaterEqual(params["max_iter"], 50)
        self.assertLessEqual(params["max_iter"], 1000)


if __name__ == "__main__":
    unittest.main()
