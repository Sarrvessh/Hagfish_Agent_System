import unittest

from adaptive_trainer.policies import FixedPolicy, RandomPolicy, GreedyPolicy


class TestPolicies(unittest.TestCase):
    def test_fixed_and_random(self):
        arm = {"pop_size": 10, "max_iter": 5, "elite_size": 1}
        fixed = FixedPolicy(arm)
        self.assertEqual(fixed.choose(), arm)

        arms = [arm, {"pop_size": 20, "max_iter": 10, "elite_size": 2}]
        rand = RandomPolicy(arms, seed=0)
        choice = rand.choose()
        self.assertIn(choice, arms)

    def test_greedy_update(self):
        initial = {"pop_size": 5, "max_iter": 2, "elite_size": 1}
        g = GreedyPolicy(initial)
        self.assertEqual(g.choose(), initial)
        g.update({"pop_size": 10}, 1.0)
        self.assertEqual(g.choose(), {"pop_size": 10})


if __name__ == "__main__":
    unittest.main()
