import unittest
from adaptive_trainer import AdaptiveTrainer


class TestLongRunStability(unittest.TestCase):

    def test_long_run(self):
        trainer = AdaptiveTrainer()

        for _ in range(200):
            plan = trainer.plan({"dataset_size": 100})
            trainer.observe(metric=0.5, cost=1000)

            self.assertLess(plan["pop_size"], 10000)
            self.assertLess(plan["max_iter"], 100000)


if __name__ == "__main__":
    unittest.main()
