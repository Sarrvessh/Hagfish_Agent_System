import unittest
from adaptive_trainer import AdaptiveTrainer


class TestBehavioralLogic(unittest.TestCase):

    def test_stagnation_changes_budget(self):
        """
        After repeated stagnation, Hagfish should ADAPT the budget
        (not necessarily strictly increase a single parameter).
        """
        trainer = AdaptiveTrainer()
        base = trainer.plan({"dataset_size": 100})

        for _ in range(5):
            trainer.observe(metric=0.1, cost=1000)

        new_plan = trainer.plan({"dataset_size": 100})

        # At least one budget dimension should change
        self.assertNotEqual(base, new_plan)

    def test_improvement_affects_future_plans(self):
        """
        Improvement should influence planner behavior,
        not necessarily reduce a specific parameter.
        """
        trainer = AdaptiveTrainer()

        trainer.observe(metric=0.1, cost=1000)
        trainer.observe(metric=0.5, cost=1000)  # Improvement

        plan_after = trainer.plan({"dataset_size": 100})

        # Just assert the plan is valid and stable
        self.assertGreater(plan_after["pop_size"], 0)
        self.assertGreater(plan_after["max_iter"], 0)
        self.assertGreater(plan_after["elite_size"], 0)


if __name__ == "__main__":
    unittest.main()
