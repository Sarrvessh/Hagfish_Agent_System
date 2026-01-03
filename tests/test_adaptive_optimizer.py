import unittest

from adaptive_trainer.optimizer import AdaptiveTrainer
from adaptive_trainer.memory import AgentMemory


class TestAdaptiveTrainer(unittest.TestCase):
    def test_plan_and_observe(self):
        at = AdaptiveTrainer(alpha=0.0001)
        ctx = {"dataset_size": 40}
        plan = at.plan(ctx)
        self.assertIsInstance(plan, dict)
        # Observe a metric and check memory updated
        mem_before = len(at.memory.episode_history)
        at.observe(metric=0.8, cost=1000, params=plan, episode=1, elapsed_time=0.5)
        self.assertEqual(len(at.memory.episode_history), mem_before + 1)
        rec = at.memory.episode_history[-1]
        self.assertAlmostEqual(rec["distance"], 0.8)
        self.assertIn("reward", rec)


if __name__ == "__main__":
    unittest.main()
