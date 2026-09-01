from adaptive_trainer import AdaptiveTrainer


def test_no_memory_does_not_record_slime_trails():
    trainer = AdaptiveTrainer(use_memory=False, seed=7)
    plan = trainer.plan({"problem_size": 40})
    trainer.observe(metric=0.2, cost=10.0, params=plan)
    trainer.observe(metric=0.2, cost=10.0, params=plan)
    assert trainer.memory.slime_trails == set()


def test_no_elite_does_not_update_elite_budget():
    trainer = AdaptiveTrainer(use_elite=False, seed=7)
    initial = trainer.memory.elite_path
    plan = trainer.plan({"problem_size": 40})
    trainer.observe(metric=0.8, cost=10.0, params=plan)
    assert trainer.memory.elite_path == initial


def test_no_cost_awareness_uses_zero_cost_penalty():
    trainer = AdaptiveTrainer(alpha=1.0, use_cost_control=False, seed=7)
    plan = trainer.plan({"problem_size": 40})
    trainer.observe(metric=0.8, cost=100.0, params=plan)
    assert trainer.memory.episode_history[-1]["reward"] == 0.8
