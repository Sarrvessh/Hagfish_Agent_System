from adaptive_trainer import AdaptiveTrainer

def test_long_run_stability():
    trainer = AdaptiveTrainer()

    for i in range(200):
        plan = trainer.plan({"dataset_size": 100})
        trainer.observe(metric=0.5, cost=1000)

        assert plan["pop_size"] < 10_000
        assert plan["max_iter"] < 100_000
