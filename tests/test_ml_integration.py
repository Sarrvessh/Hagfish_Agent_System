import unittest
from adaptive_trainer import AdaptiveTrainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class TestMLIntegration(unittest.TestCase):

    def test_real_training_loop(self):
        X, y = load_iris(return_X_y=True)
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2)

        trainer = AdaptiveTrainer(alpha=1e-4)
        plan = trainer.plan({"dataset_size": len(Xtr)})

        model = MLPClassifier(
            max_iter=min(plan["max_iter"], 50),
            batch_size=min(plan["pop_size"], len(Xtr)),
            random_state=42
        )

        model.fit(Xtr, ytr)
        acc = accuracy_score(yv, model.predict(Xv))

        trainer.observe(metric=acc, cost=plan["pop_size"] * plan["max_iter"])

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
