from __future__ import annotations

from sklearn.linear_model import SGDClassifier


class SklearnLogReg:
    """Small logistic-regression style classifier for FL smoke tests."""

    def __init__(
        self,
        max_iter: int = 1,
        learning_rate: str = "constant",
        eta0: float = 0.05,
        random_state: int = 0,
    ):
        self.model = SGDClassifier(
            loss="log_loss",
            max_iter=max_iter,
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=random_state,
            tol=None,
        )

    def __getattr__(self, item):
        return getattr(self.model, item)
