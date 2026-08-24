from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

class ONNXMLP:
    """Tiny trainable MLP-like classifier used in ONNX backend smoke tests."""

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

    def partial_fit(self, x, y, classes=None):
        kwargs = {}
        if classes is not None:
            kwargs["classes"] = classes
        self.model.partial_fit(x, y, **kwargs)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def get_weights(self):
        return {
            "coef_": np.asarray(getattr(self.model, "coef_", [])),
            "intercept_": np.asarray(getattr(self.model, "intercept_", [])),
            "classes_": np.asarray(getattr(self.model, "classes_", [])),
        }

    def set_weights(self, weights):
        if "coef_" in weights and len(np.asarray(weights["coef_"])):
            self.model.coef_ = np.asarray(weights["coef_"])
        if "intercept_" in weights and len(np.asarray(weights["intercept_"])):
            self.model.intercept_ = np.asarray(weights["intercept_"])
        if "classes_" in weights and len(np.asarray(weights["classes_"])):
            self.model.classes_ = np.asarray(weights["classes_"])

    def __getattr__(self, item):
        return getattr(self.model, item)
