import math
import pickle
import time
from typing import Callable

import numpy as np

from ml_backends.base import BackendAdapter
from ml_backends.weights import WeightPayload, unwrap_weights


class _NumpyBatchLoader:
    def __init__(self, x, y, batch_size: int, shuffle: bool = False):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]


class SklearnBackend(BackendAdapter):
    name = "sklearn"

    @staticmethod
    def _resolve_estimator(model):
        # Some examples wrap sklearn estimators under `.model`.
        return getattr(model, "model", model)

    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        estimator = self._resolve_estimator(model)
        if hasattr(estimator, "get_weights"):
            weights = estimator.get_weights()
        else:
            weights = {
                "coef_": np.asarray(getattr(estimator, "coef_", [])),
                "intercept_": np.asarray(getattr(estimator, "intercept_", [])),
                "classes_": np.asarray(getattr(estimator, "classes_", [])),
            }
        return WeightPayload(
            backend=self.name,
            model_id=model_id,
            weights=weights,
            metadata={"format": "sklearn_state"},
        )

    def set_weights(self, model, payload: WeightPayload | object) -> None:
        weights = unwrap_weights(payload)
        estimator = self._resolve_estimator(model)
        if hasattr(estimator, "set_weights"):
            estimator.set_weights(weights)
            return
        if not isinstance(weights, dict):
            raise TypeError("Sklearn backend expects dict-like weights for set_weights.")
        for key in ("coef_", "intercept_", "classes_"):
            if key in weights and weights[key] is not None and len(np.asarray(weights[key])):
                setattr(estimator, key, np.asarray(weights[key]))

    def parameter_count(self, model) -> int:
        weights = unwrap_weights(self.get_weights(model))
        if not isinstance(weights, dict):
            return 0
        return int(
            np.asarray(weights.get("coef_", [])).size
            + np.asarray(weights.get("intercept_", [])).size
        )

    def default_train_classifier(
        self,
        model,
        train_loader,
        lr: float,
        loss_func=None,
        optimizer=None,
        device=None,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict:
        from sklearn.metrics import log_loss

        start_time = time.time()
        epochs = num_epochs or 1
        total_mini_batches = 0
        total_loss = 0.0
        total = 0
        correct = 0
        float_epochs = 0.0

        if hasattr(model, "set_params") and hasattr(model, "learning_rate_init"):
            try:
                model.set_params(learning_rate_init=lr)
            except ValueError:
                pass

        train_batches = list(train_loader)
        if not train_batches:
            return {
                "time_taken_s": time.time() - start_time,
                "num_epochs": 0.0,
                "total_mini_batches": 0,
                "loss": 0,
                "accuracy": 0,
            }

        classes = np.unique(np.concatenate([batch_y for _, batch_y in train_batches]))
        for epoch in range(epochs):
            num_mini_batches = 0
            for x_batch, y_batch in train_batches:
                if max_mini_batches and total_mini_batches >= max_mini_batches:
                    break
                if max_epochs and epoch >= max_epochs:
                    break
                if timeout_duration_s and time.time() - start_time > timeout_duration_s:
                    break
                if stop_requested and stop_requested():
                    break

                if hasattr(model, "partial_fit"):
                    kwargs = {}
                    if total_mini_batches == 0:
                        kwargs["classes"] = classes
                    model.partial_fit(x_batch, y_batch, **kwargs)
                else:
                    model.fit(x_batch, y_batch)

                preds = model.predict(x_batch)
                correct += int(np.sum(preds == y_batch))
                total += len(y_batch)
                if hasattr(model, "predict_proba"):
                    total_loss += float(log_loss(y_batch, model.predict_proba(x_batch), labels=classes))
                total_mini_batches += 1
                num_mini_batches += 1
                data_entries = max(len(train_batches), 1)
                float_epochs = epoch + (num_mini_batches / data_entries)

            if (
                (max_mini_batches and total_mini_batches >= max_mini_batches)
                or (max_epochs and epoch >= max_epochs)
                or (timeout_duration_s and time.time() - start_time > timeout_duration_s)
                or (stop_requested and stop_requested())
            ):
                break

        return {
            "time_taken_s": time.time() - start_time,
            "num_epochs": float_epochs,
            "total_mini_batches": total_mini_batches,
            "loss": round(total_loss / total_mini_batches, 3) if total_mini_batches else 0,
            "accuracy": round((correct / total) * 100, 3) if total else 0,
        }

    def default_validate_classifier(
        self,
        model,
        dataloader,
        loss_func=None,
        optimizer=None,
        device=None,
        round_no=None,
    ) -> dict:
        from sklearn.exceptions import NotFittedError
        from sklearn.metrics import log_loss

        total = 0
        correct = 0
        total_loss = 0.0
        batches = 0
        classes = getattr(model, "classes_", None)
        for x_batch, y_batch in dataloader:
            try:
                preds = model.predict(x_batch)
            except NotFittedError:
                # Server-side pre-training validation may run before first fit.
                return {"accuracy": 0, "loss": 0}
            correct += int(np.sum(preds == y_batch))
            total += len(y_batch)
            if hasattr(model, "predict_proba"):
                labels = classes if classes is not None and len(classes) else np.unique(y_batch)
                total_loss += float(log_loss(y_batch, model.predict_proba(x_batch), labels=labels))
            batches += 1
        return {
            "accuracy": (correct / total) * 100 if total else 0,
            "loss": total_loss / batches if batches else 0,
        }

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        data = np.load(dataset_path)
        if {"x_train", "y_train", "x_test", "y_test"}.issubset(data.files):
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]
        elif {"x", "y"}.issubset(data.files):
            x, y = data["x"], data["y"]
            split_idx = int(0.95 * len(x))
            x_train, y_train = x[:split_idx], y[:split_idx]
            x_test, y_test = x[split_idx:], y[split_idx:]
        else:
            raise ValueError(
                "Sklearn default dataloader expects .npz with x/y or "
                "x_train/y_train/x_test/y_test arrays"
            )

        x_train = np.asarray(x_train).reshape(len(x_train), -1)
        x_test = np.asarray(x_test).reshape(len(x_test), -1)
        return (
            _NumpyBatchLoader(x_train, y_train, batch_size=batch_size, shuffle=True),
            _NumpyBatchLoader(x_test, y_test, batch_size=batch_size),
        )

    def save_weights(self, payload: WeightPayload, destination) -> None:
        pickle.dump(unwrap_weights(payload), destination)
