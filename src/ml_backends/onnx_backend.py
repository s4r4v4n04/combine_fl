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


class ONNXBackend(BackendAdapter):
    name = "onnx"

    @staticmethod
    def _resolve_model(model):
        return getattr(model, "model", model)

    @staticmethod
    def _resolve_weights_target(model):
        """Trainable wrappers (e.g. ONNXMLP) expose get/set_weights on the outer object.

        ``_resolve_model`` unwraps to ``.model`` for fit/predict; that inner sklearn
        estimator has no federated weight API, so weight ops must use the wrapper.
        """
        if hasattr(model, "get_weights") and hasattr(model, "set_weights"):
            return model
        return getattr(model, "model", model)

    @property
    def ort(self):
        import onnxruntime as ort

        return ort

    def build_model(self, model_cls: type, device=None, args: dict | None = None):
        model_obj = super().build_model(model_cls, device=device, args=args)
        if isinstance(model_obj, str):
            return self.ort.InferenceSession(model_obj)
        if hasattr(model_obj, "model_path"):
            return self.ort.InferenceSession(model_obj.model_path)
        return model_obj

    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        model_obj = self._resolve_weights_target(model)
        if hasattr(model_obj, "get_weights"):
            return WeightPayload(
                backend=self.name,
                model_id=model_id,
                weights=model_obj.get_weights(),
                metadata={"format": "onnx_custom_weights"},
            )
        return WeightPayload(
            backend=self.name,
            model_id=model_id,
            weights=None,
            metadata={"format": "onnx_runtime_session"},
        )

    def set_weights(self, model, payload: WeightPayload | object) -> None:
        model_obj = self._resolve_weights_target(model)
        if hasattr(model_obj, "set_weights"):
            model_obj.set_weights(unwrap_weights(payload))
            return
        raise NotImplementedError(
            "ONNX backend cannot mutate runtime session weights directly. "
            "Use a custom trainer/model wrapper with set_weights()."
        )

    def parameter_count(self, model) -> int:
        weights = unwrap_weights(self.get_weights(model))
        if isinstance(weights, dict):
            return int(sum(np.asarray(v).size for v in weights.values()))
        return 0

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
        model_obj = self._resolve_model(model)
        if not (hasattr(model_obj, "partial_fit") or hasattr(model_obj, "fit")):
            raise NotImplementedError(
                "ONNX backend does not provide a runtime-session trainer. "
                "Use a custom trainable wrapper with fit()/partial_fit() or a custom trainer."
            )

        start_time = time.time()
        epochs = num_epochs or 1
        total_mini_batches = 0
        correct = 0
        total = 0
        float_epochs = 0.0

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

                if hasattr(model_obj, "partial_fit"):
                    kwargs = {}
                    if total_mini_batches == 0:
                        kwargs["classes"] = classes
                    model_obj.partial_fit(x_batch, y_batch, **kwargs)
                else:
                    model_obj.fit(x_batch, y_batch)

                pred = np.asarray(model_obj.predict(x_batch))
                correct += int(np.sum(pred == y_batch))
                total += len(y_batch)
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
            "loss": 0,
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
        total = 0
        correct = 0
        total_loss = 0.0
        batches = 0

        model_obj = self._resolve_model(model)
        if hasattr(model_obj, "run"):
            input_name = model_obj.get_inputs()[0].name
            for x_batch, y_batch in dataloader:
                outputs = model_obj.run(None, {input_name: np.asarray(x_batch, dtype=np.float32)})
                logits = np.asarray(outputs[0])
                pred = np.argmax(logits, axis=1)
                correct += int(np.sum(pred == y_batch))
                total += len(y_batch)
                probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = probs / probs.sum(axis=1, keepdims=True)
                total_loss += float(
                    -np.mean(np.log(np.clip(probs[np.arange(len(y_batch)), y_batch], 1e-12, 1.0)))
                )
                batches += 1
        elif hasattr(model_obj, "predict"):
            for x_batch, y_batch in dataloader:
                pred = np.asarray(model_obj.predict(x_batch))
                correct += int(np.sum(pred == y_batch))
                total += len(y_batch)
                if hasattr(model_obj, "predict_proba"):
                    probs = np.asarray(model_obj.predict_proba(x_batch), dtype=np.float64)
                    total_loss += float(
                        -np.mean(np.log(np.clip(probs[np.arange(len(y_batch)), y_batch], 1e-12, 1.0)))
                    )
                batches += 1
        else:
            raise TypeError("ONNX validator expects an InferenceSession or model with predict().")

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
                "ONNX default dataloader expects .npz with x/y or "
                "x_train/y_train/x_test/y_test arrays"
            )
        return (
            _NumpyBatchLoader(x_train, y_train, batch_size=batch_size, shuffle=True),
            _NumpyBatchLoader(x_test, y_test, batch_size=batch_size),
        )

    def save_weights(self, payload: WeightPayload, destination) -> None:
        pickle.dump(unwrap_weights(payload), destination)
