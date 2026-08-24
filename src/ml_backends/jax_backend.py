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


class JAXBackend(BackendAdapter):
    name = "jax"

    @staticmethod
    def _resolve_model(model):
        return getattr(model, "model", model)

    @property
    def jax(self):
        import jax

        return jax

    @property
    def jnp(self):
        import jax.numpy as jnp

        return jnp

    def _devices(self, backend: str):
        try:
            return self.jax.devices(backend)
        except RuntimeError:
            return []

    def resolve_device(self, device: str | None = None, use_gpu: bool = False):
        if device:
            device = str(device).lower()
            if device.startswith("gpu") or device.startswith("cuda"):
                devices = self._devices("gpu")
                return devices[0] if devices else self.jax.devices("cpu")[0]
            return self.jax.devices("cpu")[0]
        if use_gpu:
            devices = self._devices("gpu")
            if devices:
                return devices[0]
        return self.jax.devices("cpu")[0]

    def set_seed(self, seed: int) -> None:
        self._rng = self.jax.random.PRNGKey(seed)

    def move_model_to_device(self, model, device):
        model = self._resolve_model(model)
        jax = self.jax
        if hasattr(model, "params"):
            model.params = jax.device_put(model.params, device)
        return model

    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        model = self._resolve_model(model)
        if hasattr(model, "get_weights"):
            weights = model.get_weights()
        elif hasattr(model, "params"):
            weights = model.params
        else:
            weights = model
        return WeightPayload(
            backend=self.name,
            model_id=model_id,
            weights=weights,
            metadata={"format": "jax_pytree"},
        )

    def set_weights(self, model, payload: WeightPayload | object) -> None:
        model = self._resolve_model(model)
        weights = unwrap_weights(payload)
        if hasattr(model, "set_weights"):
            model.set_weights(weights)
        elif hasattr(model, "params"):
            model.params = weights
        else:
            raise TypeError(
                "JAX default weight loading expects a model with set_weights() "
                "or a mutable params attribute."
            )

    def parameter_count(self, model) -> int:
        model = self._resolve_model(model)
        params = unwrap_weights(self.get_weights(model))
        leaves = self.jax.tree_util.tree_leaves(params)
        return int(sum(np.asarray(leaf).size for leaf in leaves))

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
        model = self._resolve_model(model)
        if not hasattr(model, "params") or not hasattr(model, "apply"):
            raise NotImplementedError(
                "JAX default trainer expects model.params and model.apply(params, x). "
                "Use a custom trainer for Flax/Haiku/Optax-specific training."
            )

        jax = self.jax
        jnp = self.jnp
        device = device or self.resolve_device("cpu")
        params = jax.device_put(model.params, device)
        epochs = num_epochs or 1
        start_time = time.time()
        total_loss = 0.0
        total = 0
        correct = 0
        total_mini_batches = 0
        float_epochs = 0.0

        def default_loss(params, x_batch, y_batch):
            logits = model.apply(params, x_batch)
            labels = y_batch.astype(jnp.int32)
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

        loss_fn = loss_func or default_loss
        grad_fn = jax.value_and_grad(loss_fn)

        for epoch in range(epochs):
            num_mini_batches = 0
            for x_batch, y_batch in train_loader:
                if max_mini_batches and total_mini_batches >= max_mini_batches:
                    break
                if max_epochs and epoch >= max_epochs:
                    break
                if timeout_duration_s and time.time() - start_time > timeout_duration_s:
                    break
                if stop_requested and stop_requested():
                    break

                x_batch = jax.device_put(jnp.asarray(x_batch), device)
                y_batch = jax.device_put(jnp.asarray(y_batch), device)
                loss, grads = grad_fn(params, x_batch, y_batch)
                params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

                logits = model.apply(params, x_batch)
                predicted = jnp.argmax(logits, axis=1)
                correct += int(jnp.sum(predicted == y_batch))
                total += int(y_batch.shape[0])
                total_loss += float(loss)
                total_mini_batches += 1
                num_mini_batches += 1
                try:
                    data_entries = len(train_loader)
                except TypeError:
                    data_entries = max(num_mini_batches, 1)
                float_epochs = epoch + (num_mini_batches / data_entries)

            if (
                (max_mini_batches and total_mini_batches >= max_mini_batches)
                or (max_epochs and epoch >= max_epochs)
                or (timeout_duration_s and time.time() - start_time > timeout_duration_s)
                or (stop_requested and stop_requested())
            ):
                break

        model.params = params
        return {
            "time_taken_s": time.time() - start_time,
            "num_epochs": float_epochs,
            "total_mini_batches": total_mini_batches,
            "loss": round(total_loss / total_mini_batches, 3)
            if total_mini_batches
            else 0,
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
        model = self._resolve_model(model)
        if not hasattr(model, "params") or not hasattr(model, "apply"):
            raise NotImplementedError(
                "JAX default validator expects model.params and model.apply(params, x)."
            )

        jax = self.jax
        jnp = self.jnp
        device = device or self.resolve_device("cpu")
        params = jax.device_put(model.params, device)
        total_loss = 0.0
        total = 0
        correct = 0
        batches = 0

        def default_loss(params, x_batch, y_batch):
            logits = model.apply(params, x_batch)
            labels = y_batch.astype(jnp.int32)
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

        loss_fn = loss_func or default_loss
        for x_batch, y_batch in dataloader:
            x_batch = jax.device_put(jnp.asarray(x_batch), device)
            y_batch = jax.device_put(jnp.asarray(y_batch), device)
            logits = model.apply(params, x_batch)
            loss = loss_fn(params, x_batch, y_batch)
            predicted = jnp.argmax(logits, axis=1)
            correct += int(jnp.sum(predicted == y_batch))
            total += int(y_batch.shape[0])
            total_loss += float(loss)
            batches += 1

        return {
            "accuracy": (correct / total) * 100 if total else 0,
            "loss": total_loss / batches if batches else 0,
        }

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        if not dataset_path:
            raise ValueError("JAX default dataloader requires dataset_path.")

        def _normalize_labels(labels):
            labels = np.asarray(labels)
            if labels.ndim > 1:
                # Handle one-hot or logits-like labels from .npz datasets.
                labels = np.argmax(labels, axis=-1)
            return labels.reshape(-1).astype(np.int64)

        with np.load(dataset_path, allow_pickle=False) as data:
            if {"x_train", "y_train", "x_test", "y_test"}.issubset(data.files):
                x_train, y_train = data["x_train"], data["y_train"]
                x_test, y_test = data["x_test"], data["y_test"]
            elif {"x", "y"}.issubset(data.files):
                x, y = data["x"], data["y"]
                if len(x) <= 1:
                    x_train, y_train = x, y
                    x_test, y_test = x, y
                else:
                    split_idx = min(max(int(0.95 * len(x)), 1), len(x) - 1)
                    x_train, y_train = x[:split_idx], y[:split_idx]
                    x_test, y_test = x[split_idx:], y[split_idx:]
            else:
                raise ValueError(
                    "JAX default dataloader expects .npz with x/y or "
                    "x_train/y_train/x_test/y_test arrays"
                )

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        y_train = _normalize_labels(y_train)
        y_test = _normalize_labels(y_test)
        return (
            _NumpyBatchLoader(x_train, y_train, batch_size=batch_size, shuffle=True),
            _NumpyBatchLoader(x_test, y_test, batch_size=batch_size),
        )

    def save_weights(self, payload: WeightPayload, destination) -> None:
        pickle.dump(unwrap_weights(payload), destination)
