import time
from typing import Callable

import numpy as np

from ml_backends.base import BackendAdapter
from ml_backends.weights import WeightPayload, unwrap_weights


class TensorFlowBackend(BackendAdapter):
    name = "tensorflow"

    @staticmethod
    def _resolve_model(model):
        return getattr(model, "model", model)

    @property
    def tf(self):
        import tensorflow as tf

        return tf

    def resolve_device(self, device: str | None = None, use_gpu: bool = False) -> str:
        if device:
            device = str(device)
            if device.startswith("/"):
                return device
            if device.startswith("cuda") or device.startswith("gpu"):
                return "/GPU:0"
            return "/CPU:0"
        if use_gpu and self.tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        return "/CPU:0"

    def set_seed(self, seed: int) -> None:
        self.tf.random.set_seed(seed)

    def build_model(self, model_cls: type, device=None, args: dict | None = None):
        with self.tf.device(device or "/CPU:0"):
            return super().build_model(model_cls, device=device, args=args)

    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        keras_model = self._resolve_model(model)
        return WeightPayload(
            backend=self.name,
            model_id=model_id,
            weights=keras_model.get_weights(),
            metadata={"format": "keras_weights"},
        )

    def set_weights(self, model, payload: WeightPayload | list) -> None:
        keras_model = self._resolve_model(model)
        keras_model.set_weights(unwrap_weights(payload))

    def parameter_count(self, model) -> int:
        keras_model = self._resolve_model(model)
        return int(np.sum([np.prod(v.shape) for v in keras_model.trainable_weights]))

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
        tf = self.tf
        device = device or self.resolve_device("cpu")
        keras_model = self._resolve_model(model)
        if loss_func is None:
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif isinstance(loss_func, type):
            loss_func = loss_func()
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        start_time = time.time()
        total_loss = 0.0
        total = 0
        correct = 0
        total_mini_batches = 0
        epochs = num_epochs or 1
        float_epochs = 0.0

        with tf.device(device):
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

                    with tf.GradientTape() as tape:
                        predictions = keras_model(x_batch, training=True)
                        loss = loss_func(y_batch, predictions)
                    gradients = tape.gradient(loss, keras_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))

                    batch_size = int(tf.shape(y_batch)[0])
                    total += batch_size
                    total_loss += float(loss.numpy())
                    predicted = tf.argmax(predictions, axis=1, output_type=y_batch.dtype)
                    correct += int(tf.reduce_sum(tf.cast(predicted == y_batch, tf.int32)).numpy())
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
        tf = self.tf
        device = device or self.resolve_device("cpu")
        keras_model = self._resolve_model(model)
        if loss_func is None:
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif isinstance(loss_func, type):
            loss_func = loss_func()

        total_loss = 0.0
        total = 0
        correct = 0
        batches = 0
        with tf.device(device):
            for x_batch, y_batch in dataloader:
                predictions = keras_model(x_batch, training=False)
                loss = loss_func(y_batch, predictions)
                batch_size = int(tf.shape(y_batch)[0])
                total += batch_size
                total_loss += float(loss.numpy())
                predicted = tf.argmax(predictions, axis=1, output_type=y_batch.dtype)
                correct += int(tf.reduce_sum(tf.cast(predicted == y_batch, tf.int32)).numpy())
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
                "TensorFlow default dataloader expects .npz with x/y or "
                "x_train/y_train/x_test/y_test arrays"
            )

        tf = self.tf
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return train.shuffle(len(x_train)).batch(batch_size), test.batch(batch_size)

    def get_server_dataloader(self, batch_size=50, dataset_path=None):
        _, test_loader = self.get_train_test_dataset_loaders(
            batch_size=batch_size, dataset_path=dataset_path
        )
        return test_loader
