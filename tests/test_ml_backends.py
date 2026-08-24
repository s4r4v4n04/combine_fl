import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ml_backends.registry import get_backend, get_backend_name
from ml_backends.weights import WeightPayload, unwrap_weights, wrap_weights
from server.load_aggregator import load_aggregator


class MemoryState:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def keys(self):
        return list(self.values.keys())

    def clear(self):
        self.values.clear()

    def deletebykey(self, key):
        self.values.pop(key, None)


def test_weight_payload_wraps_native_weights():
    weights = {"layer": np.array([1.0])}
    payload = wrap_weights(weights, backend="tensorflow", model_id="m")

    assert isinstance(payload, WeightPayload)
    assert payload.backend == "tensorflow"
    assert unwrap_weights(payload) is weights


def test_backend_name_defaults_to_pytorch():
    assert get_backend_name({}) == "pytorch"
    assert get_backend_name({"backend": "tensorflow"}) == "tensorflow"
    assert get_backend_name({"backend": "jax"}) == "jax"
    assert get_backend_name({"backend": "sklearn"}) == "sklearn"
    assert get_backend_name({"backend": "onnx"}) == "onnx"


def test_pytorch_backend_weight_roundtrip():
    torch = pytest.importorskip("torch")
    backend = get_backend("pytorch")
    model = torch.nn.Linear(2, 1)

    payload = backend.get_weights(model, model_id="linear")
    clone = torch.nn.Linear(2, 1)
    backend.set_weights(clone, payload)

    for expected, actual in zip(model.parameters(), clone.parameters()):
        assert torch.equal(expected, actual)


def test_pytorch_backend_weight_roundtrip_for_wrapped_model():
    torch = pytest.importorskip("torch")
    backend = get_backend("pytorch")

    class Wrapped:
        def __init__(self):
            self.model = torch.nn.Linear(2, 1)

    wrapped = Wrapped()
    payload = backend.get_weights(wrapped, model_id="wrapped-linear")
    clone = Wrapped()
    backend.set_weights(clone, payload)

    for expected, actual in zip(wrapped.model.parameters(), clone.model.parameters()):
        assert torch.equal(expected, actual)


def test_tensorflow_backend_weight_roundtrip():
    tf = pytest.importorskip("tensorflow")
    backend = get_backend("tensorflow")
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)]
    )

    payload = backend.get_weights(model, model_id="dense")
    clone = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)]
    )
    backend.set_weights(clone, payload)

    for expected, actual in zip(model.get_weights(), clone.get_weights()):
        np.testing.assert_array_equal(expected, actual)


def test_tensorflow_backend_weight_roundtrip_for_wrapped_model():
    tf = pytest.importorskip("tensorflow")
    backend = get_backend("tensorflow")

    class Wrapped:
        def __init__(self):
            self.model = tf.keras.Sequential(
                [tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1)]
            )

    wrapped = Wrapped()
    payload = backend.get_weights(wrapped, model_id="wrapped-dense")
    clone = Wrapped()
    backend.set_weights(clone, payload)

    for expected, actual in zip(wrapped.model.get_weights(), clone.model.get_weights()):
        np.testing.assert_array_equal(expected, actual)


def test_tensorflow_backend_default_training_and_validation():
    tf = pytest.importorskip("tensorflow")
    backend = get_backend("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]], dtype="float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    train_metrics = backend.default_train_classifier(
        model=model,
        train_loader=dataset,
        lr=0.01,
        num_epochs=1,
    )
    validation_metrics = backend.default_validate_classifier(
        model=model,
        dataloader=dataset,
    )

    assert train_metrics["total_mini_batches"] == 2
    assert "accuracy" in validation_metrics
    assert "loss" in validation_metrics


def test_jax_backend_weight_roundtrip():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = get_backend("jax")

    class Model:
        def __init__(self):
            self.params = {"w": jnp.array([[1.0], [2.0]]), "b": jnp.array([0.5])}

        def apply(self, params, x):
            return x @ params["w"] + params["b"]

    model = Model()
    payload = backend.get_weights(model, model_id="jax-linear")
    clone = Model()
    clone.params = {"w": jnp.zeros((2, 1)), "b": jnp.zeros((1,))}
    backend.set_weights(clone, payload)

    assert payload.backend == "jax"
    for expected, actual in zip(
        jax.tree_util.tree_leaves(model.params),
        jax.tree_util.tree_leaves(clone.params),
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_backend_weight_roundtrip_for_wrapped_model():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = get_backend("jax")

    class Inner:
        def __init__(self):
            self.params = {"w": jnp.array([[1.0], [2.0]]), "b": jnp.array([0.5])}

        def apply(self, params, x):
            return x @ params["w"] + params["b"]

    class Wrapped:
        def __init__(self):
            self.model = Inner()

    wrapped = Wrapped()
    payload = backend.get_weights(wrapped, model_id="wrapped-jax")
    clone = Wrapped()
    clone.model.params = {"w": jnp.zeros((2, 1)), "b": jnp.zeros((1,))}
    backend.set_weights(clone, payload)

    for expected, actual in zip(
        jax.tree_util.tree_leaves(wrapped.model.params),
        jax.tree_util.tree_leaves(clone.model.params),
    ):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_jax_backend_default_training_and_validation():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    backend = get_backend("jax")

    class Model:
        def __init__(self):
            self.params = {
                "w": jnp.zeros((2, 2)),
                "b": jnp.zeros((2,)),
            }

        def apply(self, params, x):
            return x @ params["w"] + params["b"]

    model = Model()
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]], dtype="float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    train_loader = [(x[:2], y[:2]), (x[2:], y[2:])]

    train_metrics = backend.default_train_classifier(
        model=model,
        train_loader=train_loader,
        lr=0.1,
        num_epochs=1,
    )
    validation_metrics = backend.default_validate_classifier(
        model=model,
        dataloader=train_loader,
    )

    assert train_metrics["total_mini_batches"] == 2
    assert "accuracy" in validation_metrics
    assert "loss" in validation_metrics


def test_jax_backend_dataloader_normalizes_one_hot_labels(tmp_path):
    pytest.importorskip("jax")
    backend = get_backend("jax")
    dataset_path = tmp_path / "jax_onehot.npz"

    x = np.random.rand(10, 28, 28, 1).astype("float32")
    y_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int64")
    y_onehot = np.eye(10, dtype="float32")[y_ids]
    np.savez(dataset_path, x=x, y=y_onehot)

    train_loader, test_loader = backend.get_train_test_dataset_loaders(
        batch_size=4,
        dataset_path=str(dataset_path),
    )
    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))

    assert train_x.ndim == 4
    assert test_x.ndim == 4
    assert train_y.ndim == 1
    assert test_y.ndim == 1
    assert np.issubdtype(train_y.dtype, np.integer)
    assert np.issubdtype(test_y.dtype, np.integer)


def test_tensorflow_fedavg_aggregates_keras_weight_payloads():
    aggregator = load_aggregator("test", "fedavg_tensorflow")
    training_state = MemoryState(
        {
            "c1.current_dataset_detail": {"metadata": {"num_items": 1}},
            "c2.current_dataset_detail": {"metadata": {"num_items": 3}},
        }
    )
    aggregator_state = MemoryState()
    client_info = MemoryState({"c1.is_active": True, "c2.is_active": True})
    client_selection_state = MemoryState({"selected_clients": ["c1", "c2"]})

    first = wrap_weights([np.array([1.0, 1.0])], backend="tensorflow")
    second = wrap_weights([np.array([3.0, 3.0])], backend="tensorflow")

    assert (
        aggregator.aggregate(
            "test",
            "c1",
            True,
            first,
            client_info,
            training_state,
            MemoryState(),
            aggregator_state,
            client_selection_state,
            {},
        )
        is None
    )
    result = aggregator.aggregate(
        "test",
        "c2",
        True,
        second,
        client_info,
        training_state,
        MemoryState(),
        aggregator_state,
        client_selection_state,
        {},
    )

    assert result.backend == "tensorflow"
    np.testing.assert_array_equal(result.weights[0], np.array([2.5, 2.5]))


def test_tensorflow_fedasync_updates_keras_weight_payloads():
    aggregator = load_aggregator("test", "fedasync_tensorflow")
    training_session = MemoryState(
        {
            "test.last_round_number": 2,
            "test.global_model": wrap_weights([np.array([1.0, 1.0])], backend="tensorflow"),
        }
    )
    client_selection_state = MemoryState({"c1": 1})
    result = aggregator.aggregate(
        "test",
        "c1",
        True,
        wrap_weights([np.array([3.0, 3.0])], backend="tensorflow"),
        MemoryState(),
        MemoryState(),
        training_session,
        MemoryState(),
        client_selection_state,
        {"alpha": 0.0},
    )

    assert result.backend == "tensorflow"
    np.testing.assert_array_equal(result.weights[0], np.array([3.0, 3.0]))


def test_jax_fedavg_aggregates_pytree_payloads():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    aggregator = load_aggregator("test", "fedavg_jax")
    training_state = MemoryState(
        {
            "c1.current_dataset_detail": {"metadata": {"num_items": 1}},
            "c2.current_dataset_detail": {"metadata": {"num_items": 3}},
        }
    )
    aggregator_state = MemoryState()
    client_info = MemoryState({"c1.is_active": True, "c2.is_active": True})
    client_selection_state = MemoryState({"selected_clients": ["c1", "c2"]})

    first = wrap_weights({"w": jnp.array([1.0, 1.0])}, backend="jax")
    second = wrap_weights({"w": jnp.array([3.0, 3.0])}, backend="jax")

    assert (
        aggregator.aggregate(
            "test",
            "c1",
            True,
            first,
            client_info,
            training_state,
            MemoryState(),
            aggregator_state,
            client_selection_state,
            {},
        )
        is None
    )
    result = aggregator.aggregate(
        "test",
        "c2",
        True,
        second,
        client_info,
        training_state,
        MemoryState(),
        aggregator_state,
        client_selection_state,
        {},
    )

    assert result.backend == "jax"
    np.testing.assert_array_equal(np.asarray(result.weights["w"]), np.array([2.5, 2.5]))


def test_jax_fedasync_updates_pytree_payloads():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    aggregator = load_aggregator("test", "fedasync_jax")
    training_session = MemoryState(
        {
            "test.last_round_number": 2,
            "test.global_model": wrap_weights({"w": jnp.array([1.0, 1.0])}, backend="jax"),
        }
    )
    client_selection_state = MemoryState({"c1": 1})
    result = aggregator.aggregate(
        "test",
        "c1",
        True,
        wrap_weights({"w": jnp.array([3.0, 3.0])}, backend="jax"),
        MemoryState(),
        MemoryState(),
        training_session,
        MemoryState(),
        client_selection_state,
        {"alpha": 0.0},
    )

    assert result.backend == "jax"
    np.testing.assert_array_equal(np.asarray(result.weights["w"]), np.array([3.0, 3.0]))


def test_sklearn_backend_weight_roundtrip():
    backend = get_backend("sklearn")

    class Model:
        pass

    model = Model()
    model.coef_ = np.array([[1.0, 2.0]])
    model.intercept_ = np.array([0.5])
    model.classes_ = np.array([0, 1])

    payload = backend.get_weights(model, model_id="sklinear")
    clone = Model()
    backend.set_weights(clone, payload)

    np.testing.assert_array_equal(clone.coef_, model.coef_)
    np.testing.assert_array_equal(clone.intercept_, model.intercept_)
    np.testing.assert_array_equal(clone.classes_, model.classes_)


def test_sklearn_backend_set_weights_on_wrapped_estimator():
    backend = get_backend("sklearn")

    class Wrapped:
        def __init__(self):
            from sklearn.linear_model import SGDClassifier

            self.model = SGDClassifier(loss="log_loss", max_iter=1, tol=None)

        def __getattr__(self, item):
            return getattr(self.model, item)

    wrapped = Wrapped()
    payload = wrap_weights(
        {"coef_": np.array([[0.1, 0.2]]), "intercept_": np.array([0.3]), "classes_": np.array([0, 1])},
        backend="sklearn",
    )
    backend.set_weights(wrapped, payload)

    np.testing.assert_array_equal(wrapped.model.coef_, np.array([[0.1, 0.2]]))
    np.testing.assert_array_equal(wrapped.model.intercept_, np.array([0.3]))
    np.testing.assert_array_equal(wrapped.model.classes_, np.array([0, 1]))


def test_sklearn_validate_handles_unfitted_model():
    backend = get_backend("sklearn")

    class Model:
        def predict(self, _x):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("not fit yet")

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
    y = np.array([0, 1], dtype="int64")
    metrics = backend.default_validate_classifier(Model(), [(x, y)])

    assert metrics["accuracy"] == 0
    assert metrics["loss"] == 0


def test_sklearn_fedavg_aggregates_weight_payloads():
    aggregator = load_aggregator("test", "fedavg_sklearn")
    training_state = MemoryState(
        {
            "c1.current_dataset_detail": {"metadata": {"num_items": 1}},
            "c2.current_dataset_detail": {"metadata": {"num_items": 3}},
        }
    )
    aggregator_state = MemoryState()
    client_info = MemoryState({"c1.is_active": True, "c2.is_active": True})
    client_selection_state = MemoryState({"selected_clients": ["c1", "c2"]})

    first = wrap_weights(
        {"coef_": np.array([[1.0, 1.0]]), "intercept_": np.array([1.0]), "classes_": np.array([0, 1])},
        backend="sklearn",
    )
    second = wrap_weights(
        {"coef_": np.array([[3.0, 3.0]]), "intercept_": np.array([3.0]), "classes_": np.array([0, 1])},
        backend="sklearn",
    )

    assert (
        aggregator.aggregate(
            "test",
            "c1",
            True,
            first,
            client_info,
            training_state,
            MemoryState(),
            aggregator_state,
            client_selection_state,
            {},
        )
        is None
    )
    result = aggregator.aggregate(
        "test",
        "c2",
        True,
        second,
        client_info,
        training_state,
        MemoryState(),
        aggregator_state,
        client_selection_state,
        {},
    )

    assert result.backend == "sklearn"
    np.testing.assert_array_equal(result.weights["coef_"], np.array([[2.5, 2.5]]))
    np.testing.assert_array_equal(result.weights["intercept_"], np.array([2.5]))


def test_onnx_fedavg_aggregates_native_payloads():
    aggregator = load_aggregator("test", "fedavg_onnx")
    training_state = MemoryState(
        {
            "c1.current_dataset_detail": {"metadata": {"num_items": 1}},
            "c2.current_dataset_detail": {"metadata": {"num_items": 3}},
        }
    )
    aggregator_state = MemoryState()
    client_info = MemoryState({"c1.is_active": True, "c2.is_active": True})
    client_selection_state = MemoryState({"selected_clients": ["c1", "c2"]})

    first = wrap_weights({"w": np.array([1.0, 1.0])}, backend="onnx")
    second = wrap_weights({"w": np.array([3.0, 3.0])}, backend="onnx")

    assert (
        aggregator.aggregate(
            "test",
            "c1",
            True,
            first,
            client_info,
            training_state,
            MemoryState(),
            aggregator_state,
            client_selection_state,
            {},
        )
        is None
    )
    result = aggregator.aggregate(
        "test",
        "c2",
        True,
        second,
        client_info,
        training_state,
        MemoryState(),
        aggregator_state,
        client_selection_state,
        {},
    )

    assert result.backend == "onnx"
    np.testing.assert_array_equal(result.weights["w"], np.array([2.5, 2.5]))


def test_onnx_backend_set_weights_on_wrapped_model():
    backend = get_backend("onnx")

    class Inner:
        def __init__(self):
            self._weights = {"w": np.array([1.0, 2.0])}

        def get_weights(self):
            return self._weights

        def set_weights(self, weights):
            self._weights = weights

    class Wrapped:
        def __init__(self):
            self.model = Inner()

    wrapped = Wrapped()
    payload = wrap_weights({"w": np.array([3.0, 4.0])}, backend="onnx")
    backend.set_weights(wrapped, payload)
    out = backend.get_weights(wrapped)

    np.testing.assert_array_equal(out.weights["w"], np.array([3.0, 4.0]))


def test_onnx_backend_weights_use_outer_wrapper_when_inner_has_no_weight_api():
    """Matches ONNXMLP: federated weights live on the outer object; ``.model`` is sklearn-only."""
    backend = get_backend("onnx")

    class InnerSklearnLike:
        def __init__(self):
            self.coef_ = np.array([[1.0, 2.0]], dtype=np.float64)

    class OuterTrainable:
        def __init__(self):
            self.model = InnerSklearnLike()

        def get_weights(self):
            return {"coef_": self.model.coef_.copy()}

        def set_weights(self, weights):
            self.model.coef_ = np.asarray(weights["coef_"], dtype=np.float64)

    outer = OuterTrainable()
    payload = wrap_weights({"coef_": np.array([[9.0, 10.0]])}, backend="onnx")
    backend.set_weights(outer, payload)
    out = backend.get_weights(outer)

    np.testing.assert_array_equal(out.weights["coef_"], np.array([[9.0, 10.0]]))
