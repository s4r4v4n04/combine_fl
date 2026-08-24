from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from ml_backends.base import BackendAdapter


_BACKENDS: dict[str, type[BackendAdapter]] = {}


def _normalise(name: str | None) -> str:
    return (name or "pytorch").lower().replace("-", "_")


def register_backend(name: str, adapter_cls: type[BackendAdapter]) -> None:
    _BACKENDS[_normalise(name)] = adapter_cls


def _ensure_defaults_registered() -> None:
    if "pytorch" not in _BACKENDS:
        from ml_backends.pytorch_backend import PyTorchBackend

        register_backend("pytorch", PyTorchBackend)
        register_backend("torch", PyTorchBackend)
    if "tensorflow" not in _BACKENDS:
        from ml_backends.tensorflow_backend import TensorFlowBackend

        register_backend("tensorflow", TensorFlowBackend)
        register_backend("tf", TensorFlowBackend)
    if "jax" not in _BACKENDS:
        from ml_backends.jax_backend import JAXBackend

        register_backend("jax", JAXBackend)
    if "sklearn" not in _BACKENDS:
        from ml_backends.sklearn_backend import SklearnBackend

        register_backend("sklearn", SklearnBackend)
        register_backend("scikit_learn", SklearnBackend)
    if "onnx" not in _BACKENDS:
        from ml_backends.onnx_backend import ONNXBackend

        register_backend("onnx", ONNXBackend)


@lru_cache(maxsize=None)
def get_backend(name: str | None = None) -> BackendAdapter:
    _ensure_defaults_registered()
    backend_name = _normalise(name)
    if backend_name not in _BACKENDS:
        raise ValueError(f"Unsupported ML backend: {name}")
    return _BACKENDS[backend_name]()


def get_backend_name(config: dict | None = None, model_dir: str | None = None) -> str:
    """Resolve backend from training config or model config, defaulting to PyTorch."""

    config = config or {}
    if "backend" in config:
        return _normalise(config["backend"])

    model_details = config.get("model_details")
    if isinstance(model_details, dict) and "backend" in model_details:
        return _normalise(model_details["backend"])

    if model_dir:
        try:
            with open(f"{model_dir}/config.yaml", "r") as file:
                model_config = yaml.safe_load(file) or {}
            details = model_config.get("model_details", {})
            return _normalise(details.get("backend"))
        except FileNotFoundError:
            return "pytorch"

    return "pytorch"


def get_backend_from_payload(payload: Any, fallback: str = "pytorch") -> BackendAdapter:
    backend_name = getattr(payload, "backend", fallback)
    return get_backend(backend_name)
