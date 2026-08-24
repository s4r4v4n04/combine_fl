from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightPayload:
    """Backend-tagged model weights used across transport and aggregation."""

    backend: str
    weights: Any
    model_id: str | None = None
    metadata: dict = field(default_factory=dict)


def is_weight_payload(value: Any) -> bool:
    return isinstance(value, WeightPayload)


def unwrap_weights(value: Any) -> Any:
    if isinstance(value, WeightPayload):
        return value.weights
    return value


def get_payload_backend(value: Any, default: str = "pytorch") -> str:
    if isinstance(value, WeightPayload):
        return value.backend
    return default


def wrap_weights(
    weights: Any,
    backend: str = "pytorch",
    model_id: str | None = None,
    metadata: dict | None = None,
) -> WeightPayload:
    if isinstance(weights, WeightPayload):
        return weights
    return WeightPayload(
        backend=backend,
        weights=weights,
        model_id=model_id,
        metadata=metadata or {},
    )
