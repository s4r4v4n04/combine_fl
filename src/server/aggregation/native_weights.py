from collections.abc import Mapping

import numpy as np

from ml_backends.weights import unwrap_weights


def _is_numeric_leaf(value):
    try:
        return np.issubdtype(np.asarray(value).dtype, np.number)
    except TypeError:
        return False


def num_items(dataset_detail):
    metadata = (dataset_detail or {}).get("metadata")
    if metadata is None:
        metadata = (dataset_detail or {}).get("dataset_details", {}).get("metadata")
    return (metadata or {}).get("num_items", 1)


def reported_clients(aggregator_state_keys):
    return [
        key.replace(".client_local_weights", "")
        for key in aggregator_state_keys
        if key.endswith(".client_local_weights")
    ] or list(aggregator_state_keys)


def weighted_average(weights, sample_weights):
    total = scale_tree(weights[0], sample_weights[0])
    for weight, sample_weight in zip(weights[1:], sample_weights[1:]):
        total = add_trees(total, scale_tree(weight, sample_weight))
    return total


def add_trees(left, right):
    if isinstance(left, Mapping):
        return type(left)((key, add_trees(left[key], right[key])) for key in left)
    if isinstance(left, list):
        return [add_trees(l, r) for l, r in zip(left, right)]
    if isinstance(left, tuple):
        return type(left)(add_trees(l, r) for l, r in zip(left, right))
    if _is_numeric_leaf(left) and _is_numeric_leaf(right):
        return left + right
    return right


def scale_tree(value, scalar):
    if isinstance(value, Mapping):
        return type(value)((key, scale_tree(child, scalar)) for key, child in value.items())
    if isinstance(value, list):
        return [scale_tree(child, scalar) for child in value]
    if isinstance(value, tuple):
        return type(value)(scale_tree(child, scalar) for child in value)
    if _is_numeric_leaf(value):
        return value * scalar
    return value


def average_payloads(client_payloads, sample_counts):
    weights = [unwrap_weights(payload) for payload in client_payloads]
    sample_weights = np.asarray(sample_counts, dtype=float)
    sample_weights = sample_weights / sample_weights.sum()
    return weighted_average(weights, sample_weights)
