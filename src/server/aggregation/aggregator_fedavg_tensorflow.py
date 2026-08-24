import numpy as np

from ml_backends.weights import unwrap_weights, wrap_weights
from utils.logger import FedLogger


def _num_items(dataset_detail):
    metadata = (dataset_detail or {}).get("metadata")
    if metadata is None:
        metadata = (dataset_detail or {}).get("dataset_details", {}).get("metadata")
    return (metadata or {}).get("num_items", 1)


def aggregate(
    session_id,
    client_id,
    client_active,
    client_local_weights,
    client_info,
    training_state,
    training_session,
    aggregator_state,
    client_selection_state,
    args,
):
    logger = FedLogger(id=session_id, loggername="TF_AGGREGATOR")
    if client_active:
        aggregator_state.put(f"{client_id}.client_local_weights", client_local_weights)

    finished_clients = list(aggregator_state.keys())
    selected_clients = client_selection_state.get("selected_clients") or []
    active_clients = [
        c for c in selected_clients if client_info.get(f"{c}.is_active")
    ]

    if not client_active:
        try:
            if client_id in selected_clients:
                selected_clients.remove(client_id)
            client_selection_state.put("selected_clients", selected_clients)
        except ValueError as e:
            logger.error("tensorflow.fedavg.remove_client", str(e))
            raise

    clients_to_wait_for = [c for c in selected_clients if c in active_clients]
    if len(finished_clients) == 0 or not all(
        c in finished_clients or f"{c}.client_local_weights" in finished_clients
        for c in clients_to_wait_for
    ):
        return None

    try:
        client_weights = []
        sample_counts = []
        reported_clients = [
            key.replace(".client_local_weights", "")
            for key in finished_clients
            if key.endswith(".client_local_weights")
        ] or finished_clients
        for cid in reported_clients:
            payload = aggregator_state.get(f"{cid}.client_local_weights")
            if payload is None:
                continue
            client_weights.append(unwrap_weights(payload))
            sample_counts.append(
                _num_items(training_state.get(f"{cid}.current_dataset_detail"))
            )

        if not client_weights:
            aggregator_state.clear()
            return None

        sample_weights = np.array(sample_counts, dtype=float)
        sample_weights = sample_weights / sample_weights.sum()
        global_weights = [
            np.zeros_like(weight) for weight in client_weights[0]
        ]

        for client_idx, weights in enumerate(client_weights):
            for layer_idx, layer_weights in enumerate(weights):
                global_weights[layer_idx] += layer_weights * sample_weights[client_idx]

        aggregator_state.clear()
        return wrap_weights(global_weights, backend="tensorflow")
    except Exception as e:
        aggregator_state.clear()
        logger.error("tensorflow.fedavg.exception", str(e))
        raise
