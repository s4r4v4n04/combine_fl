from ml_backends.weights import wrap_weights
from server.aggregation.native_weights import average_payloads, num_items, reported_clients
from utils.logger import FedLogger


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
    logger = FedLogger(id=session_id, loggername="JAX_AGGREGATOR")
    if client_active:
        aggregator_state.put(f"{client_id}.client_local_weights", client_local_weights)

    finished_clients = list(aggregator_state.keys())
    selected_clients = client_selection_state.get("selected_clients") or []
    active_clients = [
        cid for cid in selected_clients if client_info.get(f"{cid}.is_active")
    ]

    if not client_active:
        try:
            if client_id in selected_clients:
                selected_clients.remove(client_id)
            client_selection_state.put("selected_clients", selected_clients)
        except ValueError as e:
            logger.error("jax.fedavg.remove_client", str(e))
            raise

    clients_to_wait_for = [cid for cid in selected_clients if cid in active_clients]
    if len(finished_clients) == 0 or not all(
        cid in finished_clients or f"{cid}.client_local_weights" in finished_clients
        for cid in clients_to_wait_for
    ):
        return None

    try:
        payloads = []
        sample_counts = []
        for cid in reported_clients(finished_clients):
            payload = aggregator_state.get(f"{cid}.client_local_weights")
            if payload is None:
                continue
            payloads.append(payload)
            sample_counts.append(
                num_items(training_state.get(f"{cid}.current_dataset_detail"))
            )

        if not payloads:
            aggregator_state.clear()
            return None

        global_weights = average_payloads(payloads, sample_counts)
        aggregator_state.clear()
        return wrap_weights(global_weights, backend="jax")
    except Exception as e:
        aggregator_state.clear()
        logger.error("jax.fedavg.exception", str(e))
        raise
