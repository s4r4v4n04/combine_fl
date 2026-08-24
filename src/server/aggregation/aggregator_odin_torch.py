"""
ODIN aggregator: FedAvg weighted by client replay buffer sizes.

Ported from odin/src_odin/server_odin.py Aggregation.aggregate_fedavg()
"""

from collections import OrderedDict

import numpy as np
from torch import zeros

from ml_backends.weights import get_payload_backend, unwrap_weights, wrap_weights
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
    logger = FedLogger(id=session_id, loggername="AGGREGATION_ODIN")

    if client_active:
        aggregator_state.put(f"{client_id}.client_local_weights", client_local_weights)

        # store per-client sample count for weighted averaging
        metrics = training_state.get(f"{client_id}.training_metrics")
        if metrics:
            latest_round = max(metrics.keys())
            latest = metrics[latest_round]
            num_samples = latest.get("odin_num_samples", 0)
            aggregator_state.put(f"{client_id}.num_samples", num_samples)

    finished_clients = list(aggregator_state.keys())

    active_clients = [
        c for c in client_info.keys() if client_info.get(f"{c}.is_active")
    ]

    selected_clients = client_selection_state.get("selected_clients")

    if not client_active:
        try:
            if client_id in selected_clients:
                selected_clients.remove(client_id)
            client_selection_state.put("selected_clients", selected_clients)
        except (ValueError, TypeError) as e:
            logger.error("odin.agg.remove_client", str(e))

    clients_to_wait_for = [c for c in (selected_clients or []) if c in active_clients]

    # only aggregate when all selected clients have reported
    # filter finished_clients to only include actual weight entries
    weight_clients = [
        c.replace(".client_local_weights", "")
        for c in finished_clients
        if c.endswith(".client_local_weights")
    ]
    # simpler check: just look for keys that match expected client IDs
    reported_clients = [c for c in clients_to_wait_for if f"{c}.client_local_weights" in finished_clients]

    if len(finished_clients) > 0 and all(
        c in finished_clients for c in clients_to_wait_for
    ):
        try:
            logger.info("odin.agg.start", f"clients={finished_clients}")

            # collect weights and sample counts
            client_weights_list = []
            sample_counts = []
            participating_clients = []

            for cid in finished_clients:
                payload = aggregator_state.get(f"{cid}.client_local_weights")
                if payload is None:
                    continue
                weights = unwrap_weights(payload)
                client_weights_list.append(weights)
                n_samples = aggregator_state.get(f"{cid}.num_samples")
                sample_counts.append(n_samples if n_samples and n_samples > 0 else 1)
                participating_clients.append(cid)

            if not client_weights_list:
                aggregator_state.clear()
                return None

            # initialize global model structure
            global_model = OrderedDict()
            temp_model = client_weights_list[0]
            backend = get_payload_backend(
                aggregator_state.get(f"{participating_clients[0]}.client_local_weights")
            )
            for layer in temp_model:
                shape = temp_model[layer].shape
                global_model[layer] = zeros(shape)

            # weighted averaging (FedAvg weighted by replay buffer sizes)
            total_samples = sum(sample_counts)
            if total_samples == 0:
                # uniform averaging fallback
                n_weights = np.ones(len(client_weights_list)) / len(client_weights_list)
            else:
                n_weights = np.array(sample_counts, dtype=float) / total_samples

            for i, w in enumerate(client_weights_list):
                for layer in w.keys():
                    if global_model[layer].shape == w[layer].shape:
                        global_model[layer] += w[layer].float() * n_weights[i]
                    else:
                        logger.warn(
                            "odin.agg.shape_mismatch",
                            f"layer={layer},global={global_model[layer].shape},local={w[layer].shape}",
                        )

            aggregator_state.clear()
            logger.info(
                "odin.agg.done",
                f"num_clients={len(participating_clients)},total_samples={total_samples}",
            )
            return wrap_weights(global_model, backend=backend)

        except Exception as e:
            aggregator_state.clear()
            logger.error("odin.agg.exception", str(e))
            raise
    else:
        return None
