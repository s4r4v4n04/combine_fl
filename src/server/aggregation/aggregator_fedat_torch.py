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
    logger = FedLogger(id=session_id, loggername="AGGREGATOR_FEDAT")
    backend = get_payload_backend(client_local_weights)
    
    def get_global_model(num_tiers):
        global_model = OrderedDict()

        tier_weights = unwrap_weights(aggregator_state.get(f"tier_model_tier_0"))

        for layer in tier_weights:
            shape = tier_weights[layer].shape
            global_model[layer] = zeros(shape)

        T_k = []
        for tier in range(num_tiers):
            T_k.append(aggregator_state.get(f"update_count_tier_{tier}"))

        sorted_index = np.argsort([-x for x in T_k])

        tier_wts = np.array(
            [T_k[sorted_index[len(T_k) - 1 - i]] for i in range(len(T_k))]
        ) / sum(T_k)

        print("TIER WEIGHTS = ", tier_wts)

        for tier in range(num_tiers):
            model_wts = unwrap_weights(aggregator_state.get(f"tier_model_tier_{tier}"))

            for layer in model_wts.keys():
                # fmt: off
                global_model[layer] += (model_wts[layer] * tier_wts[tier])
                # fmt: on

        return global_model

    aggregator_state.put(f"clientweights_{client_id}", client_local_weights)

    client_to_tier_dict = client_selection_state.get("client_to_tier_id_dict")
    num_tiers = len(np.unique(list(client_to_tier_dict.values())))

    tier = client_to_tier_dict[client_id]

    selected_clients_in_tier = client_selection_state.get(
        f"selected_clients_tier_{tier}"
    )

    print("-------------------------- Aggregrator -----------------------")
    client_id_recv_weights = [
        c for c in aggregator_state.keys() if "clientweights" in c
    ]
    print("CLIENTS WHO HAVE RETURNED - ", client_id_recv_weights)
    print(f"SELECTED CLIENTS IN TIER {tier}", selected_clients_in_tier)

    if all(
        f"clientweights_{c}" in client_id_recv_weights for c in selected_clients_in_tier
    ):
        tier_model = OrderedDict()
        client_local_weights = unwrap_weights(
            aggregator_state.get(f"clientweights_{client_id}")
        )

        for layer in client_local_weights:
            shape = client_local_weights[layer].shape
            tier_model[layer] = zeros(shape)

        client_weights = list()

        N_k = np.array(list())
        for client_id in selected_clients_in_tier:
            try:
                N_k = np.append(
                    N_k,
                    training_state.get(f"{client_id}.current_dataset_detail")[
                        "metadata"
                    ]["num_items"],
                )
            except (KeyError, TypeError) as e:
                print(f"aggregator_fedat.exception:: {e}")
                print(f"CLIENT_ID DATA = {client_id}")
                logger.error("aggregator_fedat.exception", f"CLIENT_ID DATA = {client_id}, {e}")
            except Exception as e:
                print(f"aggregator_fedat.unknown_exception:: {e}")
                print(f"CLIENT_ID DATA = {client_id}")
                logger.error("aggregator_fedat.unknown_exception", f"CLIENT_ID DATA = {client_id}, {e}")
            client_weights.append(
                unwrap_weights(aggregator_state.get(f"clientweights_{client_id}"))
            )

        N_k = N_k / sum(N_k)
        for i, weights in enumerate(client_weights):
            for layer in weights.keys():
                # fmt: off
                tier_model[layer] += (weights[layer] * N_k[i])
                # fmt: on

        tier_count = aggregator_state.get(f"update_count_tier_{tier}")

        aggregator_state.put(f"update_count_tier_{tier}", tier_count + 1)
        aggregator_state.put(
            f"tier_model_tier_{tier}", wrap_weights(tier_model, backend=backend)
        )

        global_model = get_global_model(num_tiers)

        for c in selected_clients_in_tier:
            aggregator_state.deletebykey(f"clientweights_{c}")

        client_selection_state.put(f"selected_clients_tier_{tier}", [])

        print("MODEL AGGREGATED\n\n")
        print("AGGREGATION-END-STATE = ", aggregator_state.keys())
        return wrap_weights(global_model, backend=backend)

    else:
        print("AGGREGATION-END-STATE = ", aggregator_state.keys())
        return None
