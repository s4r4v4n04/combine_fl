"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""


from ml_backends.weights import get_payload_backend, unwrap_weights, wrap_weights


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
    if not client_active:
        client_selection_state.deletebykey(f"{client_id}")
        return None

    def get_alpha_t(t, tau, alpha):
        print("t:", t, "tau:", tau, "alpha:", alpha)
        alpha_t = pow((t - tau + 1), (-alpha))
        return alpha_t

    alpha = args["alpha"]
    model_version = client_selection_state.get(f"{client_id}")
    current_round = training_session.get(f"{session_id}.last_round_number")
    print(
        f"AGGREGATOR:: Round {current_round} client_id {client_id} reported in with model_version {model_version}"
    )

    alpha_t = get_alpha_t(current_round, model_version, alpha)

    backend = get_payload_backend(client_local_weights)
    global_model = unwrap_weights(training_session.get(f"{session_id}.global_model"))
    client_local_weights = unwrap_weights(client_local_weights)

    for i, layer in enumerate(global_model.keys()):
        # fmt: off
        global_model[layer] = (((1 - alpha_t) * global_model[
            layer
        ]) + (alpha_t * client_local_weights[layer]))
        # fmt: on

    client_selection_state.deletebykey(f"{client_id}")
    print("CLIENT_SELECTION_STATE.KEYS = ", client_selection_state.keys())

    return wrap_weights(global_model, backend=backend)
