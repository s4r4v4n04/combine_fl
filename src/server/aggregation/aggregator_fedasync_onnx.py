from ml_backends.weights import unwrap_weights, wrap_weights
from server.aggregation.native_weights import add_trees, scale_tree


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

    alpha = args["alpha"]
    model_version = client_selection_state.get(f"{client_id}")
    current_round = training_session.get(f"{session_id}.last_round_number")
    alpha_t = pow((current_round - model_version + 1), (-alpha))

    global_model = unwrap_weights(training_session.get(f"{session_id}.global_model"))
    client_model = unwrap_weights(client_local_weights)
    aggregated = add_trees(
        scale_tree(global_model, 1 - alpha_t),
        scale_tree(client_model, alpha_t),
    )

    client_selection_state.deletebykey(f"{client_id}")
    return wrap_weights(aggregated, backend="onnx")
