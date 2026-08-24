"""
Termination condition: stop when global validation loss falls at or below
a threshold. Useful for "train until loss < 0.1".

Args:
  loss_threshold: float, stop when global validation loss <= this value.
  metric_key: optional, key in global_validation_metrics (default "loss").
"""


def should_terminate(
    session_id,
    current_round,
    training_session,
    training_state,
    client_info,
    aggregator_state,
    client_selection_state,
    args,
):
    if current_round == 0:
        return False

    threshold = args.get("loss_threshold") if args else None
    if threshold is None:
        raise ValueError(
            "The 'loss_threshold' termination condition requires "
            "'termination_condition_args.loss_threshold'."
        )

    metric_key = args.get("metric_key", "loss") if args else "loss"

    global_metrics = training_session.get(f"{session_id}.global_validation_metrics")
    if not global_metrics or metric_key not in global_metrics:
        return False

    history = global_metrics[metric_key]
    if not history:
        return False

    return history[-1] <= threshold
