"""
Termination condition: stop when global validation accuracy reaches or exceeds
a target (e.g. 95%).

Args:
  target_accuracy: float in [0, 100] (same scale as default validator).
  metric_key: optional, key in global_validation_metrics (default "accuracy").
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

    target = args.get("target_accuracy") if args else None
    if target is None:
        raise ValueError(
            "The 'accuracy_target' termination condition requires "
            "'termination_condition_args.target_accuracy'."
        )

    metric_key = args.get("metric_key", "accuracy") if args else "accuracy"

    global_metrics = training_session.get(f"{session_id}.global_validation_metrics")
    if not global_metrics or metric_key not in global_metrics:
        return False

    history = global_metrics[metric_key]
    if not history:
        return False

    return history[-1] >= target
