"""
Termination condition: stop when global validation loss (or another metric
to minimize) has not improved for a configurable patience window.

Args:
  patience: int, number of rounds without improvement before stopping.
  min_delta: float, minimum improvement to count as "improved" (default 0.001).
  metric_key: optional, key in global_validation_metrics (default "loss").
  min_rounds: optional int; do not stop before this many rounds (default 0).
  check_interval: optional; see session_config.termination_condition_args.
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
    min_rounds = args.get("min_rounds", 0) if args else 0
    if current_round < min_rounds:
        return False

    patience = args.get("patience", 5) if args else 5
    min_delta = args.get("min_delta", 0.001) if args else 0.001
    metric_key = args.get("metric_key", "loss") if args else "loss"

    global_metrics = training_session.get(f"{session_id}.global_validation_metrics")
    if not global_metrics or metric_key not in global_metrics:
        return False

    metric_history = global_metrics[metric_key]
    if len(metric_history) < patience + 1:
        return False

    best_in_window = min(metric_history[-(patience + 1) : -1])
    latest = metric_history[-1]

    return latest >= best_in_window - min_delta
