"""
Termination condition: stop when global validation accuracy has not improved
for a configurable number of rounds (early stopping on accuracy plateau).

Args:
  patience: int, number of rounds without improvement before stopping.
  min_delta: float, minimum improvement to count as "improved" (default 0.0).
  metric_key: optional, key in global_validation_metrics (default "accuracy").
  min_rounds: optional int; do not stop before this many rounds (default 0).
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
    min_delta = args.get("min_delta", 0.0) if args else 0.0
    metric_key = args.get("metric_key", "accuracy") if args else "accuracy"

    global_metrics = training_session.get(f"{session_id}.global_validation_metrics")
    if not global_metrics or metric_key not in global_metrics:
        return False

    history = global_metrics[metric_key]
    if len(history) < patience + 1:
        return False

    best_in_window = max(history[-(patience + 1) : -1])
    latest = history[-1]

    return latest <= best_in_window + min_delta
