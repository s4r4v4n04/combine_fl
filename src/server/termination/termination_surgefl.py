"""SurgeFL termination: stop when current_round >= max_rounds (set by lifecycle)."""


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
    max_rounds = args.get("max_rounds", 100) if args else 100
    return current_round >= max_rounds
