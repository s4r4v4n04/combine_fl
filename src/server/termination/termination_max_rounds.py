"""
Default termination condition: stop when current_round >= max_rounds.
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
    max_rounds = args.get("max_rounds") if args else None
    if max_rounds is None:
        raise ValueError(
            "The 'max_rounds' termination condition requires "
            "'termination_condition_args.max_rounds'."
        )

    return current_round >= max_rounds
