"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math
import random

def client_selection(
    selectable_clients: list,
    session_id: str,
    client_info: dict,
    training_state: dict,
    training_session: dict,
    aggregate_state: dict,
    client_selection_state: dict,
    args: dict = None,
):
    if not selectable_clients:
        return None, None

    round_no = training_session.get(f"{session_id}.last_round_number") or 0
    client_fraction = (args or {}).get("client_fraction", 1.0)
    num_clients = max(1, math.floor(len(selectable_clients) * client_fraction))

    parity = round_no % 2
    parity_candidates = [
        c for c in selectable_clients if (hash(str(c)) % 2) == parity
    ]
    candidate_pool = parity_candidates or list(selectable_clients)
    num_clients = min(num_clients, len(candidate_pool))

    selected_clients = random.sample(candidate_pool, num_clients)
    client_selection_state.put("selected_clients", selected_clients)
    return selected_clients, None
