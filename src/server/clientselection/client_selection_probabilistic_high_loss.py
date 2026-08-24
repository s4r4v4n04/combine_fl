"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math
import numpy as np


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

    client_fraction = (args or {}).get("client_fraction", 1.0)
    num_clients = max(1, math.floor(len(selectable_clients) * client_fraction))
    num_clients = min(num_clients, len(selectable_clients))

    losses = []
    for client in selectable_clients:
        training_metrics = training_state.get(f"{client}.training_metrics") or {}
        if training_metrics:
            latest_round = max(training_metrics.keys())
            losses.append(float(training_metrics[latest_round].get("loss", 0.0)))
        else:
            losses.append(0.0)

    loss_sum = float(sum(losses))
    if loss_sum <= 0:
        probabilities = None
    else:
        probabilities = [loss / loss_sum for loss in losses]

    selected_clients = np.random.choice(
        a=selectable_clients,
        p=probabilities,
        size=num_clients,
        replace=False,
    ).tolist()
    client_selection_state.put("selected_clients", selected_clients)
    return selected_clients, None
