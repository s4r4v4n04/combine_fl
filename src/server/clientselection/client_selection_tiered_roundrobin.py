"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


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

    cfg = args or {}
    num_tiers = max(1, int(cfg.get("num_tiers", 1)))
    client_fraction = float(cfg.get("client_fraction", 1.0))

    benchmark_metrics = []
    for client in selectable_clients:
        model_id = training_state.get(f"{client}.current_model_id")
        bench_info = client_info.get(f"{client}.benchmark_info") or {}
        sample_count = (bench_info.get(model_id) or {}).get("num_mini_batches", 1)
        benchmark_metrics.append(float(sample_count))

    num_tiers = min(num_tiers, len(selectable_clients))
    labels = AgglomerativeClustering(n_clusters=num_tiers, metric="euclidean").fit_predict(
        np.array(benchmark_metrics).reshape(-1, 1)
    )

    tiers = {idx: [] for idx in range(num_tiers)}
    for client_id, label in zip(selectable_clients, labels):
        tiers[int(label)].append(client_id)

    next_tier = int(client_selection_state.get("tiered_roundrobin.next_tier") or 0)
    chosen_tier = next_tier % num_tiers
    chosen_clients = tiers[chosen_tier]
    num_clients = max(1, int(client_fraction * len(chosen_clients)))
    num_clients = min(num_clients, len(chosen_clients))

    selected_clients = chosen_clients[:num_clients]
    client_selection_state.put("tiered_roundrobin.next_tier", (chosen_tier + 1) % num_tiers)
    client_selection_state.put("selected_clients", selected_clients)
    return selected_clients, None
