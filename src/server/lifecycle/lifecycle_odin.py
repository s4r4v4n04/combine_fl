"""
Odin Lifecycle Module — Full Implementation

Orchestrates the continuous Odin federated learning loop:
  Phase 1: Data Ingestion & Local Discovery (edge inference via gRPC)
  Phase 2: Global Prototype Synchronization (5-Phase Merge-and-Discover)
  Phase 3: Conditional Federated Training (dynamic trigger)

Ported from: odin/src_odin/run_exp_odin.py main() + server_odin.py Server
"""

import asyncio
import math
import pickle
from copy import deepcopy

import numpy as np
from sklearn.cluster import DBSCAN

from utils.logger import FedLogger

# configuration (defaults; override via training_session odin_config)
TOTAL_TIMEFRAMES = 50
PROTO_SYNC_THRESHOLD = 0.3        # fraction of clients needed for sync
MIN_DATA_FOR_TRAIN = 500
FL_ROUNDS_DISCOVERY = 3
FL_ROUNDS_REFINEMENT = 1

# server-side prototype defaults
MIN_EPSILON = 5.0
CALIBRATION_STD_MULT = 1.6
SINGLETON_THRESHOLD = 300
SERVER_DBSCAN_MIN_SAMPLES = 3



##### ------ ENTRY POINT — called by FloSessionManager.start_session() ------ #####
async def run(session_manager):
    sm = session_manager
    logger = FedLogger(id=sm.id, loggername="LIFECYCLE_ODIN")

    # load config from training_session if available 
    odin_cfg = sm.training_session.get(f"{sm.id}.odin_config") or {}
    total_timeframes = odin_cfg.get("max_time_frames", TOTAL_TIMEFRAMES)
    proto_sync_frac = odin_cfg.get("prototype_update_threshold", PROTO_SYNC_THRESHOLD)
    fl_rounds_discovery = odin_cfg.get("num_fl_rounds_per_discovery_trigger", FL_ROUNDS_DISCOVERY)
    fl_rounds_refinement = odin_cfg.get("num_fl_rounds_per_refinement_trigger", FL_ROUNDS_REFINEMENT)
    min_epsilon = odin_cfg.get("min_epsilon", MIN_EPSILON)
    cal_std_mult = odin_cfg.get("calibration_std_multiplier", CALIBRATION_STD_MULT)
    singleton_thresh = odin_cfg.get("singleton_threshold", SINGLETON_THRESHOLD)
    dbscan_min_samples = odin_cfg.get("server_dbscan_min_samples", SERVER_DBSCAN_MIN_SAMPLES)

    # dynamic training trigger defaults
    trigger_cfg = odin_cfg.get("train_trigger_args", {})
    tau_decay = trigger_cfg.get("tau_decay", 20.0)
    delta_0 = trigger_cfg.get("delta_0", 0.1)
    k_smooth = trigger_cfg.get("smoothness_factor", 50.0)
    min_safe_batch = trigger_cfg.get("min_safe_batch_size", 64)
    moving_avg_alpha = trigger_cfg.get("moving_avg_alpha", 0.5)
    frac_clients_train = trigger_cfg.get("frac_clients_required", 0.5)
    ts_between_triggers = trigger_cfg.get("timesteps_between_triggers", 3)

    # identify edge clients 
    edge_clients = sm.get_active_clients()
    print(f"LIFECYCLE_ODIN: Found {len(edge_clients)} edge clients.")
    logger.info("odin.lifecycle.start", f"edges={len(edge_clients)},timeframes={total_timeframes}")

    # initialize global prototype state 
    sm.training_session.put("global_prototypes", {})
    sm.training_session.put("global_radii", {})
    sm.training_session.put("global_counts", {})
    sm.training_session.put("subproto_to_class", {})
    sm.training_session.put("prototype_assignments", {})

    # tracking variables
    rounds_since_discovery = 1
    avg_data_per_tf = None
    last_weight_div = 1.0
    prev_global_weights = None

    ## ----- CONTINUOUS LEARNING TIMELINE ----- ##
    for timeframe in range(total_timeframes):
        print(f"LIFECYCLE_ODIN: Starting timeframe {timeframe}")
        sm.training_state.put("_odin_meta.timeframe", timeframe)
        logger.info("odin.timeframe.start", f"timeframe={timeframe}")

        active_clients = sm.get_active_clients()
        if not active_clients:
            logger.info("odin.no_clients", "No active clients, stopping")
            break

        ### -- PHASE 1: DATA INGESTION & LOCAL DISCOVERY -- ##
        print(f"LIFECYCLE_ODIN: Gathering state for timeframe {timeframe}")
        global_protos = sm.training_session.get("global_prototypes") or {}
        global_radii = sm.training_session.get("global_radii") or {}
        global_counts = sm.training_session.get("global_counts") or {}
        proto_assignments = sm.training_session.get("prototype_assignments") or {}

        print(f"LIFECYCLE_ODIN: Extracting model weights...")
        print(f"LIFECYCLE_ODIN: Building payload...")
        payload = {
            "global_prototypes": global_protos,
            "global_radii": global_radii,
            "global_counts": global_counts,
            "prototype_assignments": proto_assignments,
            "timeframe": timeframe,
        }

        # run multiple async tasks at the same time and wait for all to finish
        inference_results = await asyncio.gather(
            *(
                sm.async_grpc_inference(
                    edge_id=cid,
                    model_wts=payload,
                    timestep=timeframe,
                )
                for cid in active_clients
            )
        )

        # process all client's results
        clients_ready = []
        total_data_since_train = 0
        data_this_tf = 0

        for i, result in enumerate(inference_results):
            if result is None:
                continue
            cid = active_clients[i]

            ready = result.get("ready_for_prototype_sync", False)
            if ready:
                clients_ready.append(cid)

            class_dist = result.get("class_distribution", {})
            sm.training_state.put(f"{cid}.odin_class_distribution", class_dist)

            data_vol = result.get("data_since_last_train", 0)
            sm.training_state.put(f"{cid}.odin_data_since_train", data_vol)
            total_data_since_train += data_vol

            tf_data = result.get("current_timeframe_data_volume", 0)
            data_this_tf += tf_data

            # store local prototypes for phase 2
            local_known = result.get("local_known_prototypes", {})
            local_unknown = result.get("local_unknown_prototypes", {})
            local_radii = result.get("local_radii", {})
            local_counts = result.get("local_counts", {})
            sm.training_state.put(f"{cid}.odin_local_known_protos", local_known)
            sm.training_state.put(f"{cid}.odin_local_unknown_protos", local_unknown)
            sm.training_state.put(f"{cid}.odin_local_radii", local_radii)
            sm.training_state.put(f"{cid}.odin_local_counts", local_counts)

            # store train data for phase 3
            train_data = result.get("odin_train_data", [])
            sm.training_state.put(f"{cid}.odin_train_data", train_data)
            class_names = result.get("odin_class_names", [])
            sm.training_state.put(f"{cid}.odin_class_names", class_names)

        logger.info(
            "odin.inference.done",
            f"timeframe={timeframe},ready={len(clients_ready)}/{len(active_clients)},data_tf={data_this_tf}",
        )

        # tracks time since last new class
        rounds_since_discovery += 1

        # total new data from all clients
        if avg_data_per_tf is None:
            avg_data_per_tf = data_this_tf
        else:
            avg_data_per_tf = moving_avg_alpha * data_this_tf + (1 - moving_avg_alpha) * avg_data_per_tf
        logger.info("odin.data.stats", f"[DATA COLLECTED CURRENT TIMEFRAME]:timeframe={timeframe}:data_collected_current_timeframe={data_this_tf}:avg_data_collected_current_timeframe={avg_data_per_tf}")

        ### -- PHASE 2: GLOBAL PROTOTYPE SYNCHRONIZATION -- ###
        required_count = max(1, int(len(active_clients) * proto_sync_frac))
        new_class_discovered = False

        if len(clients_ready) >= required_count:
            new_class_discovered = _server_merge_and_discover(
                sm=sm,
                logger=logger,
                active_clients=active_clients,
                clients_ready=clients_ready,
                timeframe=timeframe,
                min_epsilon=min_epsilon,
                cal_std_mult=cal_std_mult,
                singleton_thresh=singleton_thresh,
                dbscan_min_samples=dbscan_min_samples,
            )

        ### -- PHASE 3: CONDITIONAL FEDERATED TRAINING -- ###
        trigger_threshold = _calculate_training_trigger(rounds_since_discovery, 
                                                        last_weight_div, 
                                                        tau_decay, 
                                                        delta_0,
                                                        k_smooth, 
                                                        avg_data_per_tf or 0, 
                                                        min_safe_batch * frac_clients_train,
                                                        ts_between_triggers,)

        logger.info(
            "odin.trigger.status",
            f"timeframe={timeframe},data={total_data_since_train},threshold={trigger_threshold},new_class={new_class_discovered}",
        )

        should_train = new_class_discovered or total_data_since_train >= trigger_threshold
        if not should_train:
            logger.info("odin.train.skipped", f"[TRAINING SKIPPED]:timeframe={timeframe}:waiting_for_client_initialization")
            continue

        # determine number of FL rounds
        if new_class_discovered:
            rounds_since_discovery = 0
            num_rounds = fl_rounds_discovery
        else:
            num_rounds = fl_rounds_refinement

        logger.info("odin.train.trigger", f"timeframe={timeframe},rounds={num_rounds},reason={'discovery' if new_class_discovered else 'data_volume'}")

        # store global class names for all clients
        global_protos = sm.training_session.get("global_prototypes") or {}
        global_class_names = sorted(global_protos.keys())
        print("----------------  global_class_names ----------------", global_class_names)

        # inject training metadata for each client
        for cid in active_clients:
            train_data = sm.training_state.get(f"{cid}.odin_train_data") or []
            metadata = {
                "odin_train_data": train_data,
                "odin_class_names": global_class_names,
                "odin_data_sampling_strategy": odin_cfg.get("data_sampling_strategy", "all"),
                "odin_per_class_buffer_size": odin_cfg.get("per_class_train_data_buffer_size", 50),
            }
            sm.client_selection_state.put(f"{cid}.training_metadata", metadata)

        # save pre-training weights for divergence calculation
        prev_global_weights = deepcopy(sm.model_util.get_model_weights())

        # run FL rounds
        sm.termination_condition_args = {"max_rounds": num_rounds}
        sm.training_session.put(f"{sm.id}.last_round_number", 0)

        await sm.train()

        # calculate weight divergence
        new_weights = sm.model_util.get_model_weights()
        last_weight_div = _weight_divergence(prev_global_weights, new_weights)

        # reset per-client data counters
        for cid in active_clients:
            sm.training_state.put(f"{cid}.odin_data_since_train", 0)

        logger.info("odin.train.done", f"timeframe={timeframe},divergence={last_weight_div:.4f}")

    logger.info("odin.lifecycle.done", f"total_timeframes={total_timeframes}")


## ---- 5-PHASE MERGE-AND-DISCOVER ---- ##
def _server_merge_and_discover(
    sm, logger, active_clients, clients_ready, timeframe,
    min_epsilon, cal_std_mult, singleton_thresh, dbscan_min_samples,
):
    """
        Execute the 5-Phase Merge-and-Discover algorithm on the server.
    """
    global_protos = sm.training_session.get("global_prototypes") or {}
    global_radii = sm.training_session.get("global_radii") or {}
    global_counts = sm.training_session.get("global_counts") or {}

    new_class_added = False
    proto_assignments = {}

    # PHASE 1: Ingestion & Bucketizing
    buckets = {}
    unknown_pool = []

    valid_keys = [str(k) for k in global_protos.keys() if not str(k).startswith("unk_")]
    for gid in valid_keys:
        buckets[gid] = []

    for cid in clients_ready:
        c_known = sm.training_state.get(f"{cid}.odin_local_known_protos") or {}
        c_unknown = sm.training_state.get(f"{cid}.odin_local_unknown_protos") or {}
        c_radii = sm.training_state.get(f"{cid}.odin_local_radii") or {}
        c_counts = sm.training_state.get(f"{cid}.odin_local_counts") or {}

        c_protos = {**c_known, **c_unknown}
        c_all_radii = {**c_radii}

        for key, mu in c_protos.items():
            if key not in c_all_radii:
                continue
            mu_flat = np.array(mu).flatten()
            packet = {
                "mu": mu_flat,
                "radius": float(c_all_radii[key]),
                "n": c_counts.get(key, 1),
                "cid": cid,
                "pkey": key,
            }
            str_key = str(key)
            is_known = not str_key.startswith("unk_")
            if is_known:
                if str_key not in buckets:
                    buckets[str_key] = []
                buckets[str_key].append(packet)
                _record_assignment(proto_assignments, cid, key, str_key)
            else:
                unknown_pool.append(packet)

    # PHASE 2: Calibration
    cal_dists = []
    for gid, packets in buckets.items():
        mus = [p["mu"] for p in packets if "cid" in p]
        if len(mus) >= 2:
            for i in range(len(mus)):
                for j in range(i + 1, len(mus)):
                    cal_dists.append(np.linalg.norm(mus[i] - mus[j]))

    if cal_dists:
        unified_eps = np.mean(cal_dists) + cal_std_mult * np.std(cal_dists)
        unified_eps = max(unified_eps, min_epsilon)
    else:
        unified_eps = min_epsilon

    logger.info("odin.proto.calibration", f"timeframe={timeframe},eps={unified_eps:.4f}")

    # PHASE 3: Absorption
    remaining_unknowns = []
    for unk in unknown_pool:
        best_dist, best_gid = float("inf"), None
        for gid, packets in buckets.items():
            if gid in global_protos:
                target_mu = np.array(global_protos[gid]).flatten()
            elif packets:
                target_mu = _quick_aggregate(packets)[0]
                if target_mu is None:
                    continue
            else:
                continue
            dist = np.linalg.norm(unk["mu"] - target_mu)
            if dist < unified_eps and dist < best_dist:
                best_dist, best_gid = dist, gid

        if best_gid is not None:
            buckets[best_gid].append(unk)
            _record_assignment(proto_assignments, unk["cid"], unk["pkey"], best_gid)
        else:
            remaining_unknowns.append(unk)

    # PHASE 4: Discovery
    if remaining_unknowns:
        N = len(remaining_unknowns)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(remaining_unknowns[i]["mu"] - remaining_unknowns[j]["mu"])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        clustering = DBSCAN(eps=unified_eps, min_samples=dbscan_min_samples, metric="precomputed").fit(dist_matrix)
        labels = clustering.labels_

        all_ids = set()
        for k in global_protos:
            if str(k).isdigit():
                all_ids.add(int(k))
        for k in buckets:
            if str(k).isdigit():
                all_ids.add(int(k))
        next_gid = max(all_ids) + 1 if all_ids else 0

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            indices = np.where(labels == cluster_id)[0]
            cluster_packets = [remaining_unknowns[idx] for idx in indices]
            cluster_mus = np.vstack([p["mu"] for p in cluster_packets])
            centroid = cluster_mus.mean(axis=0)

            # Redundancy check
            is_redundant = False
            for gid in global_protos:
                existing = np.array(global_protos[gid]).flatten()
                if np.linalg.norm(centroid - existing) < unified_eps:
                    is_redundant = True
                    target = str(gid)
                    if target not in buckets:
                        buckets[target] = []
                    for pkt in cluster_packets:
                        buckets[target].append(pkt)
                        _record_assignment(proto_assignments, pkt["cid"], pkt["pkey"], target)
                    break

            if not is_redundant:
                new_gid = str(next_gid)
                buckets[new_gid] = list(cluster_packets)
                for pkt in cluster_packets:
                    _record_assignment(proto_assignments, pkt["cid"], pkt["pkey"], new_gid)
                next_gid += 1
                new_class_added = True
                logger.info("odin.proto.discovery", f"timeframe={timeframe},new_class={new_gid}")

        # Singletons
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            pkt = remaining_unknowns[idx]
            if pkt["n"] >= singleton_thresh:
                new_gid = str(next_gid)
                buckets[new_gid] = [pkt]
                _record_assignment(proto_assignments, pkt["cid"], pkt["pkey"], new_gid)
                next_gid += 1
                new_class_added = True

    # PHASE 5: Final Aggregation
    new_protos, new_radii, new_counts = {}, {}, {}
    for gid, packets in buckets.items():
        if not packets:
            if gid in global_protos:
                new_protos[gid] = global_protos[gid]
                new_radii[gid] = global_radii.get(gid, 0.5)
                new_counts[gid] = global_counts.get(gid, 0)
            continue
        mu, rad, n = _quick_aggregate(packets)
        if mu is not None:
            new_protos[gid] = mu
            new_radii[gid] = rad
            new_counts[gid] = n

    sm.training_session.put("global_prototypes", new_protos)
    sm.training_session.put("global_radii", new_radii)
    sm.training_session.put("global_counts", new_counts)
    sm.training_session.put("prototype_assignments", proto_assignments)

    logger.info("odin.proto.sync.done", f"timeframe={timeframe},classes={len(new_protos)},new_class={new_class_added}")
    return new_class_added



## ----- HELPERS FUNCTIONS ----- ##
def _quick_aggregate(packets):
    if not packets:
        return None, None, 0
    first_shape = packets[0]["mu"].shape
    valid_mus, valid_weights, valid_radii = [], [], []
    for p in packets:
        if p["mu"].shape != first_shape or not np.isfinite(p["mu"]).all():
            continue
        valid_mus.append(p["mu"])
        valid_weights.append(p["n"])
        valid_radii.append(p["radius"])
    if not valid_mus:
        return None, None, 0
    mus = np.stack(valid_mus).astype(np.float32)
    w = np.array(valid_weights, dtype=np.float32).reshape(-1, 1)
    total = w.sum()
    mu_g = (mus * w).sum(axis=0) / total
    rad_g = np.dot(valid_radii, valid_weights) / total
    return mu_g, float(rad_g), int(total)


def _record_assignment(assignments, cid, pkey, gid):
    if cid not in assignments:
        assignments[cid] = {}
    assignments[cid][pkey] = gid


def _calculate_training_trigger(
    rounds_since_disc, weight_div, tau_decay, delta_0, k_smooth,
    avg_data, min_batch, ts_between,
):
    n_base = max(int(avg_data * ts_between), int(min_batch))
    r_fresh = 1.0 - math.exp(-rounds_since_disc / tau_decay)
    sigmoid = 1.0 / (1.0 + math.exp(-k_smooth * (weight_div - delta_0)))
    r_stable = 1.0 + sigmoid
    return max(int(min_batch), int(n_base * r_fresh * r_stable))


def _weight_divergence(old_state, new_state):
    if old_state is None:
        return 0.05
    from ml_backends.weights import unwrap_weights

    old_state = unwrap_weights(old_state)
    new_state = unwrap_weights(new_state)
    if isinstance(new_state, list):
        import numpy as np

        diff_sq, mag_sq = 0.0, 0.0
        for w_new, w_old in zip(new_state, old_state):
            w_new = np.asarray(w_new)
            w_old = np.asarray(w_old)
            if w_new.shape != w_old.shape:
                slices = tuple(slice(0, min(a, b)) for a, b in zip(w_new.shape, w_old.shape))
                w_new, w_old = w_new[slices], w_old[slices]
            diff_sq += np.linalg.norm(w_new - w_old) ** 2
            mag_sq += np.linalg.norm(w_old) ** 2
        return (diff_sq ** 0.5) / (mag_sq ** 0.5) if mag_sq > 0 else 0.05

    import torch
    diff_sq, mag_sq = 0.0, 0.0
    for key in new_state:
        if "weight" in key or "bias" in key:
            if key not in old_state:
                continue
            w_new = new_state[key].float().cpu()
            w_old = old_state[key].float().cpu()
            if w_new.shape != w_old.shape:
                slices = [slice(0, min(a, b)) for a, b in zip(w_new.shape, w_old.shape)]
                w_new, w_old = w_new[tuple(slices)], w_old[tuple(slices)]
            diff_sq += torch.norm(w_new - w_old).item() ** 2
            mag_sq += torch.norm(w_old).item() ** 2
    return (diff_sq ** 0.5) / (mag_sq ** 0.5) if mag_sq > 0 else 0.05
