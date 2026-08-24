"""
SurgeFL lifecycle: timestep-based orchestration with edge inference,
drift detection, and conditional FL sessions (bootstrap/drift/retrain).
"""

import asyncio

from utils.logger import FedLogger

TOTAL_TIMESTEPS = 200
BOOTSTRAP_TIMESTEP = 10
RETRAIN_INTERVAL = 10
FRACTION_EDGES_DRIFT = 1.0

FL_ROUNDS_PER_SESSION = {
    "bootstrap": 100,
    "drift": 50,
    "retrain": 10,
}


async def run(session_manager):
    sm = session_manager
    logger = FedLogger(id=sm.id, loggername="LIFECYCLE_SURGEFL")

    edge_clients = _get_clients_by_type(sm, "edge")
    fog_clients = _get_clients_by_type(sm, "fog")

    logger.info(
        "surgefl.lifecycle.start",
        f"edges={len(edge_clients)},fogs={len(fog_clients)},timesteps={TOTAL_TIMESTEPS}",
    )

    for timestep in range(TOTAL_TIMESTEPS):
        sm.training_state.put("_sfl_meta.timestep", timestep)
        logger.info("surgefl.timestep", f"timestep={timestep}")

        model_wts = sm.model_util.get_model_weights()
        drift_count = 0

        if edge_clients:
            inference_results = await asyncio.gather(
                *(
                    sm.async_grpc_inference(
                        edge_id=eid,
                        model_wts=model_wts,
                        timestep=timestep,
                    )
                    for eid in edge_clients
                )
            )

            for result in inference_results:
                if result and result.get("drift_detected", False):
                    drift_count += 1

            sm.training_state.put("_sfl_meta.drift_count", drift_count)
            sm.training_state.put("_sfl_meta.inference_results", inference_results)

        logger.info(
            "surgefl.inference.done",
            f"timestep={timestep},drift_count={drift_count}/{len(edge_clients)}",
        )

        session_type = None
        if timestep == BOOTSTRAP_TIMESTEP:
            session_type = "bootstrap"
        elif drift_count >= FRACTION_EDGES_DRIFT * len(edge_clients) and drift_count > 0:
            session_type = "drift"
        elif timestep % RETRAIN_INTERVAL == 0 and timestep != 0:
            session_type = "retrain"

        sm.training_state.put("_sfl_meta.session_type", session_type)

        if session_type is None:
            continue

        logger.info(
            "surgefl.session.trigger",
            f"timestep={timestep},session_type={session_type}",
        )

        max_rounds = FL_ROUNDS_PER_SESSION[session_type]
        sm.termination_condition_args = {"max_rounds": max_rounds}
        sm.training_session.put(f"{sm.id}.last_round_number", 0)
        sm.client_selection_args["sfl_timestep"] = timestep
        sm.client_selection_args["sfl_session_type"] = session_type

        logger.info(
            "surgefl.train.start",
            f"session_type={session_type},max_rounds={max_rounds}",
        )

        await sm.train()

        logger.info(
            "surgefl.train.done",
            f"timestep={timestep},session_type={session_type}",
        )

        if session_type == "drift" and edge_clients:
            await asyncio.gather(
                *(
                    sm.async_grpc_reset_detector(edge_id=eid)
                    for eid in edge_clients
                )
            )
            logger.info(
                "surgefl.detectors.reset",
                f"timestep={timestep},edges_reset={len(edge_clients)}",
            )

    logger.info("surgefl.lifecycle.done", f"total_timesteps={TOTAL_TIMESTEPS}")


def _get_clients_by_type(session_manager, client_type):
    sm = session_manager
    result = []
    for cid in sm.client_info.keys():
        ctype = sm.client_info.get(f"{cid}.client_type")
        if ctype == client_type:
            result.append(cid)
    return result
