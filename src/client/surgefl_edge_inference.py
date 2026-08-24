"""Edge inference for SurgeFL (used by ClientEdgeService on edge clients)."""

from utils.logger import FedLogger


class EdgeInferenceHandler:
    def __init__(self, client_config, client_id):
        self.logger = FedLogger(id=client_id, loggername="EDGE_INFERENCE")
        self.client_id = client_id
        self.client_config = client_config
        self.detector_initialized = False

        surgefl_cfg = client_config.get("surgefl_config", {})
        self.edge_id = surgefl_cfg.get("edge_id", 0)
        self.drift_config = surgefl_cfg.get("drift_detector_config", {})

    def start_inference(self, model_wts, timestep):
        self.logger.info(
            "edge.inference.start",
            f"edge_id={self.edge_id},timestep={timestep}",
        )
        try:
            drift_type = self.drift_config.get("drift_detection_type", "oracle")
            if drift_type == "oracle":
                drift_detected = self._oracle_drift(timestep)
            else:
                drift_detected = False

            result = {
                "edge_id": self.edge_id,
                "timestep": timestep,
                "drift_detected": drift_detected,
            }
            self.logger.info(
                "edge.inference.done",
                f"edge_id={self.edge_id},drift={drift_detected}",
            )
            return result
        except Exception as e:
            self.logger.error("edge.inference.error", str(e))
            return {
                "edge_id": self.edge_id,
                "timestep": timestep,
                "drift_detected": False,
            }

    def reset_detector(self):
        self.logger.info("edge.detector.reset", f"edge_id={self.edge_id}")
        self.detector_initialized = False
        return {"edge_id": self.edge_id, "reset": True}

    def _oracle_drift(self, timestep):
        drift_timesteps = {50, 100, 150}
        return timestep in drift_timesteps
