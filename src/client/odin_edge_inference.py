"""
Odin Edge Inference Handler — Adapter for FedML-NG framework.

Dynamically loads the OdinInferenceHandler from the model directory
and exposes the same interface expected by ClientEdgeService.StartInference().
"""

import importlib
import os
import sys

from utils.logger import FedLogger


class OdinEdgeInferenceHandler:
    """
    Wrapper that loads OdinInferenceHandler and delegates start_inference() 
    to it. This keeps Odin-specific logic out of the core framework files.
    """

    def __init__(self, client_config, client_id, temp_dir_path):
        self.logger = FedLogger(id=client_id, loggername="ODIN_EDGE_INFERENCE")
        self.client_id = client_id
        self.client_config = client_config
        self.temp_dir_path = temp_dir_path

        odin_cfg = client_config.get("odin_config", {})
        model_id = odin_cfg.get("model_id", "Odin")

        OdinHandlerClass = self._load_handler_class(model_id)
        if OdinHandlerClass is None:
            raise RuntimeError(
                f"OdinInferenceHandler not found for model_id '{model_id}'. "
                f"Check that inference_handler.py exists in the model directory."
            )

        self._handler = OdinHandlerClass(client_config, client_id)
        self.logger.info(
            "odin_edge.init",
            f"OdinInferenceHandler loaded for client {client_id}",
        )

    def _load_handler_class(self, model_id):
        """
        Load OdinInferenceHandler class. Tries two locations:
        1. Model cache (temp_dir_path/model_cache/model_id/) — for streamed models
        2. Examples directory (examples/{model_id}/{model_id}/) — for colocated setups
        """
        # Always add the examples directory to sys.path if running in a colocated setup
        # This ensures dependencies like 'feature_extractors' are found even if 
        # the inference handler is loaded from the temporary model cache.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        examples_model_dir = os.path.join(project_root, "examples", model_id, model_id)
        
        if os.path.isdir(examples_model_dir):
            if examples_model_dir not in sys.path:
                sys.path.insert(0, examples_model_dir)
            parent_dir = os.path.dirname(examples_model_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

        # Attempt 1: Try model cache (standard framework path)
        try:
            from client.client_file_manager import get_model_class
            cls = get_model_class(
                path=self.temp_dir_path,
                model_id=model_id,
                class_name="OdinInferenceHandler",
                logger=self.logger,
            )
            if cls is not None:
                return cls
        except Exception:
            pass

        # Attempt 2: Try direct import from examples directory
        if os.path.isdir(examples_model_dir):
            try:
                mod = importlib.import_module("inference_handler")
                cls = getattr(mod, "OdinInferenceHandler", None)
                if cls is not None:
                    self.logger.info(
                        "odin_edge.load",
                        f"Loaded OdinInferenceHandler from {examples_model_dir}",
                    )
                    return cls
            except Exception as e:
                self.logger.error("odin_edge.load.error", str(e))

        return None

    def start_inference(self, model_wts, timestep):
        """
        Delegate to OdinInferenceHandler.start_inference().
        model_wts here is the payload dict sent by lifecycle_odin.py.
        """
        return self._handler.start_inference(model_wts, timestep)

    def reset_detector(self):
        """No-op for Odin (prototype state is managed internally)."""
        return {"edge_id": self.client_id, "reset": True}
