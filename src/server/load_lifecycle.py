import importlib
from utils.logger import FedLogger


def load_lifecycle(id, lifecycle_name):
    module_name = f"server.lifecycle.lifecycle_{lifecycle_name}"
    logger = FedLogger(id=id, loggername="LOAD_LIFECYCLE")
    try:
        module = importlib.import_module(module_name)
        logger.info("load_lifecycle.success", f"Loaded lifecycle module: {module_name}")
        return module
    except ImportError as e:
        logger.error("load_lifecycle.error", f"Failed to load lifecycle '{lifecycle_name}': {e}")
        return None
