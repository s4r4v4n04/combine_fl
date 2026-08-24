"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import importlib

from utils.logger import FedLogger


def load_termination_condition(id, termination_condition):
    logger = FedLogger(id=id, loggername="TERMINATION_CONDITION_LOADER")

    module_name = f"server.termination.termination_{termination_condition}"
    try:
        module = importlib.import_module(module_name)
        logger.info(
            "fedserver.termination_condition.module",
            f"Termination condition module name:,{module_name}",
        )
        return module
    except ImportError:
        logger.error(
            "fedserver.termination_condition.invalid.module",
            f"Could not import the module ,{module_name}",
        )
