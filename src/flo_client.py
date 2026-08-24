"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
import os
import uuid

from client.client_file_manager import OpenYaML
from client.client_manager import ClientManager
from client.utils.client_info import generate_client_info
from client.utils.monitor import Monitor
from utils.logger import FedLogger


def main():
    pid = os.getpid()
    logger = FedLogger(id="0", loggername="FLO_CLIENT")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--client_config",
        type=str,
        default=None,
        help="Path to client config YAML (default: src/config/client_config.yaml)",
    )
    parser.add_argument(
        "-m",
        "--monitor",
        action="store_true",
        default=False,
        help="Monitor CPU/RAM/Disk/Network usage.",
    )
    args, _ = parser.parse_known_args()

    _default_client_cfg = os.path.join(
        os.path.dirname(__file__), "config", "client_config.yaml"
    )
    client_config = OpenYaML(args.client_config or _default_client_cfg, logger=logger)
    temp_dir_path = os.path.join(client_config["general_config"]["temp_dir_path"])
    if os.path.isfile(os.path.join(temp_dir_path, "client_info.yaml")):
        client_info = OpenYaML(os.path.join(temp_dir_path, "client_info.yaml"), logger=logger)
        client_id: str = client_info["client_id"]
    else:
        client_id: str = str(uuid.uuid4())
        client_info = generate_client_info(client_id, temp_dir_path)

    if args.monitor:
        Monitor(client_id, pid)

    client = ClientManager(client_id, client_config, client_info)
    client.run()


if __name__ == "__main__":
    main()
