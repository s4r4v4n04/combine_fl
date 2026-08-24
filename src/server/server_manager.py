"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import threading
import time
import os
from threading import Event

from server.server_grpc_discovery import start_discovery_server
from server.server_mqtt_discovery import ServerMQTTDiscovery
from server.server_session_manager import FloSessionManager
from server.server_state_manager import StateManager
from utils.logger import FedLogger


def normalize_server_config(server_config: dict) -> dict:
    comm_config = server_config["comm_config"]
    common_config = comm_config["common"]
    mqtt_discovery_config = dict(comm_config.get("mqtt_discovery", {}))
    grpc_discovery_config = dict(comm_config.get("grpc_discovery", {}))
    grpc_runtime_config = dict(comm_config.get("grpc_runtime", {}))
    restful_config = dict(comm_config.get("restful", {}))

    if comm_config["discovery_type"] == "mqtt":
        mqtt_discovery_config["mqtt_broker"] = mqtt_discovery_config["broker_host"]
        mqtt_discovery_config["mqtt_broker_port"] = mqtt_discovery_config["broker_port"]
        mqtt_discovery_config["mqtt_heartbeat_interval_s"] = common_config[
            "heartbeat_interval_s"
        ]
        mqtt_discovery_config["num_heartbeats_timestamp_cached"] = common_config[
            "num_heartbeats_timestamp_cached"
        ]
        mqtt_discovery_config["max_heartbeat_miss_threshold"] = common_config[
            "max_heartbeat_miss_threshold"
        ]

    if comm_config["discovery_type"] == "grpc":
        grpc_discovery_config["discovery_port"] = grpc_discovery_config["port"]
        grpc_discovery_config["heartbeat_interval_s"] = common_config[
            "heartbeat_interval_s"
        ]
        grpc_discovery_config["num_heartbeats_timestamp_cached"] = common_config[
            "num_heartbeats_timestamp_cached"
        ]
        grpc_discovery_config["max_heartbeat_miss_threshold"] = common_config[
            "max_heartbeat_miss_threshold"
        ]

    restful_config["rest_hostname"] = restful_config["host"]
    restful_config["rest_port"] = restful_config["port"]

    return {
        **server_config,
        "comm_config": {
            **comm_config,
            "common": common_config,
            "mqtt_discovery": mqtt_discovery_config,
            "grpc_discovery": grpc_discovery_config,
            "grpc_runtime": grpc_runtime_config,
            "restful": restful_config,
        },
    }


class FlotillaServerManager:
    def __init__(self, server_config: dict):

        print("Server Manager Initialized")
        self.logger = FedLogger(id="0", loggername="SERVER_MANAGER")

        self.server_config = normalize_server_config(server_config)
        self.state = self.server_config["state"]

        self.client_info = StateManager(
            loc=self.state["state_location"],
            name="client_info",
            host=self.state["state_hostname"],
            port=self.state["state_port"],
            state_id="0"
        )

        self._client_available_callback_ref = [None]

        self.discovery_type = self.server_config["comm_config"]["discovery_type"]
        self.discovery_stop_event = Event()
        self.discovery_ready_event = Event()

        if self.discovery_type == "mqtt":
            mqtt_config = self.server_config["comm_config"]["mqtt_discovery"]
            mqtt_obj = ServerMQTTDiscovery(mqtt_config)
            self.discovery_task = threading.Thread(
                target=mqtt_obj.mqtt_ad,
                args=(
                    self.client_info,
                    self.discovery_stop_event,
                    self.discovery_ready_event,
                ),
            )
            self.discovery_task.name = "MQTT_Task_Thread"
        elif self.discovery_type == "grpc":
            discovery_config = self.server_config["comm_config"]["grpc_discovery"]
            self.discovery_task = threading.Thread(
                target=start_discovery_server,
                args=(
                    discovery_config,
                    self.client_info,
                    self.discovery_stop_event,
                    self.discovery_ready_event,
                ),
            )
            self.discovery_task.name = "gRPC_Discovery_Thread"
        else:
            raise ValueError(
                f"Unknown discovery_type: {self.discovery_type}. Must be 'mqtt' or 'grpc'."
            )

        self.discovery_task.start()
        self.logger.info(
            "fedserver.discovery.started",
            f"Discovery backend: {self.discovery_type}",
        )

    async def run(
        self, id: str, train_config: dict, restore=False, revive=False, file=False
    ):
        session_run_time = time.time()

        self.logger.debug("fedserver.run.started", f"{id},{session_run_time}")
        session = FloSessionManager(
            id=id,
            client_info=self.client_info,
            discovery_ready_event=self.discovery_ready_event,
            server_config=self.server_config,
            session_config=train_config,
            restore=restore,
            revive=revive,
            file=file,
            client_available_callback_ref=self._client_available_callback_ref,
        )
        self.current_session = session
        await session.start_session()
        self.current_session = None
        try:
            user_session_id = train_config.get("session_config", {}).get("session_id", "unknown")
            os.rename(f"logs/flotilla_{id}.log", f"logs/flotilla_{id}_{user_session_id}.log")
        except FileNotFoundError:
            pass
        self.logger.debug(
            "fedserver.run.finished", f"{id},{time.time()-session_run_time}"
        )

    def get_active_clients(self):
        active_clients = [
            client_id
            for client_id in self.client_info.keys()
            if self.client_info.get(f"{client_id}.is_active")
        ]

        return active_clients
