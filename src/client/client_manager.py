"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import shutil
import uuid
from concurrent import futures
from threading import Event, Thread

import grpc
import torch
from grpc import Server

import proto.grpc_pb2_grpc as grpc_pb2_grpc
from client.client_file_manager import (
    get_available_datasets,
    get_available_models,
    setup_dir,
)
from client.client_edge_service import ClientEdgeService
from client.client_grpc_discovery import ClientGRPCDiscovery
from client.client_mqtt_discovery import ClientMQTTDiscovery
from client.utils.ip import get_ip_address, get_ip_address_docker, get_advertise_ip
from client.utils.port_allocator import port_allocator
from utils.logger import FedLogger


class ClientManager:
    def __init__(self, client_id: int, client_config: dict, client_info: dict):
        self.logger = FedLogger(id=client_id, loggername="CLIENT_MANAGER")
        self.client_id: str = client_id
        self.session_id: str = str(uuid.uuid4())

        self.client_info = client_info

        if client_config["general_config"]["use_gpu"]:
            self.torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.torch_device = torch.device("cpu")
        if client_config["general_config"]["use_gpu"] and self.torch_device == "cpu":
            self.logger.warn(
                f"WARNING: GPU not available on Client{self.client_id}.",
                "Setting torch_device as 'cpu'",
            )

        try:
            ev = os.environ["DOCKER_RUNNING"]
            print("RUNNING INSIDE DOCKER")
        except KeyError:
            print("RUNNING ON BARE METAL")
            ev = False

        self.ip: str = get_ip_address_docker() if ev else get_ip_address()
        print(self.ip)

        self.client_config = client_config
        self.comm_config = self._normalize_comm_config(client_config["comm_config"])
        self.discovery_type = self.comm_config["discovery_type"]
        self.grpc_config: dict = self.comm_config["grpc_runtime"]

        self.grpc_workers: int = self.grpc_config["workers"]
        self.init_grpc_port: int = int(self.grpc_config["sync_port"])

        self.grpc_port: int = port_allocator(self.ip, self.init_grpc_port)
        # Advertise a server-reachable address (override with FLO_ADVERTISE_IP for a
        # public/VPN IP when the server is on a different network); bind on all
        # interfaces so it's reachable however the server routes to us.
        self.advertise_ip: str = get_advertise_ip(self.ip)
        self.grpc_ep: str = f"{self.advertise_ip}:{self.grpc_port}"

        self.opts: list = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.so_reuseport", 0),
            ("grpc.so_reuseaddr", 0),
        ]

        # setting up necessary directories
        self.temp_dir_path = client_config["general_config"]["temp_dir_path"]
        self.datasets_dir_path = client_config["dataset_config"]["datasets_dir_path"]
        setup_dir(dir_path=self.temp_dir_path, logger=self.logger)

        # get available datasets and models
        # self.models_available = get_available_models(self.temp_dir_path)
        self.dataset_details, self.dataset_paths = get_available_datasets(
            self.datasets_dir_path, logger=self.logger
        )

        # saving the cleanup options
        general_config = client_config["general_config"]
        self.cleanup_model_cache_on_exit: bool = general_config.get(
            "cleanup_model_cache_on_exit", False
        )
        self.cleanup_temp_on_exit: bool = general_config.get(
            "cleanup_temp_on_exit", False
        )

        # setting up client logger
        self.logger = FedLogger(id=self.client_id, loggername="CLIENT_MANAGER")

    def _normalize_comm_config(self, comm_config: dict) -> dict:
        """Normalize the structured client communication config."""

        common_config = comm_config["common"]
        grpc_runtime_config = dict(comm_config.get("grpc_runtime", {}))
        mqtt_discovery_config = dict(comm_config.get("mqtt_discovery", {}))
        grpc_discovery_config = dict(comm_config.get("grpc_discovery", {}))

        if comm_config["discovery_type"] == "grpc":
            grpc_discovery_config["client_name"] = common_config["client_name"]
            grpc_discovery_config["reconnect_interval_s"] = common_config.get(
                "reconnect_interval_s", 5
            )
            grpc_discovery_config["server_discovery_endpoint"] = (
                f"{grpc_discovery_config['host']}:{grpc_discovery_config['port']}"
            )

        if comm_config["discovery_type"] == "mqtt":
            mqtt_discovery_config["client_name"] = common_config["client_name"]
            mqtt_discovery_config["heartbeat_timeout_s"] = common_config[
                "heartbeat_timeout_s"
            ]
            mqtt_discovery_config["mqtt_broker"] = mqtt_discovery_config["broker_host"]
            mqtt_discovery_config["mqtt_broker_port"] = mqtt_discovery_config[
                "broker_port"
            ]

        return {
            **comm_config,
            "common": common_config,
            "grpc_runtime": grpc_runtime_config,
            "mqtt_discovery": mqtt_discovery_config,
            "grpc_discovery": grpc_discovery_config,
        }

    def discovery_init(self, stop_event: Event) -> Thread:
        """
        Start the discovery service (MQTT or gRPC) based on discovery_type config.
        Returns a Thread that can be joined on shutdown.
        """
        if self.discovery_type == "mqtt":
            mqtt_config = self.comm_config["mqtt_discovery"]
            self.logger.info("discovery.client.init", "backend=mqtt")
            discovery_client = ClientMQTTDiscovery(
                id=self.client_id,
                mqtt_config=mqtt_config,
                grpc_config=self.grpc_config,
                temp_dir_path=self.temp_dir_path,
                dataset_details=self.dataset_details,
                client_info=self.client_info,
            )
            task = Thread(target=discovery_client.mqtt_sub, args=(stop_event,))

        elif self.discovery_type == "grpc":
            discovery_config = self.comm_config["grpc_discovery"]
            self.logger.info("discovery.client.init", "backend=grpc")
            discovery_client = ClientGRPCDiscovery(
                client_id=self.client_id,
                discovery_config=discovery_config,
                grpc_ep=self.grpc_ep,
                temp_dir_path=self.temp_dir_path,
                dataset_details=self.dataset_details,
                client_info=self.client_info,
                client_config=self.client_config,
            )
            task = Thread(target=discovery_client.run, args=(stop_event,))

        else:
            raise ValueError(
                f"Unknown discovery_type: {self.discovery_type}. Must be 'mqtt' or 'grpc'."
            )

        task.start()
        return task

    def grpc_init(self, stop_event: Event) -> Server:
        """
        Start the client's EdgeService gRPC server (for training/validation/benchmark).
        This is independent of the discovery mechanism.
        """
        sync_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.grpc_workers), options=self.opts
        )
        grpc_pb2_grpc.add_EdgeServiceServicer_to_server(
            ClientEdgeService(
                client_id=self.client_id,
                temp_dir_path=self.temp_dir_path,
                torch_device=self.torch_device,
                dataset_paths=self.dataset_paths,
                client_info=self.client_info,
                client_config=self.client_config,
            ),
            sync_server,
        )
        sync_server.add_insecure_port(f"0.0.0.0:{self.grpc_port}")
        sync_server.start()
        self.logger.info("fedclient_gRPC.init", "")
        return sync_server

    def run(self) -> None:
        """
        Start the client's EdgeService gRPC server and discovery service,
        then wait for a kill signal.
        """
        stop_event = Event()
        grpc_sync_server = None
        discovery_task = None
        try:
            print(f"client id: {self.client_id}")

            self.logger.info("fedclient.init", "")

            grpc_sync_server = self.grpc_init(stop_event)

            discovery_task = self.discovery_init(stop_event)

            stop_event.wait()

        except KeyboardInterrupt:
            self.logger.info(
                "fedclient.Keyboard_interrupt",
                "Received KeyboardInterrupt starting exit procedure",
            )
        finally:
            self.exit_procedure(stop_event, grpc_sync_server, discovery_task)
            self.logger.info("fedclient.exit", "")

    def exit_procedure(self, stop_event, grpc_sync_server, discovery_task):
        if grpc_sync_server is not None:
            grpc_sync_server.stop(grace=None)
        stop_event.set()
        if discovery_task is not None:
            discovery_task.join()
        self._cleanup_on_exit()

    def _cleanup_on_exit(self):
        if self.cleanup_temp_on_exit:
            self._remove_path(self.temp_dir_path, "cleanup.temp")
            return

        if self.cleanup_model_cache_on_exit:
            self._remove_path(
                os.path.join(self.temp_dir_path, "model_cache"),
                "cleanup.model_cache",
            )

    def _remove_path(self, path, log_key):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            self.logger.info(log_key, f"removed,{path}")
        except FileNotFoundError:
            return
        except Exception as e:
            self.logger.error(log_key, f"failed,{path},{e}")
