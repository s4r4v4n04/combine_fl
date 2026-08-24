"""
gRPC-based discovery client for registration and heartbeat.
Replaces MQTT-based client discovery when discovery_type is set to 'grpc'.
"""

import json
import time

import grpc

import proto.grpc_pb2 as grpc_pb2
import proto.grpc_pb2_grpc as grpc_pb2_grpc
from client.client_file_manager import get_available_models
from utils.hardware_info import get_hardware_info
from utils.logger import FedLogger


class ClientGRPCDiscovery:
    def __init__(
        self,
        client_id,
        discovery_config,
        grpc_ep,
        temp_dir_path,
        dataset_details,
        client_info,
        client_config=None,
    ):
        self.client_id = client_id
        self.server_discovery_ep = discovery_config["server_discovery_endpoint"]
        self.client_name = discovery_config["client_name"]
        self.reconnect_interval_s = discovery_config.get("reconnect_interval_s", 5)
        self.grpc_ep = grpc_ep
        self.temp_dir_path = temp_dir_path
        self.dataset_details = dataset_details
        self.client_info = client_info
        self.client_config = client_config or {}
        gc = self.client_config.get("general_config") or {}
        self.client_type = gc.get("client_type", "client")
        self.hw_info = get_hardware_info()
        self.logger = FedLogger(
            id=self.client_id, loggername="CLIENT_GRPC_DISCOVERY"
        )

    def run(self, stop_event, reconnect_interval=None):
        """
        Main loop: register with the server, then stream heartbeats.
        If the connection is lost, reconnect and re-register periodically.
        Designed to be run in a Thread (same interface as ClientMQTTDiscovery.mqtt_sub).
        """
        reconnect_interval = reconnect_interval or self.reconnect_interval_s
        while not stop_event.is_set():
            channel = None
            try:
                channel = grpc.insecure_channel(self.server_discovery_ep)
                stub = grpc_pb2_grpc.DiscoveryServiceStub(channel)

                heartbeat_interval = self._register(stub, stop_event)
                if heartbeat_interval is None:
                    # stop_event was set during registration
                    break

                self.logger.info(
                    "discovery.client.registered",
                    f"Registered with server, heartbeat_interval={heartbeat_interval}s",
                )

                self._heartbeat_loop(stub, heartbeat_interval, stop_event)

                # If we reach here, heartbeat stream ended.
                # If stop_event is not set, the server likely went down — reconnect.
                if not stop_event.is_set():
                    self.logger.warn(
                        "discovery.client.reconnecting",
                        f"Heartbeat stream ended, reconnecting in {reconnect_interval}s",
                    )
                    stop_event.wait(reconnect_interval)

            except Exception as e:
                if not stop_event.is_set():
                    self.logger.error(
                        "discovery.client.connection_error",
                        f"Unexpected error, reconnecting in {reconnect_interval}s: {e}",
                    )
                    stop_event.wait(reconnect_interval)
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception:
                        pass

    def _register(self, stub, stop_event):
        """Register with the server, retrying until success or stop_event is set."""
        while not stop_event.is_set():
            try:
                response = stub.Register(
                    grpc_pb2.ClientRegistration(
                        client_id=self.client_id,
                        client_name=self.client_name,
                        grpc_ep=self.grpc_ep,
                        client_type=self.client_type,
                        hw_info=json.dumps(self.hw_info).encode(),
                        datasets=json.dumps(self.dataset_details).encode(),
                        models=json.dumps(
                            get_available_models(self.temp_dir_path)
                        ).encode(),
                        benchmark_info=json.dumps(
                            self.client_info["benchmark_info"]
                        ).encode(),
                    ),
                    timeout=10,
                )
                if response.accepted:
                    return response.heartbeat_interval_s
            except grpc.RpcError as e:
                self.logger.warn(
                    "discovery.client.register.retry",
                    f"Server not ready, retrying in 3s: {e}",
                )
                stop_event.wait(3)
        return None

    def _heartbeat_loop(self, stub, heartbeat_interval, stop_event):
        """Stream heartbeats to the server until stop_event is set."""
        def heartbeat_generator():
            while not stop_event.is_set():
                yield grpc_pb2.HeartbeatPing(
                    client_id=self.client_id,
                    timestamp=time.time(),
                )
                self.logger.debug(
                    "discovery.client.heartbeat.sent",
                    f"Heartbeat sent,{self.client_id},{time.time()}",
                )
                stop_event.wait(heartbeat_interval)

        try:
            for pong in stub.Heartbeat(heartbeat_generator()):
                if stop_event.is_set():
                    break
        except grpc.RpcError as e:
            if not stop_event.is_set():
                self.logger.warn(
                    "discovery.client.heartbeat.stream_broken",
                    f"Lost connection to server: {e}",
                )
