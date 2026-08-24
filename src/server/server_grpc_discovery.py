"""
gRPC-based discovery service for client registration and heartbeat.
Replaces MQTT-based discovery when discovery_type is set to 'grpc'.
"""

import json
import time
from concurrent import futures
from threading import Event, Thread

import grpc

import proto.grpc_pb2 as grpc_pb2
import proto.grpc_pb2_grpc as grpc_pb2_grpc
from utils.logger import FedLogger


class ServerGRPCDiscovery(grpc_pb2_grpc.DiscoveryServiceServicer):
    def __init__(self, config, client_info, heard_from_client_event):
        self.logger = FedLogger(id="0", loggername="SERVER_GRPC_DISCOVERY")
        self.client_info = client_info
        self.heard_from_client_event = heard_from_client_event
        self.heartbeat_interval_s = config["heartbeat_interval_s"]
        self.num_heartbeats_timestamp_cached = config["num_heartbeats_timestamp_cached"]
        self.max_heartbeats_miss_threshold = config["max_heartbeat_miss_threshold"]

    def Register(self, request, context):
        client_id = request.client_id

        self.client_info.put(f"{client_id}.client_name", request.client_name)
        self.client_info.put(f"{client_id}.grpc_ep", request.grpc_ep)
        self.client_info.put(
            f"{client_id}.benchmark_info", json.loads(request.benchmark_info)
        )
        self.client_info.put(
            f"{client_id}.hardware_information", json.loads(request.hw_info)
        )
        self.client_info.put(f"{client_id}.role", request.client_type)
        self.client_info.put(f"{client_id}.client_type", request.client_type)
        self.client_info.put(
            f"{client_id}.dataset_details", json.loads(request.datasets)
        )
        self.client_info.put(f"{client_id}.models", json.loads(request.models))
        self.client_info.put(f"{client_id}.is_active", True)
        self.client_info.put(f"{client_id}.is_training", False)
        self.client_info.put(f"{client_id}.heartbeat.timestamp", [time.time()])
        self.client_info.put(f"{client_id}.heartbeat.interval", 0)
        self.client_info.put(f"{client_id}.join_timestamp", time.time())

        self.logger.info(
            "discovery.register.received",
            f"{client_id},{request.client_name}",
        )
        print("discovery.register.received", f"{client_id},{request.client_name}")
        # registration_data = {
        #     "client_id": client_id,
        #     "client_name": request.client_name,
        #     "grpc_ep": request.grpc_ep,
        #     "role": request.client_type,
        #     "hardware_info": json.loads(request.hw_info),
        #     "benchmark_info": json.loads(request.benchmark_info),
        #     "datasets": json.loads(request.datasets),
        #     "models": json.loads(request.models),
        # }
        # print(f"\n[gRPC Discovery] New Client Registered:\n{json.dumps(registration_data, indent=4)}\n")
        self.heard_from_client_event.set()

        return grpc_pb2.RegistrationResponse(
            accepted=True,
            heartbeat_interval_s=self.heartbeat_interval_s,
        )

    def Heartbeat(self, request_iterator, context):
        for ping in request_iterator:
            client_id = ping.client_id

            self.logger.debug(
                "discovery.heartbeat.received",
                f"heartbeat from client:,{client_id},{ping.timestamp}",
            )

            try:
                server_time = time.time()
                
                cached_timestamp = self.client_info.get(f"{client_id}.heartbeat.timestamp")
                client_heartbeat_timestamp = (
                    cached_timestamp
                    if isinstance(cached_timestamp, list)
                    else [time.time()]
                )

                interval = round(
                    (server_time - client_heartbeat_timestamp[-1]),
                    2,
                )

                if (
                    len(client_heartbeat_timestamp)
                    >= self.num_heartbeats_timestamp_cached
                ):
                    client_heartbeat_timestamp.pop(0)
                client_heartbeat_timestamp.append(server_time)

                self.client_info.put(f"{client_id}.heartbeat.interval", interval)
                self.client_info.put(
                    f"{client_id}.heartbeat.timestamp", client_heartbeat_timestamp
                )
            except KeyError:
                self.logger.warn(
                    "discovery.heartbeat.invalid.client",
                    f"Ignoring heartbeat, client not registered:,{client_id}",
                )

            yield grpc_pb2.HeartbeatPong(acknowledged=True)

    def Deregister(self, request, context):
        client_id = request.client_id
        self.client_info.put(f"{client_id}.is_active", False)
        self.client_info.put(f"{client_id}.is_training", False)
        self.logger.info("discovery.deregister", f"{client_id}")
        return grpc_pb2.DeregisterResponse(success=True)

    def heartbeat_alive_check(self, stop_event):
        print("[gRPC] Heartbeat alive-check thread started")
        while not stop_event.is_set():
            try:
                for client in list(self.client_info.keys()):
                    if self.client_info.get(f"{client}.is_active"):
                        timestamps = self.client_info.get(f"{client}.heartbeat.timestamp")
                        if timestamps and time.time() - timestamps[-1] >= (
                            (
                                self.max_heartbeats_miss_threshold
                                * self.heartbeat_interval_s
                            )
                            + 2
                        ):
                            print(f"[gRPC] Removing client:{client} from active clients")
                            self.logger.warn(
                                "discovery.heartbeat.delayed",
                                f"Removing client:{client} from active clients",
                            )
                            self.client_info.put(f"{client}.is_active", False)
                            self.client_info.put(f"{client}.is_training", False)
            except Exception as e:
                self.logger.warn(
                    "discovery.heartbeat.check_error",
                    f"Error during heartbeat check, will retry: {e}",
                )
            stop_event.wait(self.heartbeat_interval_s)


def start_discovery_server(config, client_info, stop_event, discovery_ready_event):
    """
    Entry point for the gRPC discovery server thread.
    Mirrors the interface of ServerMQTTDiscovery.mqtt_ad(client_info, stop_event, grpc_event).
    """
    logger = FedLogger(id="0", loggername="SERVER_GRPC_DISCOVERY")
    heard_from_client_event = Event()

    service = ServerGRPCDiscovery(config, client_info, heard_from_client_event)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
    grpc_pb2_grpc.add_DiscoveryServiceServicer_to_server(service, server)

    port = config["discovery_port"]
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    logger.info(
        "discovery.server.started",
        f"gRPC discovery service listening on 0.0.0.0:{port}",
    )

    # Start heartbeat checker immediately so clients that disconnect
    # between registration and discovery_ready_event are still detected.
    heartbeat_thread = Thread(
        target=service.heartbeat_alive_check, args=(stop_event,)
    )
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    heard_from_client_event.wait()
    logger.info("discovery.server.client_registered", "At least one client registered")
    logger.debug("discovery.server.clients", f"Clients:,{client_info.keys()}")

    discovery_ready_event.set()

    stop_event.wait()
    server.stop(grace=5)
    logger.info("discovery.server.stopped", "gRPC discovery service stopped")
