"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import sys
from os.path import join
from pickle import dumps as p_dumps
from pickle import loads as p_loads
from time import time

from typing_extensions import OrderedDict

import proto.grpc_pb2 as grpc_pb2
import proto.grpc_pb2_grpc as grpc_pb2_grpc
from client.client import Client
from client.client_file_manager import setup_model_dir
from client.surgefl_edge_inference import EdgeInferenceHandler
from utils.logger import FedLogger


class ClientEdgeService(grpc_pb2_grpc.EdgeServiceServicer):
    def __init__(
        self,
        client_id: str,
        temp_dir_path: str,
        torch_device: str,
        dataset_paths: str,
        client_info: dict,
        client_config: dict = None,
    ) -> None:
        self.logger = FedLogger(id=client_id, loggername="CLIENT_EDGE_SERVICE")
        self.temp_dir_path = temp_dir_path
        self.client_id = client_id
        self.client_config = client_config 
        self._edge_inference_handler = None

        self.client = Client(
            client_id=self.client_id,
            torch_device=torch_device,
            temp_dir_path=temp_dir_path,
            dataset_paths=dataset_paths,
            client_info=client_info,
        )

    def _get_client_type(self):
        return (self.client_config.get("general_config") or {}).get("client_type", "client")

    def _get_inference_handler(self):
        if self._edge_inference_handler is None:
            if self.client_config.get("odin_config") is not None:
                from client.odin_edge_inference import OdinEdgeInferenceHandler
                self._edge_inference_handler = OdinEdgeInferenceHandler(
                    self.client_config, self.client_id, self.temp_dir_path
                )
            else:
                self._edge_inference_handler = EdgeInferenceHandler(
                    self.client_config, self.client_id
                )
        return self._edge_inference_handler

    def Echo(self, request, context) -> grpc_pb2.echoMessage:
        self.logger.debug(
            "fedclient.gRPC.echo.request", f"Received message:{request.text}"
        )
        print("fedclient.gRPC.echo.request:: Complete\n")
        if context.is_active():
            return grpc_pb2.echoMessage(text=request.text)
        else:
            self.logger.error("fedclient.gRPC.echo.request", f"fedserver not active")

    def StreamFile(self, request_iterator, context) -> None:
        model_id = str()
        file_name = str()
        data = bytearray()

        try:
            self.logger.debug("fedclient.gRPC.download.model.init", "")
            for request in request_iterator:
                if request.metadata.model_id and request.metadata.file_name:
                    model_id = request.metadata.model_id
                    file_name = request.metadata.file_name
                    self.logger.debug(
                        "fedclient.gRPC.download.model.received",
                        f"{model_id},{file_name}",
                    )
                data.extend(request.chunk_data)

            setup_model_dir(temp_dir_path=self.temp_dir_path, model_id=model_id, logger=self.logger)
            file_path = join(self.temp_dir_path, "model_cache", model_id, file_name)
            with open(file_path, "wb") as f:
                f.write(data)
        except TimeoutError as e:
            self.logger.error(f"StreamFile timeout error: {e}")
            raise
        except Exception as e:
            print(f"client_grpc_manager.StreamFile.exception:: {e}")
            self.logger.error("client_grpc_manager.StreamFile.exception", str(e))
            raise
        return grpc_pb2.StringResponse(
            text=f"{self.client_id} successfully received {model_id}/{file_name}"
        )

    def InitBench(self, request, context) -> grpc_pb2.InitBenchResponse:
        self.logger.info("fedclient.gRPC.benchmark.init", "")
        print("fedclient.gRPC.InitBench:: Benchmark Round Initiated")

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config: dict = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        batch_size: int = request.batch_size
        learning_rate: float = request.learning_rate

        optimizer = None
        loss_function = None
        timeout_duration_s = None
        max_mini_batches = None

        if request.loss_function:
            loss_function = p_loads(request.loss_function)
        if request.optimizer:
            optimizer = p_loads(request.optimizer)
        if request.timeout_duration_s:
            timeout_duration_s = request.timeout_duration_s
        if request.max_mini_batch_count:
            max_mini_batches = request.max_mini_batch_count

        if not context.is_active():
            print(f"client_grpc_manager.gRPC.InitBench:: fedserver not active")
            self.logger.error("client_grpc_manager.InitBench", f"fedserver not active")
            return
        result = self.client.Benchmark(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_function=loss_function,
            optimizer=optimizer,
            timeout_duration_s=timeout_duration_s,
            max_mini_batches=max_mini_batches,
        )

        log_str_keys = ",".join([str(key) for key in result.keys()])
        log_str_values = ",".join([str(value) for value in result.values()])

        log_string = ",".join([log_str_keys, log_str_values])
        self.logger.info("fedclient.gRPC.benchmark.results", log_string)
        self.logger.info("fedclient.gRPC.benchmark.finish", "")

        response = grpc_pb2.InitBenchResponse(
            model_id=model_id,
            num_mini_batches=result["total_mini_batches"],
            bench_duration_s=result["time_taken_s"],
        )

        print("fedclient.gRPC.InitBench:: Benchmark Round Finished")
        if context.is_active():
            return response
        else:
            print(f"client_grpc_manager.InitBench:: fedserver not active")
            self.logger.error("client_grpc_manager.InitBench", f"fedserver not active")

    def StartTraining(self, request, context) -> grpc_pb2.InitTrainResponse:
        self.logger.info("fedclient.gRPC.train.init", "")
        grpc_train_time = time()

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config: dict = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        model_wts: OrderedDict = p_loads(request.model_wts)
        batch_size: int = request.batch_size
        learning_rate: float = request.learning_rate
        num_epochs: int = request.num_epochs
        round_id: int = request.round_idx
        timeout_duration_s = None
        loss_function = p_loads(request.loss_function)
        optimizer = p_loads(request.optimizer)

        if request.timeout_duration_s:
            max_mini_batches = None
            timeout_duration_s = request.timeout_duration_s
            max_epochs = None
        elif request.max_epochs:
            max_mini_batches = None
            timeout_duration_s = None
            max_epochs = request.max_epochs
        else:
            max_mini_batches = request.max_mini_batches
            timeout_duration_s = None
            max_epochs = None

        self.logger.debug("fedclient.gRPC.train.round.model", model_id)
        print(f"\nfedclient.gRPC.train.round:: Training Round:{round_id}")

        if not context.is_active():
            print("client_grpc_manager.StartTraining:: fedserver not active")
            self.logger.error("client_grpc_manager.StartTraining", f"fedserver not active")
            return
        training_metadata = None
        if request.training_metadata:
            training_metadata = p_loads(request.training_metadata)
        result, model_weights = self.client.Train(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            model_wts=model_wts,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            loss_function=loss_function,
            optimizer=optimizer,
            timeout_duration_s=timeout_duration_s,
            max_epochs=max_epochs,
            max_mini_batches=max_mini_batches,
            training_metadata=training_metadata,
        )

        pickle_time = time()
        model_weights = p_dumps(model_weights)
        metrics = p_dumps(result)
        self.logger.info(
            "fedclient.gRPC.train.round.pickle.weights", f"{time()-pickle_time}"
        )

        response = grpc_pb2.InitTrainResponse(
            model_id=model_id,
            model_weights=model_weights,
            client_id=self.client_id,
            round_idx=round_id,
            metrics=metrics,
        )

        self.logger.info("fedclient.gRPC.train.round.complete", "")

        response_time = time()
        try:
            if context.is_active():
                return response
            else:
                print("client_grpc_manager.StartTraining:: fedserver not active")
                self.logger.error("client_grpc_manager.StartTraining", f"fedserver not active")
        finally:
            print("fedclient.gRPC.StartTraining:: Training Round Finished")
            self.logger.info("fedclient.gRPC.e2e.time", f"{time()-grpc_train_time}")
            self.logger.info(
                "fedclient.gRPC.train.response.time", f"{time()-response_time}"
            )

    def StartInference(self, request, context) -> grpc_pb2.InferenceResponse:
        if self._get_client_type() != "edge":
            return grpc_pb2.InferenceResponse(
                edge_id=str(self.client_id),
                timestep=request.timestep,
                drift_detected=False,
            )
        model_wts = p_loads(request.model_wts) if request.model_wts else None
        result = self._get_inference_handler().start_inference(
            model_wts, request.timestep
        )
        return grpc_pb2.InferenceResponse(
            edge_id=str(result.get("edge_id", self.client_id)),
            timestep=int(result.get("timestep", request.timestep)),
            drift_detected=bool(result.get("drift_detected", False)),
            result_data=p_dumps(result),
        )

    def ResetDetector(self, request, context) -> grpc_pb2.ResetDetectorResponse:
        if self._get_client_type() != "edge":
            return grpc_pb2.ResetDetectorResponse(
                edge_id=str(self.client_id), success=True
            )
        self._get_inference_handler().reset_detector()
        return grpc_pb2.ResetDetectorResponse(
            edge_id=str(self._get_inference_handler().edge_id), success=True
        )

    def StartValidation(self, request, context) -> grpc_pb2.InitValidationResponse:
        self.logger.info("fedclient.gRPC.validation.round.init", "")
        grpc_validation_time = time()

        model_id: str = request.model_id
        model_class: str = request.model_class
        model_config = p_loads(request.model_config)
        dataset_id: str = request.dataset_id
        model_wts: OrderedDict = p_loads(request.model_wts)
        batch_size: int = request.batch_size
        round_id: int = request.round_idx
        loss_function = p_loads(request.loss_function)
        optimizer = p_loads(request.optimizer)

        self.logger.debug("fedclient.gRPC.validate.round.model", model_id)
        print(f"\nfedclient.gRPC.validate.round:: Validation Round:{round_id}")

        if not context.is_active():
            print("client_grpc_manager.StartValidation:: fedserver not active")
            self.logger.error("client_grpc_manager.StartValidation", f"fedserver not active")
            return
        result = self.client.Validate(
            model_id=model_id,
            model_class=model_class,
            model_config=model_config,
            dataset_id=dataset_id,
            model_wts=model_wts,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        pickle_time = time()
        metrics = p_dumps(result)
        self.logger.info(
            "fedclient.gRPC.train.round.pickle.weights", f"{time()-pickle_time}"
        )

        response = grpc_pb2.InitValidationResponse(
            model_id=model_id,
            client_id=self.client_id,
            round_idx=round_id,
            metrics=metrics,
        )

        self.logger.info("fedclient.gRPC.validation.round.complete", "")

        return_time = time()
        try:
            if context.is_active():
                return response
            else:
                print("client_grpc_manager.StartValidation:: fedserver not active")
                self.logger.error("client_grpc_manager.StartValidation", f"fedserver not active")
        finally:
            print("fedclient.gRPC.StartValidation:: Validation Round Finished")
            self.logger.info(
                "fedclient.gRPC.e2e.time", f"{time()-grpc_validation_time}"
            )
            self.logger.info(
                "fedclient.gRPC.validation.response.time", f"{time()-return_time}"
            )
