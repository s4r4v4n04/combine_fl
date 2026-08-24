from os.path import join

import yaml

from client.client_dataset_loader import DataLoader
from client.client_file_manager import OpenYaML, get_available_models, get_model_class
from client.client_trainer import ClientTrainer
from ml_backends.registry import get_backend_name
from utils.logger import FedLogger


class Client:
    def __init__(
        self,
        client_id: str,
        torch_device: str,
        temp_dir_path: str,
        dataset_paths: dict,
        client_info: dict,
    ) -> None:
        self.torch_device: str = torch_device  # required for sending ML to device
        self.temp_dir_path: str = temp_dir_path  # required for model dir path
        self.dataset_paths: str = dataset_paths  # required for dataset path
        self.client_info: dict = client_info
        self.dataloader = DataLoader()
        self.train_loader, self.test_loader = None, None
        self.logger = FedLogger(id=client_id, loggername="CLIENT")
        self.dataset_id = None

    def _backend_name(self, model_config: dict, model_dir_path: str) -> str:
        return get_backend_name(model_config, model_dir=model_dir_path)

    def StreamFile(self):
        pass

    def Benchmark(
        self,
        model_id: str,
        model_class: str,
        model_config: dict,
        dataset_id: str,
        batch_size: int,
        learning_rate: float,
        loss_function: bytearray = None,
        optimizer: bytearray = None,
        timeout_duration_s: float = None,
        max_mini_batches: int = None,
    ):
        model_dir_path: str = join(self.temp_dir_path, "model_cache", model_id)
        model_hash: str = get_available_models(self.temp_dir_path)[model_id]

        print("client.benchmark.model_config sent:", model_config)

        if not model_config:
            model_config_path: str = join(model_dir_path, "config.yaml")
            model_config: dict = OpenYaML(model_config_path, self.logger)[
                "default_training_config"
            ]

        print("client.benchmark.model_config final:", model_config)
        backend_name = self._backend_name(model_config, model_dir_path)
        self.dataloader = DataLoader(backend=backend_name)

        use_custom_trainer = model_config["use_custom_trainer"]
        custom_trainer_args = model_config["custom_trainer_args"]
        use_custom_dataloader = model_config.get(
            "use_custom_client_dataloader", model_config.get("use_custom_dataloader")
        )
        custom_dataloader_args = model_config.get(
            "custom_client_loader_args", model_config.get("custom_loader_args")
        )
        model_args = model_config["model_args"]

        dataset_path: str = self.dataset_paths[dataset_id]
        try:
            benchmark_trainer = ClientTrainer(
                temp_dir_path=self.temp_dir_path,
                model_id=model_id,
                model_class=model_class,
                logger=self.logger,
                loss_fn=loss_function,
                optimizer=optimizer,
                device=self.torch_device,
                use_custom_trainer=use_custom_trainer,
                custom_trainer_args=custom_trainer_args,
                model_args=model_args,
                backend=backend_name,
            )
            benchmark_trainer.model = benchmark_trainer.backend.move_model_to_device(
                benchmark_trainer.model, benchmark_trainer.device
            )
        # TODO throw exception from ClientTrainer() to handle any missing not critical arguments
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"client.Benchmark.exception:: {e}")
            self.logger.error("client.Benchmark.exception", str(e))
        except Exception as e:
            print(f"client.Benchmark.unknown_exception:: {e}")
            self.logger.error("client.Benchmark.unknown_exception", str(e))

        if "use_custom_client_dataloader" in model_config and model_config["use_custom_client_dataloader"]:
            custom_dataloader_args = model_config["custom_client_loader_args"]
            custom_loader = get_model_class(
                path=self.temp_dir_path,
                model_id=model_id,
                class_name="CustomDataLoader",
                logger=self.logger,
            )()
            try:
                train_loader, test_loader = custom_loader.get_client_dataloaders(
                    batch_size=batch_size,
                    dataset_path=dataset_path,
                    args=custom_dataloader_args,
                )
            except AttributeError:
                self.logger.warn(
                    "fedclient.InitBench.DataLoader",
                    "get_client_dataloaders() not implemented, "
                    "falling back to get_train_test_dataset_loaders()"
                )
                try:
                    train_loader, test_loader = custom_loader.get_train_test_dataset_loaders(
                        batch_size=batch_size,
                        dataset_path=dataset_path,
                        args=custom_dataloader_args,
                    )
                except AttributeError:
                    raise NotImplementedError(
                        f"CustomDataLoader for model '{model_id}' must implement "
                        "'get_client_dataloaders()' or 'get_train_test_dataset_loaders()'"
                    )
            self.logger.debug(
                "fedclient.InitBench.DataLoader", "Loaded custom client Dataloader"
            )

        elif "use_custom_dataloader" in model_config and model_config["use_custom_dataloader"]:
            custom_dataloader_args = model_config["custom_loader_args"]
            custom_loader = get_model_class(
                path=self.temp_dir_path,
                model_id=model_id,
                class_name="CustomDataLoader",
                logger=self.logger,
            )()
            train_loader, test_loader = custom_loader.get_train_test_dataset_loaders(
                batch_size=batch_size,
                dataset_path=dataset_path,
                args=custom_dataloader_args,
            )
            self.logger.debug(
                "fedclient.InitBench.DataLoader", "Loaded custom Dataloader (v0.1)"
            )

        else:
            (
                train_loader,
                test_loader,
            ) = self.dataloader.get_train_test_dataset_loaders(
                batch_size=batch_size, dataset_path=dataset_path
            )
            self.logger.debug(
                "fedclient.InitBench.DataLoader", "Loaded default Dataloader"
            )

        assert train_loader is not None, "Benchmark: train_loader is None"
        assert test_loader is not None, "Benchmark: test_loader is None"

        result = benchmark_trainer.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            lr=learning_rate,
            num_epochs=100000000,
            timeout_duration_s=timeout_duration_s,
            max_mini_batches=max_mini_batches,
        )
        self.update_client_info(model_id, model_hash, result)

        return result

    def Train(
        self,
        model_id: str,
        model_class: str,
        model_config: str,
        dataset_id: str,
        model_wts: bytearray,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        loss_function,
        optimizer,
        timeout_duration_s: float = None,
        max_epochs: int = None,
        max_mini_batches: int = None,
        training_metadata=None,
    ):
        model_dir_path: str = join(self.temp_dir_path, "model_cache", model_id)

        if not model_config:
            model_config_path: str = join(model_dir_path, "config.yaml")
            model_config: dict = OpenYaML(model_config_path, self.logger)[
                "default_training_config"
            ]

        use_custom_trainer = model_config["use_custom_trainer"]
        backend_name = self._backend_name(model_config, model_dir_path)
        self.dataloader = DataLoader(backend=backend_name)
        custom_trainer_args = model_config["custom_trainer_args"]
        use_custom_dataloader = model_config.get(
            "use_custom_client_dataloader", model_config.get("use_custom_dataloader")
        )
        custom_dataloader_args = model_config.get(
            "custom_client_loader_args", model_config.get("custom_loader_args")
        )
        model_args = model_config["model_args"]

        dataset_path: str = self.dataset_paths[dataset_id]

        try:
            model_trainer = ClientTrainer(
                temp_dir_path=self.temp_dir_path,
                model_id=model_id,
                model_class=model_class,
                logger=self.logger,
                loss_fn=loss_function,
                optimizer=optimizer,
                device=self.torch_device,
                use_custom_trainer=use_custom_trainer,
                custom_trainer_args=custom_trainer_args,
                model_args=model_args,
                backend=backend_name,
            )
            model_trainer.model = model_trainer.backend.move_model_to_device(
                model_trainer.model, model_trainer.device
            )
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"client.Train.exception:: {e}")
            self.logger.error("client.Train.exception", str(e))
        except Exception as e:
            print(f"client.Train.unknown_exception:: {e}")
            self.logger.error("client.Train.unknown_exception", str(e))

        if (
            self.train_loader is None
            or self.test_loader is None
            or self.dataset_id != dataset_id
        ):
            if "use_custom_client_dataloader" in model_config and model_config["use_custom_client_dataloader"]:
                custom_dataloader_args = model_config["custom_client_loader_args"]
                custom_loader = get_model_class(
                    path=self.temp_dir_path,
                    model_id=model_id,
                    class_name="CustomDataLoader",
                    logger=self.logger,
                )()
                try:
                    (
                        self.train_loader,
                        self.test_loader,
                    ) = custom_loader.get_client_dataloaders(
                        batch_size=batch_size,
                        dataset_path=dataset_path,
                        args=custom_dataloader_args,
                    )
                except AttributeError:
                    self.logger.warn(
                        "fedclient.StartTraining.DataLoader",
                        "get_client_dataloaders() not implemented, "
                        "falling back to get_train_test_dataset_loaders()"
                    )
                    try:
                        (
                            self.train_loader,
                            self.test_loader,
                        ) = custom_loader.get_train_test_dataset_loaders(
                            batch_size=batch_size,
                            dataset_path=dataset_path,
                            args=custom_dataloader_args,
                        )
                    except AttributeError:
                        raise NotImplementedError(
                            f"CustomDataLoader for model '{model_id}' must implement "
                            "'get_client_dataloaders()' or 'get_train_test_dataset_loaders()'"
                        )
                self.logger.debug(
                    "fedclient.StartTraining.DataLoader", "Loaded custom client Dataloader"
                )

            elif "use_custom_dataloader" in model_config and model_config["use_custom_dataloader"]:
                custom_dataloader_args = model_config["custom_loader_args"]
                custom_loader = get_model_class(
                    path=self.temp_dir_path,
                    model_id=model_id,
                    class_name="CustomDataLoader",
                    logger=self.logger,
                )()
                (
                    self.train_loader,
                    self.test_loader,
                ) = custom_loader.get_train_test_dataset_loaders(
                    batch_size=batch_size,
                    dataset_path=dataset_path,
                    args=custom_dataloader_args,
                )
                self.logger.debug(
                    "fedclient.StartTraining.DataLoader", "Loaded custom Dataloader (v0.1)"
                )

            else:
                (
                    self.train_loader,
                    self.test_loader,
                ) = self.dataloader.get_train_test_dataset_loaders(
                    batch_size=batch_size, dataset_path=dataset_path
                )
                self.logger.debug(
                    "fedclient.StartTraining.DataLoader", "Loaded default Dataloader"
                )

        assert self.train_loader is not None, "Train: train_loader is None"
        assert self.test_loader is not None, "Train: test_loader is None"

        result = model_trainer.train_model(
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr=learning_rate,
            num_epochs=num_epochs,
            timeout_duration_s=timeout_duration_s,
            max_mini_batches=max_mini_batches,
            max_epochs=max_epochs,
            model_checkpoint=model_wts,
            training_metadata=training_metadata,
        )

        model_weights = model_trainer.get_model_wts()

        return result, model_weights

    def Validate(
        self,
        model_id: str,
        model_class: str,
        model_config,
        dataset_id: str,
        model_wts,
        batch_size: int,
        loss_function,
        optimizer,
    ):
        model_dir_path: str = join(self.temp_dir_path, "model_cache", model_id)

        if not model_config:
            model_config_path: str = join(model_dir_path, "config.yaml")
            model_config: dict = OpenYaML(model_config_path, self.logger)[
                "default_training_config"
            ]

        use_custom_trainer = model_config["use_custom_trainer"]
        backend_name = self._backend_name(model_config, model_dir_path)
        self.dataloader = DataLoader(backend=backend_name)
        custom_trainer_args = model_config["custom_trainer_args"]
        use_custom_dataloader = model_config.get(
            "use_custom_client_dataloader", model_config.get("use_custom_dataloader")
        )
        custom_dataloader_args = model_config.get(
            "custom_client_loader_args", model_config.get("custom_loader_args")
        )
        use_custom_validator = model_config["use_custom_validator"]
        custom_validator_args = model_config["custom_validator_args"]
        model_args = model_config["model_args"]

        dataset_path: str = self.dataset_paths[dataset_id]
        try:
            model_validator = ClientTrainer(
                temp_dir_path=self.temp_dir_path,
                model_id=model_id,
                model_class=model_class,
                logger=self.logger,
                loss_fn=loss_function,
                optimizer=optimizer,
                device=self.torch_device,
                use_custom_trainer=use_custom_trainer,
                custom_trainer_args=custom_trainer_args,
                use_custom_validator=use_custom_validator,
                custom_validator_args=custom_validator_args,
                model_args=model_args,
                backend=backend_name,
            )
            model_validator.model = model_validator.backend.move_model_to_device(
                model_validator.model, model_validator.device
            )
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"client.Validate.exception:: {e}")
            self.logger.error("client.Validate.exception", str(e))
        except Exception as e:
            print(f"client.Validate.unknown_exception:: {e}")
            self.logger.error("client.Validate.unknown_exception", str(e))

        if self.train_loader is None or self.test_loader is None:
            if "use_custom_client_dataloader" in model_config and model_config["use_custom_client_dataloader"]:
                custom_dataloader_args = model_config["custom_client_loader_args"]
                custom_loader = get_model_class(
                    path=self.temp_dir_path,
                    model_id=model_id,
                    class_name="CustomDataLoader",
                    logger=self.logger,
                )()
                try:
                    (
                        self.train_loader,
                        self.test_loader,
                    ) = custom_loader.get_client_dataloaders(
                        batch_size=batch_size,
                        dataset_path=dataset_path,
                        args=custom_dataloader_args,
                    )
                except AttributeError:
                    self.logger.warn(
                        "fedclient.StartValidation.DataLoader",
                        "get_client_dataloaders() not implemented, "
                        "falling back to get_train_test_dataset_loaders()"
                    )
                    try:
                        (
                            self.train_loader,
                            self.test_loader,
                        ) = custom_loader.get_train_test_dataset_loaders(
                            batch_size=batch_size,
                            dataset_path=dataset_path,
                            args=custom_dataloader_args,
                        )
                    except AttributeError:
                        raise NotImplementedError(
                            f"CustomDataLoader for model '{model_id}' must implement "
                            "'get_client_dataloaders()' or 'get_train_test_dataset_loaders()'"
                        )
                self.logger.debug(
                    "fedclient.StartValidation.DataLoader", "Loaded custom client Dataloader"
                )

            elif "use_custom_dataloader" in model_config and model_config["use_custom_dataloader"]:
                custom_dataloader_args = model_config["custom_loader_args"]
                custom_loader = get_model_class(
                    path=self.temp_dir_path,
                    model_id=model_id,
                    class_name="CustomDataLoader",
                    logger=self.logger,
                )()
                (
                    self.train_loader,
                    self.test_loader,
                ) = custom_loader.get_train_test_dataset_loaders(
                    batch_size=batch_size,
                    dataset_path=dataset_path,
                    args=custom_dataloader_args,
                )
                self.logger.debug(
                    "fedclient.Validate.DataLoader", f"Loaded custom Dataloader"
                )

            else:
                (
                    self.train_loader,
                    self.test_loader,
                ) = self.dataloader.get_train_test_dataset_loaders(
                    batch_size=batch_size, dataset_path=dataset_path
                )
                self.logger.debug(
                    "fedclient.StartValidation.DataLoader", "Loaded default Dataloader"
                )

        assert self.train_loader is not None, "Validate: train_loader is None"
        assert self.test_loader is not None, "Validate: test_loader is None"

        result = model_validator.validate_model(
            self.test_loader, model_checkpoint=model_wts
        )

        return result

    def update_client_info(self, model_id, model_hash, result):
        info = dict()
        info["num_mini_batches"] = result["total_mini_batches"]
        info["time_taken_s"] = result["time_taken_s"]
        info["model_hash"] = model_hash
        self.client_info["benchmark_info"][model_id] = info
        with open(join(self.temp_dir_path, "client_info.yaml"), "w") as file:
            yaml.dump(self.client_info, file)
