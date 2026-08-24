"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from tqdm import tqdm

from ml_backends.registry import get_backend
from server.load_loss import load_loss
from server.load_optimizer import load_optimizer
from server.server_file_manager import get_model_class
from utils.logger import FedLogger


class ServerModelManager:
    def __init__(
        self,
        id,
        model_dir,
        model_class,
        batch_size,
        val_data_path,
        torch_device=torch.device("cpu"),
        model_args: dict = None,
        use_custom_server_dataloader=False,
        custom_server_loader_args: dict = None,
        use_custom_dataloader=False,
        custom_dataloader_args: dict = None,
        use_custom_validator=False,
        custom_validator_args=None,
        backend: str = "pytorch",
    ) -> None:
        self.id = id
        self.backend_name = backend or "pytorch"
        self.backend = get_backend(self.backend_name)
        self.torch_device = torch_device
        self.device = self.backend.resolve_device(str(torch_device))
        self.model_dir = model_dir

        self.backend.set_seed(1122001)
        model_cls = get_model_class(path=model_dir, class_name=model_class)
        self.model = self.backend.build_model(
            model_cls, device=self.device, args=model_args
        )

        self.logger = FedLogger(id=self.id, loggername="SERVER_MODEL_MANAGER")

        if use_custom_server_dataloader:
            DataLoader = get_model_class(
                path=model_dir, class_name="CustomDataLoader"
            )()
            try:
                self.data = DataLoader.get_server_dataloader(
                    batch_size=batch_size,
                    dataset_path=val_data_path,
                    args=custom_server_loader_args,
                )
            except AttributeError:
                self.logger.warn(
                    "fedserver.DataLoader",
                    "get_server_dataloader() not implemented, "
                    "falling back to get_train_test_dataset_loaders()"
                )
                try:
                    _, self.data = DataLoader.get_train_test_dataset_loaders(
                        batch_size=batch_size,
                        dataset_path=val_data_path,
                        args=custom_server_loader_args,
                    )
                except AttributeError:
                    raise NotImplementedError(
                        "CustomDataLoader must implement "
                        "'get_server_dataloader()' or 'get_train_test_dataset_loaders()'"
                    )
            self.logger.debug(
                "fedserver.DataLoader", "Loaded custom server Dataloader"
            )

        elif use_custom_dataloader:
            DataLoader = get_model_class(
                path=model_dir, class_name="CustomDataLoader"
            )()
            self.data = DataLoader.get_server_dataloader(
                batch_size=batch_size,
                dataset_path=val_data_path,
                args=custom_server_loader_args,
            )
            self.logger.debug(
                "fedserver.DataLoader", "Loaded custom Dataloader (v0.1)"
            )

        else:
            self.data = self.backend.get_server_dataloader(
                dataset_path=val_data_path, batch_size=batch_size
            )

        self.use_custom_validator = use_custom_validator
        self.custom_validator_args = custom_validator_args

    def get_model_weights(self):
        return self.backend.get_weights(self.model)

    def set_model_weights(self, model_weights):
        self.backend.set_weights(self.model, model_weights)

    def get_model_params(self):
        return self.backend.parameter_count(self.model)

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, lr, optimizer, custom):
        if self.backend_name != "pytorch":
            self.optimizer = None
            return
        opt = load_optimizer(self.id, optimizer, custom)
        if opt is None:
            raise RuntimeError(f"Failed to load optimizer: {optimizer}")
        self.optimizer = opt.optimizer_selection(self.model.parameters(), lr=lr)

    def set_loss_fun(self, loss_fun, custom):
        if self.backend_name != "pytorch":
            self.loss_fun = None
            return
        loss = load_loss(self.id, loss_fun, custom)
        if loss is None:
            raise RuntimeError(f"Failed to load loss function: {loss_fun}")
        self.loss_fun = loss.loss_function_selection()

    def get_loss_fun(self):
        return self.loss_fun

    def test_dataset_loader(self, path: str, batch_size=50):
        return self.backend.get_server_dataloader(
            batch_size=batch_size, dataset_path=path
        )

    def validate_model(
        self,
        device: str = "cpu",
        loss_func=None,
        optimizer=None,
        round_no=None,
    ):
        if self.use_custom_validator:
            validator = get_model_class(
                path=self.model_dir, class_name="CustomModelTrainer"
            )
            res = validator.validate_model(
                self,
                model=self.model,
                dataloader=self.data,
                device=device,
                loss_func=loss_func,
                optimizer=optimizer,
                round_no=round_no,
                args=self.custom_validator_args,
            )

        else:
            res = self.backend.default_validate_classifier(
                model=self.model,
                dataloader=self.data,
                loss_func=self.loss_fun,
                optimizer=optimizer,
                device=self.device,
                round_no=round_no,
            )
            print(res)

        return res
