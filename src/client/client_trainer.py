"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import time

import torch
from tqdm import tqdm

from client.client_file_manager import get_model_class
from ml_backends.registry import get_backend
from utils.logger import FedLogger


class ClientTrainer:
    def __init__(
        self,
        temp_dir_path: str,
        model_id: str,
        model_class: str,
        logger: FedLogger,
        loss_fn=None,
        optimizer=None,
        device: str = "cpu",
        use_custom_trainer=False,
        model_args: dict = None,
        custom_trainer_args: dict = None,
        use_custom_validator=False,
        custom_validator_args: dict = None,
        backend: str = "pytorch",
    ) -> None:
        self.backend_name = backend or "pytorch"
        self.backend = get_backend(self.backend_name)
        self.device = self.backend.resolve_device(device)

        self.stop_training_flag = False
        self.loss_func = loss_fn
        self.optimizer = optimizer

        self.temp_dir_path = temp_dir_path
        self.model_id = model_id
        self.logger = logger

        model_cls = get_model_class(
            self.temp_dir_path, self.model_id, model_class, logger=self.logger
        )
        self.model = self.backend.build_model(model_cls, device=self.device, args=model_args)

        self.custom_trainer_args = None
        self.use_custom_trainer = use_custom_trainer
        if use_custom_trainer:
            self.custom_trainer_args = custom_trainer_args

        self.use_custom_validator = use_custom_validator
        if use_custom_validator:
            self.custom_validator_args = custom_validator_args

    def set_loss_function(self, loss_func) -> None:
        self.loss_func = loss_func

    def set_device(self, device: str) -> None:
        self.device = self.backend.resolve_device(device)

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

    def load_model_from_checkpoint(self, checkpoint) -> None:
        self.backend.set_weights(self.model, checkpoint)
        self.model = self.backend.move_model_to_device(self.model, self.device)

    def get_model_wts(self):
        return self.backend.get_weights(self.model, model_id=self.model_id)

    def stop_training(self):
        self.stop_training_flag = True

    def exit_check(
        self,
        epochs,
        max_epochs,
        max_mini_batches,
        num_mini_batches,
        start_time,
        timeout_duration_s,
    ):
        exit_flag = False
        if max_mini_batches and (num_mini_batches >= max_mini_batches):
            exit_flag = True
            return exit_flag
        elif max_epochs and (epochs >= max_epochs):
            exit_flag = True
            return exit_flag
        elif timeout_duration_s and (time.time() - start_time > timeout_duration_s):
            exit_flag = True
            return exit_flag

        return exit_flag

    def default_train_model_classifier(
        self,
        lr: float,
        train_loader,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
    ):
        return self.backend.default_train_classifier(
            model=self.model,
            train_loader=train_loader,
            lr=lr,
            loss_func=self.loss_func,
            optimizer=self.optimizer,
            device=self.device,
            test_loader=test_loader,
            num_epochs=num_epochs,
            timeout_duration_s=timeout_duration_s,
            max_mini_batches=max_mini_batches,
            max_epochs=max_epochs,
            stop_requested=lambda: self.stop_training_flag,
        )

    def train_model(
        self,
        lr: float,
        train_loader,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
        model_checkpoint=None,
        training_metadata=None,
    ):
        # loading model from checkpoint
        if model_checkpoint:
            self.load_model_from_checkpoint(checkpoint=model_checkpoint)

        # Setting the loss function
        # if self.loss_func is None:
        #     print("client_trainer.ClientTrainer.train_model :: WARNING - Loss was none")
        #     self.set_loss_function(torch.nn.CrossEntropyLoss)

        # cost = self.loss_func()

        # # Setting the optimizer with the model parameters and learning rate
        # if self.optimizer is None:
        #     print(
        #         "client_trainer.ClientTrainer.train_model :: WARNING - Optimizer was none"
        #     )
        #     self.set_optimizer(torch.optim.Adam(params=self.model.parameters(), lr=lr))

        # # update optimizer with current model parameters.
        # self.optimizer.param_groups.clear()
        # self.optimizer.state.clear()
        # self.optimizer.add_param_group({"params": [p for p in self.model.parameters()]})

        # setting model to train mode
        # self.model.train()

        results = dict()

        if self.use_custom_trainer:
            print("CLIIENT_TRAINER.train_model:: Using custom trainer")
            trainer = get_model_class(
                path=self.temp_dir_path,
                model_id=self.model_id,
                class_name="CustomModelTrainer",
                logger=self.logger,
            )()

            results = trainer.train_model(
                model=self.model,
                results=results,
                train_loader=train_loader,
                epochs=num_epochs,
                test_loader=test_loader,
                args=self.custom_trainer_args,
                timeout_s=timeout_duration_s,
                max_mini_batches=max_mini_batches,
                training_metadata=training_metadata,
            )
            print(f"CLIENT_TRAINER.train_model:: Results - {results}")
        else:
            print("CLIIENT_TRAINER.train_model:: Using default trainer")
            results = self.default_train_model_classifier(
                lr=lr,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                timeout_duration_s=timeout_duration_s,
                max_mini_batches=max_mini_batches,
                max_epochs=max_epochs,
            )
            print(f"CLIENT_TRAINER.train_model:: Results - {results}")
        return results

    def validate_model(
        self,
        test_loader,
        model_checkpoint=None,
    ):
        print("CLIENT_TRAINER.validate_model() called!")
        # loading model from checkpoint
        if model_checkpoint:
            self.load_model_from_checkpoint(checkpoint=model_checkpoint)

            # Setting the loss function
        if self.loss_func is None:
            print("client_trainer.ClientTrainer.train_model :: WARNING - Loss was none")
            self.set_loss_function(None)

        if self.use_custom_validator:
            print("CLIENT_TRAINER.validate_model:: Custom validator being used.")
            validator = get_model_class(
                path=self.temp_dir_path,
                model_id=self.model_id,
                class_name="CustomModelTrainer",
                logger=self.logger,
            )

            res = validator.validate_model(
                self,
                model=self.model,
                dataloader=test_loader,
                device=self.device,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                args=self.custom_validator_args,
            )

        else:
            print("CLIENT_TRAINER.validate_model:: Default validator being used.")
            res = self.backend.default_validate_classifier(
                model=self.model,
                dataloader=test_loader,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                device=self.device,
            )
        print("Result of validation : res = ", res)
        return res
