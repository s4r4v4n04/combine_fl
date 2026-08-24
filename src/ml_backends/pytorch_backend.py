import math
import time
from typing import Callable

from tqdm import tqdm

from ml_backends.base import BackendAdapter
from ml_backends.weights import WeightPayload, unwrap_weights


class PyTorchBackend(BackendAdapter):
    name = "pytorch"

    @staticmethod
    def _resolve_module(model):
        return getattr(model, "model", model)

    @property
    def torch(self):
        import torch

        return torch

    def resolve_device(self, device: str | None = None, use_gpu: bool = False):
        torch = self.torch
        if device:
            return torch.device(device)
        return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    def set_seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)

    def move_model_to_device(self, model, device):
        module = self._resolve_module(model)
        return module.to(device)

    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        module = self._resolve_module(model)
        return WeightPayload(
            backend=self.name,
            model_id=model_id,
            weights=module.to("cpu").state_dict(),
            metadata={"format": "torch_state_dict"},
        )

    def set_weights(self, model, payload: WeightPayload | dict) -> None:
        module = self._resolve_module(model)
        module.load_state_dict(unwrap_weights(payload))

    def parameter_count(self, model) -> int:
        module = self._resolve_module(model)
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _exit_check(
        self,
        epochs,
        max_epochs,
        max_mini_batches,
        num_mini_batches,
        start_time,
        timeout_duration_s,
    ) -> bool:
        if max_mini_batches and (num_mini_batches >= max_mini_batches):
            return True
        if max_epochs and (epochs >= max_epochs):
            return True
        if timeout_duration_s and (time.time() - start_time > timeout_duration_s):
            return True
        return False

    def default_train_classifier(
        self,
        model,
        train_loader,
        lr: float,
        loss_func=None,
        optimizer=None,
        device=None,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict:
        torch = self.torch
        device = device or self.resolve_device("cpu")
        module = self._resolve_module(model)
        if loss_func is None:
            print("Loss was none, using default Loss Function")
            loss_func = torch.nn.CrossEntropyLoss
        cost = loss_func()

        if optimizer is None:
            print("Optimizer was none, using default Optimizer")
            optimizer = torch.optim.Adam(params=module.parameters(), lr=lr)

        for param_group in optimizer.param_groups:
            print("Optimizer learning rate = ", param_group["lr"])

        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": [p for p in module.parameters()]})

        module.train()
        total_num_mini_batches = 0
        start_time = time.time()
        exit_flag = False
        total_loss = 0
        total = 0
        avg_loss = 0
        correct = 0
        epochs = 0
        total_accuracy = 0
        float_epochs = 0.0
        num_epochs = num_epochs or 1

        for epoch in range(num_epochs):
            num_mini_batches = 0
            for _, (train_x, train_label) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Mini Batches"
            ):
                data_entries = len(train_loader)
                exit_flag = self._exit_check(
                    epochs,
                    max_epochs,
                    max_mini_batches,
                    num_mini_batches,
                    start_time,
                    timeout_duration_s,
                )
                if exit_flag:
                    break

                train_x = train_x.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                predict_y = module(train_x)
                loss = cost(predict_y, train_label)
                loss.backward()
                optimizer.step()

                total += len(train_x)
                total_loss += loss.item()
                current_correct = (
                    (torch.argmax(predict_y, 1) == train_label).float().sum()
                ).item()
                correct += current_correct
                avg_loss = round(total_loss / (total_num_mini_batches + 1), 3)
                total_accuracy = round((correct / total) * 100, 3)
                epochs = epoch
                num_mini_batches += 1
                total_num_mini_batches += 1
                float_epochs = epoch + (num_mini_batches / data_entries)

                exit_flag = self._exit_check(
                    epochs,
                    max_epochs,
                    max_mini_batches,
                    num_mini_batches,
                    start_time,
                    timeout_duration_s,
                )
                if exit_flag:
                    break
                if stop_requested and stop_requested():
                    return {
                        "run_time": (time.time() - start_time),
                        "num_epochs": float_epochs,
                        "total_num_minibatches": total_num_mini_batches,
                        "loss": avg_loss,
                        "accuracy": total_accuracy,
                    }
            if exit_flag:
                break

        print(
            f"epochs, {float_epochs}, avg_loss ,{avg_loss}, total_accuracy, {total_accuracy}"
        )
        print(f"Training Round Finished {time.time() - start_time}sec")
        return {
            "time_taken_s": (time.time() - start_time),
            "num_epochs": float_epochs,
            "total_mini_batches": total_num_mini_batches,
            "loss": avg_loss,
            "accuracy": total_accuracy,
        }

    def default_validate_classifier(
        self,
        model,
        dataloader,
        loss_func=None,
        optimizer=None,
        device=None,
        round_no=None,
    ) -> dict:
        torch = self.torch
        device = device or self.resolve_device("cpu")
        if loss_func is None:
            loss_func = torch.nn.CrossEntropyLoss
        module = self._resolve_module(model)
        module = module.to(device)
        module.eval()

        acc = 0
        count = 0
        total_loss = 0
        batches = 0
        with torch.no_grad():
            cost = loss_func()
            for i, (x_batch, y_batch) in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Validation Round"
            ):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = module(x_batch)
                loss = cost(y_pred, y_batch)
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                count += len(y_batch)
                batches = i + 1

        module.train()
        return {
            "accuracy": (acc / count) * 100 if count else 0,
            "loss": total_loss / batches if batches else 0,
        }

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        torch = self.torch
        dataset = torch.load(dataset_path, weights_only=False).dataset
        dataset_len = len(dataset)
        split_idx = math.floor(0.95 * dataset_len)
        train_dataset = torch.utils.data.Subset(dataset, list(range(0, split_idx)))
        test_dataset = torch.utils.data.Subset(
            dataset, list(range(split_idx, dataset_len))
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=batch_size
        )
        return train_loader, test_loader

    def get_server_dataloader(self, batch_size=50, dataset_path=None):
        torch = self.torch
        test_dataset = torch.load(dataset_path, weights_only=False).dataset
        print("Length of test dataset", len(test_dataset))
        return torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )

    def save_weights(self, payload: WeightPayload, destination) -> None:
        self.torch.save(unwrap_weights(payload), destination)
