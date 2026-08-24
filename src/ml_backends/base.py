from abc import ABC, abstractmethod
from typing import Any, Callable

from ml_backends.weights import WeightPayload


class BackendAdapter(ABC):
    """Framework adapter for model, data, weight, and default loop behavior."""

    name: str

    def resolve_device(self, device: str | None = None, use_gpu: bool = False) -> Any:
        return device or ("gpu" if use_gpu else "cpu")

    def set_seed(self, seed: int) -> None:
        return None

    def build_model(self, model_cls: type, device: Any = None, args: dict | None = None):
        try:
            return model_cls(device=device, args=args)
        except TypeError:
            try:
                return model_cls(device, args=args)
            except TypeError:
                try:
                    return model_cls(args=args)
                except TypeError:
                    return model_cls()

    def move_model_to_device(self, model, device: Any):
        return model

    @abstractmethod
    def get_weights(self, model, model_id: str | None = None) -> WeightPayload:
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, model, payload: WeightPayload | Any) -> None:
        raise NotImplementedError

    def parameter_count(self, model) -> int:
        return 0

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def default_validate_classifier(
        self,
        model,
        dataloader,
        loss_func=None,
        optimizer=None,
        device=None,
        round_no=None,
    ) -> dict:
        raise NotImplementedError

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        raise NotImplementedError(
            f"Default dataloader is not implemented for backend '{self.name}'"
        )

    def get_server_dataloader(self, batch_size=50, dataset_path=None):
        _, test_loader = self.get_train_test_dataset_loaders(
            batch_size=batch_size, dataset_path=dataset_path
        )
        return test_loader

    def save_weights(self, payload: WeightPayload, destination) -> None:
        import pickle

        pickle.dump(payload, destination)
