"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

from ml_backends.registry import get_backend


class DataLoader:
    def __init__(self, backend: str = "pytorch"):
        self.backend_name = backend or "pytorch"
        self.backend = get_backend(self.backend_name)

    def get_train_loader(self, batch_size=16, dataset_path=None):
        train_loader, _ = self.get_train_test_dataset_loaders(
            batch_size=batch_size, dataset_path=dataset_path
        )
        return train_loader

    def get_test_loader(self, batch_size=16, dataset_path=None):
        _, test_loader = self.get_train_test_dataset_loaders(
            batch_size=batch_size, dataset_path=dataset_path
        )
        return test_loader

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        return self.backend.get_train_test_dataset_loaders(
            batch_size=batch_size, dataset_path=dataset_path
        )
