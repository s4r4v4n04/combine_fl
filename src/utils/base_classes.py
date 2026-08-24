"""
Authors: Daksh
Copyright 2024 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

Abstract base classes for federated learning components.
"""

from abc import ABC, abstractmethod


class AbstractDataLoader(ABC):
    """Base class for all custom dataloaders.
    
    Any custom dataloader must implement both methods to ensure
    proper data loading for both client training and server validation.
    """

    @abstractmethod
    def get_train_test_dataset_loaders(self, batch_size, dataset_path, args):
        """Load and return dataloaders for client training.
        
        Args:
            batch_size: Batch size for the dataloaders
            dataset_path: Path to the dataset
            args: Additional arguments specific to the dataloader
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        pass

    @abstractmethod
    def get_server_dataloader(self, batch_size, dataset_path, args):
        """Load and return dataloader for server validation.
        
        Args:
            batch_size: Batch size for the dataloader
            dataset_path: Path to the dataset
            args: Additional arguments specific to the dataloader
            
        Returns:
            test_loader: Dataloader for server-side validation
        """
        pass


class AbstractTrainer(ABC):
    """Base class for all custom trainers.
    
    Any custom trainer must implement both training and validation methods.
    """

    @abstractmethod
    def train_model(self, model, results, train_loader, epochs, timeout_s,
                    loss_func=None, optimizer=None, device=None,
                    test_loader=None, args=None, start_time=None):
        """Train the model on the provided data.
        
        Args:
            model: The model to train
            results: Dictionary to store results
            train_loader: DataLoader for training data
            epochs: Number of epochs to train
            timeout_s: Training timeout in seconds
            loss_func: Loss function (optional)
            optimizer: Optimizer (optional)
            device: Device to train on (optional)
            test_loader: DataLoader for test data (optional)
            args: Additional trainer-specific arguments (optional)
            start_time: Training start time (optional)
            
        Returns:
            dict: Training results containing metrics like loss, accuracy, etc.
        """
        pass

    @abstractmethod
    def validate_model(self, model, dataloader, device=None, loss_func=None,
                       optimizer=None, round_no=None, args=None):
        """Validate the model on the provided data.
        
        Args:
            model: The model to validate
            dataloader: DataLoader for validation data
            device: Device to run validation on (optional)
            loss_func: Loss function (optional)
            optimizer: Optimizer (optional)
            round_no: Current training round number (optional)
            args: Additional validator-specific arguments (optional)
            
        Returns:
            dict: Validation results containing metrics like loss, accuracy, etc.
        """
        pass
