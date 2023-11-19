from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union
import torch


class BaseTrainer(ABC):

    def __init__(self,
                 device: str,
                 save_dir: Union[str, Path],
                 batch_size: int,
                 learning_rate: float,
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every trainer needs
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def create_dataloaders(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')