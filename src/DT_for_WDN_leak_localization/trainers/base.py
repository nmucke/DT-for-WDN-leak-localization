from pickletools import optimize
import torch


class BaseTrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: optimize,
    ) -> None:
        pass
    

    def train_step(self) -> None:
        pass

    def val_step(self) -> None:
        pass