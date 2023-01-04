from typing import Protocol
from torch import nn


from DT_for_WDN_leak_localization.optimizers import Optimizers


class BaseTrainer(Protocol):

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizers,
    ) -> None:
        pass

    def _train_step(self) -> None:
        pass

    def _val_step(self) -> None:
        pass

    def fit(self) -> None:
        pass

