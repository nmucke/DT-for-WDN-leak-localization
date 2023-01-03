from enum import Enum, auto
from pathlib import Path
from typing import Protocol


import numpy as np


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):
    """
    A protocol for experiment tracking classes.    
    """

    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""
