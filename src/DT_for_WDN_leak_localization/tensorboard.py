from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from DT_for_WDN_leak_localization.tracking import Stage
from DT_for_WDN_leak_localization.utils import create_experiment_log_dir


class TensorboardExperiment:
    """
    A wrapper around the tensorboard SummaryWriter class to make it easier to use.    
    """

    def __init__(self, log_path: str, create: bool = True):

        log_dir = create_experiment_log_dir(root=log_path)
        self.stage = Stage.TRAIN
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    @staticmethod
    def collapse_batches(
        y_true: list[np.array], y_pred: list[np.array]
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)
