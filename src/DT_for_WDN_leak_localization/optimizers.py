import torch
from torch import nn
from torch import optim
import pdb

from DT_for_WDN_leak_localization.schedulers import CosineWarmupScheduler

class Optimizers:
    """
    Optimizers
    """

    def __init__(
        self, 
        model: nn.Module,
        params: dict,
        ) -> None:

        if model.encoder:
            self.encoder_optimizer = get_optimizer(
                model=model.encoder,
                optimizer_type=params['optimizer_params']['type'],
                optimizer_args=params['optimizer_params']['encoder_args'],
            )
            self.encoder_scheduler = get_scheduler(
                optimizer=self.encoder_optimizer,
                scheduler_type=params['scheduler_params']['type'],
                scheduler_args=params['scheduler_params']['args'],
            )
        
        if model.decoder:
            self.decoder_optimizer = get_optimizer(
                model=model.decoder,
                optimizer_type=params['optimizer_params']['type'],
                optimizer_args=params['optimizer_params']['decoder_args'],
            )
            self.decoder_scheduler = get_scheduler(
                optimizer=self.decoder_optimizer,
                scheduler_type=params['scheduler_params']['type'],
                scheduler_args=params['scheduler_params']['args'],
            )

def get_optimizer(
    model: nn.Module,
    optimizer_type: str,
    optimizer_args: dict,
    ) -> optim.Optimizer:
    """Get optimizer"""
    
    optimizer_factory = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
    }
    return optimizer_factory[optimizer_type](
        model.parameters(),
        **optimizer_args,
    )

def get_scheduler(
    optimizer: Optimizers,
    scheduler_type: str,
    scheduler_args: dict,
    ):

    scheduler_factory = {
        'StepLR': optim.lr_scheduler.StepLR,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CosineWarmupScheduler': CosineWarmupScheduler
    }

    return scheduler_factory[scheduler_type](
        optimizer=optimizer,
        **scheduler_args,
    )