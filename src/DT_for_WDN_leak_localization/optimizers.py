import torch
from torch import nn
from torch import optim
import pdb

class Optimizers:
    """
    Optimizers
    """

    def __init__(
        self, 
        model: nn.Module,
        optimizer_params: dict,
        ) -> None:

        if model.encoder:
            self.encoder_optimizer = self._get_optimizer(
                model=model.encoder,
                optimizer_type=optimizer_params['type'],
                optimizer_args=optimizer_params['encoder_args'],
            )
        
        if model.decoder:
            self.decoder_optimizer = get_optimizer(
                model=model.decoder,
                optimizer_type=optimizer_params['type'],
                optimizer_args=optimizer_params['decoder_args'],
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
