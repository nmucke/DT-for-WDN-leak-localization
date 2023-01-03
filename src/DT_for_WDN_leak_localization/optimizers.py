import torch
from torch import nn
from torch import optim

class Optimizers:
    """
    Optimizers
    """

    def __init__(
        self, 
        model: nn.Module,
        optimizer_args: dict,
        optimizer_type: str='Adam',
        ) -> None:

        if model.encoder:
            self.encoder_optimizer = self._get_optimizer(
                model=model.encoder,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args['encoder'],
            )
        
        if model.decoder:
            self.decoder_optimizer = self._get_optimizer(
                model=model.decoder,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args['decoder'],
            )

    def _get_optimizer(
        self,
        model: nn.Module,
        optimizer_type: str,
        optimizer_args: dict,
        ):
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
