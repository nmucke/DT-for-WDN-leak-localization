import pdb
import torch
from torch import nn
from torch import optim
from DT_for_WDN_leak_localization.model_architectures import (
    dense,
    transformers,
)
from DT_for_WDN_leak_localization.models.wasserstein_AE import (
    SupervisedWassersteinAE,
    UnsupervisedWassersteinAE,
)
from DT_for_WDN_leak_localization.optimizers import Optimizers

from DT_for_WDN_leak_localization.trainers import (
    WAE_trainer
)

def create_diffusion_model(
    model_params: dict,
    AE: nn.Module,
):

    return 2

def create_train_stepper(
    model: nn.Module,
    optimizer: Optimizers,
    params: dict,
):
    """Get trainer"""

    trainer_factory = {
        'WAE': WAE_trainer.WAETrainStepper,
    }

    return trainer_factory[params['model_params']['type']](
        model=model,
        optimizer=optimizer,
        params=params,
        )

def create_AE(model_params: dict) -> nn.Module:
    """Create model based on model_params."""

    # Create encoder
    encoder = create_encoder(
        encoder_architecture=model_params['architecture'],
        encoder_args=model_params['encoder'],
    )

    # Check if decoder is should be supervised
    if model_params['type'] in ['WAE']:
        supervised = True
    else:
        supervised = False

    # Create decoder
    decoder = create_decoder(
        decoder_architecture=model_params['architecture'],
        decoder_args=model_params['decoder'],
        supervised=supervised,
    )

    # Create model
    model_factory = {
        'WAE': SupervisedWassersteinAE,
    }

    return model_factory[model_params['type']](encoder, decoder)
    
def create_encoder(
    encoder_architecture: str,
    encoder_args: dict,
    ):

    encoder_architecture_factory = {
        'transformer': transformers.Encoder,
        'dense': dense.Encoder,
    }

    return encoder_architecture_factory[encoder_architecture](**encoder_args)

def create_decoder(
    decoder_architecture: str,
    decoder_args: dict,
    supervised: bool = False,
    ):

    if supervised:
        decoder_architecture_factory = {
            'transformer': transformers.CrossAttentionDecoder,
            'dense': dense.SupervisedDecoder,
        }
    else:
        decoder_architecture_factory = {
            'transformer': transformers.Decoder,
            'dense': dense.Decoder,
        }

    return decoder_architecture_factory[decoder_architecture](**decoder_args)

