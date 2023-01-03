import pdb
import torch
from torch import nn
from DT_for_WDN_leak_localization.model_architectures import (
    dense,
    transformers,
)

def create_model(
    model_type: str, 
    model_architecture: str,
    model_args: dict
    ):
    
    model_type_facotry = {
        'WAE': WAE,
        'VAE': VAE,
        'AE': AE,
    }

    model_architecture_factory = {
        'WAE': create_WAE,
        'VAE': create_VAE,
        'AE': create_AE,
    }
    

    return 2

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
            'transformer': transformers.SupervisedDecoder,
            'dense': dense.SupervisedDecoder,
        }
    else:
        decoder_architecture_factory = {
            'transformer': transformers.Decoder,
            'dense': dense.Decoder,
        }

    return decoder_architecture_factory[decoder_architecture](**decoder_args)