import torch
from torch import nn
from DT_for_WDN_leak_localization.factories import (
    create_encoder,
    create_decoder
)


class SupervisedWassersteinAE(nn.Module):
    """
    Wasserstein Autoencoder
    """

    def __init__(
        self,
        model_architecture: str,
        model_args: dict,
        ):
        super().__init__()
        
        self.model_architecture = model_architecture
        self.model_args = model_args

        self.encoder = create_encoder(
            encoder_architecture=model_args['encoder']['architecture'],
            encoder_args=model_args['encoder']['args'],
        )

        self.decoder = create_decoder(
            decoder_architecture=model_args['decoder']['architecture'],
            decoder_args=model_args['decoder']['args'],
            supervised=True,
        )
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, x, pars):
        """Decode"""
        return self.decoder(x, pars)
            
    def forward(self, x, pars):
        """Forward pass"""
        return self.decoder(self.encoder(x), pars)


class UnsupervisedWassersteinAE(nn.Module):
    """
    Wasserstein Autoencoder
    """

    def __init__(
        self,
        model_architecture: str,
        model_args: dict,
        ):
        super().__init__()
        
        self.model_architecture = model_architecture
        self.model_args = model_args

        self.encoder = create_encoder(
            encoder_architecture=model_args['encoder']['architecture'],
            encoder_args=model_args['encoder']['args'],
        )

        self.decoder = create_decoder(
            decoder_architecture=model_args['decoder']['architecture'],
            decoder_args=model_args['decoder']['args'],
            supervised=False,
        )
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, x):
        """Decode"""
        return self.decoder(x)
            
    def forward(self, x):
        """Forward pass"""
        return self.decoder(self.encoder(x))
