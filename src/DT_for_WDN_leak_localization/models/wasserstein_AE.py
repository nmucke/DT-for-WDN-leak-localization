import torch
from torch import nn
import pdb


class SupervisedWassersteinAE(nn.Module):
    """
    Wasserstein Autoencoder
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, x, pars):
        """Decode"""
        return self.decoder(x, pars)
            
    def forward(self, z, pars):
        """Forward pass"""
        return self.decoder(z, pars)
    
    @property
    def device(self):
        return next(self.parameters()).device.type


class UnsupervisedWassersteinAE(nn.Module):
    """
    Wasserstein Autoencoder
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, x):
        """Decode"""
        return self.decoder(x)
            
    def forward(self, z):
        """Forward pass"""
        return self.decoder(z)
    
    @property
    def device(self):
        return next(self.parameters()).device.type
