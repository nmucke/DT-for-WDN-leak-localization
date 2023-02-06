import torch
from torch import nn
import pdb


class GAN(nn.Module):
    """
    Wasserstein GAN
    """

    def __init__(
        self,
        critic: nn.Module,
        generator: nn.Module,
        ):
        super().__init__()

        self.critic = critic
        self.generator = generator

        self.latent_dim = self.generator.latent_dim
    
    def generate(self, x, pars):
        """Decode"""
        return self.generator(x, pars)
            
    def forward(self, z, pars):
        """Forward pass"""
        return self.generator(z, pars)
    
    @property
    def device(self):
        return next(self.parameters()).device.type

