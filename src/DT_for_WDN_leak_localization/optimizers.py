import torch
from torch import nn
from torch import optim
import pdb

from DT_for_WDN_leak_localization.schedulers import CosineWarmupScheduler

class AEOptimizers:
    """
    Optimizers
    """

    def __init__(
        self, 
        model: nn.Module,
        args: dict,
        ) -> None:

        if model.encoder:
            self.encoder_optimizer = optim.Adam(
                model.encoder.parameters(),
                **args['optimizer']
            )
            self.encoder_scheduler = CosineWarmupScheduler(
                optimizer=self.encoder_optimizer,
                **args['scheduler']
            )
        
        if model.decoder:
            self.decoder_optimizer = optim.Adam(
                model.decoder.parameters(),
                **args['optimizer']
            )
            self.decoder_scheduler = CosineWarmupScheduler(
                optimizer=self.decoder_optimizer,
                **args['scheduler']
            )
    
    def step(self) -> None:
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
    
    def step_scheduler(self) -> None:
        self.encoder_scheduler.step()
        self.decoder_scheduler.step()
    
    def zero_grad(self) -> None:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


class GANOptimizers:
    """
    Optimizers
    """

    def __init__(
        self, 
        model: nn.Module,
        args: dict,
        ) -> None:

        if model.generator:
            self.generator_optimizer = optim.Adam(
                model.generator.parameters(),
                **args['optimizer']
            )
            self.generator_scheduler = CosineWarmupScheduler(
                optimizer=self.generator_optimizer,
                **args['scheduler']
            )
        
        if model.critic:
            self.critic_optimizer = optim.Adam(
                model.critic.parameters(),
                **args['optimizer']
            )
            self.critic_scheduler = CosineWarmupScheduler(
                optimizer=self.critic_optimizer,
                **args['scheduler']
            )
    
    def step(self) -> None:
        self.generator_optimizer.step()
        self.critic_optimizer.step()
    
    def step_scheduler(self) -> None:
        self.generator_scheduler.step()
        self.critic_scheduler.step()
    
    def zero_grad(self) -> None:
        self.generator_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
