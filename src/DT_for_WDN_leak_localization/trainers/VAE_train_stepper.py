from attr import dataclass
from torch import nn
import torch
import pdb

from DT_for_WDN_leak_localization.optimizers import AEOptimizers


class SupervisedWAETrainStepper():

    def __init__(
        self,
        model: nn.Module,
        optimizer: AEOptimizers,
        kernel: str,
        kernel_regu: float,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.kernel = kernel
        self.kernel_regu = kernel_regu

        self.device = model.device

        # Loss function
        self.loss = nn.MSELoss()

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def step_scheduler(self) -> None:
        self.optimizer.step_scheduler()
    
    def _compute_losses(
        self, 
        state: torch.Tensor, 
        pars: torch.Tensor
        ):

        latent_state = self.model.encode(state)

        # MMD loss
        mmd_loss = MMD(
            x=latent_state, 
            y=self._sample_latent(latent_state.shape), 
            kernel=self.kernel, 
            device=self.device
            )

        # Reconstruct state
        recon_state = self.model.decode(latent_state, pars)

        # Reconstruction loss
        recon_loss = self.loss(state, recon_state)

        return mmd_loss, recon_loss

    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.optimizer.zero_grad()

        # Compute losses
        mmd_loss, recon_loss = self._compute_losses(state, pars)

        # Total loss
        loss = recon_loss + self.kernel_regu * mmd_loss

        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 0.5)

        self.optimizer.step()

        return {
            'latent_loss': mmd_loss.item(),
            'recon_loss': recon_loss.item()
        }


    def val_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
    ) -> None:
        
        # Compute losses
        mmd_loss, recon_loss = self._compute_losses(state, pars)

        return {
            'latent_loss': mmd_loss.item(),
            'recon_loss': recon_loss.item()
        }
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
    
