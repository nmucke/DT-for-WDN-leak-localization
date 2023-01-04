from torch import nn
import torch

from DT_for_WDN_leak_localization.optimizers import Optimizers

def MMD(
    x: torch.Tensor, 
    y: torch.Tensor,
    kernel: str,
    device: str
    ) -> torch.Tensor:
    """
    Emprical maximum mean discrepancy. The lower the result, 
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
        '''
        C = 2*x.shape[-1]*1
        XX += C * (C + dxx)**-1
        YY += C * (C + dyy)**-1
        XY += C * (C + dxy)**-1
        '''
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


class WAETrainStepper():

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizers,
        params: dict,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.kernel = params['training_params']['kernel']
        self.kernel_regu = params['training_params']['kernel_regu']

        self.device = model.device

        self.loss = nn.MSELoss()

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def _compute_losses(
        self, 
        state: torch.Tensor, 
        pars: torch.Tensor
        ):

        latent_state = self.model.encoder(state)

        # MMD loss
        mmd_loss = MMD(
            x=latent_state, 
            y=self._sample_latent(latent_state.shape), 
            kernel=self.kernel, 
            device=self.device
            )

        # Reconstruct state
        recon_state = self.model.decoder(latent_state, pars)

        # Reconstruction loss
        recon_loss = self.loss(state, recon_state)

        return mmd_loss, recon_loss

    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.optimizer.encoder_optimizer.zero_grad()
        self.optimizer.decoder_optimizer.zero_grad()

        # Compute losses
        mmd_loss, recon_loss = self._compute_losses(state, pars)

        # Total loss
        loss = recon_loss + self.kernel_regu * mmd_loss

        # Backpropagation
        loss.backward()
        self.optimizer.encoder_optimizer.step()
        self.optimizer.decoder_optimizer.step()

        return {
            'mmd_loss': mmd_loss.item(),
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
            'mmd_loss': mmd_loss.item(),
            'recon_loss': recon_loss.item()
        }
    
    def scheduler_step(self):
        self.optimizer.encoder_scheduler.step()
        self.optimizer.decoder_scheduler.step()