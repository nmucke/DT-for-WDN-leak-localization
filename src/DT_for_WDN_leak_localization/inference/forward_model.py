import pdb
import torch
import torch.nn as nn


class BaseForwardModel(nn.Module):

    def __init__(
        self,
        ):
        super(BaseForwardModel, self).__init__()
        
        pass
    
    def forward(
        self,
        num_samples: int,
        pars: torch.Tensor,
        ):

        pass

class ForwardModel(BaseForwardModel):

    def __init__(
        self,
        generator: nn.Module,
        device: torch.device = torch.device('cpu'),
        ):
        super(ForwardModel, self).__init__()

        self.generator = generator
        self.latent_dim = generator.latent_dim
        self.device = device

        self.pars_init = False
    
    def _sample_latent(
        self,
        shape: tuple,
        ):
        return torch.randn(shape, device=self.device)
    
    def forward(
        self,
        num_samples: int,
        leak_location: torch.Tensor,
        time: torch.Tensor,
        ):

        latent_samples = self._sample_latent(
            shape=(num_samples, self.latent_dim)
        )

        if not self.pars_init:

            pars = torch.cat([leak_location, time], dim=0)
            #pars = pars.unsqueeze(0).repeat(num_samples, 1)

            if self.generator.transformer:
                pars = pars.unsqueeze(0)

                pars_1, pars_2 = self.generator.pars_forward(pars)

                self.pars_1 = pars_1.repeat(num_samples, 1, 1)
                self.pars_2 = pars_2.repeat(num_samples, 1, 1)
            
            else:
                pars = pars.unsqueeze(0)

                pars_1, pars_2 = self.generator.pars_forward(pars)

                self.pars_1 = pars_1.repeat(num_samples, 1)
                self.pars_2 = pars_2.repeat(num_samples, 1)

            self.pars_init = True

        return self.generator.state_forward(
            latent_samples, 
            pars_attn_1=self.pars_1,
            pars_attn_2=self.pars_2,
            )
