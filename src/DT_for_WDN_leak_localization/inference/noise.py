

import pdb
import torch
import matplotlib.pyplot as plt


class ObservationNoise():
    def __init__(
        self, 
        noise: float,
        ) -> None:
        super().__init__()

        self.noise = noise
    
    def add_noise(self, observations: torch.Tensor) -> torch.Tensor:

        self.distribution = torch.distributions.Normal(
            loc=torch.tensor(0.),
            scale=torch.tensor(self.noise),
            )
        noise_sample = self.distribution.sample(observations.shape)
        
        return observations + noise_sample