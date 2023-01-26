

import pdb
import torch


class ObservationNoise():
    def __init__(
        self, 
        noise: float,
        ) -> None:
        super().__init__()

        self.noise = noise

        self.distribution = torch.distributions.Normal(
            loc=torch.tensor([0.0]),
            scale=torch.tensor([self.noise]),
            )
    
    def _sample(self, shape: int) -> torch.Tensor:
        return self.distribution.sample(shape).squeeze(-1)
    
    def add_noise(self, observations: torch.Tensor) -> torch.Tensor:
        return observations + self._sample(shape=observations.shape)
    