

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


        '''
        self.distribution = torch.distributions.Normal(
            loc=torch.tensor(0.),
            scale=torch.tensor(1.),
            )
        noise_sample = self.distribution.sample(observations.shape)
        lol = observations + self.noise*observations.mean(axis=1)*noise_sample
        '''

        self.distribution = torch.distributions.Normal(
            loc=torch.tensor(0.),
            scale=torch.tensor(self.noise),
            )
        noise_sample = self.distribution.sample(observations.shape)
        '''
        lal = observations + noise_sample
        pdb.set_trace()

        lol = observations + self.noise*observations*noise_sample
        plt.figure()
        for i in range(5):
            plt.plot(lol[0, :, i].detach().numpy())
            plt.plot(observations[0, :, i].detach().numpy())
        plt.show()
        pdb.set_trace()
        print(torch.abs(lol-observations).mean())
        '''

        return observations + noise_sample#self.noise*observations.mean(axis=1)*noise_sample
    