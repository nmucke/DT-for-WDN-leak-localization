import pdb
import torch
from torch import nn
from DT_for_WDN_leak_localization.inference.observation import ObservationModel
from DT_for_WDN_leak_localization.inference.noise import ObservationNoise

import matplotlib.pyplot as plt

class Likelihood():
    def __init__(
        self, 
        observation_model: ObservationModel,
        observation_noise: ObservationNoise,
        ) -> None:
        super().__init__()

        self.observation_model = observation_model
        self.observation_noise = observation_noise

    def _compute_log_likelihood(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Computes the log-likelihood of the observations given the state.
        """
        # Compute the log-likelihood of the observations given the state
        log_likelihood = self.observation_noise.distribution.log_prob(
            obs-self.observation_model.get_observations(state),
            ).sum(dim=-1)

        return log_likelihood
    
    def compute_likelihood(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        ) -> torch.Tensor:
        """
        Computes the likelihood of the observations given the state.
        """
        return torch.exp(self._compute_log_likelihood(state, obs))
        