import pdb
import torch
import matplotlib.pyplot as plt
import ray

from DT_for_WDN_leak_localization.inference.forward_model import BaseForwardModel
from DT_for_WDN_leak_localization.inference.likelihood import Likelihood
from DT_for_WDN_leak_localization.inference.true_data import TrueData

@ray.remote
def compute_posterior_k(
    forward_model: BaseForwardModel,
    likelihood: Likelihood,
    obs: TrueData,
    prior: torch.Tensor,
    num_samples: int,
    leak_location: int,
    t_idx: int,
    batch_size: int = None,
):


    if batch_size is not None:

        likelihood_mean_k = []
        for _ in range(0, num_samples, batch_size):

            # Compute ensemble of states
            state_pred = forward_model(
                num_samples=num_samples, 
                leak_location=torch.tensor([leak_location]), 
                time=torch.tensor([t_idx])
            ).detach()

            # Likelihood of each sample
            likelihood_k = likelihood.compute_likelihood(
                state=state_pred,
                obs=obs
                )
            
            likelihood_mean_k.append(torch.mean(likelihood_k))
        
        likelihood_mean_k = torch.stack(likelihood_mean_k)
        mean_likelihood_k = torch.mean(likelihood_mean_k) + 1e-8
            
    else:
        # Compute ensemble of states
        state_pred = forward_model(
            num_samples=num_samples, 
            leak_location=torch.tensor([leak_location]), 
            time=torch.tensor([t_idx])
            ).detach()

        # Likelihood of each sample
        likelihood_k = likelihood.compute_likelihood(
            state=state_pred,
            obs=obs
            )

        # Compute posterior
        mean_likelihood_k = torch.mean(likelihood_k) + 1e-8

    posterior_k = mean_likelihood_k * prior

    return posterior_k

def solve_inverse_problem(
    true_data: TrueData,
    forward_model: BaseForwardModel,
    likelihood: Likelihood,
    num_samples: int,
    time: list,
    prior: torch.Tensor = None,
    batch_size: bool = False,
    ) -> torch.Tensor:
    """Solve the inverse problem"""

    if prior is None:
        prior = torch.ones(len(true_data.wdn.edges.ids)) / len(true_data.wdn.edges.ids)

    posterior_list = []
    for t_idx in time:
        posterior_k = []
        for leak_location in true_data.wdn.edges.ids:

            posterior_k.append(compute_posterior_k.remote(
                forward_model=forward_model,
                likelihood=likelihood,
                obs=true_data.obs[0, t_idx],
                prior=prior[leak_location],
                num_samples=num_samples,
                leak_location=leak_location,
                t_idx=t_idx,
                batch_size=batch_size,
            ))

        posterior_k = torch.stack(ray.get(posterior_k))

        posterior_k = posterior_k / torch.sum(posterior_k)

        # check convergence
        kl_divergence = torch.sum(prior * torch.log(prior / posterior_k))

        if kl_divergence < 1e-3:
            posterior_list.append(posterior_k.detach())
            return posterior_list
        
        prior = posterior_k
        posterior_list.append(prior.detach())
        
    return posterior_list