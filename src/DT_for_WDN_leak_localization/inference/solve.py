import pdb
import torch
import matplotlib.pyplot as plt
import ray

from DT_for_WDN_leak_localization.inference.forward_model import BaseForwardModel
from DT_for_WDN_leak_localization.inference.likelihood import Likelihood
from DT_for_WDN_leak_localization.inference.true_data import TrueData

#@ray.remote(num_gpus=1)
#@ray.remote
def compute_posterior_k(
    forward_model: BaseForwardModel,
    likelihood: Likelihood,
    obs: TrueData,
    prior: torch.Tensor,
    num_samples: int,
    leak_location: int,
    t_idx: int,
    batch_size: int = None,
    device=torch.device("cpu"),
):
    """Compute posterior for a given leak location"""

    with torch.no_grad():
        if batch_size is not None:

            likelihood_mean_k = []
            for _ in range(0, num_samples, batch_size):
                
                # Compute ensemble of states
                state_pred = forward_model(
                    num_samples=batch_size, 
                    leak_location=torch.tensor([leak_location], device=device), 
                    time=torch.tensor([t_idx], device=device)
                )


                state_pred = state_pred
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
                leak_location=torch.tensor([leak_location], device=device), 
                time=torch.tensor([t_idx], device=device)
                )

            state_pred = state_pred
            # Likelihood of each sample
            likelihood_k = likelihood.compute_likelihood(
                state=state_pred,
                obs=obs
                )

            # Compute posterior
            mean_likelihood_k = torch.mean(likelihood_k) + 1e-8

    posterior_k = mean_likelihood_k * prior

    forward_model.pars_init = False

    return posterior_k

def solve_inverse_problem(
    true_data: TrueData,
    forward_model: BaseForwardModel,
    likelihood: Likelihood,
    num_samples: int,
    time: list,
    prior: torch.Tensor = None,
    batch_size: bool = False,
    device=torch.device("cpu"),
    ) -> torch.Tensor:
    """Solve the inverse problem"""

    uniform_prior = 1 / len(true_data.wdn.edges.ids)
    if prior is not None:
        if prior[prior==0.0] is not None:
            prior[prior==0.0] = 1e-12
    if prior is None:
        prior = torch.ones(len(true_data.wdn.edges.ids)) / len(true_data.wdn.edges.ids)
    prior = prior.to(device)

    forward_model = forward_model.to(device)

    potential_leak_locations = true_data.wdn.edges.ids

    posterior_list = []
    for t_idx in time:

        obs = true_data.obs[0, t_idx].to(device)

        posterior_k = []
        for leak_location in potential_leak_locations:
            
            '''
            state_pred = forward_model(
                num_samples=100, 
                leak_location=torch.tensor([leak_location], device=device), 
                time=torch.tensor([t_idx], device=device)
            )
            forward_model.pars_init = False

            state_pred_true = forward_model(
                num_samples=100, 
                leak_location=torch.tensor([true_data.leak.item()], device=device), 
                time=torch.tensor([t_idx], device=device)
            )
            state_pred_true_1 = forward_model(
                num_samples=100, 
                leak_location=torch.tensor([true_data.leak.item() + 20], device=device), 
                time=torch.tensor([t_idx], device=device)
            )
            state_pred_true_2 = forward_model(
                num_samples=100, 
                leak_location=torch.tensor([true_data.leak.item() + 10], device=device), 
                time=torch.tensor([t_idx], device=device)
            )

            plt.figure()
            plt.plot(true_data.state[0, t_idx].detach().to('cpu').numpy(), label='true')
            plt.plot(state_pred[0, :].detach().to('cpu').numpy(), label='pred')
            plt.plot(state_pred_true[0, :].detach().to('cpu').numpy(), label='pred_true')
            plt.plot(state_pred_true[20, :].detach().to('cpu').numpy(), label='pred_true_1')
            plt.plot(state_pred_true[10, :].detach().to('cpu').numpy(), label='pred_true_2')
            plt.legend()
            plt.show()

            pdb.set_trace()
            '''

            if prior[leak_location] < uniform_prior/2:
                posterior_k.append(torch.tensor(1e-12, device=device))
                continue

            posterior_k.append(compute_posterior_k(
                forward_model=forward_model,
                likelihood=likelihood,
                obs=obs,
                prior=prior[leak_location],
                num_samples=num_samples,
                leak_location=leak_location,
                t_idx=t_idx,
                batch_size=batch_size,
                device=device
            ))

            #print(leak_location)


        #posterior_k = torch.stack(ray.get(posterior_k))
        posterior_k = torch.stack(posterior_k)

        posterior_k = posterior_k / torch.sum(posterior_k)

        #print(f"KL: {len(posterior_k[posterior_k > uniform_prior/2])}")
        #print(torch.sum(posterior_k))

        # check convergence
        kl_divergence = torch.sum(prior * torch.log(prior / posterior_k))
        #print(f"KL divergence: {kl_divergence.item()}")
        if kl_divergence.item() < 1e-3:
            posterior_list.append(posterior_k.detach().to('cpu'))
            return posterior_list
        

        if len(posterior_k[posterior_k > uniform_prior/2]) < 2:
            posterior_list.append(posterior_k.detach().to('cpu'))
            return posterior_list

        prior = posterior_k

        posterior_list.append(prior.detach().to('cpu'))
        
        
    return posterior_list