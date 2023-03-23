import pickle
import numpy as np
import ray
import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import wntr
import networkx as nx

from DT_for_WDN_leak_localization.inference.forward_model import ForwardModel
from DT_for_WDN_leak_localization.inference.likelihood import Likelihood
from DT_for_WDN_leak_localization.inference.metrics import InverseProblemMetrics
from DT_for_WDN_leak_localization.inference.noise import ObservationNoise
from DT_for_WDN_leak_localization.inference.observation import ObservationModel
from DT_for_WDN_leak_localization.inference.prior import get_prior
from DT_for_WDN_leak_localization.inference.solve import solve_inverse_problem
from DT_for_WDN_leak_localization.inference.true_data import TrueData
from DT_for_WDN_leak_localization.network import WDN

torch.set_default_dtype(torch.float32)

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

NET = 4
CONFIG_PATH = f"conf/net_{str(NET)}/inverse_problem.yml"
DATA_PATH = f"data/raw_data/net_{str(NET)}/test_data"

NUM_SAMPLES = 100

NUM_WORKERS = 30

DENSE = False

PLOT = False

CUDA = True
if CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Set batch size to 1 if NET == 3. This is necessary if the memory is not enough
if NET == 3:
    BATCH_SIZE = 500
elif NET == 2:
    BATCH_SIZE = 30000
elif NET == 4:
    BATCH_SIZE = 2000
else:
    BATCH_SIZE = None

MODEL_LOAD_PATH = f"trained_models/net_{str(NET)}/"

if DENSE:
    MODEL_LOAD_NAME = f"dense_Supervised_WAE_net_{str(NET)}.pt"
else:
    MODEL_LOAD_NAME = f"Supervised_WAE_net_{str(NET)}.pt"
MODEL_LOAD_PATH = os.path.join(MODEL_LOAD_PATH, MODEL_LOAD_NAME)

PREPROCESSOR_LOAD_PATH = f"trained_preprocessors/net_{str(NET)}_preprocessor.pkl"

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

PRIOR = config['prior']
if PRIOR:
    prior = get_prior(net=NET, data_path=DATA_PATH)
    prior = torch.tensor(prior, dtype=torch.float32)

    DATA_PATH += "_prior"
else:
    prior = None

if DENSE:
    save_string = f"results/dense/net_{str(NET)}"
else:
    save_string = f"results/net_{str(NET)}"

def main():

    model = torch.load(MODEL_LOAD_PATH).to("cpu")
    model.eval()
    preprocessor = pickle.load(open(PREPROCESSOR_LOAD_PATH, "rb"))

    #pytorch_total_params = sum(p.numel() for p in model.decoder.parameters())
    #print(f"Number of parameters: {pytorch_total_params}")
    #pdb.set_trace()

    wdn = WDN(
        data_path=f"{DATA_PATH}/network_0",
    )
    
    for obs_case_key in config['observation_args'].keys():
        for noise in config['noise_args']['noise']:
            
            observation_model = ObservationModel(
                wdn=wdn,
                **config['observation_args'][obs_case_key],        
            )
            observation_noise = ObservationNoise(
                noise=noise,
            )

            forward_model = ForwardModel(
                generator=model.decoder,
                device=device,
            )

            topological_distance_list = []
            correct_leak_location_list = []
            entropy_list = []

            pbar = tqdm(
                range(0, NUM_SAMPLES),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
            for i in pbar:
                
                # Load true data
                wdn = WDN(
                    data_path=f"{DATA_PATH}/network_{str(i)}",
                )

                # Set up true data
                true_data = TrueData(
                    wdn=wdn,
                    preprocessor=preprocessor,
                    observation_model=observation_model,
                    observation_noise=observation_noise,
                )
                
                # Set up likelihood
                likelihood = Likelihood(
                    observation_model=observation_model,
                    observation_noise=observation_noise,
                    #**config['likelihood_args'],
                )

                # Solve inverse problem
                posterior = solve_inverse_problem(
                    true_data=true_data,
                    forward_model=forward_model,
                    likelihood=likelihood,
                    time=range(6, 17),
                    prior=prior,
                    batch_size=BATCH_SIZE,
                    device=device,
                    **config['solve_args'],
                )

                # Compute metrics
                metrics = InverseProblemMetrics(
                    true_data=true_data,
                    posterior=posterior
                )

                topological_distance_list.append(metrics.topological_distance)
                correct_leak_location_list.append(metrics.is_correct)

                entropy = metrics.entropy
                entropy_list.append(entropy)

                #time_topological_distance_list.append(metrics.time_topological_distance)

                pbar.set_postfix({
                    'topological_distance': np.mean(topological_distance_list),
                    'Accuracy': np.sum(correct_leak_location_list)/len(correct_leak_location_list),
                })
                
                if PLOT:
                    metrics.plot_posterior_on_graph(
                        #edge_obs_labels=observation_model.edge_obs_labels,
                        node_obs_labels=observation_model.node_obs_labels
                        )

            topological_distance_list = np.array(topological_distance_list)
            correct_leak_location_list = np.array(correct_leak_location_list)
            entropy_list = np.array(entropy_list)
            
            # Save results
            if PRIOR:
                save_string_case = f"_noise_{noise}_sensors_{len(config['observation_args'][obs_case_key]['edge_obs'])}_prior.npy"
            else:
                save_string_case  = f"_noise_{noise}_sensors_{len(config['observation_args'][obs_case_key]['edge_obs'])}.npy"

            np.save(
                f"{save_string}/topological_distance{save_string_case}", 
                topological_distance_list
                )
            np.save(
                f"{save_string}/accuracy{save_string_case}",
                correct_leak_location_list
                )
            np.save(
                f"{save_string}/entropy{save_string_case}",
                entropy_list
                )
    
if __name__ == "__main__":
    #ray.shutdown()
    #ray.init(num_cpus=NUM_WORKERS)
    main()
    #ray.shutdown()