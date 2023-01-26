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

from DT_for_WDN_leak_localization.inference.forward_model import ForwardModel

from DT_for_WDN_leak_localization.inference.likelihood import Likelihood
from DT_for_WDN_leak_localization.inference.metrics import InverseProblemMetrics
from DT_for_WDN_leak_localization.inference.noise import ObservationNoise

from DT_for_WDN_leak_localization.inference.observation import ObservationModel
from DT_for_WDN_leak_localization.inference.solve import solve_inverse_problem
from DT_for_WDN_leak_localization.inference.true_data import TrueData
from DT_for_WDN_leak_localization.network import WDN

torch.set_default_dtype(torch.float32)

NET = 1
CONFIG_PATH = f"conf/net_{str(NET)}/inverse_problem.yml"
DATA_PATH = f"data/raw_data/net_{str(NET)}/test_data"

NUM_SAMPLES = 100

NUM_WORKERS = 25

MODEL_LOAD_PATH = f"trained_models/net_{str(NET)}/"
#MODEL_LOAD_NAME = f"GAN_net_{str(NET)}.pt"
MODEL_LOAD_NAME = f"Supervised_WAE_net_{str(NET)}.pt"
MODEL_LOAD_PATH = os.path.join(MODEL_LOAD_PATH, MODEL_LOAD_NAME)

PREPROCESSOR_LOAD_PATH = f"trained_preprocessors/net_{str(NET)}_preprocessor.pkl"

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

def main():

    model = torch.load(MODEL_LOAD_PATH).to("cpu")
    preprocessor = pickle.load(open(PREPROCESSOR_LOAD_PATH, "rb"))

    wdn = WDN(
        data_path=f"{DATA_PATH}/network_0",
    )
    observation_model = ObservationModel(
        wdn=wdn,
        **config['observation_args'],        
    )
    observation_noise = ObservationNoise(
        **config['noise_args'],
    )

    #forward_model = ForwardModel(
    #    generator=model.generator,
    #)
    forward_model = ForwardModel(
        generator=model.decoder,
    )

    topological_distance_list = []
    correct_leak_location_list = []

    pbar = tqdm(
        range(NUM_SAMPLES),
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    for i in pbar:
        
        # Load true data
        wdn = WDN(
            data_path=f"{DATA_PATH}/network_{str(i)}",
        )
        true_data = TrueData(
            wdn=wdn,
            preprocessor=preprocessor,
            observation_model=observation_model,
            observation_noise=observation_noise,
        )
        
        likelihood = Likelihood(
            observation_model=observation_model,
            observation_noise=observation_noise,
            #**config['likelihood_args'],
        )

        posterior = solve_inverse_problem(
            true_data=true_data,
            forward_model=forward_model,
            likelihood=likelihood,
            time=range(2,5),
            **config['solve_args'],
        )

        metrics = InverseProblemMetrics(
            true_data=true_data,
            posterior=posterior,
        )

        topological_distance_list.append(metrics.topological_distance)
        correct_leak_location_list.append(metrics.is_correct)

        pbar.set_postfix({
            'topological_distance': np.mean(topological_distance_list),
            'Accuracy': np.sum(correct_leak_location_list)/len(correct_leak_location_list),
        })



        '''
        plt.figure()
        plt.plot(posterior[0].detach().numpy())
        plt.axvline(x=true_data.leak, color='r')
        plt.show()
        pdb.set_trace()
        '''
        


    
if __name__ == "__main__":
    ray.init(num_cpus=NUM_WORKERS)
    main()
    ray.shutdown()