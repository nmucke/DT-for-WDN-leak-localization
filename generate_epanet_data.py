import ray
import os
import wntr
import numpy as np
import networkx as nx
import pdb
import os

from DT_for_WDN_leak_localization.generate_data import simulate_WDN

NUM_CPUS = 10

NET = 1
WITH_LEAK = True
TRAIN_OR_TEST = 'train'

NUM_SAMPLES = 10

LOWER_LEAK_AREA = 0.001
UPPER_LEAK_AREA = 0.005

if NET == 1:
    total_inflow_based_demand = True
else:
    total_inflow_based_demand = False

save_data_path = \
    f"data/raw_data/net_{str(NET)}/{TRAIN_OR_TEST}_data/"
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)
save_data_path += "network_"

epanet_data_path = f"EPANET_input_files/net_{str(NET)}.inp"

def main():

    # Reading the input file into EPANET
    wn = wntr.network.WaterNetworkModel(epanet_data_path)

    # Remove pump and valve links
    link_list = wn.link_name_list
    pump_list = wn.pump_name_list
    valve_list = wn.valve_name_list
    link_list = \
        [link for link in link_list if link not in pump_list + valve_list]
    
    ray.shutdown()
    ray.init(num_cpus=NUM_CPUS)
    sample_ids = range(0, NUM_SAMPLES)
    if WITH_LEAK:
        if TRAIN_OR_TEST == 'train':
            leak_pipes_id = np.random.randint(
                low=1, 
                high=len(link_list), 
                size=NUM_SAMPLES
                )
        else:
            leak_pipes_id = np.arange(0, len(link_list), 1)
            extra_samples = np.random.randint(
                low=1, 
                high=len(link_list), 
                size=NUM_SAMPLES-len(link_list)
                )
            leak_pipes_id = np.concatenate((leak_pipes_id, extra_samples))

        leak_pipes = [link_list[i] for i in leak_pipes_id]
        leak_areas = np.random.uniform(low=0.001, high=0.005, size=NUM_SAMPLES)
        
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            result_dict_leak = simulate_WDN(
            #result_dict_leak = simulate_WDN.remote(
                inp_file=epanet_data_path,
                leak={
                    'pipe': leak_pipe,
                    'area': leak_area
                },
                id=id,
                data_save_path=save_data_path,
                total_inflow_based_demand=total_inflow_based_demand
                )

    else:
        for id in sample_ids:
            result_dict = simulate_WDN(
                inp_file=epanet_data_path,
                )

    ray.shutdown()
if __name__ == "__main__":
    main()
