from matplotlib import pyplot as plt
import ray
import os
import wntr
import numpy as np
import networkx as nx
import pdb
import os

from DT_for_WDN_leak_localization.generate_data import simulate_WDN
from DT_for_WDN_leak_localization.network import WDN

NUM_CPUS = 1

NET = 1
WITH_LEAK = True
TRAIN_OR_TEST = 'test'

NUM_SAMPLES = 100

LOWER_LEAK_AREA = 0.002
UPPER_LEAK_AREA = 0.004

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

WITH_PRIOR = True
if WITH_PRIOR:
    prior = np.arange(0,34)
    prior = prior/np.sum(prior)

    save_data_path = \
        f"data/raw_data/net_{str(NET)}/{TRAIN_OR_TEST}_data_prior/"
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    save_data_path += "network_"

DATA_PATH = f"data/raw_data/net_{str(NET)}/test_data"

'''
wn = WDN(
    data_path=f"{DATA_PATH}/network_{str(0)}",
)

posterior = prior
pos = {}
for key in wn.nodes.label_to_index.keys():
    pos[key] = nx.get_node_attributes(wn.graph, 'pos')[str(key)]

# Reorder posterior list for plotting on graph
posterior_list = [posterior[i] for i in range(len(posterior))]
posterior_for_plot = []
for edges in wn.graph.edges:
    posterior_for_plot.append(posterior_list[wn.edges.label_to_index[edges[-1]]])
    
edge_cmap = plt.get_cmap('Reds')
nx.draw_networkx(
    G=wn.graph, 
    pos=pos, 
    edge_vmin=np.min(posterior),
    edge_vmax=np.max(posterior),
    edge_color=posterior_for_plot, 
    edge_cmap=edge_cmap, 
    width=2,
    node_size=10, #node_color=head, node_cmap=node_cmap,
    with_labels=False
    )
sm = plt.cm.ScalarMappable(
    cmap=edge_cmap,
    norm=plt.Normalize(vmin=np.min(posterior), vmax=np.max(posterior)))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label(
    'P(c)', 
    rotation=270, 
    fontsize=15,
    labelpad=20
    )
plt.savefig(f"figures/network_leak_prior.pdf")
plt.show()
pdb.set_trace()
'''

'''
wn = wntr.network.WaterNetworkModel(epanet_data_path)
DMA1 = []
DMA2 = []
DMA3 = []
DMA4 = []
DMA5 = []
node_DMA = {}
for node_label in wn.nodes.junction_names:
    node = wn.get_node(node_label)
    DMA = node.demand_timeseries_list[0].pattern_name[0:4]

    if DMA == 'DMA1':
        DMA1.append(node_label)
    elif DMA == 'DMA2':
        DMA2.append(node_label)
    elif DMA == 'DMA3':
        DMA3.append(node_label)
    elif DMA == 'DMA4':
        DMA4.append(node_label)
    elif DMA == 'DMA5':
        DMA5.append(node_label)

pdb.set_trace()
'''

def main():

    # Reading the input file into EPANET
    wn = wntr.network.WaterNetworkModel(epanet_data_path)

    # Remove pump and valve links
    link_list = wn.link_name_list
    pump_list = wn.pump_name_list
    valve_list = wn.valve_name_list
    link_list = \
        [link for link in link_list if link not in pump_list + valve_list]
    
    sample_ids = range(0, NUM_SAMPLES)
    if WITH_LEAK:
        if TRAIN_OR_TEST == 'train':
            leak_pipes_id = np.random.randint(
                low=1, 
                high=len(link_list), 
                size=NUM_SAMPLES
                )
        else:

            if WITH_PRIOR:
                leak_pipes_id = np.random.multinomial(
                    n=1,
                    pvals=prior,
                    size=NUM_SAMPLES
                    )
                leak_pipes_id = np.argmax(leak_pipes_id, axis=1)
            else:
                leak_pipes_id = np.arange(0, len(link_list), 1)
                extra_samples = np.random.randint(
                    low=1, 
                    high=len(link_list), 
                    size=NUM_SAMPLES-len(link_list)
                    )
                leak_pipes_id = np.concatenate((leak_pipes_id, extra_samples))

        leak_pipes = [link_list[i] for i in leak_pipes_id]

        leak_areas = np.random.uniform(
            low=LOWER_LEAK_AREA,
            high=UPPER_LEAK_AREA,
            size=NUM_SAMPLES
            )
        results = []
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            #result_dict_leak = simulate_WDN(
            result_dict_leak = simulate_WDN.remote(
                inp_file=epanet_data_path,
                leak={
                    'pipe': leak_pipe,
                    'area': leak_area
                },
                id=id,
                data_save_path=save_data_path,
                total_inflow_based_demand=total_inflow_based_demand
                )
            results.append(result_dict_leak)
        results = ray.get(results)

    else:
        for id in sample_ids:
            result_dict = simulate_WDN(
                inp_file=epanet_data_path,
                )

if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_cpus=NUM_CPUS)
    main()
    ray.shutdown()
