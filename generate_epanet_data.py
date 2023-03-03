from matplotlib import pyplot as plt
import ray
import os
import wntr
import numpy as np
import networkx as nx
import pdb
import os

from DT_for_WDN_leak_localization.generate_data import simulate_WDN
from DT_for_WDN_leak_localization.inference.prior import get_prior
from DT_for_WDN_leak_localization.network import WDN

NUM_CPUS = 30

NET = 4
WITH_LEAK = True
TRAIN_OR_TEST = 'test'

NUM_SAMPLES = 500

LOWER_LEAK_AREA = 0.002
UPPER_LEAK_AREA = 0.004

demand_scaler = None
if NET == 1 or NET == 4:
    total_inflow_based_demand = True
    if NET == 4:
        demand_scaler = 0.1
else:
    total_inflow_based_demand = False

save_data_path = \
    f"data/raw_data/net_{str(NET)}/{TRAIN_OR_TEST}_data/"
if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)
save_data_path += "network_"

epanet_data_path = f"EPANET_input_files/net_{str(NET)}.inp"

WITH_PRIOR = False
if WITH_PRIOR:

    DATA_PATH = f"data/raw_data/net_{str(NET)}/test_data"

    wn = WDN(
        data_path=f"{DATA_PATH}/network_{str(0)}",
    )

    prior = get_prior(net=NET, data_path=DATA_PATH)
    prior = np.asarray(prior)
    prior = prior.astype(np.float64)
    prior = prior / np.sum(prior)

    save_data_path = \
        f"data/raw_data/net_{str(NET)}/{TRAIN_OR_TEST}_data_prior/"
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    save_data_path += "network_"
#else:
#    prior = np.ones(len(wn.edges.labels)) / len(wn.edges.labels)
'''
DATA_PATH = f"data/raw_data/net_{str(NET)}/train_data"
wdn = WDN(
    data_path=f"{DATA_PATH}/network_{str(0)}",
)
wn = wntr.network.WaterNetworkModel(epanet_data_path)
pos = {}
for key in wdn.nodes.label_to_index.keys():
    pos[key] = nx.get_node_attributes(wdn.graph, 'pos')[str(key)]
nx.draw_networkx(
    G=wdn.graph, 
    pos=pos, 
    width=0.2,
    node_size=5,
    with_labels=True,
    arrows=False,
    node_color='black'
    )
lol = {}
#for key in wdn.edges.labels:
#    lol[key] = key
for edges in wdn.graph.edges:
    lol[(edges[0], edges[1])] = edges[-1]
nx.draw_networkx_edge_labels(
                    G=wdn.graph, pos=pos,
                    edge_labels=lol, font_size=10,
                    bbox={'alpha':0.0})
plt.show()
pdb.set_trace()
'''
'''
wdn = wntr.network.WaterNetworkModel(epanet_data_path)

posterior = np.asarray(prior)
pos = {}
for key in wn.nodes.label_to_index.keys():
    pos[key] = nx.get_node_attributes(wn.graph, 'pos')[str(key)]

# Reorder posterior list for plotting on graph
posterior = [posterior[i] for i in range(len(posterior))]

posterior_for_plot = []
link_list = wdn.link_name_list
for edges in wn.graph.edges:
    posterior_for_plot.append(posterior[wn.edges.label_to_index[edges[-1]]])

edge_cmap = plt.get_cmap('Reds')
nx.draw_networkx(
    G=wn.graph, 
    pos=pos, 
    edge_vmin=np.min(posterior)-0.0001,
    edge_vmax=np.max(posterior),
    edge_color=posterior_for_plot, 
    edge_cmap=edge_cmap, 
    width=2,
    node_size=5,
    with_labels=False,
    arrows=False,
    node_color='black'
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
if WITH_PRIOR:
    plt.savefig(f"figures/net_{NET}/network_leak_prior.pdf")
else:
    plt.savefig(f"figures/net_{NET}/network_leak_no_prior.pdf")
plt.show()
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
                '''
                leak_pipes_id = np.arange(0, len(link_list), 1)
                extra_samples = np.random.randint(
                    low=1, 
                    high=len(link_list), 
                    size=NUM_SAMPLES-len(link_list)
                    )
                leak_pipes_id = np.concatenate((leak_pipes_id, extra_samples))
                '''
                leak_pipes_id = np.random.randint(
                    low=1, 
                    high=len(link_list), 
                    size=NUM_SAMPLES
                )

        leak_pipes = [link_list[i] for i in leak_pipes_id]

        leak_areas = np.random.uniform(
            low=LOWER_LEAK_AREA,
            high=UPPER_LEAK_AREA,
            size=NUM_SAMPLES
            )
        results = []
        #sample_ids = []
        for id, leak_pipe, leak_area in zip(sample_ids, leak_pipes, leak_areas):
            #result_dict_leak = simulate_WDN(
            result_dict_leak = simulate_WDN(#).remote(
                inp_file=epanet_data_path,
                leak={
                    'pipe': leak_pipe,
                    'area': leak_area
                },
                id=id,
                data_save_path=save_data_path,
                total_inflow_based_demand=total_inflow_based_demand,
                demand_scaler=demand_scaler,
                )
            results.append(result_dict_leak)
        results = ray.get(results)

    else:
        for id in sample_ids:
            result_dict = simulate_WDN(
                inp_file=epanet_data_path,
                )

if __name__ == "__main__":
    #ray.shutdown()
    #ray.init(num_cpus=NUM_CPUS)
    main()
    #ray.shutdown()
