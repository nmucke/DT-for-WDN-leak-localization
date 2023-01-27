import pickle
import numpy as np
import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

from DT_for_WDN_leak_localization.network import WDN

torch.set_default_dtype(torch.float32)

NET = 2
CONFIG_PATH = f"conf/net_{str(NET)}/data_preprocessing.yml"
DATA_PATH = f"data/raw_data/net_{str(NET)}/train_data"

FIGURE_SAVE_PATH = f"figures/net_{str(NET)}/"

NUM_SAMPLES = 10000

PREPROCESSOR_LOAD_PATH = f"trained_preprocessors/net_{str(NET)}_preprocessor.pkl"

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

num_pipes = config['num_pipes']
num_nodes = config['num_nodes']
num_time_steps = config['num_time_steps']

def main():

    demand = np.zeros((NUM_SAMPLES, num_time_steps, num_nodes))

    for i in range(NUM_SAMPLES):
        wdn = WDN(
            data_path=f"{DATA_PATH}/network_{i}",
        )
        _demand = wdn.nodes.demand.iloc[0:num_time_steps].values

        demand[i] = _demand

    demand[demand < 0] = 0
    demand *= 1e3
    total_demand = np.sum(demand, axis=-1)

    demand_mean = np.mean(total_demand, axis=0)
    demand_std = np.std(total_demand, axis=0)

    plt.figure()
    plt.plot(demand_mean, linewidth=2)
    plt.fill_between(
        np.arange(num_time_steps), 
        demand_mean - demand_std, 
        demand_mean + demand_std, 
        alpha=0.25
        )
    plt.xlabel("Time [Hours]")
    plt.ylabel("Total demand [l/s]")
    plt.grid()
    plt.savefig(f"{FIGURE_SAVE_PATH}total_demand.pdf")
    plt.show()

    if NET == 1:
        node_size = 30
    else:
        node_size = 15
        
    plt.figure()
    nx.draw_networkx(
        G=wdn.graph, 
        pos=nx.get_node_attributes(wdn.graph, 'pos'), 
        with_labels=False,
        node_size=node_size,
        width=1.5,
        node_color='k',
        arrowstyle='-',
        )
    plt.savefig(f"{FIGURE_SAVE_PATH}network.pdf")
    plt.show()


    if NET == 2:
        
        plotting_nodes = [92, 93, 94]

        plt.figure()
        for node_id in plotting_nodes:
            
            demand_node = demand[:, :, node_id]
            demand_node_mean = np.mean(demand_node, axis=0)
            demand_node_std = np.std(demand_node, axis=0)

            node_label = wdn.nodes.index_to_label[node_id]

            plt.plot(
                np.arange(num_time_steps),
                demand_node_mean, 
                linewidth=2,
                label=f"Tank {node_label}"
                )
            plt.fill_between(
                np.arange(num_time_steps), 
                demand_node_mean - demand_node_std, 
                demand_node_mean + demand_node_std, 
                alpha=0.25
                )
        plt.xlabel("Time [Hours]")
        plt.ylabel("Total demand [l/s]")
        plt.legend()
        plt.grid()
        plt.savefig(f"{FIGURE_SAVE_PATH}tank_demand.pdf")
        plt.show()

        plotting_nodes = [1, 10, 45, 60, 80]

        plt.figure()
        for node_id in plotting_nodes:
            
            demand_node = demand[:, :, node_id]
            demand_node_mean = np.mean(demand_node, axis=0)
            demand_node_std = np.std(demand_node, axis=0)

            node_label = wdn.nodes.index_to_label[node_id]

            plt.plot(
                np.arange(num_time_steps),
                demand_node_mean, 
                linewidth=2,
                label=f"Node {node_label}"
                )
            plt.fill_between(
                np.arange(num_time_steps), 
                demand_node_mean - demand_node_std, 
                demand_node_mean + demand_node_std, 
                alpha=0.5
                )
        plt.xlabel("Time [Hours]")
        plt.ylabel("Total demand [l/s]")
        plt.legend()
        plt.grid()
        plt.savefig(f"{FIGURE_SAVE_PATH}node_demand.pdf")
        plt.show()


if __name__ == "__main__":
    main()