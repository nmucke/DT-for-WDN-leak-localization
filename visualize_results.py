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

torch.set_default_dtype(torch.float32)

NET = 1

RESULT_PATH = f"results/net_{str(NET)}"

NOISE_LIST = [0.001, 0.005, 0.01, 0.05, 0.1] 
if NET == 1:
    NUM_SENSORS_LIST = [3, 5, 7]

def get_result_paths(noise, num_sensors):
    topological_distance_path = f"{RESULT_PATH}/topological_distance_noise_{str(noise)}_sensors_{str(num_sensors)}.npy"

    accuracy_path = f"{RESULT_PATH}/accuracy_noise_{str(noise)}_sensors_{str(num_sensors)}.npy"

    return topological_distance_path, accuracy_path

def main():

    topological_distance_mean = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))
    topological_distance_std = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))
    accuracy_list = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))

    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        for i, noise in enumerate(NOISE_LIST):
            topological_distance = np.load(
                get_result_paths(noise, num_sensors)[0]
            )
            topological_distance_mean[j, i] = np.mean(topological_distance)
            topological_distance_std[j, i] = np.std(topological_distance)

            accuracy = np.load(
                get_result_paths(noise, num_sensors)[1]
            )
            accuracy_list[j, i] = np.sum(accuracy)/len(accuracy) * 100
    
    plt.figure()
    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        plt.semilogx(
            NOISE_LIST, 
            topological_distance_mean[j], 
            '.-',
            linewidth=3, 
            markersize=20,
            label=f"{num_sensors} sensors"
            )
        plt.fill_between(
            NOISE_LIST,
            topological_distance_mean[j] - topological_distance_std[j],
            topological_distance_mean[j] + topological_distance_std[j],
            alpha=0.2
        )
    plt.xticks(NOISE_LIST)
    plt.xlabel("Noise")
    plt.ylabel("Average Topological Distance")
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/net_{str(NET)}/topological_distance.pdf")
    plt.show()

    plt.figure()
    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        plt.semilogx(
            NOISE_LIST, 
            accuracy_list[j], 
            '.-',
            linewidth=3,
            markersize=20, 
            label=f"{num_sensors} sensors"
            )
    plt.xticks(NOISE_LIST)
    plt.xlabel("Noise")
    plt.ylabel("Accuracy [%]")
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/net_{str(NET)}/accuracy.pdf")
    plt.show()

if __name__ == "__main__":
    main()