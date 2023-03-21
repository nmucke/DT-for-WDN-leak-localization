import pickle
import numpy as np
import ray
import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({'font.size': 16})


torch.set_default_dtype(torch.float32)

NET = 2

PRIOR = True

NUM_SAMPLES = 200

DENSE = False

if DENSE:
    RESULT_PATH = f"results/dense/net_{str(NET)}"
else:
    RESULT_PATH = f"results/net_{str(NET)}"

NOISE_LIST = [0.005, 0.01, 0.025, 0.05, 0.1]
if NET == 1:
    NUM_SENSORS_LIST = [3, 5, 7]
elif NET == 2:
    NUM_SENSORS_LIST = [9, 11]
elif NET == 4:
    NUM_SENSORS_LIST = [9, 14]
    NOISE_LIST = [0.01, 0.025, 0.05]

def get_result_paths(noise, num_sensors, prior=False):
    if prior:
        topological_distance_path = f"{RESULT_PATH}/topological_distance_noise_{str(noise)}_sensors_{str(num_sensors)}_prior.npy"

        accuracy_path = f"{RESULT_PATH}/accuracy_noise_{str(noise)}_sensors_{str(num_sensors)}_prior.npy"

        entropy_path = f"{RESULT_PATH}/entropy_noise_{str(noise)}_sensors_{str(num_sensors)}_prior.npy"

    else:
        topological_distance_path = f"{RESULT_PATH}/topological_distance_noise_{str(noise)}_sensors_{str(num_sensors)}.npy"

        accuracy_path = f"{RESULT_PATH}/accuracy_noise_{str(noise)}_sensors_{str(num_sensors)}.npy"

        entropy_path = f"{RESULT_PATH}/entropy_noise_{str(noise)}_sensors_{str(num_sensors)}.npy"

    return topological_distance_path, accuracy_path, entropy_path

def main():

    topological_distance_mean = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))
    topological_distance_std = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))
    accuracy_list = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST)))
    entropy_list = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST), NUM_SAMPLES))

    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        for i, noise in enumerate(NOISE_LIST):
            topological_distance = np.load(
                get_result_paths(noise, num_sensors, prior=PRIOR)[0]
            )
            topological_distance_mean[j, i] = np.mean(topological_distance)
            topological_distance_std[j, i] = np.std(topological_distance)

            accuracy = np.load(
                get_result_paths(noise, num_sensors, prior=PRIOR)[1]
            )
            accuracy_list[j, i] = np.sum(accuracy)/len(accuracy) * 100

            #entropy = np.load(
            #    get_result_paths(noise, num_sensors)[2]
            #)
            
            #entropy_list[j, i, :] = entropy

            print(f"Num sensors: {num_sensors}, Noise: {noise}, Topological distance: {topological_distance_mean[j, i]:.2f}, Accuracy: {accuracy_list[j, i]:.2f} %")

    print(f"Topological distance mean: {topological_distance_mean.mean()}")
    print(f"Accuracy: {accuracy_list.mean()}")
    

    plot_x_ticks = [f'{noise*100}%' for noise in NOISE_LIST]
    fig, ax = plt.subplots(1,1) 
    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        plt.semilogx(
            NOISE_LIST, 
            topological_distance_mean[j], 
            '.-',
            linewidth=3, 
            markersize=20,
            label=f"{num_sensors} sensors"
            )
        '''
        plt.fill_between(
            np.log(NOISE_LIST),
            topological_distance_mean[j] - topological_distance_std[j],
            topological_distance_mean[j] + topological_distance_std[j],
            alpha=0.2
        )
        '''

    ax.set_xticks(NOISE_LIST)
    ax.set_xticklabels(plot_x_ticks)
    plt.xlabel("Noise")
    plt.ylabel("Average Topological Distance")
    plt.grid()
    plt.legend()
    if not DENSE:
        if PRIOR:
            plt.savefig(f"figures/net_{str(NET)}/topological_distance_prior.pdf")
        else:
            plt.savefig(f"figures/net_{str(NET)}/topological_distance.pdf")
    plt.show()

    fig, ax = plt.subplots(1,1) 
    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        plt.semilogx(
            NOISE_LIST, 
            accuracy_list[j], 
            '.-',
            linewidth=3,
            markersize=20, 
            label=f"{num_sensors} sensors"
            )
    ax.set_xticks(NOISE_LIST)
    ax.set_xticklabels(plot_x_ticks)
    plt.xlabel("Noise")
    plt.ylabel("Accuracy [%]")
    plt.grid()
    plt.legend()
    if not DENSE:
        if PRIOR:
            plt.savefig(f"figures/net_{str(NET)}/accuracy_prior.pdf")
        else:
            plt.savefig(f"figures/net_{str(NET)}/accuracy.pdf")
    plt.show()

    topological_distance_list = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST), NUM_SAMPLES))
    entropy_list = np.zeros((len(NUM_SENSORS_LIST), len(NOISE_LIST), NUM_SAMPLES))

    true_entropy_list = []
    false_entropy_list = []

    for j, num_sensors in enumerate(NUM_SENSORS_LIST):
        for i, noise in enumerate(NOISE_LIST):
            topological_distance = np.load(
                get_result_paths(noise, num_sensors, prior=PRIOR)[0]
            )
            topological_distance_list[j, i, :] = topological_distance


            accuracy = np.load(
                get_result_paths(noise, num_sensors, prior=PRIOR)[1]
            )

            entropy = np.load(
                get_result_paths(noise, num_sensors, prior=PRIOR)[2]
            )
            
            entropy_list[j, i, :] = entropy

            true_entropy_list.append(entropy[accuracy])
            false_entropy_list.append(entropy[~accuracy])

    true_entropy_list = np.concatenate(true_entropy_list)
    false_entropy_list = np.concatenate(false_entropy_list)

    true_entropy_kde = gaussian_kde(true_entropy_list)
    false_entropy_kde = gaussian_kde(false_entropy_list)

    x = np.linspace(0, 4, 1000)

    true_entropy_kde = true_entropy_kde(x)
    false_entropy_kde = false_entropy_kde(x)
    
    fig, ax = plt.subplots(1,1)
    plt.plot(x, true_entropy_kde, color="tab:blue")
    plt.plot(x, false_entropy_kde, color="tab:orange")
    plt.hist(
        true_entropy_list,
        bins=30,
        density=True,
        alpha=0.5,
        label="Correct Prediction",
        color="tab:blue"
        )
    plt.hist(
        false_entropy_list,
        bins=30,
        density=True,
        alpha=0.5,
        label="Incorrect Prediction",
        color="tab:orange"
        )
    plt.xlim([0, 3.2])
    plt.xlabel("Entropy")
    plt.ylabel("Probability")
    plt.grid()
    plt.legend()

    print(f"True entropy mean: {np.mean(true_entropy_list)}")
    print(f"False entropy mean: {np.mean(false_entropy_list)}")
    
    if not DENSE:
        if PRIOR:
            plt.savefig(f"figures/net_{str(NET)}/entropy_hist_prior.pdf")
        else:
            plt.savefig(f"figures/net_{str(NET)}/entropy_hist.pdf")

    plt.show()

    xy_stack = np.stack((topological_distance_list.flatten(), entropy_list.flatten()))
    z = gaussian_kde(xy_stack)(xy_stack)

    x_lim = [np.min(topological_distance_list), np.max(topological_distance_list)]
    y_lim = [np.min(entropy_list), np.max(entropy_list)]
    fig, ax = plt.subplots(1,1)
    plt.scatter(
        topological_distance_list.flatten(), 
        entropy_list.flatten(),
        c='tab:blue',
        s=z*1000,
        )
    plt.xlabel("Topological Distance")
    plt.ylabel("Entropy")
    plt.grid()
    if not DENSE:
        if PRIOR:
            plt.savefig(f"figures/net_{str(NET)}/entropy_prior.pdf")
        else:
            plt.savefig(f"figures/net_{str(NET)}/entropy.pdf")
    plt.show()

    '''
    fig, ax = plt.subplots(1,1)
    plt.plot(
        topological_distance_list.flatten(), 
        entropy_list.flatten(),
        '.',
        markersize=10,
        )
    plt.xlabel("Topological Distance")
    plt.ylabel("Entropy")
    plt.grid()
    if PRIOR:
        plt.savefig(f"figures/net_{str(NET)}/entropy_prior.pdf")
    else:
        plt.savefig(f"figures/net_{str(NET)}/entropy.pdf")
    plt.show()
    '''

if __name__ == "__main__":
    main()