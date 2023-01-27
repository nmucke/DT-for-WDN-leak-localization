from email.mime import base
import pdb
from re import S
from unittest import TextTestRunner

import numpy as np
import os
import wntr
import networkx as nx
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton, curve_fit
import torch
import ray

def get_demand_time_series_noise(t_start, t_end, t_step, base_value):

    noise_std = 5e-2*base_value
    demand_noise = np.random.normal(
        loc=0,
        scale=noise_std, 
        size=int((t_end - t_start) / t_step)
        )
    return demand_noise


@ray.remote(num_returns=1)
def simulate_WDN(
    inp_file, 
    leak=None, 
    data_save_path=None, 
    total_inflow_based_demand=False,
    id=0
    ):


    wn = wntr.network.WaterNetworkModel(inp_file)

    base_demands = []
    for node_key in wn.nodes.junction_names:
        base_demands.append(
            wn.get_node(node_key).demand_timeseries_list[0].base_value
            )
    base_demands = np.asarray(base_demands)


    wn.options.time.duration = 1*60*60*24
    wn.options.time.hydraulic_timestep = 60
    wn.options.time.pattern_timestep = 60*60
    wn.options.time.report_timestep = 60*60

    duration = wn.options.time.duration
    pattern_timestep = wn.options.time.pattern_timestep

    if total_inflow_based_demand:
        total_inflow = [
            2.1226, 1.7893, 1.5880, 1.4802, 1.4173, 1.3968, 1.8873, 3.6701,
            4.5846, 4.0725, 4.0539, 4.0444, 3.8519, 3.7347, 3.7883, 3.7886,
            3.3522, 3.1466, 3.0435, 3.0730, 3.4929, 3.2028, 2.7843, 2.4205
            ]
        total_inflow_with_noise = \
            total_inflow + \
            0.1*np.random.normal(0, 1, len(total_inflow)) * \
            total_inflow

        total_base_demand = base_demands.sum()
        demand_fraction = base_demands/total_base_demand

        j = 1
        for n in wn.nodes.junction_names:
            wn.add_pattern(n, total_inflow_with_noise)
            pat = wn.get_pattern(n)

            wn.get_node(n).demand_timeseries_list.clear()
            wn.get_node(n).demand_timeseries_list.append(
                (demand_fraction[j-1],  pat)
                )
            j = j + 1
    else:
        for node_key in wn.nodes.junction_names:
            base_value = \
                wn.get_node(node_key).demand_timeseries_list[0].base_value
            demand_noise = get_demand_time_series_noise(
                t_start=0, 
                t_end=duration,
                t_step=pattern_timestep,
                base_value=base_value
                )
            wn.add_pattern(node_key, demand_noise)
            pat = wn.get_pattern(node_key)
            wn.get_node(node_key).demand_timeseries_list.append((1., pat))

    if leak is not None:
        wn_leak = copy.deepcopy(wn)

        leak_start_time = 0
        leak_pipe = leak['pipe']

        leak_area = leak['area']
        wn_leak = wntr.morph.link.split_pipe(
            wn_leak, leak_pipe,
            'leak_pipe',
            'leak_node'
            )
        leak_node = wn_leak.get_node('leak_node')
        
        leak_node.add_leak(wn_leak, area=leak_area, start_time=leak_start_time)

        wn_leak.options.hydraulic.demand_model = 'DDA'

        sim = wntr.sim.WNTRSimulator(wn_leak)
    else:
        sim = wntr.sim.WNTRSimulator(wn)

    results = sim.run_sim()

    G = wn.get_graph()
    pipe_flowrates = copy.deepcopy(results.link['flowrate'])

    if leak is not None:
        leak['demand'] = results.node['leak_demand']['leak_node'][0]

        pipe_flowrates[f'{leak_pipe}'] = 0.5 * (pipe_flowrates[f'{leak_pipe}']
                                                + pipe_flowrates[f'leak_pipe'])
        head_df = results.node['head'].drop('leak_node', axis=1)
        demand_df = results.node['demand'].drop('leak_node', axis=1)
    else:
        head_df = results.node['head']
        demand_df = results.node['demand']

    if leak is not None:
        flowrate_df = results.link['flowrate'].drop('leak_pipe', axis=1)
        leak['start_time'] = leak_start_time

    if leak is not None:
        result_dict = {
            'graph': G,
            'head': head_df,
            'demand': demand_df,
            'flow_rate': flowrate_df,
            'leak': leak
        }
    else:
        result_dict = {
            'graph': G,
            'head': head_df,
            'demand': demand_df,
            'flow_rate': flowrate_df
        }

    nx.write_gpickle(result_dict, f'{data_save_path}{id}')
    
    print(id)
    return None

