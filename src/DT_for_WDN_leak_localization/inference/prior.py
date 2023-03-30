import pdb
import wntr
import torch
import networkx as nx

from DT_for_WDN_leak_localization.network import WDN

def get_prior(net, data_path):
    epanet_data_path =  f"EPANET_input_files/net_{str(net)}.inp"
    
    if net == 1:
        return get_prior_net_1(data_path, epanet_data_path)
    
    elif net == 2:
        return get_prior_net_2(data_path, epanet_data_path)

    elif net == 3:
        return get_prior_net_3(data_path, epanet_data_path)

    elif net == 4:
        return get_prior_net_4(data_path, epanet_data_path)
        


def get_prior_net_1(data_path, epanet_data_path):

    prior = torch.arange(0, 34)
    prior = prior/torch.sum(prior)

    return prior
    
def get_prior_net_2(data_path, epanet_data_path):

    wdn = wntr.network.WaterNetworkModel(epanet_data_path)

    wn = WDN(
        data_path=f"{data_path}/network_{str(0)}",
    )
    pump_list = wdn.pump_name_list

    non_leak_edges = [wn.edges.label_to_index[pump] for pump in pump_list]

    prior = torch.arange(0,119)
    prior = torch.pow(prior,3)
    prior[non_leak_edges] = 0
    prior = prior/torch.sum(prior)

    return prior

def get_prior_net_4(data_path, epanet_data_path):

    wdn = wntr.network.WaterNetworkModel(epanet_data_path)

    wn = WDN(
        data_path=f"{data_path}/network_{str(0)}",
    )
    pump_list = wdn.pump_name_list

    non_leak_edges = [wn.edges.label_to_index[pump] for pump in pump_list]

    prior = torch.arange(0,313)
    prior = torch.cat((prior, torch.zeros(4)))
    prior = torch.pow(prior,3)
    prior[non_leak_edges] = 0
    prior = prior/torch.sum(prior)

    return prior

def get_prior_net_3(data_path, epanet_data_path):

    wn = wntr.network.WaterNetworkModel(epanet_data_path)

    wdn = WDN(
        data_path=f"{data_path}/network_{str(0)}",
    )

    G = wn.get_graph()

    edges = G.edges

    DMA1 = []
    DMA2 = []
    DMA3 = []
    DMA4 = []
    DMA5 = []
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

    DMA1_edges = []
    DMA2_edges = []
    DMA3_edges = []
    DMA4_edges = []
    DMA5_edges = []
    no_DMA_edges = []

    prior = []
    for edge in edges:
        if edge[0] in DMA1 and edge[1] in DMA1:
            DMA1_edges.append(edge[-1])
        elif edge[0] in DMA2 and edge[1] in DMA2:
            DMA2_edges.append(edge[-1])
        elif edge[0] in DMA3 and edge[1] in DMA3:
            DMA3_edges.append(edge[-1])
        elif edge[0] in DMA4 and edge[1] in DMA4:
            DMA4_edges.append(edge[-1])
        elif edge[0] in DMA5 and edge[1] in DMA5:
            DMA5_edges.append(edge[-1])
        else:
            no_DMA_edges.append(edge[-1])

    pos = {}
    for key in wdn.nodes.label_to_index.keys():
        pos[key] = nx.get_node_attributes(wdn.graph, 'pos')[str(key)]

    posterior_for_plot = []

    for edges in wdn.graph.edges:
        if edges[-1] in DMA1_edges:
            posterior_for_plot.append(1)
        elif edges[-1] in DMA2_edges:
            posterior_for_plot.append(2)
        elif edges[-1] in DMA3_edges:
            posterior_for_plot.append(3)
        elif edges[-1] in DMA4_edges:
            posterior_for_plot.append(4)
        elif edges[-1] in DMA5_edges:
            posterior_for_plot.append(5)
        else:
            posterior_for_plot.append(0)

    total_sum = sum(posterior_for_plot)
    posterior_for_plot = [val/total_sum for val in posterior_for_plot]
    prior = posterior_for_plot

    return prior