import pdb
from matplotlib import pyplot as plt
import torch
import networkx as nx

from DT_for_WDN_leak_localization.inference.true_data import TrueData

class InverseProblemMetrics():
    def __init__(
        self,
        true_data: TrueData,
        posterior: list
        ) -> None:

        self.true_data = true_data
        self.posterior = posterior

        self.G = self.true_data.wdn.graph

        self.true_leak_location_id = self.true_data.leak.item()
        self.true_leak_location_label = \
            self.true_data.wdn.edges.index_to_label[self.true_leak_location_id]
                
        self.predicted_leak_location_id = self._get_predictid_leak_location_id(posterior)
        self.predicted_leak_location_label = \
            self.true_data.wdn.edges.index_to_label[self.predicted_leak_location_id]
        
        self.topological_distance = \
            self._get_topological_distance(self.G, self.predicted_leak_location_label)

        self.is_correct = self._is_predicted_leak_location_correct(posterior)

    def _get_predictid_leak_location_id(self, posterior: list) -> int:
        return torch.argmax(posterior[-1]).item()
    
    def _get_predictid_leak_location_label(self, posterior: list) -> int:
        predicted_leak_location_id = self._get_predictid_leak_location_id(posterior)
        return self.true_data.wdn.edges.index_to_label[predicted_leak_location_id]
    
    def _is_predicted_leak_location_correct(self, posterior: list) -> bool:
        predicted_leak_location_id = self._get_predictid_leak_location_id(posterior)
        return predicted_leak_location_id == self.true_data.leak
    
    def _get_topological_distance(
        self, 
        G: nx.DiGraph,
        pred_leak_location_label: str,
        ) -> int:

        G = G.copy()

        if self._is_predicted_leak_location_correct(self.posterior):
            return 0

        for edge in G.edges:
            if edge[-1] == self.true_leak_location_label:
                true_leak_location_edge = edge
            if edge[-1] == pred_leak_location_label:
                pred_leak_location_edge = edge

        G.add_node('pred_leak_node')
        G.add_edge(
            pred_leak_location_edge[0], 
            'pred_leak_node',
            )
        G.add_edge(
            'pred_leak_node', 
            pred_leak_location_edge[1], 
            )

        G.add_node('true_leak_node')
        G.add_edge(
            true_leak_location_edge[0], 
            'true_leak_node', 
             )
        G.add_edge(
            'true_leak_node', 
            true_leak_location_edge[1],
            )
        G = G.to_undirected()

        topological_distance = nx.shortest_path_length(
            G=G,
            source='true_leak_node',
            target='pred_leak_node',
        )
        return topological_distance - 1
    
    def plot_posterior_on_graph(self):
        
        edge_cmap = plt.get_cmap('Reds')


        pos = {}
        for i in self.true_data.wdn.nodes.label_to_index.keys():
            pos[i] = nx.get_node_attributes(self.G, 'pos')[str(i)]

        posterior_list = [self.posterior[-1][i].item() for i in range(len(self.posterior[-1]))]
        lol = []
        for i, key in enumerate(self.true_data.wdn.edges.label_to_index.keys()):
            lol.append(posterior_list[self.true_data.wdn.edges.label_to_index[key]])
            
        nx.draw_networkx(
            G=self.G, 
            pos=pos, 
            edge_vmin=self.posterior[-1].min(), 
            edge_vmax=self.posterior[-1].max(),
            edge_color=lol, 
            edge_cmap=edge_cmap, 
            width=2,
            node_size=10, #node_color=head, node_cmap=node_cmap,
            with_labels=False
            )

        for edges in self.G.edges:
            if edges[-1] == self.true_leak_location_label:
                nx.draw_networkx_edge_labels(
                    G=self.G, pos=pos,
                    edge_labels={(edges[0], edges[1]): 'X'},
                    font_color='tab:red', font_size=25,
                    bbox={'alpha':0.0})
            if edges[-1] == self.predicted_leak_location_label:
                nx.draw_networkx_edge_labels(
                    G=self.G, pos=pos,
                    edge_labels={(edges[0], edges[1]): 'X'},
                    font_color='tab:green', font_size=20,
                    bbox={'alpha':0.0})
        plt.show()
    
