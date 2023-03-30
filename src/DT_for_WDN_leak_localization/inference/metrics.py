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

        self.entropy = self._get_entropy(self.posterior[-1])

        self.time_topological_distance = \
            self._get_time_topological_distance()

    def _get_entropy(self, posterior: list) -> float:
        """Get the entropy of the posterior distribution."""

        return torch.sum(- posterior * torch.log(posterior)).item()

    def _get_predictid_leak_location_id(self, posterior: list) -> int:
        """Get the predicted leak location id from the posterior distribution."""

        return torch.argmax(posterior[-1]).item()
    
    def _get_predictid_leak_location_label(self, posterior: list) -> int:
        """Get the predicted leak location label from the posterior distribution."""

        predicted_leak_location_id = self._get_predictid_leak_location_id(posterior)
        return self.true_data.wdn.edges.index_to_label[predicted_leak_location_id]
    
    def _is_predicted_leak_location_correct(self, posterior: list) -> bool:
        """Check if the predicted leak location is correct."""

        predicted_leak_location_id = self._get_predictid_leak_location_id(posterior)
        return predicted_leak_location_id == self.true_data.leak
    
    def _get_time_topological_distance(
        self, 
        ) -> int:
        """Get the topological distance between the true leak location 
        and the predicted leak location for all time steps."""

        time_topological_distance = []
        for posterior_t in self.posterior:
            top_dist = self._get_topological_distance(
                self.G, 
                self.true_data.wdn.edges.index_to_label[torch.argmax(posterior_t).item()],
                )
            
            time_topological_distance.append(top_dist)            
            
        return time_topological_distance

    def _get_topological_distance(
        self, 
        G: nx.DiGraph,
        pred_leak_location_label: str,
        ) -> int:
        """Get the topological distance between the true leak location 
        and the predicted leak location."""

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
    
    def plot_posterior_on_graph(
        self,
        edge_obs_labels: list = None,
        node_obs_labels: list = None
        ):
        """Plot the posterior distribution on the graph."""
        
        edge_cmap = plt.get_cmap('Reds')

        # Get node positions
        pos = {}
        for key in self.true_data.wdn.nodes.label_to_index.keys():
            pos[key] = nx.get_node_attributes(self.G, 'pos')[str(key)]

        # Reorder posterior list for plotting on graph
        posterior_list = [self.posterior[-1][i].item() for i in range(len(self.posterior[-1]))]
        posterior_for_plot = []
        for edges in self.G.edges:
            posterior_for_plot.append(posterior_list[self.true_data.wdn.edges.label_to_index[edges[-1]]])
            
        # Plot
        nx.draw_networkx(
            G=self.G, 
            pos=pos, 
            edge_vmin=self.posterior[-1].min(), 
            edge_vmax=self.posterior[-1].max(),
            edge_color=posterior_for_plot, 
            edge_cmap=edge_cmap, 
            width=2,
            node_size=10, #node_color=head, node_cmap=node_cmap,
            with_labels=False
            )
        
        # Plot true leak location
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
        
        if node_obs_labels is not None:
            node_obs_dict = {key: key for key in node_obs_labels}
            nx.draw_networkx_labels(
                G=self.G, 
                pos=pos,
                labels=node_obs_dict,
                font_color='tab:purple', 
                font_size=10,
                horizontalalignment='left',
                verticalalignment='top',
                )
        if edge_obs_labels is not None:
            for edges in self.G.edges:
                for key in edge_obs_labels:
                    if key == edges[-1]:
                        nx.draw_networkx_edge_labels(
                            G=self.G, 
                            pos=pos,
                            edge_labels={(edges[0], edges[1]): key},
                            font_color='tab:purple', 
                            font_size=10,
                            horizontalalignment='left',
                            verticalalignment='top',
                            )
        plt.show()
    
