import pdb
import torch
from torch import nn

from DT_for_WDN_leak_localization.network import WDN

class ObservationModel(nn.Module):
    def __init__(
        self, 
        wdn: WDN,
        ids_or_labels: str = 'labels',
        edge_obs: list = None,
        node_obs: list = None,
        ) -> None:
        super().__init__()

        self.wdn = wdn

        self.num_edges = len(self.wdn.edges.ids)
        self.num_nodes = len(self.wdn.nodes.ids)

        #edge_obs = self.wdn.edges.labels[0::5]
        #node_obs = self.wdn.nodes.labels[0::5]

        if ids_or_labels == 'labels':
            self._prepare_observation_ids(
                edge_obs_labels=edge_obs,
                node_obs_labels=node_obs,
                )
            self.edge_obs_labels = edge_obs
            self.node_obs_labels = node_obs
        elif ids_or_labels == 'ids':
            self._prepare_observation_labels(
                edge_obs_ids=edge_obs,
                node_obs_ids=node_obs,
                )
            self.edge_obs_ids = edge_obs
            self.node_obs_ids = node_obs

        self.edge_obs_tensor = torch.tensor(self.edge_obs_ids)
        self.node_obs_tensor = torch.tensor(self.node_obs_ids)
        self.obs_tensor = torch.cat(
            (self.edge_obs_tensor, self.num_edges+self.node_obs_tensor-1)
            )

    def _prepare_observation_ids(
        self, 
        edge_obs_labels: list,
        node_obs_labels: list,
        ) -> None:

            self.edge_obs_ids = \
                [self.wdn.edges.label_to_index[label] for label in edge_obs_labels]
            self.edge_obs_labels = edge_obs_labels
            self.node_obs_ids = \
                [self.wdn.nodes.label_to_index[label] for label in node_obs_labels]
            self.node_obs_labels = node_obs_labels

    def _prepare_observation_labels(
        self, 
        edge_obs_ids: list,
        node_obs_ids: list,        
    ) -> None:

        self.edge_obs_labels = \
            [self.wdn.edges.index_to_label[idx] for idx in edge_obs_ids]
        self.edge_obs_ids = edge_obs_ids

        self.node_obs_labels = \
            [self.wdn.nodes.index_to_label[idx] for idx in node_obs_ids]
        self.node_obs_ids = node_obs_ids
    

    def get_observations(
        self,
        state: torch.Tensor,
        ) -> torch.Tensor:

        if len(state.shape) == 2:
            return state[:, self.obs_tensor]
        elif len(state.shape) == 3:
            return state[:, :, self.obs_tensor]