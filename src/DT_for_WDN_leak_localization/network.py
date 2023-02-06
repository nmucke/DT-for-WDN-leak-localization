import networkx as nx
from dataclasses import dataclass, field
import pdb
import pandas as pd
from typing import Optional

def get_label_to_index_dict(labels):
    """Returns a dictionary mapping labels to indices."""

    index = [labels.index(node) for node in labels]
    label_to_index_dict = dict(zip(labels, index))

    return label_to_index_dict

def get_index_to_label_dict(label_to_index_dict):
    """Returns a dictionary mapping indices to labels."""

    index = list(label_to_index_dict.values())
    labels = list(label_to_index_dict.keys())
    index_to_label_dict = dict(zip(index, labels))

    return index_to_label_dict

@dataclass
class Leak:
    pipe_id: int
    pipe_label: str
    area: float

@dataclass
class Edges:
    flow_rate: pd.DataFrame
    labels: list = field(init=False)
    ids: list = field(init=False)
    label_to_index: dict = field(init=False)
    index_to_label: dict = field(init=False)

    def __post_init__(self):
        self.labels = list(self.flow_rate.columns)
        self.label_to_index = get_label_to_index_dict(self.labels)
        self.index_to_label = get_index_to_label_dict(self.label_to_index)
        self.ids = list(self.label_to_index.values())
    

@dataclass
class Nodes:
    head: pd.DataFrame
    demand: pd.DataFrame
    labels: list = field(init=False)
    ids: list = field(init=False)
    label_to_index: dict = field(init=False)
    index_to_label: dict = field(init=False)

    def __post_init__(self):
        self.labels = list(self.head.columns)
        self.label_to_index = get_label_to_index_dict(self.labels)
        self.index_to_label = get_index_to_label_dict(self.label_to_index)
        self.ids = list(self.label_to_index.values())

@dataclass
class WDN:
    data_path: Optional[str] = None
    data_dict: Optional[dict] = None
    edges: Edges = field(init=False)
    nodes: Nodes = field(init=False)
    graph: nx.DiGraph = field(init=False)
    leak: Leak = field(init=False)

    def __post_init__(self):
        if self.data_path is not None:
            data_dict = nx.read_gpickle(self.data_path)

            self.edges = Edges(
                flow_rate=data_dict['flow_rate']
            )
            self.nodes = Nodes(
                head=data_dict['head'],
                demand=data_dict['demand']
            )

            self.graph = data_dict['graph']

            self.leak = Leak(
                pipe_label=data_dict['leak']['pipe'],
                pipe_id=self.edges.label_to_index[data_dict['leak']['pipe']],
                area=data_dict['leak']['area']
            )
        else:
            self.edges = Edges(
                flow_rate=self.data_dict['flow_rate']
            )
            self.nodes = Nodes(
                head=self.data_dict['head'],
                demand=self.data_dict['demand']
            )

            self.graph = self.data_dict['graph']

            self.leak = Leak(
                pipe_label=self.data_dict['leak']['pipe'],
                pipe_id=self.edges.label_to_index[self.data_dict['leak']['pipe']],
                area=self.data_dict['leak']['area']
            )
