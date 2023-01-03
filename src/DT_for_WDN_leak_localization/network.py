import networkx as nx


class WDN:
    def __init__(self, data_path: str) -> None:
        
        self.data_dict = nx.read_gpickle(data_path)

        self.nodes = GraphElements(
            data_dict=self.data_dict,
            type='nodes'
        )
        self.edges = GraphElements(
            data_dict=self.data_dict,
            type='edges'
        )

        self.G = self.data_dict['G']


class GraphElements:
    def __init__(self, data_dict: str, type: str) -> None:

        if type == 'nodes':
            self.data = data_dict['head']
        elif type == 'edges':
            self.data = data_dict['flow_rate']
        
        self.labels = list(self.data.columns)
        self.label_to_index = self.label_to_index(self.labels)
        self.index_to_label = self.index_to_label(self.label_to_index)
        

    
def get_label_to_index_dict(labels):

    index = [labels.index(node) for node in labels]
    label_to_index_dict = dict(zip(labels, index))

    return label_to_index_dict

def get_index_to_label_dict(label_to_index_dict):

    index = list(label_to_index_dict.values())
    labels = list(label_to_index_dict.keys())
    index_to_label_dict = dict(zip(index, labels))

    return index_to_label_dict

