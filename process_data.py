import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import torch
import networkx as nx
from torch.utils.data import DataLoader

from DT_for_WDN_leak_localization.preprocess import Preprocessor

torch.set_default_dtype(torch.float32)

NET = 1
DATA_PATH = f"data/raw_data/net_{str(NET)}/training_data/network_"
DATA_PARAMS_PATH = f"conf/net_{str(NET)}/config.yml"

with open(DATA_PARAMS_PATH) as f:
    params = yaml.load(f, Loader=SafeLoader)
params = params['data_params']

class NetworkDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: str,
            file_ids=range(10000),
    ):

        self.data_path_state = data_path
        self.file_ids = file_ids

        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):

        idx = self.file_ids[idx]

        data_dict = nx.read_gpickle(self.data_path_state + str(idx))

        flow_rate = torch.tensor(
            data_dict['flow_rate'].values, 
            dtype=self.dtype
            )
        head = torch.tensor(
            data_dict['head'].values, 
            dtype=self.dtype
            )

        flow_rate = flow_rate[0:24]
        head = head[0:24]

        data = torch.cat([flow_rate, head], dim=1)

        return data


def main():

    dataset = NetworkDataset(
        data_path=DATA_PATH,
        file_ids=range(30000),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    preprocessor = Preprocessor(
        num_pipes=params['num_pipes'],
        num_nodes=params['num_nodes'],
    )
    
    for i, data in enumerate(dataloader):
        preprocessor.partial_fit(data)

    print("DONE")

if __name__ == "__main__":
    main()