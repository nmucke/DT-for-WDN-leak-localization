import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import torch
import networkx as nx
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import os

from DT_for_WDN_leak_localization.preprocess import Preprocessor
from DT_for_WDN_leak_localization.network import WDN

torch.set_default_dtype(torch.float32)

NET = 4
NUM_SAMPLES = 30000
BATCH_SIZE = 100

TEST_OR_TRAIN = 'train'

PARS_DIM = 3

DATA_CONFIG_PATH = f"conf/net_{str(NET)}/data_preprocessing.yml"
DATA_PATH = f"data/raw_data/net_{str(NET)}/train_data/network_"
#DATA_PATH = f"data/raw_data/net_{str(NET)}/test_data/network_"

with open(DATA_CONFIG_PATH) as f:
    params = yaml.load(f, Loader=SafeLoader)
STATE_DIM = params['num_pipes'] + params['num_nodes']
NUM_TIME_STEPS = params['num_time_steps']

TRAINED_PREPROCESSOR_SAVE_PATH = \
    f'trained_preprocessors/'
if not os.path.exists(TRAINED_PREPROCESSOR_SAVE_PATH):
    os.makedirs(TRAINED_PREPROCESSOR_SAVE_PATH)
TRAINED_PREPROCESSOR_SAVE_PATH += f"net_{str(NET)}_preprocessor.pkl"

PROCESSED_DATA_SAVE_PATH = \
    f'data/processed_data/net_{str(NET)}/{TEST_OR_TRAIN}_data'
if not os.path.exists(PROCESSED_DATA_SAVE_PATH):
    os.makedirs(PROCESSED_DATA_SAVE_PATH)

class NetworkDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: str,
            file_ids=range(NUM_SAMPLES),
    ):

        self.data_path_state = data_path
        self.file_ids = file_ids

        self.dtype = torch.get_default_dtype()
    
    def get_pars_tensor(self, wdn: WDN) -> torch.Tensor:
        pars = torch.zeros((NUM_TIME_STEPS, PARS_DIM), dtype=self.dtype)
        pars[:, 0] = torch.tensor(wdn.leak.pipe_id)
        pars[:, 1] = torch.tensor(wdn.leak.area)
        pars[:, 2] = torch.arange(0, NUM_TIME_STEPS)

        return pars
    
    def get_state_tensor(self, wdn: WDN) -> torch.Tensor:
        flow_rate = torch.tensor(
            wdn.edges.flow_rate.values, 
            dtype=self.dtype
            )
        head = torch.tensor(
            wdn.nodes.head.values, 
            dtype=self.dtype
            )

        flow_rate = flow_rate[0:NUM_TIME_STEPS]
        head = head[0:NUM_TIME_STEPS]

        state = torch.cat([flow_rate, head], dim=1)

        return state

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):

        wdn = WDN(self.data_path_state + str(self.file_ids[idx]))

        return self.get_state_tensor(wdn), self.get_pars_tensor(wdn)

'''
for i in range(0, 50000):
    wdn = WDN(DATA_PATH + str(i))
    if wdn.edges.flow_rate.shape[0] != 25:
        print(i)
pdb.set_trace()
'''

def main():

    dataset = NetworkDataset(
        data_path=DATA_PATH,
        file_ids=range(NUM_SAMPLES),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    if TEST_OR_TRAIN == 'test':    
        preprocessor = pickle.load(open(TRAINED_PREPROCESSOR_SAVE_PATH, "rb"))
    else:
        preprocessor = Preprocessor(
            num_pipes=params['num_pipes'],
            num_nodes=params['num_nodes'],
        )
        
        for i, (state, _) in enumerate(dataloader):
            preprocessor.partial_fit(state)

        print("Done Training Preprocessor")

        with open(TRAINED_PREPROCESSOR_SAVE_PATH, "wb") as f:
            pickle.dump(preprocessor, f)
    
    state_tensor_to_be_saved = torch.zeros((NUM_SAMPLES, NUM_TIME_STEPS, STATE_DIM))
    pars_tensor_to_be_saved = torch.zeros((NUM_SAMPLES, NUM_TIME_STEPS, PARS_DIM))
    for i, (state, pars) in enumerate(dataloader):
        iter_batch_size = state.shape[0]

        processed_data = preprocessor.transform_state(state)

        state_tensor_to_be_saved[i*BATCH_SIZE:i*BATCH_SIZE+iter_batch_size] = processed_data
        
        pars_tensor_to_be_saved[i*BATCH_SIZE:i*BATCH_SIZE+iter_batch_size] = pars
        
    torch.save(state_tensor_to_be_saved, f'{PROCESSED_DATA_SAVE_PATH}/state.pt')
    torch.save(pars_tensor_to_be_saved, f'{PROCESSED_DATA_SAVE_PATH}/pars.pt')

    print("Done Saving Processed Data")


if __name__ == "__main__":
    main()