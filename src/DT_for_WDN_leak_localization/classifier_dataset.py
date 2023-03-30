
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pdb

from typing import Any
from DT_for_WDN_leak_localization.inference.noise import ObservationNoise

from DT_for_WDN_leak_localization.inference.observation import ObservationModel
from DT_for_WDN_leak_localization.inference.true_data import TrueData
from DT_for_WDN_leak_localization.network import WDN

class WDNDataset(Dataset[Any]):
    """Dataset for processed WDN data."""

    def __init__(
        self, 
        data_path: str, 
        sample_ids: int,
        config: dict,
        obs_case_key: str,
        preprocessor,
    ) -> None:

        self.data_path_state = data_path
        self.sample_ids = sample_ids
        self.config = config
        self.obs_case_key = obs_case_key
        self.preprocessor = preprocessor

        wdn = WDN(
            data_path=f"{data_path}/network_0",
        )
        
        self.observation_model = ObservationModel(
            wdn=wdn,
            **config['observation_args'][obs_case_key],        
        )


    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> any:

        wdn = WDN(
            data_path=f"{self.data_path_state}/network_{idx}",
        )
        # Set up true data
        true_data = TrueData(
            wdn=wdn,
            preprocessor=self.preprocessor,
            observation_model=self.observation_model,
        )

        features = true_data.obs.squeeze(0)
        features = features.type(torch.get_default_dtype())

        target = torch.zeros(self.config['data_args']['num_pipes'], dtype=torch.get_default_dtype())
        target[true_data.leak] = 1
        target = target.unsqueeze(0)
        target = target.repeat((features.shape[0], 1))

        return features, target

def get_dataloader(
    data_path: str, 
    sample_ids: int,
    config: dict,
    obs_case_key: str,
    preprocessor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:

    return DataLoader(
        dataset=WDNDataset(
            data_path=data_path, 
            sample_ids=sample_ids,
            config=config,
            obs_case_key=obs_case_key,
            preprocessor=preprocessor,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )