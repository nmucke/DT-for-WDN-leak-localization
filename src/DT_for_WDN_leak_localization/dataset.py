
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pdb

from typing import Any

class WDNDataset(Dataset[Any]):
    """Dataset for processed WDN data."""

    def __init__(
        self, 
        data_path: str, 
        sample_ids: int,
        include_leak_area: bool = False,
    ) -> None:

        self.data_path_state = data_path
        self.sample_ids = sample_ids
        self.include_leak_area = include_leak_area

        self.state = self._prepare_state(
            state_path=f'{self.data_path_state}/state.pt'
            )
        
        self.pars = self._prepare_pars(
            pars_path=f'{self.data_path_state}/pars.pt'
            )

    def _prepare_state(self, state_path: str) -> torch.Tensor:
        """Prepare state tensor from state.pt file."""
        
        state = torch.load(state_path)

        return state[self.sample_ids ].reshape(-1, state.shape[-1])
    
    def _prepare_pars(self, pars_path: str) -> torch.Tensor:
        """Prepare pars tensor from pars.pt file."""

        pars = torch.load(pars_path)

        if self.include_leak_area:
            pars_discrete = pars[self.sample_ids][:, :, [0, 2]].reshape(-1, 2)
            pars_continuous = pars[self.sample_ids][:, :, 1].reshape(-1, 1)
            return pars_discrete.type(torch.int32), pars_continuous
        else:
            return pars[self.sample_ids][:, :, [0, 2]].reshape(-1, 2).type(torch.int32)

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, idx: int) -> any:
        return self.state[idx], self.pars[idx]

def create_dataloader(
    data_path: str,
    sample_ids: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    include_leak_area: bool = False,
) -> DataLoader[Any]:

    return DataLoader(
        dataset=WDNDataset(
            data_path=data_path,
            sample_ids=sample_ids,
            include_leak_area=include_leak_area,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )