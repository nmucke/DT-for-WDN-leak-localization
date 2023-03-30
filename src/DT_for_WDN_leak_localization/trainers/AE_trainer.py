from attr import dataclass
from torch import nn
import torch
from tqdm import tqdm
import pdb

from DT_for_WDN_leak_localization.trainers.base import BaseTrainStepper
from DT_for_WDN_leak_localization.trainers.logging import AEMetricLogger

@dataclass
class EarlyStopping:
    num_non_improving_epochs: int = 0
    best_loss: float = float('inf')
    patience: int = 10


def train_AE(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    model_save_path: str,
    train_stepper: BaseTrainStepper,
    print_progress: bool = True,
    patience: int = None
) -> None:

    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    device = train_stepper.device

    for epoch in range(num_epochs):

        if print_progress:
            pbar = tqdm(
                    enumerate(train_dataloader),
                    total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        else:
            pbar = enumerate(train_dataloader)
        
        train_logger = AEMetricLogger()

        for i, (state, pars) in pbar:

            state = state.to(device)
            pars = pars.to(device)

            loss = train_stepper.train_step(
                state=state,
                pars=pars,
                )

            train_logger.update(
                recon_loss=loss['recon_loss'],
                latent_loss=loss['latent_loss'],
                )

            if i % 100 == 0:
                pbar.set_postfix(loss)
        
        train_stepper.step_scheduler()

        val_logger = AEMetricLogger()
        for i, (state, pars) in enumerate(val_dataloader):
                
                state = state.to(device)
                pars = pars.to(device)
    
                loss  = train_stepper.val_step(
                    state=state,
                    pars=pars,
                    )
                
                val_logger.update(
                    recon_loss=loss['recon_loss'],
                    latent_loss=loss['latent_loss'],
                    )
        
        if print_progress:
            print(f'Epoch {epoch} of {num_epochs}')
            print(f'Val recon loss: {val_logger.total_recon_loss}')
            print(f'Val latent loss: {val_logger.total_latent_loss}')
            
        if patience is not None:
            if loss['recon_loss'] < early_stopper.best_loss:
                early_stopper.best_loss = loss['recon_loss']
                early_stopper.num_non_improving_epochs = 0
                train_stepper.save_model(model_save_path)
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    break
        
    if patience is None:
        train_stepper.save_model(model_save_path)
        


