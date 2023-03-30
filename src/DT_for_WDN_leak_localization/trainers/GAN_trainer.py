from attr import dataclass
from torch import nn
import torch
from tqdm import tqdm
import pdb

from DT_for_WDN_leak_localization.trainers.base import BaseTrainStepper
from DT_for_WDN_leak_localization.trainers.logging import GANMetricLogger

@dataclass
class EarlyStopping:
    num_non_improving_epochs: int = 0
    best_loss: float = float('inf')
    patience: int = 10


def train_GAN(
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
        
        train_logger = GANMetricLogger()

        for i, (state, pars) in pbar:

            state = state.to(device)
            pars = pars.to(device)

            loss = train_stepper.train_step(
                state=state,
                pars=pars,
                )

            train_logger.update(
                gen_loss=loss['gen_loss'],
                critic_loss=loss['critic_loss'],
                )

            if i % 5 == 0:
                pbar.set_postfix({
                    'gen_loss': train_logger.total_gen_loss,
                    'critic_loss': train_logger.total_critic_loss,
                })
        
        train_stepper.step_scheduler()

        val_logger = GANMetricLogger()
        for i, (state, pars) in enumerate(val_dataloader):
                
                state = state.to(device)
                pars = pars.to(device)
    
                loss = train_stepper.val_step(
                    state=state,
                    pars=pars,
                    )
                
                val_logger.update(
                gen_loss=loss['gen_loss'],
                critic_loss=loss['critic_loss'],
                )
        
        if print_progress:
            print(f'Epoch {epoch} of {num_epochs}')
            print(f'Val gen loss: {val_logger.total_gen_loss}')
            print(f'Val crtic loss: {val_logger.total_critic_loss}')
            
        if patience is not None:
            if loss['gen_loss'] < early_stopper.best_loss:
                early_stopper.best_loss = loss['gen_loss']
                early_stopper.num_non_improving_epochs = 0
                train_stepper.save_model(model_save_path)
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    break
        else:
            train_stepper.save_model(model_save_path)
        


