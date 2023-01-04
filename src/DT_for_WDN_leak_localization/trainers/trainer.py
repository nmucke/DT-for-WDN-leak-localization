from torch import nn
import torch
from tqdm import tqdm
import pdb

from DT_for_WDN_leak_localization.early_stopping import EarlyStopper
from DT_for_WDN_leak_localization.factories import create_train_stepper
from DT_for_WDN_leak_localization.optimizers import Optimizers



class AETrainer():

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizers,
        params: dict,
        model_save_path: str,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.params = params
        
        self.model_save_path = model_save_path

        self.device = model.device

        self.early_stopper = EarlyStopper(
            patience=params['early_stopping_params']['patience'],
            min_delta=0,
        )

        # Get trainer
        self.train_stepper = create_train_stepper(
            model=model,
            optimizer=optimizer,
            params=params,
        )

    def _train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> None:

            pbar = tqdm(
                enumerate(train_dataloader),
                total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
            for i, (state, pars) in pbar:

                state = state.to(self.device)
                pars = pars.to(self.device)

                loss = self.train_stepper.train_step(state, pars)

                if i % 100 == 0:
                    pbar.set_postfix(loss)

    def _val_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        total_loss = {}
        for i, (state, pars) in enumerate(val_dataloader):

            state = state.to(self.device)
            pars = pars.to(self.device)

            loss = self.train_stepper.val_step(state, pars)

            for k in loss.keys():
                total_loss[k] = total_loss.get(k, 0) + loss[k]
        
        print('Validation losses', end=': ')
        for k in total_loss.keys():
            total_loss[k] = total_loss[k]
            print(f'{k}: {total_loss[k]/ len(val_dataloader):.4f}', end=', ')
        print()

        return total_loss
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:

        for epoch in range(self.params['training_params']['num_epochs']):
            self.model.train()
            self._train_epoch(train_dataloader)

            self.model.eval()
            val_loss = self._val_epoch(val_dataloader)

            early_stop, is_best_model = \
                self.early_stopper.early_stop(val_loss['recon_loss'])
            
            if early_stop:
                print('Early stopping')
                break
            if is_best_model:
                torch.save(self.model, self.model_save_path)

            self.train_stepper.scheduler_step()
            



