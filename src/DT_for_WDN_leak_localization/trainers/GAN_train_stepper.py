from attr import dataclass
from torch import nn
import torch
import pdb

from DT_for_WDN_leak_localization.optimizers import GANOptimizers

class GANTrainStepper():

    def __init__(
        self,
        model: nn.Module,
        optimizer: GANOptimizers,
        gradient_penalty_regu: str,
        num_critic_steps: int,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.gradient_penalty_regu = gradient_penalty_regu

        self.device = model.device

        # Loss function
        self.loss = nn.MSELoss()

        self.critic_train_count = 0
        self.num_critic_steps = num_critic_steps

    def _sample_latent(self, shape: torch.Tensor) -> torch.Tensor:
        return torch.randn(shape, device=self.device)
    
    def step_scheduler(self) -> None:
        self.optimizer.step_scheduler()

    def _compute_gradient_penalty(
        self, 
        real_state: torch.Tensor,
        fake_state: torch.Tensor,
        pars: torch.Tensor
    ):
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real_state.size(0), 1), device=self.device)

        # Get random interpolation between real and fake data
        interpolates = (
            alpha * real_state + ((1 - alpha) * fake_state)
            ).requires_grad_(True)

        model_interpolates = self.model.critic(interpolates, pars)
        grad_outputs = torch.ones(
            model_interpolates.size(), 
            device=self.device, 
            requires_grad=False
            )

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty

    def _critic_train_step(
        self, 
        state: torch.Tensor, 
        pars: torch.Tensor        
    ):
        self.model.critic.train()
        self.model.generator.eval()

        # compute critic loss for real data
        critic_output_real_data = self.model.critic(state, pars)
        critic_loss_real = torch.mean(critic_output_real_data)

        # compute critic loss for fake data
        latent_samples = self._sample_latent(
            shape=(state.shape[0], self.model.latent_dim)
        )

        generated_state = self.model.generator(latent_samples, pars)

        critic_output_fake_data = self.model.critic(generated_state, pars)
        critic_loss_fake = torch.mean(critic_output_fake_data)

        # compute gradient penalty
        gradient_penalty = self._compute_gradient_penalty(
            real_state=state,
            fake_state=generated_state, 
            pars=pars
            )
        # compute critic loss
        critic_loss = -critic_loss_real + critic_loss_fake \
            + self.gradient_penalty_regu * gradient_penalty

        # update critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 0.5)
        self.optimizer.critic_optimizer.step()

        return critic_loss.detach().item()

    def _generator_train_step(
        self, 
        state: torch.Tensor, 
        pars: torch.Tensor  
    ):
        self.model.critic.eval()
        self.model.generator.train()

        # compute critic loss for fake data
        latent_samples = self._sample_latent(
            shape=(state.shape[0], self.model.latent_dim)
        )

        generated_state = self.model.generator(latent_samples, pars)

        critic_output_data = self.model.critic(generated_state, pars)
        generator_loss = -torch.mean(critic_output_data)

        # update generator
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 0.5)
        self.optimizer.generator_optimizer.step()

        return generator_loss.detach().item()

    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.optimizer.critic_optimizer.zero_grad()

        # train critic
        critic_loss = self._critic_train_step(state, pars)

        self.critic_train_count += 1

        # train generator
        if self.critic_train_count == self.num_critic_steps:
            self.optimizer.generator_optimizer.zero_grad()
            generator_loss = self._generator_train_step(state, pars)
            self.critic_train_count = 0

            return {
                'gen_loss': generator_loss,
                'critic_loss': critic_loss
            }
        else:
            return {
                'gen_loss': None,
                'critic_loss': critic_loss
            }

    def val_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
    ) -> None:
        
        return {
            'gen_loss': 0,
            'critic_loss': 0
        }

    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
    
