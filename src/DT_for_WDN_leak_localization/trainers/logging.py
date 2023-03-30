

from attr import dataclass


@dataclass
class AEMetricLogger:
    recon_loss: float = 0
    latent_loss: float = 0
    counter: int = 0

    def update(
        self, 
        recon_loss: float,
        latent_loss: float,
        ) -> None:

        self.recon_loss += recon_loss
        self.latent_loss += latent_loss
        self.counter += 1
    
    @property
    def total_recon_loss(self) -> float:
        return self.recon_loss/self.counter
    
    @property
    def total_latent_loss(self) -> float:
        return self.latent_loss/self.counter

@dataclass
class GANMetricLogger:
    gen_loss: float = 0
    critic_loss: float = 0
    gen_counter: int = 1
    critic_counter: int = 1

    def update(
        self, 
        critic_loss: float,
        gen_loss: float = None,
        ) -> None:

        if gen_loss is not None:
            self.gen_loss += gen_loss
            self.gen_counter += 1
        self.critic_loss += critic_loss
        self.critic_counter += 1
    
    @property
    def total_gen_loss(self) -> float:
        return self.gen_loss/self.gen_counter
    
    @property
    def total_critic_loss(self) -> float:
        return self.critic_loss/self.critic_counter


