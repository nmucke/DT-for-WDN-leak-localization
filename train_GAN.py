import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import mlflow
import os

from DT_for_WDN_leak_localization.dataset import get_dataloader
from DT_for_WDN_leak_localization.trainers.GAN_trainer import train_GAN

from DT_for_WDN_leak_localization.models.generator import Generator
from DT_for_WDN_leak_localization.models.critic import Critic
from DT_for_WDN_leak_localization.models.GAN import GAN
from DT_for_WDN_leak_localization.optimizers import GANOptimizers

from DT_for_WDN_leak_localization.trainers.GAN_train_stepper import GANTrainStepper

torch.set_default_dtype(torch.float32)

NET = 1
CONFIG_PATH = f"conf/net_{str(NET)}/GAN.yml"
DATA_PATH = f"data/processed_data/net_{str(NET)}/train_data"

NUM_SAMPLES = 30000
NUM_TRAIN_SAMPLES = 25000
NUM_VAL_SAMPLES = 5000

NUM_WORKERS = 4
CUDA = True

MODEL_SAVE_PATH = f"trained_models/net_{str(NET)}/"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

model_save_name = f"GAN_net_{str(NET)}.pt"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, model_save_name)

if CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_sample_ids = range(NUM_TRAIN_SAMPLES)
val_sample_ids = range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES)

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

def main():


    train_dataloader = get_dataloader(
        data_path=DATA_PATH,
        sample_ids=train_sample_ids,
        include_leak_area=config['data_args']['include_leak_area'],
        **config['dataloader_args']
    )
    
    val_dataloader = get_dataloader(
        data_path=DATA_PATH,
        sample_ids=val_sample_ids,
        include_leak_area=config['data_args']['include_leak_area'],
        **config['dataloader_args']
    )

    generator = Generator(
        **config['model_args']['generator'],
    )
    critic = Critic(
        **config['model_args']['critic'],
    )

    model = GAN(
        generator=generator,
        critic=critic,        
    )
    model.to(device)

    optimizer = GANOptimizers(
        model=model,
        args=config['optimizer_args'],
    )
    train_stepper = GANTrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args']        
    )
    
    train_GAN(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_stepper=train_stepper,
        model_save_path=MODEL_SAVE_PATH,
        **config['trainer_args']
    )
    
if __name__ == "__main__":
    main()