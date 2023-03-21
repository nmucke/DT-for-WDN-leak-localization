import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import mlflow
import os

from DT_for_WDN_leak_localization.dataset import get_dataloader
from DT_for_WDN_leak_localization.trainers.AE_trainer import train_AE

from DT_for_WDN_leak_localization.models.dense_decoder import Decoder as DenseDecoder
from DT_for_WDN_leak_localization.models.dense_encoder import Encoder as DenseEncoder

from DT_for_WDN_leak_localization.models.decoder import Decoder
from DT_for_WDN_leak_localization.models.encoder import Encoder
from DT_for_WDN_leak_localization.models.wasserstein_AE import SupervisedWassersteinAE
from DT_for_WDN_leak_localization.optimizers import AEOptimizers

from DT_for_WDN_leak_localization.trainers.WAE_train_stepper import SupervisedWAETrainStepper

torch.set_default_dtype(torch.float32)


WITH_MLFLOW = False

DENSE = False

NET = 4
if DENSE:
    CONFIG_PATH = f"conf/net_{str(NET)}/dense_Supervised_WAE.yml"
else:
    CONFIG_PATH = f"conf/net_{str(NET)}/Supervised_WAE.yml"
DATA_PATH = f"data/processed_data/net_{str(NET)}/train_data"

NUM_SAMPLES = 30000
NUM_TRAIN_SAMPLES = 25000
NUM_VAL_SAMPLES = NUM_SAMPLES - NUM_TRAIN_SAMPLES

NUM_WORKERS = 4
CUDA = True

MODEL_SAVE_PATH = f"trained_models/net_{str(NET)}/"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if DENSE:
    model_save_name = f"dense_Supervised_WAE_net_{str(NET)}.pt"
else:
    model_save_name = f"Supervised_WAE_net_{str(NET)}.pt"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, model_save_name)

if CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

train_sample_ids = range(NUM_TRAIN_SAMPLES)
val_sample_ids = range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES)

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

if WITH_MLFLOW:
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.start_run()
def main():

    if WITH_MLFLOW:
        mlflow.log_params(config)

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

    if DENSE:
        encoder = DenseEncoder(
            **config['model_args']['encoder'],
        )
        decoder = DenseDecoder(
            **config['model_args']['decoder'],
        )
    else:
        encoder = Encoder(
            **config['model_args']['encoder'],
        )
        decoder = Decoder(
            **config['model_args']['decoder'],
        )

    model = SupervisedWassersteinAE(
        encoder=encoder,
        decoder=decoder,        
    )
    model.to(device)


    pytorch_total_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    optimizer = AEOptimizers(
        model=model,
        args=config['optimizer_args'],
    )
    
    train_stepper = SupervisedWAETrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args']        
    )
    
    train_AE(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_stepper=train_stepper,
        model_save_path=MODEL_SAVE_PATH,
        **config['trainer_args']
    )

    if WITH_MLFLOW:
        mlflow.end_run()
    
if __name__ == "__main__":
    main()