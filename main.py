import torch
import yaml
from yaml.loader import SafeLoader
import pdb
import mlflow

from DT_for_WDN_leak_localization.factories import create_model
from DT_for_WDN_leak_localization.optimizers import Optimizers
from DT_for_WDN_leak_localization.dataset import create_dataloader

torch.set_default_dtype(torch.float32)

WITH_MLFLOW = False

NET = 1
PARAMS_PATH = f"conf/net_{str(NET)}/config.yml"
DATA_PATH = f"data/processed_data/net_{str(NET)}/train_data"

NUM_SAMPLES = 10
NUM_TRAIN_SAMPLES = 8
NUM_VAL_SAMPLES = 2

NUM_WORKERS = 4
CUDA = True

if CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_sample_ids = range(NUM_TRAIN_SAMPLES)
val_sample_ids = range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES)

with open(PARAMS_PATH) as f:
    params = yaml.load(f, Loader=SafeLoader)

if WITH_MLFLOW:
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.start_run()
def main():

    if WITH_MLFLOW:
        mlflow.log_params(params)

    train_dataloader = create_dataloader(
        data_path=DATA_PATH,
        sample_ids=train_sample_ids,
        batch_size=params['training_params']["batch_size"],
        num_workers=4,
        shuffle=True,
        include_leak_area=params['data_params']['include_leak_area'],
    )
    
    val_dataloader = create_dataloader(
        data_path=DATA_PATH,
        sample_ids=val_sample_ids,
        batch_size=params['training_params']["batch_size"],
        num_workers=NUM_WORKERS,
        shuffle=False,
        include_leak_area=params['data_params']['include_leak_area'],
    )

    state, pars = next(iter(train_dataloader))

    model = create_model(model_params=params['model_params'])
    model.to(device)

    optimizers = Optimizers(
        model=model,
        optimizer_params=params['optimizer_params'],
    )

    if WITH_MLFLOW:
        mlflow.end_run()
    
if __name__ == "__main__":
    main()