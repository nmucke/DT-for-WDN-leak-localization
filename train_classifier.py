import pickle
import torch
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
import pdb
import mlflow
import os

from DT_for_WDN_leak_localization.classifier_dataset import get_dataloader
from DT_for_WDN_leak_localization.models.dense_classifier import DenseClassifier


torch.set_default_dtype(torch.float32)

DENSE = True
OBS_CASE_KEY = "obs_case_1"

NET = 2
if DENSE:
    CONFIG_PATH = f"conf/net_{str(NET)}/dense_classifier.yml"
else:
    CONFIG_PATH = f"conf/net_{str(NET)}/transformer_classifier.yml"

DATA_CONFIG_PATH = f"conf/net_{str(NET)}/inverse_problem.yml"

DATA_PATH = f"data/raw_data/net_{str(NET)}/train_data"

NUM_SAMPLES = 30000
NUM_TRAIN_SAMPLES = 25000
NUM_VAL_SAMPLES = NUM_SAMPLES - NUM_TRAIN_SAMPLES

NUM_WORKERS = 4
CUDA = True

MODEL_SAVE_PATH = f"trained_models/net_{str(NET)}/"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if DENSE:
    model_save_name = f"dense_classifier_net_{str(NET)}_{OBS_CASE_KEY}.pt"
else:
    model_save_name = f"transformer_net_{str(NET)}_{OBS_CASE_KEY}.pt"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, model_save_name)

if CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

train_sample_ids = range(NUM_TRAIN_SAMPLES)
val_sample_ids = range(NUM_TRAIN_SAMPLES, NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES)

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

with open(DATA_CONFIG_PATH) as f:
    data_config = yaml.load(f, Loader=SafeLoader)


PREPROCESSOR_LOAD_PATH = f"trained_preprocessors/net_{str(NET)}_preprocessor.pkl"
preprocessor = pickle.load(open(PREPROCESSOR_LOAD_PATH, "rb"))


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs,
    patience,
    device,
    ):

    best_val_loss = float("inf")
    best_val_epoch = 0

    loss_function = torch.nn.BCELoss()

    pbar = tqdm(
            range(num_epochs),
            total=num_epochs,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            postfix=dict,
        )
    for epoch in pbar:

        model.train()
        train_loss = 0


        for batch_idx, (features, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()

            features = features.view(-1, features.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_dataloader):
                features = features.view(-1, features.shape[-1])
                targets = targets.view(-1, targets.shape[-1])

                features = features.to(device)
                targets = targets.to(device)

                preds = model(features)
                loss = loss_function(preds, targets)

                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        pbar.set_postfix(
            {
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model, MODEL_SAVE_PATH)

        if epoch - best_val_epoch > patience:
            break

def main():


    train_dataloader = get_dataloader(
        data_path=DATA_PATH, 
        sample_ids=train_sample_ids,
        config=data_config,
        obs_case_key=OBS_CASE_KEY,
        preprocessor=preprocessor,
        **config['dataloader_args']
    )
    
    val_dataloader = get_dataloader(
        data_path=DATA_PATH, 
        sample_ids=val_sample_ids,
        config=data_config,
        obs_case_key=OBS_CASE_KEY,
        preprocessor=preprocessor,
        **config['dataloader_args']
    )

    config['model_args']['state_dim'] = \
        len(data_config['observation_args'][OBS_CASE_KEY]['edge_obs']) + \
        len(data_config['observation_args'][OBS_CASE_KEY]['node_obs'])
        
    if DENSE:
        model = DenseClassifier(
            **config['model_args'],
        )
    else:
        model = TransformerClassifier(
            **config['model_args'],
        )
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        **config['optimizer_args']
    )

    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        **config['trainer_args']
    )

if __name__ == "__main__":
    main()