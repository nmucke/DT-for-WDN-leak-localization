import torch
from DT_for_WDN_leak_localization.factories import create_model
from DT_for_WDN_leak_localization.optimizers import Optimizers
import yaml
from yaml.loader import SafeLoader
import pdb
torch.set_default_dtype(torch.float32)

NET = 1
PARAMS_PATH = f"conf/net_{str(NET)}/config.yml"


with open(PARAMS_PATH) as f:
    params = yaml.load(f, Loader=SafeLoader)

def main():

    model = create_model(
        model_type=params["model_type"],
        model_architecture=params["model_architecture"],
        model_args=params["model_args"],
    )

    optimizers = Optimizers(
        model=model,
        optimizer_type=params["optimizer_type"],
        optimizer_args=params["optimizer_args"],
    )
    
if __name__ == "__main__":
    main()