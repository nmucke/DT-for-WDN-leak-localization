import pdb
import torch
from DT_for_WDN_leak_localization.inference.noise import ObservationNoise
from DT_for_WDN_leak_localization.inference.observation import ObservationModel
from DT_for_WDN_leak_localization.network import WDN
from DT_for_WDN_leak_localization.preprocess import Preprocessor

class TrueData():
    def __init__(
        self,
        wdn: WDN,
        preprocessor: Preprocessor,
        observation_model: ObservationModel,
        observation_noise: ObservationNoise = None,
        ):
        
        self.wdn = wdn
        
        self.state, self.leak = self._prepare_data(wdn, preprocessor)

        self.obs = observation_model.get_observations(self.state)
        
        if observation_noise is not None:
            self.obs = observation_noise.add_noise(self.obs)


    def _prepare_data(
        self,
        wdn: WDN,
        preprocessor = None
        ):
        flow_rate = torch.tensor(wdn.edges.flow_rate.values)
        head = torch.tensor(wdn.nodes.head.values)
        true_state = torch.cat((flow_rate, head), dim=-1)[:-1].unsqueeze(0)

        true_state = preprocessor.transform_state(true_state)

        true_leak = torch.tensor(wdn.leak.pipe_id)

        return true_state, true_leak