from typing import Dict, List, Union

from gymnasium import Space
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torch
from torch import nn

class DualVisionNetwork(VisionNetwork):

    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: ModelConfigDict, name: str, **custom_model_kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # state_dict = super().state_dict()

        # self._offset_network: nn.Module = VisionNetwork(obs_space, action_space, num_outputs, model_config, name+"_offset")
        # self._offset_network.load_state_dict(state_dict)

        # for param in self._offset_network.parameters():  # freeze the offset network
        #     param.requires_grad = False

        self._dual_network = VisionNetwork(obs_space, action_space, num_outputs, model_config, name+"_dual")
        dual_state_dict = torch.load(model_config["custom_model_config"]["state_dict_path"], map_location = torch.device("cpu"))
        self._dual_network.load_state_dict(dual_state_dict, strict=False)  # load state dict from the sleep model
        
        for name, param in self._dual_network.named_parameters():  # freeze the dual network
            param.requires_grad = False

        self.combining_layer: nn.Sequential = nn.Sequential(
            nn.Linear(num_outputs*2, num_outputs),
            nn.ReLU()
        )

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
    
        # self._features = input_dict["obs"].float()

        # # Permuate b/c data comes in as [B, dim, dim, channels]:
        # self._features = self._features.permute(0, 3, 1, 2)
        # conv_out = self._convs(self._features)
        # dual_conv_out = self._dual_network._convs(self._features)

        # # Store features to save forward pass when getting value_function out.
        # if not self._value_branch_separate:
        #     self._features = conv_out

        # if not self.last_layer_is_flattened:
        #     if self._logits:
        #         conv_out = self._logits(conv_out)
        #     if len(conv_out.shape) == 4:
        #         if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
        #             raise ValueError(
        #                 "Given `conv_filters` ({}) do not result in a [B, {} "
        #                 "(`num_outputs`), 1, 1] shape (but in {})! Please "
        #                 "adjust your Conv2D stack such that the last 2 dims "
        #                 "are both 1.".format(
        #                     self.model_config["conv_filters"],
        #                     self.num_outputs,
        #                     list(conv_out.shape),
        #                 )
        #             )
        #         logits = conv_out.squeeze(3)
        #         logits = logits.squeeze(2)
        #     else:
        #         logits = conv_out
        #     return logits, state
        # else:
        #     return conv_out, state
        
        device = input_dict["obs"].device
        self._dual_network.to(device)

        features, _ = super().forward(input_dict, state, seq_lens)
    
        #features_offset, _ = self._offset_network.forward(input_dict, state, seq_lens)
        features_dual, _ = self._dual_network.forward(input_dict, state, seq_lens)

        # just a clumsy way of concatenating, as I previously had problems with using torch.cat where for some reason it prevented any learning
        concatenated_features = torch.zeros((features.shape[0], features.shape[1]*2), device=device)
        concatenated_features[:, :features.shape[1]] = features
        concatenated_features[:, features.shape[1]:] = features_dual

        return self.combining_layer(concatenated_features), state