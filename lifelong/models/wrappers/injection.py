from typing import Callable, Union

import torch
import torch.nn as nn
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel

from lifelong.models.dqn import DQNModelWrapper

# class PlasticityInjectionDQNModelWrapper(DQNModelWrapper):

#     # a method for resetting the plasticity of a neural network without affecting its immediate performance.
#     # works by creating two additional networks at the time of injection, both of which having the same weights
#     # (and thus the same outputs for all inputs). then, the old network as well as one of the new networks (offset network) are frozen,
#     # meaning the one remaining new network is the only network that learns from this point on.

#     # then, when passing inputs through this architecture, the formula is as follows:
#     # Y = old(X) + new(X) - offset(X)
#     # 
#     # (note that just after the point of plasticity injection, the 'offset' and 'new' network are the exact same, so the last two terms in
#     # this equation cancel out leaving the old networks dynamics unchanged. this will not be true as the new network adjusts its weights, but
#     # it allows for a smoother transition to higher plasticity)

#     # this allows for the 'new' network to learn as if it was a fresh model (with high plasiticty), while having its initial outputs being the same
#     # as the old network (ideally being accurate / good).

#     def __init__(self, model: Union[nn.Module, DQNTorchModel], model_instantiator: Callable[[], nn.Module], add_value: bool, device: str):
#         super().__init__(model, add_value, device)

#         # create two additional models, and freeze the old model together with the new offset model

#         new_model = model_instantiator()
#         self.prev_model = self.model
#         self.model = new_model

#         self.offset_model = model_instantiator()
#         self.offset_model.load_state_dict(self.model.state_dict())

#         # freeze the new offset model, as well as the old model
#         for param in self.prev_model.parameters():
#             param.requires_grad = False
#         for param in self.offset_model.parameters():
#             param.requires_grad = False

#     def forward(self, x: torch.Tensor):
#         x = x.to(self.device)
#         fwd = lambda model: DQNModelWrapper.static_forward(x, model, self.add_value)
#         return fwd(self.prev_model) + fwd(self.model) - fwd(self.offset_model)
