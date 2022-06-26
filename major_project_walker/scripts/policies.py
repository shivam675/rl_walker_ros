
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


# class CustomHiddenLayers(nn.Module):
#     """
#     Custom hidden network architecture for policy and value function.
#     It receives as input the features extracted by the feature extractor.

#     :param feature_dim: dimension of the input features
#     """
#     def __init__(
#         self,
#         feature_dim: int
#     ):
#         # self.hid_layer_sizes = [512]*2
#         self.activation_fns = [th.nn.Tanh]*2
#         super(CustomHiddenLayers, self).__init__()

#         # IMPORTANT:
#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = self.hid_layer_sizes[-1]
#         self.latent_dim_vf = self.hid_layer_sizes[-1]

#         # Build the hidden layers of a fully connected neural network
#         # currently we're using the same architecture for pi and vf
#         layers = []
#         for size, activation_fn in zip(self.hid_layer_sizes, self.activation_fns):
#             layers += [nn.Linear(feature_dim, size), activation_fn()]
#             feature_dim = size

#         # build the Policy network hidden layers
#         self.policy_net = nn.Sequential(*layers)
#         # build the Value network hidden layers
#         self.value_net = nn.Sequential(*layers)

        # log('Hidden Layer Network Architecture:\n' + str(self.policy_net))


    # def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     """
    #     :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
    #         If all layers are shared, then ``latent_policy == latent_value``
    #     """
    #     return self.policy_net(features), self.value_net(features)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 786,
        last_layer_dim_vf: int = 786,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)






class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # ('Using our CustomActorCriticPolicy!')

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)