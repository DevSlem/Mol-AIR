from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init

import modules.mol_air.drl.agent as agent
import modules.mol_air.drl.net as net
from modules.mol_air.drl.policy import CategoricalPolicy
from modules.mol_air.drl.policy_dist import CategoricalDist


def init_linear_weights(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.orthogonal_(layer.weight, 2.0 ** 0.5)
            layer.bias.data.zero_()


### ===================== ORIGINAL SelfiesRecurrentPPOSharedNet ===================== #
class SelfiesRecurrentPPOSharedNet(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.hidden_state_dim = 64 * 2
        self.n_recurrent_layers = 2
        self.out_features = 256

        self.recurrent_layers = nn.LSTM(
            in_features,
            self.hidden_state_dim // 2,  # H_cell = 64, H_hidden = 64
            batch_first=True,
            num_layers=self.n_recurrent_layers
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(self.hidden_state_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_features),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feed forward to the recurrent layers
        h, c = net.unwrap_lstm_hidden_state(hidden_state)
        embedding_seq, (h_n, c_n) = self.recurrent_layers(x, (h, c))
        next_seq_hidden_state = net.wrap_lstm_hidden_state(h_n, c_n)

        # feed forward to the linear layers
        embedding_seq = self.linear_layers(embedding_seq)

        return embedding_seq, next_seq_hidden_state


### ===================== MODIFIED SelfiesRecurrentPPOSharedNet ===================== #

class SelfiesRecurrentPPOSharedNetV2(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        # smaller model to reduce overfitting
        self.hidden_state_dim = 32  # down from 64
        self.n_recurrent_layers = 1
        self.out_features = 64  # down from 128

        # variational (per-sequence) input dropout and output dropout
        self.input_dropout = nn.Dropout(p=0.25)
        self.output_dropout = nn.Dropout(p=0.25)

        # LSTM: keep single layer (less params); keep batch_first
        # note: hidden_size is half of hidden_state_dim if you want to keep split convention
        self.recurrent_layers = nn.LSTM(
            in_features,
            self.hidden_state_dim // 2,  # internal hidden cell size
            batch_first=True,
            num_layers=self.n_recurrent_layers,
            dropout=0.0  # no inter-layer dropout for 1 layer
        )

        # layer norm across feature dimension (applied per time-step)
        self.layer_norm = nn.LayerNorm(self.hidden_state_dim // 2)

        # small MLP with dropout and residual projection
        self.linear_proj = nn.Linear(self.hidden_state_dim // 2, self.out_features)
        self.mlp = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.out_features, self.out_features),
            nn.GELU(),
            nn.Dropout(p=0.15),
        )

        # optional residual projection if dims differ
        if self.out_features != (self.hidden_state_dim // 2):
            self._residual = nn.Linear(self.hidden_state_dim // 2, self.out_features)
        else:
            self._residual = None

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, in_features)
        # apply same dropout mask across time steps (nn.Dropout applied to full tensor is fine)
        x = self.input_dropout(x)

        h, c = net.unwrap_lstm_hidden_state(hidden_state)
        embedding_seq, (h_n, c_n) = self.recurrent_layers(x, (h, c))
        next_seq_hidden_state = net.wrap_lstm_hidden_state(h_n, c_n)

        # layer norm per time-step
        # embedding_seq: (B, T, feat)
        embedding_seq = self.layer_norm(embedding_seq)

        # project to out_features and apply small MLP with residual
        proj = self.linear_proj(embedding_seq)  # (B, T, out_features)
        mlp_out = self.mlp(proj)

        if self._residual is not None:
            res = self._residual(embedding_seq)
        else:
            res = embedding_seq

        embedding_seq = self.output_dropout(mlp_out + res)

        return embedding_seq, next_seq_hidden_state


### ===================== ================================================ ===================== #

class SelfiesEmbeddedConcatRND(nn.Module):
    def __init__(self, obs_features, hidden_state_shape) -> None:
        super().__init__()

        hidden_state_features = hidden_state_shape[0] * hidden_state_shape[1]

        # predictor network
        self._predictor_obs_embedding = nn.Sequential(
            nn.Linear(obs_features, 64),
            nn.ReLU(),
        )

        self._predictor = nn.Sequential(
            nn.Linear(64 + hidden_state_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # target network
        self._target_obs_embedding = nn.Sequential(
            nn.Linear(obs_features, 64),
            nn.ReLU(),
        )

        self._target = nn.Sequential(
            nn.Linear(64 + hidden_state_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        for param in self._target_obs_embedding.parameters():
            param.requires_grad = False

        for param in self._target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_embedding = self._predictor_obs_embedding(obs)
        predicted_features = self._predictor(torch.cat([obs_embedding, hidden_state], dim=1))

        obs_embedding = self._target_obs_embedding(obs)
        target_features = self._target(torch.cat([obs_embedding, hidden_state], dim=1))

        return predicted_features, target_features


# ==================================================================================================== #

class SelfiesPretrainedNet(nn.Module, agent.PretrainedRecurrentNetwork):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        # Actor only
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(vocab_size)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, vocab_size)

        init_linear_weights(self)

    def load_state_dict(self, state_dict, strict=True):
        try:
            return super().load_state_dict(state_dict, strict)
        except RuntimeError:
            # Fallback to modified architecture if loading fails
            self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNetV2(self._vocab_size)
            self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, self._vocab_size)
            return super().load_state_dict(state_dict, strict)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[CategoricalDist, torch.Tensor]:
        # feed forward to the shared network
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(x, hidden_state)
        # feed forward to the actor layer
        policy_dist_seq = self._actor(embedding_seq)
        return policy_dist_seq, next_seq_hidden_state

    def hidden_state_shape(self) -> Tuple[int, int]:
        return (
            self._actor_critic_shared_net.n_recurrent_layers,
            self._actor_critic_shared_net.hidden_state_dim
        )

    def model(self) -> nn.Module:
        return self


class SelfiesRecurrentPPONet(nn.Module, agent.RecurrentPPONetwork):
    def __init__(self, in_features: int, num_actions: int) -> None:
        super().__init__()
        self._in_features = in_features
        self._num_actions = num_actions

        # Actor-Critic
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions)
        self._critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)

        init_linear_weights(self)

    def load_state_dict(self, state_dict, strict=True):
        try:
            return super().load_state_dict(state_dict, strict)
        except RuntimeError:
            self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNetV2(self._in_features)
            self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, self._num_actions)
            self._critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
            return super().load_state_dict(state_dict, strict)

    def model(self) -> nn.Module:
        return self

    def hidden_state_shape(self) -> Tuple[int, int]:
        return (
            self._actor_critic_shared_net.n_recurrent_layers,
            self._actor_critic_shared_net.hidden_state_dim
        )

    def forward(
            self,
            obs_seq: torch.Tensor,
            hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:
        # feed forward to the shared network
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(obs_seq, hidden_state)

        # feed forward to actor-critic layer
        policy_dist_seq = self._actor(embedding_seq)
        state_value_seq = self._critic(embedding_seq)

        return policy_dist_seq, state_value_seq, next_seq_hidden_state


class SelfiesRecurrentPPORNDNet(nn.Module, agent.RecurrentPPORNDNetwork):
    def __init__(self, in_features: int, num_actions: int, temperature: float = 1.0) -> None:
        super().__init__()
        self._in_features = in_features
        self._num_actions = num_actions
        self._temperature = temperature

        # Actor-Critic
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions, temperature=temperature)
        self._ext_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        self._int_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)

        # RND
        self._rnd_net = SelfiesEmbeddedConcatRND(in_features, self.hidden_state_shape())

        init_linear_weights(self)

    def load_state_dict(self, state_dict, strict=True):
        try:
            return super().load_state_dict(state_dict, strict)
        except RuntimeError:
            self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNetV2(self._in_features)
            self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, self._num_actions, temperature=self._temperature)
            self._ext_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
            self._int_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
            # RND must be re-initialized because hidden_state_shape changed
            self._rnd_net = SelfiesEmbeddedConcatRND(self._in_features, self.hidden_state_shape())
            return super().load_state_dict(state_dict, strict)

    def model(self) -> nn.Module:
        return self

    def hidden_state_shape(self) -> Tuple[int, int]:
        return (
            self._actor_critic_shared_net.n_recurrent_layers,
            self._actor_critic_shared_net.hidden_state_dim
        )

    def forward_actor_critic(
            self,
            obs_seq: torch.Tensor,
            hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor, torch.Tensor]:
        # feed forward to the shared net
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(obs_seq, hidden_state)

        # feed forward to actor-critic layer
        policy_dist_seq = self._actor(embedding_seq)
        ext_state_value_seq = self._ext_critic(embedding_seq)
        int_state_value_seq = self._int_critic(embedding_seq)

        return policy_dist_seq, ext_state_value_seq, int_state_value_seq, next_seq_hidden_state

    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._rnd_net(obs, hidden_state)
