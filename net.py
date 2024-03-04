from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from noisy_net import FactorisedNoisyLayer
import drl.agent as agent
import drl.net as net
from drl.policy import CategoricalPolicy
from drl.policy_dist import CategoricalDist

def init_linear_weights(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.orthogonal_(layer.weight, 2.0**0.5)
            layer.bias.data.zero_()

class SelfiesRecurrentPPOSharedNet(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        
        self.hidden_state_dim = 64 * 2
        self.n_recurrent_layers = 2
        self.out_features = 256
        
        self.recurrent_layers = nn.LSTM(
            in_features,
            self.hidden_state_dim // 2, # H_cell = 64, H_hidden = 64
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
    
class SelfiesNaiveRND(nn.Module):
    def __init__(self, obs_features) -> None:
        super().__init__()
        
        # predictor network
        self._predictor = nn.Sequential(
            nn.Linear(obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # target network
        self._target = nn.Sequential(
            nn.Linear(obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        for param in self._target.parameters():
            param.requires_grad = False
        
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_features = self._predictor(obs)
        target_features = self._target(obs)
        return predicted_features, target_features
    
class SelfiesImmediateConcatRND(nn.Module):
    def __init__(self, obs_features, hidden_state_shape) -> None:
        super().__init__()

        rnd_in_features = obs_features + hidden_state_shape[0] * hidden_state_shape[1]
            
        # predictor network
        self._predictor = nn.Sequential(
            nn.Linear(rnd_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # target network
        self._target = nn.Sequential(
            nn.Linear(rnd_in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        for parm in self._target.parameters():
            parm.requires_grad = False
        
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rnd_input = torch.cat([obs, hidden_state], dim=1)
        predicted_features = self._predictor(rnd_input)
        target_features = self._target(rnd_input)
        return predicted_features, target_features
    
class SelfiesEmbeddedConcatRND(nn.Module):
    def __init__(self, obs_features, hidden_state_shape) -> None:
        super().__init__()
        
        hidden_state_features = hidden_state_shape[0] * hidden_state_shape[1]
        
        # predictor network
        self._predictor_obs_embedding = nn.Sequential(
            nn.Linear(obs_features, 64),
            nn.ReLU(),
        )
        
        # self._predictor_hidden_state_embedding = nn.Sequential(
        #     nn.Linear(hidden_state_features, 128),
        #     nn.ReLU(),
        # )
        
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
        
        # self._target_hidden_state_embedding = nn.Sequential(
        #     nn.Linear(hidden_state_features, 128),
        #     nn.ReLU(),
        # )
         
        self._target = nn.Sequential(
            nn.Linear(64 + hidden_state_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        for param in self._target_obs_embedding.parameters():
            param.requires_grad = False
            
        # for param in self._target_hidden_state_embedding.parameters():
        #     param.requires_grad = False
        
        for param in self._target.parameters():
            param.requires_grad = False
            
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_embedding = self._predictor_obs_embedding(obs)
        # hidden_state_embedding = self._predictor_hidden_state_embedding(hidden_state)
        predicted_features = self._predictor(torch.cat([obs_embedding, hidden_state], dim=1))
        
        obs_embedding = self._target_obs_embedding(obs)
        # hidden_state_embedding = self._target_hidden_state_embedding(hidden_state)
        target_features = self._target(torch.cat([obs_embedding, hidden_state], dim=1))
        
        return predicted_features, target_features
    
# ==================================================================================================== #
    
class SelfiesRecurrentPPONet(nn.Module, agent.RecurrentPPONetwork):
    def __init__(self, in_features: int, num_actions: int) -> None:
        super().__init__()
        
        # Actor-Critic
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions)
        self._critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        
        init_linear_weights(self)
        
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
    def __init__(self, in_features: int, num_actions: int, rnd_hidden_state: bool = True) -> None:
        super().__init__()
        
        # Actor-Critic
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions)
        self._ext_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        self._int_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        
        # RND
        if rnd_hidden_state:
            # self._rnd_net = SelfiesImmediateConcatRND(in_features, self.hidden_state_shape())
            self._rnd_net = SelfiesEmbeddedConcatRND(in_features, self.hidden_state_shape())
        else:
            self._rnd_net = SelfiesNaiveRND(in_features)
                        
        init_linear_weights(self)
        
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
    
class SelfiesRecurrentPPOEpisodicRNDNet(nn.Module, agent.RecurrentPPOEpisodicRNDNetwork):
    def __init__(self, in_features: int, num_actions: int, rnd_hidden_state: bool = True) -> None:
        super().__init__()
        
        # Actor-Critic
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNet(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions)
        self._critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        
        # RND
        if rnd_hidden_state:
            # self._rnd_net = SelfiesImmediateConcatRND(in_features, self.hidden_state_shape())
            self._rnd_net = SelfiesEmbeddedConcatRND(in_features, self.hidden_state_shape())
        else:
            self._rnd_net = SelfiesNaiveRND(in_features)
        
        init_linear_weights(self)
        
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
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:        
        # feed forward to the shared network
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(obs_seq, hidden_state)
        
        # feed forward to actor-critic layer
        policy_dist_seq = self._actor(embedding_seq)
        state_value_seq = self._critic(embedding_seq)
        
        return policy_dist_seq, state_value_seq, next_seq_hidden_state
    
    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._rnd_net(obs, hidden_state)

class SelfiesRecurrentPPONoisyRNDNet(nn.Module, agent.RecurrentPPORNDNetwork):
    def __init__(self, in_features: int, num_actions: int) -> None:
        super().__init__()
        
        self._lstm_hidden_features = 64
        self._encoding_hidden_features = 256
        self._num_recurrent_layers = 2
        
        self._recurrent_layer = nn.LSTM(
            in_features,
            self._lstm_hidden_features,
            batch_first=True,
            num_layers=self._num_recurrent_layers
        )
        
        self._encoding_layer = nn.Sequential(
            nn.Linear(self._lstm_hidden_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self._encoding_hidden_features),
            nn.ReLU()
        )
        
        self._actor = CategoricalPolicy(self._encoding_hidden_features, num_actions)
        self._ext_critic = nn.Linear(self._encoding_hidden_features, 1)
        self._int_critic = nn.Linear(self._encoding_hidden_features, 1)
        
        rnd_in_features = in_features + self.hidden_state_shape()[0] * self.hidden_state_shape()[1]
        self._rnd_predictor = nn.Sequential(
            FactorisedNoisyLayer(rnd_in_features, 128),
            nn.ReLU(),
            FactorisedNoisyLayer(128, 256),
            nn.ReLU(),
            FactorisedNoisyLayer(256, 256),
            nn.ReLU(),
            FactorisedNoisyLayer(256, 256)
        )
        self._rnd_target = nn.Sequential(
            nn.Linear(rnd_in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.orthogonal_(layer.weight, 2.0**0.5)
                layer.bias.data.zero_()
                
        for parm in self._rnd_target.parameters():
            parm.requires_grad = False
        
    def model(self) -> nn.Module:
        return self
        
    def hidden_state_shape(self) -> Tuple[int, int]:
        return (self._num_recurrent_layers, self._lstm_hidden_features * 2)
        
    def forward_actor_critic(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor, torch.Tensor]:
        # feed forward to recurrent layer
        h, c = net.unwrap_lstm_hidden_state(hidden_state)
        encoding_seq, (h_n, c_n) = self._recurrent_layer(obs_seq, (h, c))
        next_seq_hidden_state = net.wrap_lstm_hidden_state(h_n, c_n)
        
        # feed forward to encoding layer
        encoding_seq = self._encoding_layer(encoding_seq)
        
        # feed forward to actor-critic layer
        policy_dist_seq = self._actor(encoding_seq)
        ext_state_value_seq = self._ext_critic(encoding_seq)
        int_state_value_seq = self._int_critic(encoding_seq)
        
        return policy_dist_seq, ext_state_value_seq, int_state_value_seq, next_seq_hidden_state
    
    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rnd_input = torch.cat([obs, hidden_state], dim=1)
        predicted_features = self._rnd_predictor(rnd_input)
        target_features = self._rnd_target(rnd_input)
        return predicted_features, target_features
