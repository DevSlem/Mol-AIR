from abc import abstractmethod
from typing import Tuple

import torch

from drl.net import RecurrentNetwork
from drl.policy_dist import CategoricalDist


class RecurrentPPONetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network.
    
    Since it uses the recurrent network, 
    you must consider the hidden state which can acheive the action-observation history.
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    """
        
    @abstractmethod
    def forward(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution 
        and state value using the recurrent network.
        
        It's recommended to set your recurrent network to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (CategoricalDist): policy distribution sequences
            state_value_seq (Tensor): state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
            
        ## Input/Output Details
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError
    
class RecurrentPPORNDNetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network with Random Network Distillation (RND).
    
    Since it uses the recurrent network, you must consider the hidden state which can acheive the action-observation history.
    
    Note that since PPO uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Be careful not to share parameters between PPO and RND networks.
    
    RND uses extrinsic and intrinsic reward streams. 
    Each stream can be different episodic or non-episodic, and can have different discount factors. 
    RND constitutes of the predictor and target networks. 
    Both of them should have the similar architectures (not must same) but their initial parameters should not be the same.
    The target network is determinsitic, which means it will be never updated. 
    """
        
    @abstractmethod
    def forward_actor_critic(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution , 
        extinrisc state value and intrinsic state value using the recurrent network.
                
        It's recommended to set your recurrent network to `batch_first=True`.
        
        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (Tensor): policy distribution sequences
            ext_state_value_seq (Tensor): extrinsic state value sequences
            int_state_value_seq (Tensor): intrinsic state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
        
        ## Input/Output Details
                
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |ext_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |int_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method tp to compute both predicted feature and target feature. 
        You can use the hidden state by concatenating it with either the observation or the feature extracted from the observation. 

        Args:
            obs (Tensor): observation batch
            hidden_state (Tensor): hidden state batch with flattened features

        Returns:
            predicted_feature (Tensor): predicted feature whose gradient flows
            target_feature (Tensor): target feature whose gradient doesn't flow
            
        ## Input/Output Details
        
        The value of `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        |hidden_state|`(batch_size, D x num_layers x H)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted_feature|`(batch_size, out_features)`|
        |target_feature|`(batch_size, out_features)`|
        """
        raise NotImplementedError

class RecurrentPPOEpisodicRNDNetwork(RecurrentNetwork):
    @abstractmethod
    def forward_actor_critic(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution 
        and state value using the recurrent network.
        
        It's recommended to set your recurrent network to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (CategoricalDist): policy distribution sequences
            state_value_seq (Tensor): state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
            
        ## Input/Output Details
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method tp to compute both predicted feature and target feature. 
        You can use the hidden state by concatenating it with either the observation or the feature extracted from the observation. 

        Args:
            obs (Tensor): observation batch
            hidden_state (Tensor): hidden state batch with flattened features

        Returns:
            predicted_feature (Tensor): predicted feature whose gradient flows
            target_feature (Tensor): target feature whose gradient doesn't flow
            
        ## Input/Output Details
        
        The value of `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        |hidden_state|`(batch_size, D x num_layers x H)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted_feature|`(batch_size, out_features)`|
        |target_feature|`(batch_size, out_features)`|
        """
        raise NotImplementedError