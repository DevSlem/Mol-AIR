from typing import Optional

import torch

from drl.agent.agent import Agent, agent_config
from drl.agent.net import PretrainedRecurrentNetwork
from drl.exp import Experience

@agent_config(name="Pretrained Agent")
class PretrainedRecurrentAgent(Agent):
    def __init__(
        self,
        network: PretrainedRecurrentNetwork,
        num_envs: int,
        device: Optional[str] = None,
    ):
        super().__init__(num_envs, network, device)
        
        self._network = network
        
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        policy_dist_seq, next_hidden_state = self._network.forward(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        self._next_hidden_state = next_hidden_state
        return policy_dist_seq.sample().squeeze(dim=1)
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._prev_terminated = exp.terminated

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return PretrainedRecurrentAgent(self._network, num_envs, device or str(self.device))
