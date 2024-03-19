from abc import ABC, abstractmethod
from typing import Dict, Optional, TypeVar, Type, Callable

import torch

from drl.exp import Experience
from drl.net import Network

T = TypeVar('T', bound='Agent')

def agent_config(name: str) -> Callable[[Type[T]], Type[T]]:    
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, Agent):
            raise TypeError("Class must inherit from Agent")
        cls.name = name
        return cls
    return decorator

class Agent(ABC):
    """
    Deep reinforcement learning agent.
    """
    
    name: str
    
    def __init__(
        self,
        num_envs: int, 
        network: Network,
        device: Optional[str] = None,
    ) -> None:
        assert num_envs >= 1, "The number of environments must be greater than or euqal to 1."
                
        self._model = network.model()
        if device is not None:
            self._model = self._model.to(device=torch.device(device))
        self._device = network.device
        self._num_envs = num_envs
        
        self._training_steps = 0
                
    @abstractmethod
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Select actions from the observation. 

        Args:
            obs (Tensor): observation `(num_envs, *obs_shape)`

        Returns:
            action (Tensor): `(num_envs, num_action_branches)`
        """
        raise NotImplementedError()
        
    @abstractmethod
    def update(self, exp: Experience) -> Optional[dict]:
        """
        Update the agent. It stores data and trains the agent.

        Args:
            exp (Experience): one-step experience tuple.
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def config_dict(self) -> dict:
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def training_steps(self) -> int:
        return self._training_steps
    
    def _tick_training_steps(self):
        self._training_steps += 1
        
    @property
    def log_data(self) -> Dict[str, tuple]:
        """
        Returns log data and reset it.

        Returns:
            dict[str, tuple]: key: (value, time)
        """
        return dict()
            
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        return dict(
            training_steps=self._training_steps,
            model=self._model.state_dict()
        )
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        self._training_steps = state_dict["training_steps"]
        self._model.load_state_dict(state_dict["model"])
