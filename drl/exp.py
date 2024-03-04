from typing import Callable
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class Experience:
    """
    Experience data type. 
    
    Args:
        obs (Tensor): `(num_envs, *obs_shape)`
        action (Tensor): `(num_envs, 1)`
        next_obs (Tensor): `(num_envs, *obs_shape)`
        reward (Tensor): `(num_envs, 1)`
        terminated (Tensor): `(num_envs, 1)`
    """
    obs: Tensor
    action: Tensor
    next_obs: Tensor
    reward: Tensor
    terminated: Tensor
    
    def transform(self, func: Callable[[Tensor], Tensor]) -> "Experience":
        """
        Transform the experience data type.

        Args:
            func (Callable[[TSource], TResult]): transform function

        Returns:
            Experience[TResult]: transformed experience
        """
        return Experience(
            obs=func(self.obs),
            action=func(self.action),
            next_obs=func(self.next_obs),
            reward=func(self.reward),
            terminated=func(self.terminated)
        )
        