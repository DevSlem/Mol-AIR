from typing import Optional

import torch
from torch.distributions import Categorical

class CategoricalDist:
    """
    Categorical policy distribution for the discrete action type.
    
    `*batch_shape` depends on the input of the algorithm you are using.
    
    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(num_seq, seq_len)`
    
    Args:
        probs (Tensor): categorical probabilities `(*batch_shape, num_actions)` which is typically the output of neural network
    """
    def __init__(
        self, 
        probs: Optional[torch.Tensor] = None, 
        logits: Optional[torch.Tensor] = None
    ) -> None:
        self._dist = Categorical(probs=probs, logits=logits)

    def sample(self) -> torch.Tensor:
        """Sample an action `(*batch_shape, 1)` from the policy distribution."""
        return self._dist.sample().unsqueeze_(dim=-1)
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Returns the log of the porability mass/density function accroding to the `action`.

        Args:
            action (Tensor): `(*batch_shape, 1)`

        Returns:
            log_prob (Tensor): `(*batch_shape, 1)`
        """
        action = action.squeeze(dim=-1)
        return self._dist.log_prob(action).unsqueeze_(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of this distribution `(*batch_shape, num_branches)`. 
        """
        return self._dist.entropy().unsqueeze_(dim=-1)