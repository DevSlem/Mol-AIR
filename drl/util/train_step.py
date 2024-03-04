from typing import Iterator
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils.clip_grad import clip_grad_norm_

class TrainStep:
    def __init__(self,
                 optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer
        
        self._parameters = None
        self._grad_clip_max_norm = 0.0
    
    def enable_grad_clip(self, parameters: Iterator[Parameter], grad_clip_max_norm: float):
        """
        Enable gradient clipping.

        Args:
            parameters (Iterator[Parameter]): an iterable of Tensors or a single Tensor that will have gradients normalized
            grad_clip_max_norm (float): max norm of the gradients
        """
        self._parameters = parameters
        self._grad_clip_max_norm = grad_clip_max_norm
        
    def train_step(self, loss: torch.Tensor):
        """
        Gradient step method.

        Args:
            loss (torch.Tensor): single loss value
        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._parameters is not None:
            clip_grad_norm_(self._parameters, self._grad_clip_max_norm)
        self._optimizer.step()
