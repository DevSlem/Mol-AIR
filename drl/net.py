from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_


class Trainer:
    """
    PyTorch optimizer wrapper for single scalar loss.
    """
    @dataclass(frozen=True)
    class __ClipGradNormConfig:
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]]
        max_norm: float
        norm_type: float = 2.0
        error_if_nonfinite: bool = False
    
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self._clip_grad_norm_config = None
        
    def enable_grad_clip(
        self,
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ) -> "Trainer":
        """
        Enable gradient clipping when parameter update. Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Tensor | Iterable[Tensor]): an iterable of Tensors or a single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float, optional): type of the used p-norm. Can be `inf` for infinity norm.
            error_if_nonfinite (bool, optional): if True, an error is thrown if the total norm of the gradients from `parameters` is `nan`, `inf`, or `-inf`. Default: False (will switch to True in the future)

        Returns:
            Trainer: self
        """
        self._clip_grad_norm_config = Trainer.__ClipGradNormConfig(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite
        )
        return self
    
    def step(self, loss: torch.Tensor, training_steps: int):
        """
        Performs a single optimization step (parameter update).
        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_grad_norm_config is not None:
            clip_grad_norm_(**self._clip_grad_norm_config.__dict__)
        self._optimizer.step()

class Network(ABC):
    @abstractmethod
    def model(self) -> nn.Module:
        raise NotImplementedError
    
    @property
    def device(self) -> torch.device:
        return next(self.model().parameters()).device

class RecurrentNetwork(Network):
    """
    Recurrent neural network (RNN).
    """
    
    @abstractmethod
    def hidden_state_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the rucurrent hidden state `(D x num_layers, H)`.
        
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network

        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError

def wrap_lstm_hidden_state(
    h: torch.Tensor, 
    c: torch.Tensor
) -> torch.Tensor:
    """
    Wrap the hidden state of LSTM.
    
    Args:
        h (Tensor): `(D x num_layers, seq_batch_size, H_out)`
        c (Tensor): `(D x num_layers, seq_batch_size, H_cell)`
    
    Returns:
        hc (Tensor): `(D x num_layers, seq_batch_size, H_out + H_cell)`
    """
    return torch.cat((h, c), dim=2)

def unwrap_lstm_hidden_state(
    hc: torch.Tensor, 
    h_size: Union[int, None] = None, 
    c_size: Union[int, None] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unwrap the hidden state of LSTM.
    
    Note if `H_out` and `H_cell` are different size, you must specify both of them.

    Args:
        hc (Tensor): `(D x num_layers, seq_batch_size, H_out + H_cell)`
        h_size (int | None): `H_out`. Defaults to `H_out` = `H_cell`.
        c_size (int | None): `H_cell`. Defaults to `H_cell` = `H_out`.

    Returns:
        h (Tensor): `(D x num_layers, seq_batch_size, H_out)`
        c (Tensor): `(D x num_layers, seq_batch_size, H_cell)`
    """
    if (h_size is None) ^ (c_size is None):
        raise ValueError("if `H_out` and `H_cell` are different size, you must specify both of them.")
    
    if (h_size is None) and (c_size is None):
        h_size = c_size = hc.shape[2] // 2
    
    h, c = hc.split([h_size, c_size], dim=2)
    return h.contiguous(), c.contiguous()