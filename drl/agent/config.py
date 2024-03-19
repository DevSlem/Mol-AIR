from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """
    Recurrent PPO configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    
@dataclass(frozen=True)
class RecurrentPPORNDConfig:
    """
    Recurrent PPO with RND configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: Optional[int] = None
    padding_value: float = 0.0
    gamma: float = 0.99
    gamma_n: float = 0.99
    nonepi_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: Optional[int] = 50
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
