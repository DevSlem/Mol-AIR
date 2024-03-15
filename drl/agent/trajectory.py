from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class RecurrentPPOExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    hidden_state: torch.Tensor
    
class RecurrentPPOTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        
    def add(self, exp: RecurrentPPOExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        
    def sample(self) -> RecurrentPPOExperience:
        self._obs_buffer.append(self._final_next_obs)
        exp_batch = RecurrentPPOExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._state_value_buffer),
            torch.concat(self._hidden_state_buffer, dim=1),
        )
        self.reset()
        return exp_batch
        
    def _make_buffer(self) -> list:
        return [None] * self._n_steps
    
@dataclass(frozen=True)
class RecurrentPPORNDExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    rnd_int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    epi_state_value: torch.Tensor
    nonepi_state_value: torch.Tensor
    hidden_state: torch.Tensor
    next_hidden_state: torch.Tensor
    
class RecurrentPPORNDTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._rnd_int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._epi_state_value_buffer = self._make_buffer()
        self._nonepi_state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        self._final_next_hidden_state = None
        
    def add(self, exp: RecurrentPPORNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.reward
        self._rnd_int_reward_buffer[self._recent_idx] = exp.rnd_int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._epi_state_value_buffer[self._recent_idx] = exp.epi_state_value
        self._nonepi_state_value_buffer[self._recent_idx] = exp.nonepi_state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        self._final_next_hidden_state = exp.next_hidden_state
        
    def sample(self) -> RecurrentPPORNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        self._hidden_state_buffer.append(self._final_next_hidden_state)
        exp_batch = RecurrentPPORNDExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._rnd_int_reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._epi_state_value_buffer),
            torch.concat(self._nonepi_state_value_buffer),
            torch.concat(self._hidden_state_buffer[:-1], dim=1),
            torch.concat(self._hidden_state_buffer[1:], dim=1)
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps # type: ignore

@dataclass(frozen=True)
class RecurrentPPOEpisodicRNDExperience:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    ext_reward: torch.Tensor
    int_reward: torch.Tensor
    terminated: torch.Tensor
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    hidden_state: torch.Tensor
    next_hidden_state: torch.Tensor
    
class RecurrentPPOEpisodicRNDTrajectory:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps
        self.reset()
        
    @property
    def reached_n_steps(self) -> bool:
        return self._recent_idx + 1 >= self._n_steps
        
    def reset(self):
        self._recent_idx = -1
        
        self._obs_buffer = self._make_buffer()
        self._action_buffer = self._make_buffer()
        self._reward_buffer = self._make_buffer()
        self._int_reward_buffer = self._make_buffer()
        self._terminated_buffer = self._make_buffer()
        self._action_log_prob_buffer = self._make_buffer()
        self._state_value_buffer = self._make_buffer()
        self._hidden_state_buffer = self._make_buffer()
        
        self._final_next_obs = None
        self._final_next_hidden_state = None
        
    def add(self, exp: RecurrentPPOEpisodicRNDExperience):
        self._recent_idx += 1
        
        self._obs_buffer[self._recent_idx] = exp.obs
        self._action_buffer[self._recent_idx] = exp.action
        self._reward_buffer[self._recent_idx] = exp.ext_reward
        self._int_reward_buffer[self._recent_idx] = exp.int_reward
        self._terminated_buffer[self._recent_idx] = exp.terminated
        self._action_log_prob_buffer[self._recent_idx] = exp.action_log_prob
        self._state_value_buffer[self._recent_idx] = exp.state_value
        self._hidden_state_buffer[self._recent_idx] = exp.hidden_state
        
        self._final_next_obs = exp.next_obs
        self._final_next_hidden_state = exp.next_hidden_state
        
    def sample(self) -> RecurrentPPOEpisodicRNDExperience:
        self._obs_buffer.append(self._final_next_obs)
        self._hidden_state_buffer.append(self._final_next_hidden_state)
        exp_batch = RecurrentPPOEpisodicRNDExperience(
            torch.concat(self._obs_buffer[:-1]),
            torch.concat(self._action_buffer),
            torch.concat(self._obs_buffer[1:]),
            torch.concat(self._reward_buffer),
            torch.concat(self._int_reward_buffer),
            torch.concat(self._terminated_buffer),
            torch.concat(self._action_log_prob_buffer),
            torch.concat(self._state_value_buffer),
            torch.concat(self._hidden_state_buffer[:-1], dim=1),
            torch.concat(self._hidden_state_buffer[1:], dim=1)
        )
        self.reset()
        return exp_batch
    
    def _make_buffer(self) -> list:
        return [None] * self._n_steps # type: ignore
