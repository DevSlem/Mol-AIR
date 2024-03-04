import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import drl
import drl.agent as agent
import envs
from drl.util import IncrementalMean
from util import logger, try_create_dir

warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard.writer import SummaryWriter

warnings.filterwarnings(action="default")


@dataclass(frozen=True)
class InferenceConfig:
    episodes: int
    agent_file_path: Optional[str] = None
    results_dir: Optional[str] = None

class Inference:
    _TRACE_ENV: int = 0
    
    def __init__(
        self,
        id: str,
        config: InferenceConfig,
        agent: agent.Agent,
        env: envs.Env,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("Inference only supports single environment.")
        
        self._id = id
        self._config = config
        self._agent = agent
        self._env = env
        
        self._dtype = torch.float32
        self._device = self._agent.device
        
        self._enabled = True
        
        self._tb_logger = None
        
    def inference(self) -> "Inference":
        if not self._enabled:
            raise RuntimeError("Inference is already closed.")
        
        with agent.BehaviorScope(self._agent, agent.BehaviorType.INFERENCE):
            if not logger.enabled():
                logger.enable(self._id, enable_log_file=False)
                
            self._load_inference() 
            
            logger.disable()
            logger.enable(self._id, enable_log_file=False)
            
            try:
                # reset the environment
                # since it automatically resets, we just call reset only once
                obs = self._agent_tensor(self._env.reset())
                
                for e in range(self._config.episodes):
                    not_terminated = True
                    cumulative_reward = 0.0
                    epi_len = 0
                    
                    while not_terminated:
                        # take action and observe
                        action = self._agent.select_action(obs)
                        next_obs, reward, terminated, real_final_next_obs = self._env.step(action.detach().cpu())
                        
                        # update the agent
                        real_next_obs = next_obs.clone()
                        real_next_obs[terminated.squeeze(dim=-1)] = real_final_next_obs
                        
                        next_obs = self._agent_tensor(next_obs)
                        real_next_obs = self._agent_tensor(real_next_obs)
                        
                        exp = drl.Experience(
                            obs,
                            action,
                            real_next_obs,
                            self._agent_tensor(reward),
                            self._agent_tensor(terminated),
                        )
                        self._agent.update(exp)
                        
                        # take next step
                        obs = next_obs
                        not_terminated = not terminated[self._TRACE_ENV].item()
                        cumulative_reward += reward[self._TRACE_ENV].item()
                        epi_len += 1
                        
                    logger.print(f"inference - episode: {e}, cumulative reward: {cumulative_reward:.2f}")
                    self._save_inference(e, cumulative_reward, epi_len)
                    
                logger.print(f"Inference is finished.")
            except KeyboardInterrupt:
                logger.print(f"Inference interrupted.")
                    
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        if logger.enabled():
            logger.disable()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            self._tb_logger.close()
        
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _save_inference(self, episode: int, cumulative_reward: float, episode_length: int):
        base_dir = self._config.results_dir if self._config.results_dir is not None else f"{logger.dir()}/Inference"
        
        # log data
        if self._tb_logger is None:
            self._tb_logger = SummaryWriter(log_dir=base_dir)
        
        self._tb_logger.add_scalar("Environment/Cumulative Reward", cumulative_reward, episode)
        self._tb_logger.add_scalar("Environment/Episode Length", episode_length, episode)
        
        for key, (value, t) in self._env.log_data.items():
            if t is None:
                self._tb_logger.add_scalar(key, value, episode)
            else:
                self._tb_logger.add_scalar(key, value, t)
                
        # save data
        self._env.save_data(base_dir)

    def _load_inference(self):
        agent_file_path = self._config.agent_file_path if self._config.agent_file_path is not None else f"{logger.dir()}/agent.pt"
        try:
            state_dict = torch.load(agent_file_path)
            logger.print(f"Agent is successfully loaded from: {agent_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Agent file is not found at: {agent_file_path}")
        
        try:
            self._agent.load_state_dict(state_dict["agent"])
        except:
            raise RuntimeError("the loaded agent is not compatible wsith the current agent")
        
@dataclass(frozen=True)
class InferenceCkptConfig:
    episodes: int
    ckpt_dir: Optional[str] = None
    results_dir: Optional[str] = None
        
class InferenceCkpt:
    _TRACE_ENV: int = 0
    
    def __init__(
        self,
        id: str,
        config: InferenceCkptConfig,
        agent: agent.Agent,
        env: envs.Env,
    ) -> None:
        self._id = id
        self._config = config
        self._agent = agent
        self._env = env
        
        self._dtype = torch.float32
        self._device = self._agent.device
        
        self._enabled = True
        
        self._tb_logger = None
        
    def inference(self) -> "InferenceCkpt":
        if not self._enabled:
            raise RuntimeError("InferenceCkpt is already closed.")
        
        with agent.BehaviorScope(self._agent, agent.BehaviorType.INFERENCE):
            if not logger.enabled():
                logger.enable(self._id, enable_log_file=False)
                
            ckpt_paths = self._load_ckpt_paths()
            
            logger.disable()
            logger.enable(self._id, enable_log_file=False)
            
            try:
                # reset the environment
                # since it automatically resets, we just call reset only once
                obs = self._agent_tensor(self._env.reset())
                
                for ckpt_path in ckpt_paths:
                    self._load_inference(ckpt_path)
                    cumulative_reward_mean = IncrementalMean()
                    episode_len_mean = IncrementalMean()
                    log_data_mean = dict()
                    
                    for _ in range(self._config.episodes):
                        # reset the environment
                        not_terminated = True
                        cumulative_reward = 0.0
                        epi_len = 0
                        
                        while not_terminated:
                            # take action and observe
                            action = self._agent.select_action(obs)
                            next_obs, reward, terminated, real_final_next_obs = self._env.step(action.detach().cpu())
                            
                            # update the agent
                            real_next_obs = next_obs.clone()
                            real_next_obs[terminated.squeeze(dim=-1)] = real_final_next_obs
                            
                            next_obs = self._agent_tensor(next_obs)
                            real_next_obs = self._agent_tensor(real_next_obs)
                            
                            exp = drl.Experience(
                                obs,
                                action,
                                real_next_obs,
                                self._agent_tensor(reward),
                                self._agent_tensor(terminated),
                            )
                            self._agent.update(exp)
                            
                            # take next step
                            obs = next_obs
                            not_terminated = not terminated[self._TRACE_ENV].item()
                            cumulative_reward += reward[self._TRACE_ENV].item()
                            epi_len += 1
                            
                        cumulative_reward_mean.update(cumulative_reward)
                        episode_len_mean.update(epi_len)
                        for key, (value, _) in self._env.log_data.items():
                            if key not in log_data_mean:
                                log_data_mean[key] = IncrementalMean()
                            log_data_mean[key].update(value)
                        
                    logger.print(f"inference - average cumulative reward: {cumulative_reward_mean.mean:.2f}")
                    
                    log_data_mean["Environment/Cumulative Reward"] = cumulative_reward_mean
                    log_data_mean["Environment/Episode Length"] = episode_len_mean
                    
                    self._save_inference(log_data_mean)
                
                logger.print(f"Inference is finished.")
            except KeyboardInterrupt:
                logger.print(f"Inference interrupted.")
                    
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        if logger.enabled():
            logger.disable()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            self._tb_logger.close()
            
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _save_inference(self, log_data: Dict[str, IncrementalMean]):
        base_dir = self._config.results_dir if self._config.results_dir is not None else f"{logger.dir()}/Inference on {self._config.episodes} Episodes"
        
        # log data
        if self._tb_logger is None:
            self._tb_logger = SummaryWriter(log_dir=base_dir)
        
        for key, incr_mean in log_data.items():
            self._tb_logger.add_scalar(key, incr_mean.mean, self._time_steps)
            self._tb_logger.add_scalar(f"{key} per Episode", incr_mean.mean, self._episodes)
                
        # save data
        env_base_dir = f"{base_dir}/ckpt_{self._time_steps}"
        try_create_dir(env_base_dir)
        self._env.save_data(env_base_dir)
        
    def _load_ckpt_paths(self) -> Tuple[str, ...]:
        import glob
        import re

        def extract_prefix_and_time_steps(file_name):
            match = re.search(r'(\D+)(\d+)\.pt', file_name)
            if match:
                return match.group(1), int(match.group(2))
            else:
                return '', float('inf')
        ckpt_dir = self._config.ckpt_dir if self._config.ckpt_dir is not None else f"{logger.dir()}/agent_ckpt"
        return tuple(sorted(glob.glob(f"{ckpt_dir}/*.pt"), key=extract_prefix_and_time_steps))

    def _load_inference(self, agent_file_path: str):
        try:
            state_dict = torch.load(agent_file_path)
            logger.print(f"Agent is successfully loaded from: {agent_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Agent file is not found at: {agent_file_path}")
        
        try:
            self._agent.load_state_dict(state_dict["agent"])
        except:
            raise RuntimeError("the loaded agent is not compatible wsith the current agent")
        
        self._time_steps = state_dict["train"]["time_steps"]
        self._episodes = state_dict["train"]["episodes"