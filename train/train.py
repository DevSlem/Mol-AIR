import time
from queue import Queue
from typing import Optional

import numpy as np
import pandas as pd
import torch

import drl
import drl.agent as agent
import metric
from drl.util import IncrementalMean
from envs import Env
from util import CSVSyncWriter, TextInfoBox, logger, try_create_dir


class Train:
    _TRACE_ENV: int = 0
    
    def __init__(
        self,
        env: Env,
        agent: agent.Agent,
        id: str,
        total_time_steps: int,
        summary_freq: Optional[int] = None,
        agent_save_freq: Optional[int] = None,
        inference_env: Optional[Env] = None,
        n_inference_episodes: int = 1,
    ) -> None:
        self._env = env
        self._agent = agent
        
        self._id = id
        self._total_time_steps = total_time_steps
        self._summary_freq = total_time_steps // 20 if summary_freq is None else summary_freq
        self._agent_save_freq = self._summary_freq * 10 if agent_save_freq is None else agent_save_freq
        self._inference_env = inference_env
        self._n_inference_episodes = n_inference_episodes
        
        self._dtype = torch.float32
        self._device = self._agent.device
        self._best_score = float("-inf")
        
        self._time_steps = 0
        self._episodes = 0
        self._episode_len = 0
        self._real_start_time = time.time()
        self._real_time = 0.0
        
        self._cumulative_reward_mean = IncrementalMean()
        self._episode_len_mean = IncrementalMean()
                
        # helps to synchronize final molecule results of each episode
        self._metric_csv_sync_writer_dict = dict()
        # self._episode_molecule_sync_buffer_dict = defaultdict(lambda: SyncFixedBuffer(max_size=self._env.num_envs, callback=self._record_molecule))
        # to save final molecule results periodically
        self._molecule_queue = Queue() # (episode, env_id, score, selfies)
        self._best_molecule = None
        self._best_molecule_queue = Queue() # (episode, best_score, selfies)
        self._intrinsic_reward_queue = Queue() # (episode, env_id, time_step, intrinsic_rewards...)
        
        self._enabled = True
        
    def train(self) -> "Train":
        if not self._enabled:
            raise RuntimeError("Train is already closed.")
        
        if not logger.enabled():
            logger.enable(self._id, enable_log_file=False)
            
        if self._time_steps == self._total_time_steps:
            self._save_train()
            return self
            
        self._load_train()
        
        if self._time_steps >= self._total_time_steps:  
            logger.print(f"Training is already finished.")
            return self
        
        logger.disable()
        logger.enable(self._id, enable_log_file=True)
        
        self._print_train_info()            
        
        try:
            obs = self._env.reset()
            cumulative_reward = 0.0
            last_agent_save_t = 0
            for _ in range(self._time_steps, self._total_time_steps):
                # take action and observe
                obs = self._numpy_to_tensor(obs)
                action = self._agent.select_action(obs)
                next_obs, reward, terminated, real_final_next_obs, env_info = self._env.step(action.detach().cpu().numpy())
                
                # update the agent
                real_next_obs = next_obs.copy()
                real_next_obs[terminated] = real_final_next_obs
                real_next_obs = self._numpy_to_tensor(real_next_obs)
                
                exp = drl.Experience(
                    obs,
                    action,
                    real_next_obs,
                    self._numpy_to_tensor(reward[..., np.newaxis]),
                    self._numpy_to_tensor(terminated[..., np.newaxis]),
                )
                agent_info = self._agent.update(exp)
                
                # process info dict
                self._process_info_dict(env_info, agent_info)
                
                # take next step
                obs = next_obs
                cumulative_reward += reward[self._TRACE_ENV].item()
                self._tick_time_steps()
                
                if terminated[self._TRACE_ENV].item():
                    self._cumulative_reward_mean.update(cumulative_reward)
                    self._episode_len_mean.update(self._episode_len)
                    
                    cumulative_reward = 0.0
                    self._tick_episode()
                
                # summary
                if self._time_steps % self._summary_freq == 0:
                    self._summary_train()
                    
                # save the agent
                if self._time_steps % self._agent_save_freq == 0:
                    score = self._inference(self._n_inference_episodes)
                    self._save_train(score)
                    last_agent_save_t = self._time_steps
            logger.print(f"Training is finished.")
            if self._time_steps > last_agent_save_t:
                self._save_train()
            
        except KeyboardInterrupt:
            logger.print(f"Training interrupted at the time step {self._time_steps}.")
            self._save_train()
                    
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        
        if self._inference_env is not None:
            self._inference_env.close()
        
        if logger.enabled():
            logger.disable()
            
    def _make_csv_sync_writer(self, metric_name: str, metric_info_dict: dict):
        key_fields = metric_info_dict["keys"].keys()
        value_fields = metric_info_dict["values"].keys()
        
        return CSVSyncWriter(
            file_path=f"{logger.dir()}/{metric_name}.csv",
            key_fields=key_fields,
            value_fields=value_fields,
        )
        
    def _write_metric_info_dict(self, metric_name: str, metric_info_dict: dict):
        if metric_name not in self._metric_csv_sync_writer_dict:
            self._metric_csv_sync_writer_dict[metric_name] = self._make_csv_sync_writer(metric_name, metric_info_dict)
        self._metric_csv_sync_writer_dict[metric_name].add(
            keys=metric_info_dict["keys"],
            values=metric_info_dict["values"],
        )
        
    def _write_metric_dicts(self, metric_dicts, include_time_step=False):
        for metric_dict in metric_dicts:
            if metric_dict is None:
                continue
            for metric_name, metric_info in metric_dict.items():
                if include_time_step:
                    metric_info["values"]["time_step"] = self._time_steps
                if metric_name not in self._metric_csv_sync_writer_dict:
                    self._metric_csv_sync_writer_dict[metric_name] = self._make_csv_sync_writer(metric_name, metric_info)
                if not any(value_field in set(self._metric_csv_sync_writer_dict[metric_name].value_fields) for value_field in metric_info["values"].keys()):
                    self._metric_csv_sync_writer_dict[metric_name].value_fields += tuple(metric_info["values"].keys())
                self._metric_csv_sync_writer_dict[metric_name].add(
                    keys=metric_info["keys"],
                    values=metric_info["values"],
                )
            
    def _process_info_dict(self, env_info: dict, agent_info: Optional[dict]):        
        if "metric" in env_info: 
            self._write_metric_dicts(env_info["metric"], True)
                
        if agent_info is not None and "metric" in agent_info:
            self._write_metric_dicts(agent_info["metric"])
        
    def _tick_time_steps(self):
        self._episode_len += 1
        self._time_steps += 1
        self._real_time = time.time() - self._real_start_time
    
    def _tick_episode(self):
        self._episode_len = 0
        self._episodes += 1
        
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _numpy_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(device=self._device, dtype=self._dtype)
        
    def _print_train_info(self):
        text_info_box = TextInfoBox() \
            .add_text(f"SELFIES Training Start!") \
            .add_line(marker="=") \
            .add_text(f"ID: {self._id}") \
            .add_text(f"Output Path: {logger.dir()}")
        
        # display environment info
        env_config_dict = self._env.config_dict
        if len(env_config_dict.keys()) > 0:
            text_info_box.add_line() \
                .add_text(f"Environment INFO:")
            for key, value in env_config_dict.items():
                text_info_box.add_text(f"    {key}: {value}")
                
        # display training info
        text_info_box.add_line() \
            .add_text(f"Training INFO:") \
            .add_text(f"    number of environments: {self._env.num_envs}") \
            .add_text(f"    total time steps: {self._total_time_steps}") \
            .add_text(f"    summary frequency: {self._summary_freq}") \
            .add_text(f"    agent save frequency: {self._agent_save_freq}")
        
        # display agent info
        agent_config_dict = self._agent.config_dict
        agent_config_dict["device"] = self._device
        
        text_info_box.add_line() \
            .add_text(f"Agent:")
        for key, value in agent_config_dict.items():
            text_info_box.add_text(f"    {key}: {value}")
            
        logger.print(text_info_box.make(), prefix="")
        logger.print("", prefix="")
        
    def _summary_train(self):
        metric_df = pd.read_csv(f"{logger.dir()}/episode_metric.csv")
        current_metric_df = metric_df[(metric_df["time_step"] < self._time_steps) & (metric_df["time_step"] >= self._time_steps - self._summary_freq)]
        current_metric_df = current_metric_df.dropna(subset=["score", "smiles"])
        if len(current_metric_df) == 0:
            info = "episode has not terminated yet"
        else:
            score = current_metric_df["score"].mean()
            diversity = metric.calc_diversity(current_metric_df["smiles"].tolist())
            uniqueness = metric.calc_uniqueness(current_metric_df["smiles"].tolist())
            info = f"score: {score:.3f}, diversity: {diversity:.3f}, uniqueness: {uniqueness:.3f}"
            logger.log_data("Environment/Score", score, self._time_steps)
            logger.log_data(f"Environment/Diversity", diversity, self._time_steps)
            logger.log_data(f"Environment/Uniqueness", uniqueness, self._time_steps)
            
        logger.print(f"training time: {self._real_time:.2f}, time steps: {self._time_steps}/{self._total_time_steps}, {info}")
        
        for key, (value, t) in self._agent.log_data.items():
            logger.log_data(key, value, t)
            
        logger.plot_logs()
                    
    def _save_train(self, score=None):
        train_dict = dict(
            time_steps=self._time_steps,
            episodes=self._episodes,
            episode_len=self._episode_len,
        )
        state_dict = dict(
            train=train_dict,
            agent=self._agent.state_dict,
        )
        
        agent_save_path = f"{logger.dir()}/agent.pt"
        torch.save(state_dict, agent_save_path)
        
        agent_ckpt_dir = f"{logger.dir()}/agent_ckpt"
        try_create_dir(agent_ckpt_dir)
        torch.save(state_dict, f"{agent_ckpt_dir}/agent_{self._time_steps}.pt")
        
        if score is not None and score > self._best_score:
            self._best_score = score
            best_agent_save_path = f"{logger.dir()}/best_agent.pt"
            torch.save(state_dict, best_agent_save_path)
            logger.print(f"Agent is successfully saved ({self._time_steps} steps): {agent_save_path} and {best_agent_save_path}")
        else:
            logger.print(f"Agent is successfully saved ({self._time_steps} steps): {agent_save_path}")
        
        # self._env.save_data(logger.dir())
        # self._save_molecules(logger.dir())
        
    def _inference(self, n_episodes: int):
        if self._inference_env is None:
            return None
        
        episodes = np.zeros((self._inference_env.num_envs,), dtype=int)
        score_list = []
        smiles_list = []
        
        inference_agent = self._agent.inference_agent(self._inference_env.num_envs)
        inference_agent.model.eval()
        
        obs = self._inference_env.reset()
        
        while np.sum(episodes) < n_episodes:
            obs = self._numpy_to_tensor(obs)
            with torch.no_grad():
                action = inference_agent.select_action(obs)
            next_obs, reward, terminated, real_final_next_obs, env_info = self._inference_env.step(action.detach().cpu().numpy())
            
            # update the agent
            real_next_obs = next_obs.copy()
            real_next_obs[terminated] = real_final_next_obs
            real_next_obs = self._numpy_to_tensor(real_next_obs)
            
            exp = drl.Experience(
                obs,
                action,
                real_next_obs,
                self._numpy_to_tensor(reward[..., np.newaxis]),
                self._numpy_to_tensor(terminated[..., np.newaxis]),
            )
            with torch.no_grad():
                _ = inference_agent.update(exp)
                            
            obs = next_obs
            episodes += terminated.astype(int)
            
            scores, smiles = self._inference_metric(env_info)
            score_list += scores
            smiles_list += smiles
            
        if len(score_list) == 0:
            logger.print(f"=== Inference ({self._time_steps} steps) -> # molecules: 0 ===")
            return None
            
        avg_score = np.mean(score_list)
        diversity = metric.calc_diversity(smiles_list)
        uniqueness = metric.calc_uniqueness(smiles_list)
        logger.print(f"=== Inference ({self._time_steps} steps) -> # molecules: {len(score_list)}, score: {avg_score:.3f}, diversity: {diversity:.3f}, uniqueness: {uniqueness:.3f} ===")
        
        logger.log_data("Inference/Score", avg_score, self._time_steps)
        logger.log_data("Inference/Diversity", diversity, self._time_steps)
        logger.log_data("Inference/Uniqueness", uniqueness, self._time_steps)
        
        logger.plot_logs()
            
        self._agent.model.train()
        
        return avg_score
        
    def _inference_metric(self, env_info: dict):
        scores = []
        smiles = []
        
        if "metric" not in env_info:
            return scores, smiles
        
        metric_dicts = env_info["metric"]
        
        for metric_dict in metric_dicts:
            if metric_dict is None:
                continue
        
            if "episode_metric" not in metric_dict:
                continue
            
            score = metric_dict["episode_metric"]["values"]["score"]
            smile = metric_dict["episode_metric"]["values"]["smiles"]
            if score is not None and smile is not None:
                scores.append(score)
                smiles.append(smile)
            
        return scores, smiles
        
        
    def _load_train(self):
        try:
            state_dict = torch.load(f"{logger.dir()}/agent.pt")
        except FileNotFoundError:
            return
        
        train_dict = state_dict["train"]
        self._time_steps = train_dict["time_steps"]
        self._episodes = train_dict["episodes"]
        self._episode_len = train_dict["episode_len"]
        self._agent.load_state_dict(state_dict["agent"])
