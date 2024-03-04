import time
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from queue import Queue
from util import SyncFixedBuffer
import csv
from util import file_exists

import torch

import drl
import drl.agent as agent
import envs
from envs.chem_env import ChemEnv
from drl.util import IncrementalMean
from util import TextInfoBox, logger, try_create_dir, CSVSyncWriter


@dataclass(frozen=True)
class TrainConfig:
    time_steps: int
    summary_freq: int
    agent_save_freq: int
    
    def __init__(
        self,
        time_steps: int,
        summary_freq: Optional[int] = None,
        agent_save_freq: Optional[int] = None,
    ) -> None:
        summary_freq = time_steps // 20 if summary_freq is None else summary_freq
        agent_save_freq = summary_freq * 10 if agent_save_freq is None else agent_save_freq
        
        object.__setattr__(self, "time_steps", time_steps)
        object.__setattr__(self, "summary_freq", summary_freq)
        object.__setattr__(self, "agent_save_freq", agent_save_freq)

class Train:
    _TRACE_ENV: int = 0
    
    def __init__(
        self,
        id: str,
        config: TrainConfig,
        agent: agent.Agent,
        env: ChemEnv,
    ) -> None:
        self._id = id
        self._config = config
        self._agent = agent
        self._env = env
        
        self._dtype = torch.float32
        self._device = self._agent.device
        
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
        
        with agent.BehaviorScope(self._agent, agent.BehaviorType.TRAIN):
            if not logger.enabled():
                logger.enable(self._id, enable_log_file=False)
                
            self._load_train()
            
            if self._time_steps >= self._config.time_steps:  
                logger.print(f"Training is already finished.")
                return self
            
            logger.disable()
            logger.enable(self._id)
            
            self._print_train_info()            
            
            try:
                obs = self._agent_tensor(self._env.reset())
                cumulative_reward = 0.0
                last_agent_save_t = 0
                for _ in range(self._time_steps, self._config.time_steps):
                    # take action and observe
                    action = self._agent.select_action(obs)
                    next_obs, reward, terminated, real_final_next_obs, env_info = self._env.step(action.detach().cpu())
                    
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
                    if self._time_steps % self._config.summary_freq == 0:
                        self._summary_train()
                        
                    # save the agent
                    if self._time_steps % self._config.agent_save_freq == 0:
                        self._save_train()
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
        
    def _write_metric_dicts(self, metric_dicts):
        for metric_dict in metric_dicts:
            if metric_dict is None:
                continue
            for metric_name, metric_info in metric_dict.items():
                if metric_name not in self._metric_csv_sync_writer_dict:
                    self._metric_csv_sync_writer_dict[metric_name] = self._make_csv_sync_writer(metric_name, metric_info)
                if not any(value_field in set(self._metric_csv_sync_writer_dict[metric_name].value_fields) for value_field in metric_info["values"].keys()):
                    self._metric_csv_sync_writer_dict[metric_name].value_fields += tuple(metric_info["values"].keys())
                self._metric_csv_sync_writer_dict[metric_name].add(
                    keys=metric_info["keys"],
                    values=metric_info["values"],
                )
            
    def _process_info_dict(self, env_info: dict, agent_info: dict):        
        if "metric" in env_info: 
            self._write_metric_dicts(env_info["metric"])
                
        if "metric" in agent_info:
            self._write_metric_dicts(agent_info["metric"])

        # since molecules are generated non-episodically,
        # needs to synchronize final molecule results of each episode
        # for env_id, final_molecule in enumerate(env_info["final_molecule"]):
        #     if final_molecule is None:
        #         continue
        #     episode = final_molecule["episode"]
        #     self._episode_molecule_sync_buffer_dict[episode][env_id] = final_molecule
        
    # def _record_molecule(self, molecule_iter):    
    #     molecules = tuple(molecule_iter)
    #     episode = molecules[0]["episode"]
    #     # remove the buffer
    #     self._episode_molecule_sync_buffer_dict.pop(episode)
        
    #     # final molecules of current episode
    #     for env_id, molecule in enumerate(molecules):
    #         self._molecule_queue.put((episode, env_id, molecule["score"], molecule["selfies"]))
    #         # intrinsic rewards of all time steps
    #         int_reward_dict = molecule["intrinsic_reward"]
    #         # e.g., int_reward_lists = (count_reward_list, memory_reward_list, ...)
    #         self._enabled_intrinsic_reward_types = tuple(int_reward_dict.keys())
    #         # e.g., int_rewards = (count_reward, memory_reward, ...)
    #         for time_step, int_rewards in enumerate(zip(*int_reward_dict.values())):
    #             self._intrinsic_reward_queue.put((episode, env_id, time_step) + int_rewards)
        
    #     valid_molecules = tuple(filter(lambda m: m["score"], molecules))
    #     if len(valid_molecules) == 0:
    #         return
        
    #     # best final molecule during training
    #     best_molecule = max(valid_molecules, key=lambda m: m["score"])
    #     if self._best_molecule is None or best_molecule["score"] > self._best_molecule["score"]:
    #         self._best_molecule = best_molecule
    #         self._best_molecule_queue.put(tuple(best_molecule.values()))
        
    # def _save_molecules(self, base_dir: str):
    #     # final molecule of each episode
    #     molecule_file_path = f"{base_dir}/molecules.csv"
    #     molecule_file_exists = file_exists(molecule_file_path)
    #     with open(molecule_file_path, "a", newline="") as f:
    #         wr = csv.writer(f)
    #         if not molecule_file_exists:
    #             wr.writerow(("Episode", "Env ID", "Average Score", "Selfies"))
    #         while not self._molecule_queue.empty():
    #             wr.writerow(self._molecule_queue.get())
        
    #     # best molecule during training
    #     best_molecule_file_path = f"{base_dir}/best_molecules.csv"
    #     best_molecule_file_exists = file_exists(best_molecule_file_path)
    #     with open(best_molecule_file_path, "a", newline="") as f:
    #         wr = csv.writer(f)
    #         if not best_molecule_file_exists:
    #             wr.writerow(("Episode", "Best Score") + tuple(ChemEnv.format_prop_name(x) for x in self._env._prop_keys) + ("Selfies",))
    #         while not self._best_molecule_queue.empty():
    #             wr.writerow(self._best_molecule_queue.get())
                
    #     intrinsic_reward_file_path = f"{base_dir}/intrinsic_rewards.csv"
    #     intrinsic_reward_file_exists = file_exists(intrinsic_reward_file_path)
    #     with open(intrinsic_reward_file_path, "a", newline="") as f:
    #         wr = csv.writer(f)
    #         if not intrinsic_reward_file_exists:
    #             wr.writerow(("episode", "env_id", "time_step") + self._enabled_intrinsic_reward_types)
    #         while not self._intrinsic_reward_queue.empty():
    #             wr.writerow(self._intrinsic_reward_queue.get())
        
    def _tick_time_steps(self):
        self._episode_len += 1
        self._time_steps += 1
        self._real_time = time.time() - self._real_start_time
    
    def _tick_episode(self):
        self._episode_len = 0
        self._episodes += 1
        
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
        
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
            .add_text(f"    total time steps: {self._config.time_steps}") \
            .add_text(f"    summary frequency: {self._config.summary_freq}") \
            .add_text(f"    agent save frequency: {self._config.agent_save_freq}")
        
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
        if self._cumulative_reward_mean.count == 0:
            reward_info = "episode has not terminated yet"
        else:
            reward_info = f"cumulative reward: {self._cumulative_reward_mean.mean:.2f}"
            logger.log_data("Environment/Cumulative Reward", self._cumulative_reward_mean.mean, self._time_steps)
            logger.log_data("Environment/Cumulative Reward per Episode", self._cumulative_reward_mean.mean, self._episodes)
            logger.log_data("Environment/Episode Length", self._episode_len_mean.mean, self._time_steps)
            self._cumulative_reward_mean.reset()
            self._episode_len_mean.reset()
        logger.print(f"training time: {self._real_time:.2f}, time steps: {self._time_steps}, {reward_info}")
        
        for key, (value, t) in self._agent.log_data.items():
            logger.log_data(key, value, t)
            
        for key, (value, t) in self._env.log_data.items():
            if t is None:
                logger.log_data(key, value, self._time_steps)
                logger.log_data(f"{key} per Episode", value, self._episodes)
            else:
                logger.log_data(key, value, t)
                    
    def _save_train(self):
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
        
        logger.print(f"Agent is successfully saved ({self._time_steps} steps): {agent_save_path}")
        
        # self._env.save_data(logger.dir())
        # self._save_molecules(logger.dir())
        
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
