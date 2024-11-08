from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import drl
import metric
from drl.agent import Agent
from envs import Env
from util import logger, try_create_dir


class Inference:
    def __init__(
        self,
        id: str,
        env: Env,
        agent: Agent,
        n_episodes: int = 1,
    ):
        self._id = id
        self._env = env
        self._agent = agent
        self._n_episodes = n_episodes
        self._device = agent.device
        
        self._dtype = torch.float32
        
        self._enabled = True
    
    def inference(self) -> "Inference":
        if not self._enabled:
            raise RuntimeError("Inference is already closed.")        

        logger.enable(self._id, enable_log_file=False)
        logger.print(f"Inference started (ID: {self._id}).")
        
        episodes = np.zeros((self._env.num_envs,), dtype=int)
        metric_list_dict = defaultdict(list)
        
        self._agent.model.eval()
        
        obs = self._env.reset()
        
        with tqdm(total=self._n_episodes, desc="Episode") as pbar:
            while np.sum(episodes) < self._n_episodes:
                obs = self._numpy_to_tensor(obs)
                with torch.no_grad():
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
                with torch.no_grad():
                    _ = self._agent.update(exp)
                                
                for key, value in self._inference_metric(env_info).items():
                    metric_list_dict[key].extend(value)
                                
                obs = next_obs
                episodes += terminated.astype(int)
                pbar.update(np.sum(terminated))
            
        metric_df = pd.DataFrame(metric_list_dict)
        metric_df = metric_df[:self._n_episodes]
        n_total = len(metric_df)
        metric_df.dropna(inplace=True)
        n_valid = len(metric_df)
        if n_valid == 0:
            logger.print("No valid molecule is generated.")
            return self
        
        avg_scores = metric_df.mean(numeric_only=True)
            
        smiles_list = metric_df["smiles"].tolist()
        avg_scores["diversity"] = metric.calc_diversity(smiles_list)
        avg_scores["uniqueness"] = metric.calc_uniqueness(smiles_list)
        
        try_create_dir(f"{logger.dir()}/inference")
        metric_df.to_csv(f"{logger.dir()}/inference/molecules.csv", index=False)
        
        avg_scores = pd.concat([pd.Series([n_total, n_valid], index=["n_total", "n_valid"]), avg_scores])
        avg_scores.index.name = "Metric"
        avg_scores.name = "Score"
        avg_scores.to_csv(f"{logger.dir()}/inference/metrics.csv", header=True)
        
        logger.print("===== Inference Result =====")
        logger.print(avg_scores.to_frame().T.to_string(index=False), prefix="")
        
        best_i = metric_df["score"].argmax()
        best_row = metric_df.iloc[best_i:best_i+1]
        logger.print("===== Best Molecule =====")
        best_scores = best_row.mean(numeric_only=True)
        logger.print(best_scores.to_frame().T.to_string(index=False), prefix="")
        
        best_row = best_row.iloc[0]
        best_row.index.name = "Metric"
        best_row.name = "Score"
        best_row.to_csv(f"{logger.dir()}/inference/best_molecule.csv", header=True)
        
        return self
    
    def close(self):
        self._enabled = False
        self._env.close()
        
        if logger.enabled():
            logger.disable()
    
    def _inference_metric(self, env_info: dict):
        metric_list_dict = defaultdict(list)
        
        if "metric" not in env_info:
            return metric_list_dict
        
        metric_dicts = env_info["metric"]
        
        for metric_dict in metric_dicts:
            if metric_dict is None:
                continue
        
            if "episode_metric" not in metric_dict:
                continue
            
            for key, value in metric_dict["episode_metric"]["values"].items():
                metric_list_dict[key].append(value)
            
        return metric_list_dict
    
    def _agent_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self._device, dtype=self._dtype)
    
    def _numpy_to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(device=self._device, dtype=self._dtype)