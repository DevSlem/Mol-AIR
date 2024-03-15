import unittest

import numpy as np

from envs.chem_env import ChemEnv, make_async_chem_env
from train import MolRLTrainFactory
import shutil

class Test(unittest.TestCase):
    def test_chem_env(self):
        print("=== test env ===")
        
        env = ChemEnv(
            plogp_coef=1.0,
            max_str_len=10
        )
        
        obs = env.reset()
        # check all elements of obs are -1
        self.assertTrue(np.all(obs == -1), f"obs: {obs}")
        terminated = False
        time_step = 0
        while not terminated:
            action = np.random.randint(env.num_actions, size=(1, 1))
            next_obs, reward, terminated, real_final_next_obs, _ = env.step(action)
            self.assertTrue(reward.shape == (1,), f"reward: {reward}")
            self.assertTrue(terminated.shape == (1,), f"terminated: {terminated}")
            if not terminated:
                self.assertTrue(next_obs[0, time_step] == action[0], f"time_step: {time_step}, next_obs: {next_obs}, action: {action}")
            else:
                self.assertTrue(np.all(next_obs == -1), f"next_obs: {next_obs}")
                self.assertTrue(real_final_next_obs[0, time_step] == action[0], f"time_step: {time_step}, real_final_next_obs: {real_final_next_obs}, action: {action}")
            time_step += 1
            
        env.close()
        
        print("=== test env completed ===")
        
    def test_async_chem_env(self):
        print("=== test make_async_chem_env ===")
        env = make_async_chem_env(
            num_envs=3,
            count_int_reward_coef=1.0,
            plogp_coef=1.0,
            max_str_len=5,
        )
        
        self.assertTrue(env.num_envs == 3, f"num_envs: {env.num_envs}")
        self.assertTrue(env.obs_shape == (31,), f"obs_shape: {env.obs_shape}")
        self.assertTrue(env.num_actions == 31, f"num_actions: {env.num_actions}")
        
        obs = env.reset()
        self.assertTrue(obs.shape == (3, 31), f"obs shape: {obs.shape}")
        for _ in range(10):
            next_obs, reward, terminated, real_final_next_obs, info = env.step(
                np.random.randint(env.num_actions, size=(env.num_envs, 1))
            )
            self.assertTrue(next_obs.shape == (3, 31), f"next_obs shape: {next_obs.shape}")
            self.assertTrue(reward.shape == (3,), f"reward shape: {reward.shape}")
            self.assertTrue(terminated.shape == (3,), f"terminated shape: {terminated.shape}")
            self.assertTrue(real_final_next_obs.shape == (terminated.sum(), 31), f"real_final_next_obs shape: {real_final_next_obs.shape}")
            
            if "valid_termination" in info:
                print(f"valid_termination: {info['valid_termination']}")
                
            if "metric" in info:
                for metric_info in info["metric"]:
                    if metric_info is not None and "avg_count_int_reward" in metric_info["episode_metric"]["values"]:
                        print(f"avg_count_int_reward: {metric_info['episode_metric']['values']['avg_count_int_reward']}")
                
        env.close()
        print("=== test make_async_chem_env completed ===")
        
    def test_train_ppo(self):
        print("=== test train PPO ===")
        
        id = "Test_PPO"
        shutil.rmtree(f"results/{id}", ignore_errors=True)
        config = {
            "Agent": {
                "type": "PPO",
                "n_steps": 64,
                "seq_len": 35,
                "seq_mini_batch_size": 16,
                "epoch": 3,
            },
            "Env": {
                "plogp_coef": 1.0,
                "max_str_len": 10
            },
            "Train": {
                "num_envs": 3,
                "seed": 0,
                "total_time_steps": 1000,
                "summary_freq": 200,
            },
        }
        MolRLTrainFactory(id, config) \
            .create_train() \
            .train() \
            .close()
            
        print("=== test train PPO completed ===")
        
    def test_train_mol_air(self):
        print("=== test train MolAIR ===")
        
        id = "Test_MolAIR"
        shutil.rmtree(f"results/{id}", ignore_errors=True)
        config = {
            "Agent": {
                "type": "RND",
                "n_steps": 64,
                "seq_len": 35,
                "seq_mini_batch_size": 16,
                "epoch": 3,
            },
            "Env": {
                "plogp_coef": 1.0,
                "max_str_len": 10
            },
            "Train": {
                "num_envs": 3,
                "seed": 0,
                "total_time_steps": 1000,
                "summary_freq": 200,
            },
            "CountIntReward": {
                "count_int_reward_coef": 1.0
            }
        }
        MolRLTrainFactory(id, config) \
            .create_train() \
            .train() \
            .close()
            
        print("=== test train MolAIR completed ===")
    
if __name__ == '__main__':
    unittest.main()