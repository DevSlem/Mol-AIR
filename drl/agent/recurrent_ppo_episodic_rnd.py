from typing import Dict, Tuple, Iterable
from numbers import Number

import torch
import torch.nn.functional as F
import numpy as np

import drl.rl_loss as L
import drl.util.func as util_f
from drl.agent.agent import Agent, BehaviorType
from drl.agent.config import RecurrentPPOEpisodicRNDConfig
from drl.agent.net import RecurrentPPOEpisodicRNDNetwork
from drl.agent.trajectory import (RecurrentPPOEpisodicRNDExperience,
                                  RecurrentPPOEpisodicRNDTrajectory)
from drl.exp import Experience
from drl.net import Trainer
from drl.util import (IncrementalMean, IncrementalMeanVarianceFromBatch,
                      TruncatedSequenceGenerator)
from drl.util.scheduler import ConstantScheduler, LinearScheduler


class RecurrentPPOEpisodicRND(Agent):
    def __init__(
        self, 
        config: RecurrentPPOEpisodicRNDConfig,
        network: RecurrentPPOEpisodicRNDNetwork,
        trainer: Trainer,
        num_envs: int, 
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPOEpisodicRNDTrajectory(self._config.n_steps)
        
        # training data
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._state_value: torch.Tensor = None # type: ignore
        self._prev_discounted_int_return = 0.0
        self._current_init_norm_steps = 0
        # compute normalization parameters of intrinic reward of each env along time steps
        self._int_return_mean_var = IncrementalMeanVarianceFromBatch(dim=1)
        # compute normalization parameters of each feature of next observation along batches
        self._obs_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        # compute normalization parameters of each feature of next hidden state along batches
        self._hidden_state_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
                
        # inferenece data
        infer_hidden_state_shape = (network.hidden_state_shape()[0], 1, network.hidden_state_shape()[1])
        self._infer_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_next_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_prev_terminated = torch.zeros((1, 1), device=self.device)
                
        # log data
        self._actor_avg_loss = IncrementalMean()
        self._critic_avg_loss = IncrementalMean()
        self._rnd_avg_loss = IncrementalMean()
        self._int_avg_reward = IncrementalMean()
        self._int_adv_coef_avg = IncrementalMean()
        self._episodes = np.zeros((self.num_envs, 1), dtype=np.int32)
        self._rnd_int_rewards = []
        self._avg_rnd_int_rewards = [IncrementalMeanVarianceFromBatch() for _ in range(self.num_envs)]
        
        self._time_steps = 0
            
        
    @property
    def name(self) -> str:
        return "Recurrent PPO Episodic RND"
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
        
    @torch.no_grad()
    def _select_action_train(self, obs: torch.Tensor) -> torch.Tensor:
        # update hidden state H_t
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        
        # feed forward
        # when interacting with environment, sequence_length must be 1
        # that is (seq_batch_size, seq_len) = (num_envs, 1)
        policy_dist_seq, state_value_seq, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        
        # (num_envs, 1, *shape) -> (num_envs, *shape)
        action = action_seq.squeeze(dim=1)
        self._action_log_prob = policy_dist_seq.log_prob(action_seq).squeeze_(dim=1)
        self._state_value = state_value_seq.squeeze_(dim=1)
        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: torch.Tensor) -> torch.Tensor:
        self._infer_hidden_state = self._infer_next_hidden_state * (1.0 - self._infer_prev_terminated)
        policy_dist_seq, _, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._infer_hidden_state
        )
        action_seq = policy_dist_seq.sample()
        self._infer_next_hidden_state = next_hidden_state
        return action_seq.squeeze(dim=1)
    
    def _update_train(self, exp: Experience) -> dict:
        info_dict = dict()
        
        self._time_steps += 1
        self._prev_terminated = exp.terminated
        
        # (D x num_layers, num_envs, H) -> (num_envs, D x num_layers, H)
        next_hidden_state_along_envs = self._next_hidden_state.swapdims(0, 1)
        
        # initialize normalization parameters
        if (self._config.init_norm_steps is not None) and (self._current_init_norm_steps < self._config.init_norm_steps):
            self._current_init_norm_steps += 1
            self._obs_feature_mean_var.update(exp.next_obs)
            self._hidden_state_feature_mean_var.update(next_hidden_state_along_envs)
            return info_dict
        
        # compute intrinsic reward
        normalized_next_obs = self._normalize_obs(exp.next_obs.to(device=self.device))
        normalized_next_hidden_state = self._normalize_hidden_state(next_hidden_state_along_envs)
        int_reward = self._compute_intrinsic_reward(normalized_next_obs, normalized_next_hidden_state)
        self._rnd_int_rewards.append(int_reward.detach().cpu().numpy())
        
        # add an experience
        self._trajectory.add(RecurrentPPOEpisodicRNDExperience(
            obs=exp.obs,
            action=exp.action,
            next_obs=exp.next_obs,
            ext_reward=exp.reward,
            int_reward=int_reward,
            terminated=exp.terminated,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value,
            hidden_state=self._hidden_state,
            next_hidden_state=self._next_hidden_state
        ))
        
        if self._trajectory.reached_n_steps:
            metric_info_dicts = self._train()
            info_dict["metric"] = metric_info_dicts
        
        return info_dict
        
    def _update_inference(self, exp: Experience):
        self._infer_prev_terminated = exp.terminated
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        # compute advantage, extrinsic and intrinsic target state value
        advantage, target_state_value, metric_info_dicts = self._compute_adv_target(exp_batch)
        
        # batch (batch_size, *shape) to truncated sequence (seq_batch_size, seq_len, *shape)
        seq_generator = TruncatedSequenceGenerator(
            self._config.seq_len,
            self._num_envs,
            self._config.n_steps,
            self._config.padding_value
        )
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(util_f.batch_to_perenv(batch, self._num_envs), start_idx, seq_len)
    
        add_to_seq_gen(exp_batch.hidden_state.swapdims(0, 1), seq_len=1)
        add_to_seq_gen(exp_batch.next_hidden_state.swapdims(0, 1))
        add_to_seq_gen(exp_batch.obs)
        add_to_seq_gen(exp_batch.next_obs)
        add_to_seq_gen(exp_batch.action)
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(target_state_value)
        
        sequences = seq_generator.generate(util_f.batch_to_perenv(exp_batch.terminated, self._num_envs).squeeze_(dim=-1))
        (mask, seq_init_hidden_state, next_hidden_state_seq, obs_seq, next_obs_seq, action_seq, old_action_log_prob_seq, 
         advantage_seq, target_state_value_seq) = sequences

        entire_seq_batch_size = len(mask)
        # (entire_seq_batch_size, 1, D x num_layers, H) -> (D x num_layers, entire_seq_batch_size, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapdims_(0, 1)
        
        # update the normalization parameters of the observation and the hidden state
        # when masked by mask, (entire_seq_batch_size, seq_len,) -> (masked_batch_size,)
        masked_next_obs = next_obs_seq[mask]
        masked_next_hidden_state = next_hidden_state_seq[mask]
        self._obs_feature_mean_var.update(masked_next_obs)
        self._hidden_state_feature_mean_var.update(masked_next_hidden_state)
        
        # normalize the observation and the hidden state
        normalized_next_obs_seq = next_obs_seq
        normalized_hidden_state_seq = next_hidden_state_seq
        normalized_next_obs_seq[mask] = self._normalize_obs(masked_next_obs)
        normalized_hidden_state_seq[mask] = self._normalize_hidden_state(masked_next_hidden_state)
        
        for _ in range(self._config.epoch):
            # if seq_mini_batch_size is None, use the entire sequence batch to executes iteration only once
            # otherwise, use the randomly shuffled mini batch to executes iteration multiple times
            if self._config.seq_mini_batch_size is None:
                shuffled_seq = torch.arange(entire_seq_batch_size)
                seq_mini_batch_size = entire_seq_batch_size
            else:
                shuffled_seq = torch.randperm(entire_seq_batch_size)
                seq_mini_batch_size = self._config.seq_mini_batch_size
                
            for i in range(entire_seq_batch_size // seq_mini_batch_size):
                # when sliced by sample_seq, (entire_seq_batch_size,) -> (seq_mini_batch_size,)
                sample_seq = shuffled_seq[seq_mini_batch_size * i : seq_mini_batch_size * (i + 1)]
                # when masked by sample_mask, (seq_mini_batch_size, seq_len) -> (masked_batch_size,)
                sample_mask = mask[sample_seq]
                
                # feed forward
                sample_policy_dist_seq, sample_state_value_seq, _ = self._network.forward_actor_critic(
                    obs_seq[sample_seq],
                    seq_init_hidden_state[:, sample_seq]
                )
                predicted_feature, target_feature = self._network.forward_rnd(
                    normalized_next_obs_seq[sample_seq][sample_mask],
                    normalized_hidden_state_seq[sample_seq][sample_mask].flatten(1, 2)
                )
                
                # compute actor loss
                sample_new_action_log_prob_seq = sample_policy_dist_seq.log_prob(action_seq[sample_seq])
                actor_loss = L.ppo_clipped_loss(
                    advantage_seq[sample_seq][sample_mask],
                    old_action_log_prob_seq[sample_seq][sample_mask],
                    sample_new_action_log_prob_seq[sample_mask],
                    self._config.epsilon_clip
                )
                
                # compute critic loss
                critic_loss = L.bellman_value_loss(
                    sample_state_value_seq[sample_mask],
                    target_state_value_seq[sample_seq][sample_mask],
                )
                
                # compute entropy
                entropy = sample_policy_dist_seq.entropy()[sample_mask].mean()
                
                # compute RND loss
                rnd_loss = F.mse_loss(
                    predicted_feature,
                    target_feature.detach(),
                    reduction="none"
                ).mean(dim=-1)
                # the proportion of experiences to keep the effective batch size
                rnd_loss_mask = torch.rand(len(rnd_loss), device=rnd_loss.device)
                rnd_loss_mask = (rnd_loss_mask < self._config.rnd_pred_exp_proportion).to(dtype=rnd_loss.dtype)
                rnd_loss = (rnd_loss * rnd_loss_mask).sum() / torch.max(rnd_loss_mask.sum(), torch.tensor(1.0, device=rnd_loss.device))
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy + rnd_loss
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_avg_loss.update(actor_loss.item())
                self._critic_avg_loss.update(critic_loss.item())
                self._rnd_avg_loss.update(rnd_loss.item())
                
        return metric_info_dicts
    
    def _compute_adv_target(self, exp_batch: RecurrentPPOEpisodicRNDExperience):
        """
        Compute advantage `(batch_size, 1)`, extrinsic and intrinsic target state value `(batch_size, 1)`.
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        with torch.no_grad():
            # compute final next state value
            _, final_next_state_value_seq, _ = self._network.forward_actor_critic(
                final_next_obs.unsqueeze(dim=1), # (num_envs, 1, *obs_shape) because sequence length is 1
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_next_state_value = final_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_state_value = torch.cat((exp_batch.state_value, final_next_state_value), dim=0)
        
        # (num_envs x T, 1) -> (num_envs, T)
        b2e = lambda x: util_f.batch_to_perenv(x, self._num_envs)
        entire_state_value = b2e(entire_state_value).squeeze_(dim=-1)
        reward = b2e(exp_batch.ext_reward).squeeze_(dim=-1)
        int_reward = b2e(exp_batch.int_reward).squeeze_(dim=-1)
        terminated = b2e(exp_batch.terminated).squeeze_(dim=-1)
        
        # compute discounted intrinsic reward
        discounted_int_return = torch.empty_like(int_reward)
        for t in range(self._config.n_steps):
            self._prev_discounted_int_return = int_reward[:, t] + self._config.gamma * self._prev_discounted_int_return
            discounted_int_return[:, t] = self._prev_discounted_int_return
        
        # update intrinsic reward normalization parameters
        self._int_return_mean_var.update(discounted_int_return)
        
        # normalize intrinsic reward
        int_reward /= torch.sqrt(self._int_return_mean_var.variance).unsqueeze(dim=-1) + 1e-8
        self._int_avg_reward.update(int_reward.mean().item())
        
        reward += self._config.int_reward_coef * int_reward
        
        metric_info_dicts = []
        for env_id in range(self.num_envs):
            end_idxes = torch.where(terminated[env_id])[0].cpu() + 1
            start = 0
            for end in end_idxes:
                mean, _ = self._avg_rnd_int_rewards[env_id].update(int_reward[env_id, start:end].cpu())
                metric_info_dicts.append({
                    "episode_metric": {
                        "keys": {
                            "episode": self._episodes[env_id].item(),
                            "env_id": env_id
                        },
                        "values": {
                            "avg_rnd_int_reward": mean.item()
                        }
                    }
                })
                self._episodes[env_id] += 1
                self._avg_rnd_int_rewards[env_id].reset()
                start = end
            if start < len(int_reward[env_id]):
                self._avg_rnd_int_rewards[env_id].update(int_reward[env_id, start:].cpu())
        
        # compute advantage (num_envs, n_steps) using GAE
        advantage = L.gae(
            entire_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        
        # compute target state value (num_envs, n_steps)
        target_state_value = advantage + entire_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: util_f.perenv_to_batch(x).unsqueeze_(dim=-1)
        advantage = e2b(advantage)
        target_state_value = e2b(target_state_value)
        
        return advantage, target_state_value, metric_info_dicts
    
    def _compute_intrinsic_reward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward.

        Args:
            obs (Tensor): `(batch_size, *obs_shape)`
            hidden_state (Tensor): `(batch_size, D x num_layers, H)`

        Returns:
            int_reward (Tensor): intrinsic reward `(batch_size, 1)`
        """
        with torch.no_grad():
            predicted_feature, target_feature = self._network.forward_rnd(obs, hidden_state.flatten(1, 2))
            int_reward = 0.5 * ((target_feature - predicted_feature)**2).sum(dim=1, keepdim=True)
            return int_reward
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._config.init_norm_steps is None:
            return obs
        obs_feature_mean = self._obs_feature_mean_var.mean
        obs_feature_std = torch.sqrt(self._obs_feature_mean_var.variance) + 1e-8
        normalized_obs = (obs - obs_feature_mean) / obs_feature_std
        return normalized_obs.clamp(self._config.obs_norm_clip_range[0], self._config.obs_norm_clip_range[1])

    def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._config.init_norm_steps is None:
            return hidden_state
        hidden_state_feature_mean = self._hidden_state_feature_mean_var.mean
        hidden_state_feature_std = torch.sqrt(self._hidden_state_feature_mean_var.variance) + 1e-8
        normalized_hidden_state = (hidden_state - hidden_state_feature_mean) / hidden_state_feature_std
        return normalized_hidden_state.clamp(self._config.hidden_state_norm_clip_range[0], self._config.hidden_state_norm_clip_range[1])
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + (
            "Training/Actor Loss", 
            "Training/Critic Loss", 
            "Training/RND Loss", 
            "Environment/Intrinsic Average Reward"
        )
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self._actor_avg_loss.count > 0:
            ld["Training/Actor Loss"] = (self._actor_avg_loss.mean, self.training_steps)
            ld["Training/Critic Loss"] = (self._critic_avg_loss.mean, self.training_steps)
            ld["Training/RND Loss"] = (self._rnd_avg_loss.mean, self.training_steps)
            self._actor_avg_loss.reset()
            self._critic_avg_loss.reset()
            self._rnd_avg_loss.reset()
        if self._int_avg_reward.count > 0:
            ld["Environment/Intrinsic Average Reward"] = (self._int_avg_reward.mean, self.training_steps)
            self._int_avg_reward.reset()
        if self._int_adv_coef_avg.count > 0:
            ld["Training/Intrinsic Advantage Coefficient"] = (self._int_adv_coef_avg.mean, self._time_steps)
            self._int_adv_coef_avg.reset()
        return ld
