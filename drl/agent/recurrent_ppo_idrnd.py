from typing import Dict, Tuple, Iterable
from numbers import Number

import torch
import torch.nn.functional as F

import drl.rl_loss as L
import drl.util.func as util_f
from drl.agent.agent import Agent, BehaviorType
from drl.agent.config import RecurrentPPORNDConfig
from drl.agent.net import RecurrentPPORNDNetwork
from drl.agent.trajectory import (RecurrentPPORNDExperience,
                                  RecurrentPPORNDTrajectory)
from drl.exp import Experience
from drl.net import Trainer
from drl.util import (IncrementalMean, IncrementalMeanVarianceFromBatch,
                      TruncatedSequenceGenerator)
from drl.util.scheduler import ConstantScheduler, LinearScheduler


class RecurrentPPOIDRND(Agent):
    def __init__(
        self, 
        config: RecurrentPPORNDConfig,
        network: RecurrentPPORNDNetwork,
        trainer: Trainer,
        num_envs: int, 
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPORNDTrajectory(self._config.n_steps)
        
        # training data
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._ext_state_value: torch.Tensor = None # type: ignore
        self._int_state_value: torch.Tensor = None # type: ignore
        self._prev_discounted_int_return = 0.0
        self._current_init_norm_steps = 0
        # compute normalization parameters of intrinic reward of each env along time steps
        self._int_return_mean_var = IncrementalMeanVarianceFromBatch(dim=1)
        # compute normalization parameters of each feature of next observation along batches
        self._obs_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        self._hidden_state_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        self._next_obs_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        self._next_hidden_state_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
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
        self._ext_critic_avg_loss = IncrementalMean()
        self._int_critic_avg_loss = IncrementalMean()
        self._rnd_avg_loss = IncrementalMean()
        self._int_avg_reward = IncrementalMean()
        self._int_adv_coef_avg = IncrementalMean()
        
        self._time_steps = 0
        if isinstance(self._config.int_adv_coef, Iterable):
            self._int_adv_coef_scheduler = LinearScheduler(
                t0=self._config.int_adv_coef[0],
                t1=self._config.int_adv_coef[1],
                y0=self._config.int_adv_coef[2],
                y1=self._config.int_adv_coef[3]
            )
        else:
            self._int_adv_coef_scheduler = ConstantScheduler(self._config.int_adv_coef)
            
        
    @property
    def name(self) -> str:
        return "Recurrent PPO IDRND"
    
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
        policy_dist_seq, ext_state_value_seq, int_state_value_seq, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        
        # (num_envs, 1, *shape) -> (num_envs, *shape)
        action = action_seq.squeeze(dim=1)
        self._action_log_prob = policy_dist_seq.log_prob(action_seq).squeeze_(dim=1)
        self._ext_state_value = ext_state_value_seq.squeeze_(dim=1)
        self._int_state_value = int_state_value_seq.squeeze_(dim=1)
        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    @torch.no_grad()
    def _select_action_inference(self, obs: torch.Tensor) -> torch.Tensor:
        self._infer_hidden_state = self._infer_next_hidden_state * (1.0 - self._infer_prev_terminated)
        policy_dist_seq, _, _, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._infer_hidden_state
        )
        action_seq = policy_dist_seq.sample()
        self._infer_next_hidden_state = next_hidden_state
        return action_seq.squeeze(dim=1)
    
    def _update_train(self, exp: Experience):
        self._time_steps += 1
        self._prev_terminated = exp.terminated
        
        # (D x num_layers, num_envs, H) -> (num_envs, D x num_layers, H)
        hidden_state_along_envs = self._hidden_state.swapdims(0, 1)
        next_hidden_state_along_envs = self._next_hidden_state.swapdims(0, 1)
        
        # initialize normalization parameters
        if (self._config.init_norm_steps is not None) and (self._current_init_norm_steps < self._config.init_norm_steps):
            self._current_init_norm_steps += 1
            self._obs_feature_mean_var.update(exp.obs)
            self._hidden_state_feature_mean_var.update(hidden_state_along_envs)
            self._next_obs_feature_mean_var.update(exp.next_obs)
            self._next_hidden_state_feature_mean_var.update(next_hidden_state_along_envs)
            return
        
        # compute intrinsic reward
        normalized_obs = self._normalize(
            exp.obs.to(device=self.device),
            self._obs_feature_mean_var.mean,
            self._obs_feature_mean_var.variance,
            self._config.obs_norm_clip_range[0],
            self._config.obs_norm_clip_range[1]
        )
        normalized_hidden_state = self._normalize(
            hidden_state_along_envs,
            self._hidden_state_feature_mean_var.mean,
            self._hidden_state_feature_mean_var.variance,
            self._config.hidden_state_norm_clip_range[0],
            self._config.hidden_state_norm_clip_range[1]
        )
        normalized_next_obs = self._normalize(
            exp.next_obs.to(device=self.device),
            self._next_obs_feature_mean_var.mean,
            self._next_obs_feature_mean_var.variance,
            self._config.obs_norm_clip_range[0],
            self._config.obs_norm_clip_range[1]
        )
        normalized_next_hidden_state = self._normalize(
            next_hidden_state_along_envs,
            self._next_hidden_state_feature_mean_var.mean,
            self._next_hidden_state_feature_mean_var.variance,
            self._config.hidden_state_norm_clip_range[0],
            self._config.hidden_state_norm_clip_range[1]
        )
        # normalized_next_obs = self._normalize_obs(exp.next_obs.to(device=self.device))
        # normalized_next_hidden_state = self._normalize_hidden_state(next_hidden_state_along_envs)
        int_reward = self._compute_intrinsic_reward(
            normalized_obs,
            normalized_hidden_state,
            normalized_next_obs, 
            normalized_next_hidden_state
        )
        
        # add an experience
        self._trajectory.add(RecurrentPPORNDExperience(
            obs=exp.obs,
            action=exp.action,
            next_obs=exp.next_obs,
            ext_reward=exp.reward,
            int_reward=int_reward,
            terminated=exp.terminated,
            action_log_prob=self._action_log_prob,
            ext_state_value=self._ext_state_value,
            int_state_value=self._int_state_value,
            hidden_state=self._hidden_state,
            next_hidden_state=self._next_hidden_state
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
        
    def _update_inference(self, exp: Experience):
        self._infer_prev_terminated = exp.terminated
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        # compute advantage, extrinsic and intrinsic target state value
        advantage, ext_target_state_value, int_target_state_value = self._compute_adv_target(exp_batch)
        
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
        add_to_seq_gen(exp_batch.hidden_state.swapdims(0, 1))
        add_to_seq_gen(exp_batch.next_hidden_state.swapdims(0, 1))
        add_to_seq_gen(exp_batch.obs)
        add_to_seq_gen(exp_batch.next_obs)
        add_to_seq_gen(exp_batch.action)
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(ext_target_state_value)
        add_to_seq_gen(int_target_state_value)
        
        sequences = seq_generator.generate(util_f.batch_to_perenv(exp_batch.terminated, self._num_envs).squeeze_(dim=-1))
        (mask, seq_init_hidden_state, hidden_state_seq, next_hidden_state_seq, obs_seq, next_obs_seq, action_seq, old_action_log_prob_seq, 
         advantage_seq, ext_target_state_value_seq, int_target_state_value_seq) = sequences

        entire_seq_batch_size = len(mask)
        # (entire_seq_batch_size, 1, D x num_layers, H) -> (D x num_layers, entire_seq_batch_size, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapdims_(0, 1)
        
        # update the normalization parameters of the observation and the hidden state
        # when masked by mask, (entire_seq_batch_size, seq_len,) -> (masked_batch_size,)
        masked_next_obs = next_obs_seq[mask]
        masked_next_hidden_state = next_hidden_state_seq[mask]
        self._obs_feature_mean_var.update(obs_seq[mask])
        self._hidden_state_feature_mean_var.update(hidden_state_seq[mask])
        self._next_obs_feature_mean_var.update(masked_next_obs)
        self._next_hidden_state_feature_mean_var.update(masked_next_hidden_state)
        
        # normalize the observation and the hidden state
        normalized_next_obs_seq = next_obs_seq
        normalized_hidden_state_seq = next_hidden_state_seq
        # normalized_next_obs_seq[mask] = self._normalize_obs(masked_next_obs)
        # normalized_hidden_state_seq[mask] = self._normalize_hidden_state(masked_next_hidden_state)
        normalized_next_obs_seq[mask] = self._normalize(
            masked_next_obs,
            self._next_obs_feature_mean_var.mean,
            self._next_obs_feature_mean_var.variance,
            self._config.obs_norm_clip_range[0],
            self._config.obs_norm_clip_range[1]
        )
        normalized_hidden_state_seq[mask] = self._normalize(
            masked_next_hidden_state,
            self._next_hidden_state_feature_mean_var.mean,
            self._next_hidden_state_feature_mean_var.variance,
            self._config.hidden_state_norm_clip_range[0],
            self._config.hidden_state_norm_clip_range[1]
        )
        
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
                sample_policy_dist_seq, sample_ext_state_value_seq, sample_int_state_value_seq, _ = self._network.forward_actor_critic(
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
                ext_critic_loss = L.bellman_value_loss(
                    sample_ext_state_value_seq[sample_mask],
                    ext_target_state_value_seq[sample_seq][sample_mask],
                )
                int_critic_loss = L.bellman_value_loss(
                    sample_int_state_value_seq[sample_mask],
                    int_target_state_value_seq[sample_seq][sample_mask],
                )
                critic_loss = ext_critic_loss + int_critic_loss
                
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
                self._ext_critic_avg_loss.update(ext_critic_loss.item())
                self._int_critic_avg_loss.update(int_critic_loss.item())
                self._rnd_avg_loss.update(rnd_loss.item())
    
    def _compute_adv_target(self, exp_batch: RecurrentPPORNDExperience):
        """
        Compute advantage `(batch_size, 1)`, extrinsic and intrinsic target state value `(batch_size, 1)`.
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        with torch.no_grad():
            # compute final next state value
            _, final_ext_next_state_value_seq, final_int_next_state_value_seq, _ = self._network.forward_actor_critic(
                final_next_obs.unsqueeze(dim=1), # (num_envs, 1, *obs_shape) because sequence length is 1
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_ext_next_state_value = final_ext_next_state_value_seq.squeeze_(dim=1)
        final_int_next_state_value = final_int_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_ext_state_value = torch.cat((exp_batch.ext_state_value, final_ext_next_state_value), dim=0)
        entire_int_state_value = torch.cat((exp_batch.int_state_value, final_int_next_state_value), dim=0)
        
        # (num_envs x T, 1) -> (num_envs, T)
        b2e = lambda x: util_f.batch_to_perenv(x, self._num_envs)
        entire_ext_state_value = b2e(entire_ext_state_value).squeeze_(dim=-1)
        entire_int_state_value = b2e(entire_int_state_value).squeeze_(dim=-1)
        reward = b2e(exp_batch.ext_reward).squeeze_(dim=-1)
        int_reward = b2e(exp_batch.int_reward).squeeze_(dim=-1)
        terminated = b2e(exp_batch.terminated).squeeze_(dim=-1)
        
        # compute discounted intrinsic reward
        discounted_int_return = torch.empty_like(int_reward)
        for t in range(self._config.n_steps):
            self._prev_discounted_int_return = int_reward[:, t] + self._config.int_gamma * self._prev_discounted_int_return
            discounted_int_return[:, t] = self._prev_discounted_int_return
        
        # update intrinsic reward normalization parameters
        self._int_return_mean_var.update(discounted_int_return)
        
        # normalize intrinsic reward
        int_reward /= torch.sqrt(self._int_return_mean_var.variance).unsqueeze(dim=-1) + 1e-8
        self._int_avg_reward.update(int_reward.mean().item())
        
        # compute advantage (num_envs, n_steps) using GAE
        ext_advantage = L.gae(
            entire_ext_state_value,
            reward,
            terminated,
            self._config.ext_gamma,
            self._config.lam
        )
        int_advantage = L.gae(
            entire_int_state_value,
            int_reward,
            torch.zeros_like(terminated), # non-episodic
            self._config.int_gamma,
            self._config.lam
        )
        int_adv_coef = self._int_adv_coef_scheduler(self._time_steps)
        advantage = self._config.ext_adv_coef * ext_advantage + int_adv_coef * int_advantage
        self._int_adv_coef_avg.update(int_adv_coef)
        
        # compute target state value (num_envs, n_steps)
        ext_target_state_value = ext_advantage + entire_ext_state_value[:, :-1]
        int_target_state_value = int_advantage + entire_int_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: util_f.perenv_to_batch(x).unsqueeze_(dim=-1)
        advantage = e2b(advantage)
        ext_target_state_value = e2b(ext_target_state_value)
        int_target_state_value = e2b(int_target_state_value)
        
        return advantage, ext_target_state_value, int_target_state_value
    
    @torch.no_grad()
    def _compute_intrinsic_reward(
        self, 
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        next_obs: torch.Tensor, 
        next_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute intrinsic reward.

        Args:
            obs (Tensor): `(batch_size, *obs_shape)`
            hidden_state (Tensor): `(batch_size, D x num_layers, H)`

        Returns:
            int_reward (Tensor): intrinsic reward `(batch_size, 1)`
        """
        predicted_feature, target_feature = self._network.forward_rnd(obs, hidden_state.flatten(1, 2))
        next_predicted_feature, next_target_feature = self._network.forward_rnd(next_obs, next_hidden_state.flatten(1, 2))
        # # case 1)
        # prediction_error = 0.5 * ((predicted_feature - target_feature)**2).sum(dim=1, keepdim=True)
        # next_prediction_error = 0.5 * ((next_predicted_feature - next_target_feature)**2).sum(dim=1, keepdim=True)
        # int_reward = torch.max(next_prediction_error - prediction_error, torch.tensor(0.0))
        # case 2)
        next_prediction_error = 0.5 * ((next_predicted_feature - next_target_feature)**2).sum(dim=1, keepdim=True)
        impact = torch.norm(next_predicted_feature - predicted_feature, p=2, dim=1, keepdim=True) # type: ignore
        int_reward = next_prediction_error + impact
        return int_reward
    
    # def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
    #     if self._config.init_norm_steps is None:
    #         return obs
    #     obs_feature_mean = self._next_obs_feature_mean_var.mean
    #     obs_feature_std = torch.sqrt(self._next_obs_feature_mean_var.variance) + 1e-8
    #     normalized_obs = (obs - obs_feature_mean) / obs_feature_std
    #     return normalized_obs.clamp(self._config.obs_norm_clip_range[0], self._config.obs_norm_clip_range[1])

    # def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
    #     if self._config.init_norm_steps is None:
    #         return hidden_state
    #     hidden_state_feature_mean = self._next_hidden_state_feature_mean_var.mean
    #     hidden_state_feature_std = torch.sqrt(self._next_hidden_state_feature_mean_var.variance) + 1e-8
    #     normalized_hidden_state = (hidden_state - hidden_state_feature_mean) / hidden_state_feature_std
    #     return normalized_hidden_state.clamp(self._config.hidden_state_norm_clip_range[0], self._config.hidden_state_norm_clip_range[1])
    
    def _normalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        min: float,
        max: float
    ) -> torch.Tensor:
        if self._config.init_norm_steps is None:
            return x
        normalized_x = (x - mean) / torch.sqrt(var + 1e-8)
        return normalized_x.clamp(min, max)
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + (
            "Training/Actor Loss", 
            "Training/Extrinsic Critic Loss", 
            "Training/Intrinsic Critic Loss", 
            "Training/RND Loss", 
            "Environment/Intrinsic Average Reward"
        )
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self._actor_avg_loss.count > 0:
            ld["Training/Actor Loss"] = (self._actor_avg_loss.mean, self.training_steps)
            ld["Training/Extrinsic Critic Loss"] = (self._ext_critic_avg_loss.mean, self.training_steps)
            ld["Training/Intrinsic Critic Loss"] = (self._int_critic_avg_loss.mean, self.training_steps)
            ld["Training/RND Loss"] = (self._rnd_avg_loss.mean, self.training_steps)
            self._actor_avg_loss.reset()
            self._ext_critic_avg_loss.reset()
            self._int_critic_avg_loss.reset()
            self._rnd_avg_loss.reset()
        if self._int_avg_reward.count > 0:
            ld["Environment/Intrinsic Average Reward"] = (self._int_avg_reward.mean, self.training_steps)
            self._int_avg_reward.reset()
        if self._int_adv_coef_avg.count > 0:
            ld["Training/Intrinsic Advantage Coefficient"] = (self._int_adv_coef_avg.mean, self._time_steps)
            self._int_adv_coef_avg.reset()
        return ld
