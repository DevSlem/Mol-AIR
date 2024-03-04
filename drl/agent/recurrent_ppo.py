from typing import Dict, Tuple

import torch

import drl.rl_loss as L
import drl.util.func as util_f
from drl.agent.agent import Agent, BehaviorType
from drl.agent.config import RecurrentPPOConfig
from drl.agent.net import RecurrentPPONetwork
from drl.agent.trajectory import RecurrentPPOExperience, RecurrentPPOTrajectory
from drl.exp import Experience
from drl.net import Trainer
from drl.util import IncrementalMean, TruncatedSequenceGenerator


class RecurrentPPO(Agent):
    def __init__(
        self, 
        config: RecurrentPPOConfig,
        network: RecurrentPPONetwork,
        trainer: Trainer,
        num_envs: int, 
        behavior_type: BehaviorType = BehaviorType.TRAIN
    ) -> None:
        super().__init__(num_envs, network, config.device, behavior_type)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPOTrajectory(self._config.n_steps)
        
        # training data
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._state_value: torch.Tensor = None # type: ignore    
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
        # inference data
        infer_hidden_state_shape = (network.hidden_state_shape()[0], 1, network.hidden_state_shape()[1])
        self._infer_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_next_hidden_state = torch.zeros(infer_hidden_state_shape, device=self.device)
        self._infer_prev_terminated = torch.zeros((1, 1), device=self.device)
                
        # log data
        self._actor_average_loss = IncrementalMean()
        self._critic_average_loss = IncrementalMean()
        
    @property
    def name(self) -> str:
        return "Recurrent PPO"
    
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
        policy_dist_seq, state_value_seq, next_hidden_state = self._network.forward(
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
        policy_dist_seq, _, next_hidden_state = self._network.forward(
            obs.unsqueeze(dim=1),
            self._infer_hidden_state
        )
        action_seq = policy_dist_seq.sample()
        self._infer_next_hidden_state = next_hidden_state
        return action_seq.squeeze(dim=1)
    
    def _update_train(self, exp: Experience) -> dict:
        self._prev_terminated = exp.terminated
        
        self._trajectory.add(RecurrentPPOExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value,
            hidden_state=self._hidden_state,
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
            
        return dict()
        
    def _update_inference(self, exp: Experience):
        self._infer_prev_terminated = exp.terminated
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        # compute advantage and target state value
        advantage, target_state_value = self._compute_adv_target(exp_batch)
        
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
        add_to_seq_gen(exp_batch.obs)
        add_to_seq_gen(exp_batch.action)
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(target_state_value)
        
        sequences = seq_generator.generate(util_f.batch_to_perenv(exp_batch.terminated, self._num_envs).squeeze_(dim=-1))
        mask, seq_init_hidden_state, obs_seq, action_seq, old_action_log_prob_seq, advantage_seq, target_state_value_seq = sequences

        entire_seq_batch_size = len(mask)
        # (entire_seq_batch_size, 1, D x num_layers, H) -> (D x num_layers, entire_seq_batch_size, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapdims_(0, 1)
        
        for _ in range(self._config.epoch):
            shuffled_seq = torch.randperm(entire_seq_batch_size)
            for i in range(entire_seq_batch_size // self._config.seq_mini_batch_size):
                # when sliced by sample_seq, (entire_seq_batch_size,) -> (mini_seq_batch_size,)
                sample_seq = shuffled_seq[self._config.seq_mini_batch_size * i : self._config.seq_mini_batch_size * (i + 1)]
                # when masked by sample_mask, (mini_seq_batch_size, seq_len) -> (masked_batch_size,)
                sample_mask = mask[sample_seq]
                
                # feed forward
                sample_policy_dist_seq, sample_state_value_seq, _ = self._network.forward(
                    obs_seq[sample_seq],
                    seq_init_hidden_state[:, sample_seq]
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
                
                # train step
                loss = actor_loss + self._config.value_loss_coef * critic_loss - self._config.entropy_coef * entropy
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_average_loss.update(actor_loss.item())
                self._critic_average_loss.update(critic_loss.item())
    
    def _compute_adv_target(self, exp_batch: RecurrentPPOExperience):
        """
        Compute advantage `(batch_size, 1)` and target state value `(batch_size, 1)`.
        """
        
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        with torch.no_grad():
            # compute final next state value
            _, final_next_state_value_seq, _ = self._network.forward(
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
        reward = b2e(exp_batch.reward).squeeze_(dim=-1)
        terminated = b2e(exp_batch.terminated).squeeze_(dim=-1)
        
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
        
        return advantage, target_state_value
    
    @property
    def log_keys(self) -> Tuple[str, ...]:
        return super().log_keys + ("Training/Actor Loss", "Training/Critic Loss")
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self._actor_average_loss.count > 0:
            ld["Training/Actor Loss"] = (self._actor_average_loss.mean, self.training_steps)
            ld["Training/Critic Loss"] = (self._critic_average_loss.mean, self.training_steps)
            self._actor_average_loss.reset()
            self._critic_average_loss.reset()
        return ld
