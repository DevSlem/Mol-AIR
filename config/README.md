# Mol-AIR Experiment Configuration

Mol-AIR experiment configuration is written in YAML format. You can make a new configuration or modify the existing one. Configuration starts with an experiment ID and contains six sections: 

<!-- no toc -->
|Section|Description|Required|
|---|---|---|
|[Env](#env)|Chemical environment|✅|
|[Agent](#agent)|RL agent|✅|
|[CountIntReward](#countintreward)|Count-based intrinsic reward|❌|
|[Pretrain](#pretrain)|Pre-training|❌|
|[Train](#train)|RL training|✅|
|[Inference](#inference)|RL Inference|❌|

For example, the configuration of the pLogP optimization by Mol-AIR is as follows:

```yaml
PLogP-MolAIR:
  Env:
    plogp_coef: 1.0
  Agent:
    type: RND
    n_steps: 64
    epoch: 6
    seq_len: 35
    seq_mini_batch_size: 16
    gamma: 1.0
    nonepi_adv_coef: 1.0
  CountIntReward:
    crwd_coef: 0.01
  Train:
    num_envs: 64
    seed: 0
    total_time_steps: 120000
    summary_freq: 1000
    device: cuda
```

where `PLogP-MolAIR` is the experiment ID. You can check the details of the parameters for each configuration section below.

## Configuration Parameters

In this section, we describe the configuration parameters for each configuration section. The most important variables are `Agent/nonepi_adv_coef` and `CountIntReward/crwd_coef`. The `nonepi_adv_coef` is equal to $\alpha$ and the `crwd_coef` is equal to $\alpha\beta$ in the context of our paper. **Since these variables determine the power of the intrinsic reward, it affects the performance of the agent.**

> Note: If the configuration variable has no default value, you must specify the value explicitly in the configuration.

### Env

`Env` section contains the parameters for the chemical environment which evaluates the generated molecules. You must set at least one property score coefficients. If you don't set the value, the property is never evaluated.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`plogp_coef`|`float`|`0.0`|pLogP score coefficient.|
|`qed_coef`|`float`|`0.0`|QED score coefficient.|
|`similarity_coef`|`float`|`0.0`|Similarity score coefficient.|
|`gsk3b_coef`|`float`|`0.0`|GSK3B score coefficient.|
|`jnk3_coef`|`float`|`0.0`|JNK3 score coefficient.|
|`drd2_coef`|`float`|`0.0`|DRD2 score coefficient.|
|`sa_coef`|`float`|`0.0`|Synthetic accessibility (SA) score coefficient.|
|`max_str_len`|`int`|`35`|Maximum length of SELFIES strings without `[STOP]` token.|
|`vocab_path`|`str\|None`|`None`|Vocabulary path for SELFIES. It overwrites `max_str_len`. Defaults to SELFIES default alphabets.|
|`init_selfies`|`str\|list[str]\|None`|`None`|The agent generates molecules from the initial SELFIES strings. It is useful either when the agent struggle to learn the policy with no reference or when you want to generate molecules (inference) with the specific substructure the well-known drug has (i.e., the substring of the drug). If it is a list, the environment provides the initial SELFIES randomly from the list. Defaults to no initial SELFIES.|

We highly recommend to set the coefficient value to `1.0` for the property you want to optimize. For example, if you want to optimize the pLogP score, set `plogp_coef` to `1.0` and the other coefficients to `0.0`. If you want to optimize multiple properties at the same time, we recommend that the weighted sum of the values is `1.0`.

### Agent

`Agent` section contains the parameters for the RL agent. First of all, you should choose the agent type:

|Parameters|Type|Default|Description|
|---|---|---|---|
|`type`|`str`||Which agent to use. <br><br> Options: `PPO`, `RND`, `Pretrained`|

where `Pretrained` agent has no additional parameters. Since both PPO and RND use recurrent neural networks, you should consider sequence-based parameters to train the agent:

|Parameters|Type|Default|Description|
|---|---|---|---|
|`n_steps`|`int`||The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|`int`||The number of times that the entire experience batch is used to update parameters.|
|`seq_len`|`int`||The sequence length of the experience sequence batch when **training**. The entire experience batch is truncated by `seq_len` unit then results in the experience sequences with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. We highly recommend to set the value equal to the `max_str_len`.|
|`seq_mini_batch_size`|`int`||The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size / seq_mini_batch_size`.|
|`padding_value`|`float`|`0.0`|Pad sequences to the value for the same `seq_len`.|

PPO and RND have shared common parameters:

|Parameters|Type|Default|Description|
|---|---|---|---|
|`gamma`|`float`|`0.99`|Discount factor $\gamma$ of future episodic rewards.|
|`lam`|`float`|`0.95`|Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|`float`|`0.2`|Clamps the probability ratio ($\frac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1-\epsilon, 1+\epsilon]$.|
|`critic_loss_coef`|`float`|`0.5`|State value loss (critic loss) multiplier.|
|`entropy_coef`|`float`|`0.001`|Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|

RND has unique parameters (**RND only**):

|Parameters|Type|Default|Description|
|---|---|---|---|
|`gamma_n`|`float`|`0.99`|Discount factor $\gamma_N$ of future non-episodic rewards.|
|`nonepi_adv_coef`|`float`|`1.0`|Non-episodic advantage multiplier. It is equal to $\alpha$ in the context of our paper.|
|`rnd_pred_exp_proportion`|`float`|`0.25`|The proportion of experiences used to train RND predictor to keep the effective batch size.|
|`init_norm_steps`|`int\|None`|`50`|The initial time steps to initialize normalization parameters of both observation and hidden state. If the value is `None`, the algorithm never normalize them during training.|
|`obs_norm_clip_range`|`[float,float]`|`[-5.0,5.0]`|Clamps the normalized observation into the range `[min, max]`.|
|`hidden_state_norm_clip_range`|`[float,float]`|`[-5.0,5.0]`|Clamps the normalized hidden state into the range `[min, max]`.|

### CountIntReward

`CountIntReward` section contains the parameters for the count-based intrinsic reward. If you exclude this section, the count-based intrinsic reward is not used.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`crwd_coef`|`float`|`0.0`|Count-based intrinsic reward coefficient. It is equal to $\alpha\beta$ in the context of our paper.|
|`max_mol_count`|`int`|`10`|Maximum count $\tau$ of the same molecule. The count starts from 0.|
|`fingerprint_bits`|`int`|`256`||
|`fingerprint_radius`|`int`|`2`||
|`lsh_bits`|`int`|`32`||

### Pretrain

`Pretrain` section contains the parameters for the pre-training process.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`dataset_path`|`str`||Dataset path for pre-training. It doesn't matter whether the dataset is SMILES or SELFIES.|
|`epoch`|`int`|`50`|The number of epochs to train.|
|`batch_size`|`int`|`256`|Batch size.|
|`lr`|`float`|`1e-3`|Learning rate.|
|`device`|`str\|None`|`None`|Device on which the pre-trained network works. Defaults to `cpu`.|
|`seed`|`int\|None`|`None`|Random seed for pre-training.|
|`save_agent`|`bool`|`True`|Whether to save the pre-trained agent as `agent.pt` by wrapping the best network with `PretrainedRecurrentAgent` at the end of the pre-training.|

### Train

`Train` section contains the parameters for the RL training process.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`num_envs`|`int`||The number of parallel environments.|
|`total_time_steps`|`int`||The number of total time steps to train. The number of total experiences is `total_time_steps` * `num_envs`.|
|`summary_freq`|`int\|None`|`None`|Summary frequency. Defaults to `total_time_steps` / `20`.|
|`agent_save_freq`|`int\|None`|`None`|Agent save frequency. Inference is performed at this frequency. Defaults to `summary_freq` * `10`.|
|`num_inference_envs`|`int`|`0`|The number of parallel environments for inference. If you have a lower performance CPU or small amount of memory, we highly recommend that the value is lower. Defaults to no inference.|
|`n_inference_episodes`|`int`|`1`|The total number of episodes (generated molecules) at each inference frequency. It is similar to validation. The greater `num_inference_envs` is, the faster `n_inference_episodes` is reached. We highly recommend that the value is greater.|
|`seed`|`int\|None`|`None`|Random seed for RL training.|
|`lr`|`float`|`1e-3`|Learning rate. This default value is for training from scratch. If you optimize a pre-trained model with the default value, **policy collapse (catatrophic forgetting)** may occur due to the large learning rate.|
|`grad_clip_max_norm`|`float`|`5.0`|Maximum gradient norm.|
|`pretrained_path`|`str\|None`|`None`|Pre-trained model path with the `.pt` extension. If you don't specify the path explicitly, it trains either the pre-trained model `results/[EXPERIMENT_ID]/pretrained_models/best.pt` or from the scratch.|
|`refset_path`|`str\|None`|`None`|Reference set path for calculating novelty metric. Typically, the reference set is the training set. Defaults to no reference set.|
|`device`|`str\|None`|`None`|Device on which the agent works. <br><br> Options: `None`, `cpu`, `cuda`, `cuda:0` and other devices of `torch.device()` argument|

### Inference

`Inference` section contains the parameters for the RL inference process after RL training.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`num_envs`|`int`|`1`|The number of parallel environments for inference. It speeds up the inference process.|
|`n_episodes`|`int`|`1`|The total number of episodes (generated molecules). The greater `num_envs` is, the faster `n_episodes` is reached. We highly recommend that the value is larger.|
|`n_unique_molecules`|`int\|None`|`None`|The threshold of the number of unique molecules to stop the inference. If the value is `None`, the inference is performed until the `n_episodes` is reached.|
|`ckpt`|`str\|int`|`best`|Checkpoint name or step number. If the value is `best`, the best checkpoint `best_agent.pt` is used. If the value is `final`, `agent.pt` is used. Otherwise, the checkpoint with the step number is used.|
|`refset_path`|`str\|None`|`None`|Reference set path for calculating novelty metric. Typically, the reference set is the training set. Defaults to `Train.refset_path`.|
|`temperature`|`float`|`1.0`|Temperature for sampling. The higher the temperature, the more diverse the generated molecules are.|
|`seed`|`int\|None`|`None`|Random seed for inference.|
|`device`|`str\|None`|`None`|Device on which the agent works. Defaults to `Train.device`.|
