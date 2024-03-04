# Mol-AIR

This repository is the implementation of our paper: 

Mol-AIR: Molecular Reinforcement Learning with Adaptive Intrinsic Reward for Goal-directed Molecular Generation

We optimized the pLogP score and could find sulfur-phosphorus-nitrogen chain using Mol-AIR:

![](img/molair-fig6.jpg)

For other properties, we outperformed the previous methods. Please refer to our paper for more details.

> This repository will be updated soon to be more user-friendly. The new updated repository will reproduce the same results of our paper.

## Setup

First of all, **Python 3.7 (specifically 3.7.16)** is required.

Install required packages:

TODO: setup.py

or manually:

```bash
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard==2.11.2
pip install PyYAML==6.0
pip install selfies==0.2.4
pip install rdkit==2022.9.5
pip install PyTDC==0.4.0
pip install networkx==2.6.3
pip install pandas==1.3.5
```

<!-- ```bash
pip install CairoSVG==2.7.0
pip install matplotlib==3.5.3
``` -->

## Run

Enter the below command:

```bash
python run.py <CONFIG_PATH>
```

You can reproduce the following properties, pLogP:

```bash
python run.py config/plogp/ppo.yaml # PPO
python run.py config/plogp/rnd.yaml # RND
python run.py config/plogp/count_ppo.yaml # Count+PPO
python run.py config/plogp/count_rnd.yaml # Count+RND
```

QED:

```bash
python run.py config/qed/ppo.yaml # PPO
python run.py config/qed/rnd.yaml # RND
python run.py config/qed/count_ppo.yaml # Count+PPO
python run.py config/qed/count_rnd.yaml # Count+RND
```

Similarity:

```bash
python run.py config/similarity/ppo.yaml # PPO
python run.py config/similarity/rnd.yaml # RND
python run.py config/similarity/count_ppo.yaml # Count+PPO
python run.py config/similarity/count_rnd.yaml # Count+RND
```

DRD2:

```bash
python run.py config/drd2/ppo.yaml # PPO
python run.py config/drd2/rnd.yaml # RND
python run.py config/drd2/count_ppo.yaml # Count+PPO
python run.py config/drd2/count_rnd.yaml # Count+RND
```

GSK3B:

```bash
python run.py config/gsk3b/ppo.yaml # PPO
python run.py config/gsk3b/rnd.yaml # RND
python run.py config/gsk3b/count_ppo.yaml # Count+PPO
python run.py config/gsk3b/count_rnd.yaml # Count+RND
```

JNK3:

```bash
python run.py config/jnk3/ppo.yaml # PPO
python run.py config/jnk3/rnd.yaml # RND
python run.py config/jnk3/count_ppo.yaml # Count+PPO
python run.py config/jnk3/count_rnd.yaml # Count+RND
```

<!-- ### Inference

Enter the below command:

```bash
python run.py <CONFIG_PATH> -i
``` -->

## Configuration

### Env

|Setting|Description|
|---|---|
|`max_str_len`|(`int`, default = `35`) Maximum length of SELFIES strings.|
|`plogp_coef`|(`float`, default = `0.0`) pLogP reward coefficient.|
|`qed_coef`|(`float`, default = `0.0`) QED reward coefficient.|
|`similarity_coef`|(`float`, default = `0.0`) Similarity reward coefficient.|
|`only_final`|(`bool`, default = `False`) If `False`, $r_t = p(\text{mol}(t)) - p(\text{mol}(t-1))$. If `True`, $r_t = p(\text{mol}(T))$ when SELFIES string finished, otherwise $0$.|
|`intrinsic_reward_type`|(`str`, default = `independent`) Whether to use either independent or dependent intrinsic reward. TODO: more descriptions|
|`count_coef`|(`float`, default = `0.0`) Count-based intrinsic reward coefficient.|
|`memory_coef`|(`float`, default = `0.0`) Memory-based intrinsic reward coefficient.|
|`memory_size`|(`int`, default = `1000`) Morgan fingerprint memory size for memory-based reward.|
|`fingerprint_bits`|(`int`, default = `16`) TODO|
|`fingerprint_radius`|(`int`, default = `2`) TODO|
|`lsh_bits`|(`int`, default = `16`) TODO|

### Train

|Setting|Description|
|---|---|
|`num_envs`|(`int`) The number of environments. The environments work asynchronously.|
|`time_steps`|(`int`) The number of total time steps to train.|
|`summary_freq`|(`int \| None`, default = `None`) Summary frequency. Defaults to `time_steps` / `20`.|
|`agent_save_freq`|(`int \| None`, default = `None`) Agent save frequency. Defaults to `summary_freq` * `10`.|
|`seed`|(`int \| None`, default = `None`) Random seed.|

### Inference

TODO

### Agent

Common settings:

|Setting|Description|
|---|---|
|`type`|(`str`) Which agent to use. <br><br> Options: `RecurrentPPO`, `RecurrentPPORND`|
|`device`|(`str \| None`, default = `None`) Device on which the agent works. If this setting is `None`, the agent device is same as your network's one. Otherwise, the network device changes to this device. <br><br> Options: `None`, `cpu`, `cuda`, `cuda:0` and other devices of `torch.device()` argument|

Recurrent PPO settings:

|Setting|Description|
|---|---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs` * `n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters.|
|`seq_len`|(`int`) The sequence length of the experience sequence batch when **training**. The entire experience batch is split by `seq_len` unit then results in the experience sequences with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|(`int`) The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size` / `seq_mini_batch_size`.|
|`padding_value`|(`float`, default = `0.0`) Pad sequences to the value for the same `seq_len`.|
|`gamma`|(`float`, default = `0.99`) Discount factor $\gamma$ of future rewards.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($\frac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1-\epsilon, 1+\epsilon]$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|

Recurrent PPO RND settings:

|Setting|Description|
|---|---|
|`n_steps`|(`int`) The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs` * `n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|(`int`) The number of times the entire experience batch is used to update parameters.|
|`seq_len`|(`int`) The sequence length of the experience sequence batch when **training**. The entire experience batch is split by `seq_len` unit then results in the experience sequences with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|(`int`) The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size` / `seq_mini_batch_size`.|
|`padding_value`|(`float`, default = `0.0`) Pad sequences to the value for the same `seq_len`.|
|`ext_gamma`|(`float`, default = `0.999`) Discount factor $\gamma_E$ of future extrinsic rewards.|
|`int_gamma`|(`float`, default = `0.99`) Discount factor $\gamma_I$ of future intrinsic rewards.|
|`ext_adv_coef`|(`float`, default = `1.0`) Extrinsic advantage multiplier.|
|`int_adv_coef`|(`float`, default = `1.0`) Intrinsic advantage multiplier.|
|`lam`|(`float`, default = `0.95`) Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|(`float`, default = `0.2`) Clamps the probability ratio ($\frac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1-\epsilon, 1+\epsilon]$.|
|`value_loss_coef`|(`float`, default = `0.5`) State value loss (critic loss) multiplier.|
|`entropy_coef`|(`float`, default = `0.001`) Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|
|`rnd_pred_exp_proportion`|(`float`, default = `0.25`) The proportion of experiences used to train RND predictor to keep the effective batch size.|
|`init_norm_steps`|(`int \| None`, default = `50`) The initial time steps to initialize normalization parameters of both observation and hidden state. If the value is `None`, the algorithm never normalize them during training.|
|`obs_norm_clip_range`|(`[float, float]`, default = `[-5.0, 5.0]`) Clamps the normalized observation into the range `[min, max]`.|
|`hidden_state_norm_clip_range`|(`[float, float]`, default = `[-5.0, 5.0]`) Clamps the normalized hidden state into the range `[min, max]`.|