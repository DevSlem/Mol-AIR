# Mol-AIR

This repository is the implementation of our paper: 

[Mol-AIR: Molecular Reinforcement Learning with Adaptive Intrinsic Reward for Goal-directed Molecular Generation](https://arxiv.org/abs/2403.20109)

We optimized the pLogP score using Mol-AIR without any prior knowledge and could find sulfur-phosphorus-nitrogen chain:

![](img/molair-fig6.jpg)

For other properties, we outperformed the previous methods. Please refer to our paper for more details.

## Setup

Our environment settings:

* OS: Ubuntu 20.04.6 LTS
* GPU: NVIDIA RTX A6000
* Python: 3.7 (ours: 3.7.16)

> Note: If you don't use Python 3.7, you may have dependency issues.

Packages:

* [PyTorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-21): 1.11.0+cu113
* [TensorBoard](https://pypi.org/project/tensorboard/2.11.2/): 2.11.2
* [PyYAML](https://pypi.org/project/PyYAML/6.0/): 6.0
* [SELFIES](https://pypi.org/project/selfies/0.2.4/): 0.2.4
* [RDKit](https://pypi.org/project/rdkit/2022.9.5/): 2022.9.5
* [PyTDC](https://pypi.org/project/PyTDC/0.4.0/): 0.4.0
* [NetworkX](https://pypi.org/project/networkx/2.6.3/): 2.6.3
* [Pandas](https://pypi.org/project/pandas/1.3.5/): 1.3.5

If you use conda, create a new Python 3.7.16 environment:

```bash
conda create -n mol-air python=3.7.16 -y
conda activate mol-air
```

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

> Note: The [requirements.txt](requirements.txt) file was tested only on Ubuntu (Linux) OS.

## Run

All experimental configurations are in the [config](config/) directory. `ppo.yaml` is the configuration for PPO, `hir.yaml` is for HIR (Count-based intrinsic reward used), and `lir.yaml` is for LIR (RND intrinsic reward used) and `molair.yaml` is for Mol-AIR (Both Count-based and RND intrinsic rewards used). For all molecular properties, we used the same configuration except for the intrinsic reward coefficients. 

You can run our experiments by entering the below command:

```bash
python run.py [CONFIG_PATH]
```

For example, you can run Mol-AIR ($\alpha$ differs for each property, $\beta = 0.01$):

```bash
python run.py config/plogp/molair.yaml # pLogP
python run.py config/qed/molair.yaml # QED
python run.py config/similarity/molair.yaml # Similarity
python run.py config/gsk3b/molair.yaml # GSK3B
python run.py config/jnk3/molair.yaml # JNK3
python run.py config/gsk3b_jnk3/molair.yaml # GSK3B+JNK3
```

Experiment results are saved in the `results` directory. For example, if you run the pLogP with Mol-AIR, the result is saved in `results/PLogP-MolAIR`. 

There are two methods to view the results: TensorBoard and CSV. You can see the simple results such as cumulative reward, training loss, etc by Tensorboard. To do so, enter the command: 

```bash
tensorboard --logdir=[EXPERIMENT_RESULT_DIR]
```

`[EXPERIMENT_RESULT_DIR]` is the directory where the experiment result is saved. For example, `results/PLogP-MolAIR`.

If you want to see the detailed results of the molecules generated, you should see the CSV file `[EXPERIMENT_RESULT_DIR]/episode_metric.csv`. Using the CSV files, you can compare the results of different methods by executing the `export.py`:

```
usage: export.py [-h] [-e EPISODE] [-m MOVING_AVERAGE]
                 TITLE EXPERIMENT_RESULT_DIR [EXPERIMENT_RESULT_DIR ...]

positional arguments:
  TITLE                 Title for the data
  EXPERIMENT_RESULT_DIR
                        Directories of experiment results

optional arguments:
  -h, --help            show this help message and exit
  -e EPISODE, --episode EPISODE
                        Episode number (default = max)
  -m MOVING_AVERAGE, --moving_average MOVING_AVERAGE
                        Moving average n (default = max_episode / 100)
```

For example, if you want to compare the results of PPO, HIR, LIR and Mol-AIR for pLogP (until 3000 episodes), enter the command like this: `$ python export.py -e 3000 pLogP results/PLogP-PPO results/PLogP-HIR results/PLogP-LIR results/PLogP-MolAIR`. Then, the comparison results are saved in the `exports/[TITLE]` directory. In this case, the directory is `exports/pLogP`.

## Configuration

If you want to customize the experiment configuration, you should modify the configuration file or create a new one. The configuration file is written in YAML format. The configuration file starts with experiment ID and contains four sections: `Agent`, `Env`, `Train`, and `CountIntReward`. For example, the pLogP with Mol-AIR configuration is as follows:

```yaml
PLogP-MolAIR:
  Agent:
    type: RND
    n_steps: 64
    epoch: 6
    seq_len: 35
    seq_mini_batch_size: 16
    gamma: 1.0
    nonepi_adv_coef: 1.0
  Env:
    plogp_coef: 1.0
  Train:
    num_envs: 64
    seed: 0
    total_time_steps: 120000
    summary_freq: 1000
    device: cuda
  CountIntReward:
    crwd_coef: 0.01
```

In this case, `PLogP-MolAIR` is the experiment ID. 

The most important variables are `Agent/nonepi_adv_coef` and `CountIntReward/crwd_coef`. The `nonepi_adv_coef` is equal to $\alpha$ and the `crwd_coef` is equal to $\alpha\beta$ in the context of our paper. Since these variables determine the power of the intrinsic reward, it affects the performance of the agent.

The details of configuration variables are described below.

> Note: If the configuration variable has no default value, you must set the value explicitly in the configuration file.

### Train

`Train` section contains the settings for the training process.

|Setting|Type|Default|Description|
|---|---|---|---|
|`num_envs`|`int`||The number of parallel environments.|
|`total_time_steps`|`int`||The number of total time steps to train. The number of total experiences is `total_time_steps` * `num_envs`.|
|`summary_freq`|`int \| None`|`None`|Summary frequency. Defaults to `total_time_steps` / `20`.|
|`agent_save_freq`|`int \| None`|`None`|Agent save frequency. Inference is performed at this frequency. Defaults to `summary_freq` * `10`.|
|`num_inference_envs`|`int`|`1`|The number of parallel environments for inference. If you have a lower performance CPU or small amount of memory, we highly recommend that the value is lower.|
|`n_inference_episodes`|`int`|`1`|The total number of episodes for inference. The greater `num_inference_envs` is, the faster `n_inference_episodes` is reached. We highly recommend that the value is greater.|
|`pretrained_path`|`str \| None`|`None`|Pre-trained model path. Defaults to training from scratch.|
|`seed`|`int \| None`|`None`|Random seed.|
|`lr`|`float`|`1e-3`|Learning rate. This default value is for training from scratch. If you optimize a pre-trained model with the default value, **policy collapse (catatrophic forgetting)** may occur due to the large learning rate.|
|`grad_clip_max_norm`|`float`|`5.0`|Maximum gradient norm.|
|`device`|`str \| None`|`None`|Device on which the agent works. If this setting is `None`, the agent device is same as your network's one. Otherwise, the network device changes to this device. <br><br> Options: `None`, `cpu`, `cuda`, `cuda:0` and other devices of `torch.device()` argument|

### Env

`Env` section contains the settings for the chemical environment which evaluates the generated molecules. You must set at least one property score coefficients. If you don't set the value, the property is never evaluated.

|Setting|Type|Default|Description|
|---|---|---|---|
|`plogp_coef`|`float`|`0.0`|pLogP score coefficient.|
|`qed_coef`|`float`|`0.0`|QED score coefficient.|
|`similarity_coef`|`float`|`0.0`|Similarity score coefficient.|
|`gsk3b_coef`|`float`|`0.0`|GSK3B score coefficient.|
|`jnk3_coef`|`float`|`0.0`|JNK3 score coefficient.|
|`drd2_coef`|`float`|`0.0`|DRD2 score coefficient.|
|`sa_coef`|`float`|`0.0`|Synthetic accessibility (SA) score coefficient.|
|`max_str_len`|`int`|`35`|Maximum length of SELFIES strings. If you optimize your pre-trained model, we highly recommend that the value is the **maximum sequence length** without `[STOP]` token in the pre-training dataset.|

We highly recommend to set the coefficient value to `1.0` for the property you want to optimize. For example, if you want to optimize the pLogP score, set `plogp_coef` to `1.0` and the other coefficients to `0.0`. If you want to optimize multiple properties at the same time, we recommend that the weighted sum of the values is `1.0`.

### Agent

`Agent` section contains the settings for the RL agent. First of all, you should choose the agent type:

|Setting|Type|Default|Description|
|---|---|---|---|
|`type`|`str`||Which agent to use. <br><br> Options: `PPO`, `RND`|


Since both PPO and RND use recurrent neural networks, you should consider sequence-based settings to train the agent:

|Setting|Type|Default|Description|
|---|---|---|---|
|`n_steps`|`int`||The number of time steps to collect experiences until training. The number of total experiences (`entire_batch_size`) is `num_envs * n_steps`. Since PPO is on-policy method, the experiences are discarded after training.|
|`epoch`|`int`||The number of times that the entire experience batch is used to update parameters.|
|`seq_len`|`int`||The sequence length of the experience sequence batch when **training**. The entire experience batch is truncated by `seq_len` unit then results in the experience sequences with `padding_value`. This is why the entire sequence batch size (`entire_seq_batch_size`) is greater than `entire_batch_size`. Typically `8` or greater value are recommended.|
|`seq_mini_batch_size`|`int`||The sequence mini-batches are selected randomly and independently from the entire experience sequence batch during one epoch. The number of parameters updates at each epoch is the integer value of `entire_seq_batch_size / seq_mini_batch_size`.|
|`padding_value`|`float`|`0.0`|Pad sequences to the value for the same `seq_len`.|

RND is based on PPO, so it has common settings with PPO:

|Setting|Type|Default|Description|
|---|---|---|---|
|`gamma`|`float`|`0.99`|Discount factor $\gamma$ of future episodic rewards.|
|`lam`|`float`|`0.95`|Regularization parameter $\lambda$ which controls the bias-variance trade-off of Generalized Advantage Estimation (GAE).|
|`epsilon_clip`|`float`|`0.2`|Clamps the probability ratio ($\frac{\pi_{\text{new}}}{\pi_{\text{old}}}$) into the range $[1-\epsilon, 1+\epsilon]$.|
|`critic_loss_coef`|`float`|`0.5`|State value loss (critic loss) multiplier.|
|`entropy_coef`|`float`|`0.001`|Entropy multiplier used to compute loss. It adjusts exploration-exploitation trade-off.|

RND has unique settings (**RND only**):

|Setting|Type|Default|Description|
|---|---|---|---|
|`gamma_n`|`float`|`0.99`|Discount factor $\gamma_N$ of future non-episodic rewards.|
|`nonepi_adv_coef`|`float`|`1.0`|Non-episodic advantage multiplier. It is equal to $\alpha$ in the context of our paper.|
|`rnd_pred_exp_proportion`|`float`|`0.25`|The proportion of experiences used to train RND predictor to keep the effective batch size.|
|`init_norm_steps`|`int \| None`|`50`|The initial time steps to initialize normalization parameters of both observation and hidden state. If the value is `None`, the algorithm never normalize them during training.|
|`obs_norm_clip_range`|`[float, float]`|`[-5.0, 5.0]`|Clamps the normalized observation into the range `[min, max]`.|
|`hidden_state_norm_clip_range`|`[float, float]`|`[-5.0, 5.0]`|Clamps the normalized hidden state into the range `[min, max]`.|

### CountIntReward

`CountIntReward` section contains the settings for the count-based intrinsic reward. If you exclude this section, the count-based intrinsic reward is not used.

|Setting|Type|Default|Description|
|---|---|---|---|
|`crwd_coef`||`float`|`0.0`|Count-based intrinsic reward coefficient. It is equal to $\alpha\beta$ in the context of our paper.|
|`max_mol_count`|`int`|`10`|Maximum count $\tau$ of the same molecule. The count starts from 0.|
|`fingerprint_bits`|`int`|`256`||
|`fingerprint_radius`|`int`|`2`||
|`lsh_bits`|`int`|`32`||

## Citation

Please cite our work if you find it useful:

```
@misc{park2024molair,
      title={Mol-AIR: Molecular Reinforcement Learning with Adaptive Intrinsic Rewards for Goal-directed Molecular Generation}, 
      author={Jinyeong Park and Jaegyoon Ahn and Jonghwan Choi and Jibum Kim},
      year={2024},
      eprint={2403.20109},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
