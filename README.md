# Mol-AIR

This repository is the implementation of our paper: 

[Mol-AIR: Molecular Reinforcement Learning with Adaptive Intrinsic Reward for Goal-directed Molecular Generation](https://arxiv.org/abs/2403.20109)

We optimized the pLogP score using Mol-AIR without any prior knowledge and could find sulfur-phosphorus-nitrogen chain:

![](img/molair-fig6.jpg)

For other properties, we outperformed the previous methods. Please refer to our paper for more details.

## Setup & Run Experiments

We have two types of experiments: **pure RL** and **pre-training + RL**. The pure RL experiments were conducted with SELFIES 0.2.4 for comparison with the previous methods. While the pre-training + RL experiments were conducted with SELFIES 2.1.2 since SELFIES 0.2.4 is too old to construct the vocabulary for training. Therefore, we highly recommend to install the desired version of SELFIES to reproduce our experiments, even if it is compatible with any version and customizable. Also, we recommend to use conda to create a new environment for each experiment type.

### Pure RL Experiments

The experiments of pure RL show that our proposed intrinsic reward method, **Mol-AIR**, can help the agent to explore the chemical space effectively so that the agent can find better molecules. If you use conda, create a new Python 3.7.16 environment:

```bash
conda create -n mol-air python=3.7.16 -y
conda activate mol-air
```

> Note: If you don't use Python 3.7, you may have dependency issues.

Install the required packages using pip (SELFIES 0.2.4 is included):

```bash
pip install -r requirements.txt
```

> Note: The [requirements.txt](requirements.txt) file was tested only on Ubuntu (Linux) OS.

Pure RL experiments were conducted on the 6 target properties: **pLogP, QED, Similarity, GSK3B, JNK3, and GSK3B+JNK3 (two objectives)**. All experiment configurations are in the [config/](config/) directory. `ppo.yaml` is the configuration for vanilla PPO, `hir.yaml` is for HIR (history-based intrinsic reward), and `lir.yaml` is for LIR (learning-based intrinsic reward) and `molair.yaml` is for Mol-AIR (both history-based and learning-based intrinsic rewards). You can run the experiments by entering the below command:

```bash
python run.py [CONFIG_PATH]
```

For example, pLogP with PPO is `$ python run.py config/plogp/ppo.yaml` and pLogP with Mol-AIR is `$ python run.py config/plogp/molair.yaml`.

### Pre-training + RL Experiments (Generative Model)

The experiments of pre-training + RL show that Mol-AIR works well in optimizing the pre-trained model to generate the various molecules with the desired properties. We recommend to create a different environment for the pre-training + RL experiments:

```bash
conda create -n mol-air-gen python=3.7.16 -y
conda activate mol-air-gen
```

> Note: If you don't use Python 3.7, you may have dependency issues.

Install the required packages using pip (SELFIES 2.1.2 is included):

```bash
pip install -r requirements_gen.txt
```

> Note: The [requirements_gen.txt](requirements_gen.txt) file was tested only on Ubuntu (Linux) OS.

Pre-training + RL experiments were conducted on the DRD2+QED+SA property (three objectives). You can run the end-to-end experiment of the DRD2+QED+SA property from pre-training to Mol-AIR by entering the below commands ([molair_end2end.yaml](config/drd2+qed+sa/molair_end2end.yaml)):

```bash
python run.py config/drd2+qed+sa/molair_end2end.yaml -p # Pre-training
python run.py config/drd2+qed+sa/molair_end2end.yaml # RL training
python run.py config/drd2+qed+sa/molair_end2end.yaml -i # Inference
```

where Inference is to generate the molecules with the trained agent. But, it doesn't guarantee to reproduce the results since the order of the tokens in the vocabulary constructed at the pre-training stage can be different from our experiment. If you want to reproduce the results of our experiments, we highly recommend to run the experiments by the configuration files: [pretrained.yaml](config/drd2+qed+sa/pretrained.yaml), [ppo.yaml](config/drd2+qed+sa/ppo.yaml) and [molair.yaml](config/drd2+qed+sa/molair.yaml), which use the provided [pre-trained model](data/drd2+qed+sa/pretrained.pt) and [vocabulary](data/drd2+qed+sa/vocab.json). So, if you want to reproduce the results of Mol-AIR, enter the below commands:

```bash
python run.py config/drd2+qed+sa/molair.yaml # RL training
python run.py config/drd2+qed+sa/molair.yaml -i # Inference
```

Also, you can reproduce the results of PPO by:

```bash
python run.py config/drd2+qed+sa/ppo.yaml # RL training
python run.py config/drd2+qed+sa/ppo.yaml -i # Inference
```

and the pre-trained agent:

```bash
python run.py config/drd2+qed+sa/pretrained.yaml # Just wrapping
python run.py config/drd2+qed+sa/pretrained.yaml -i # Inference
```

> Note: RL training with the pre-trained agent configuration ([pretrained.yaml](config/drd2+qed+sa/pretrained.yaml)) is just wrapping the pre-trained model with the `PretrainedRecurrentAgent` class for the compatibility with the RL agent. The pre-trained model is never updated by the command.

## Experiment Results

All experiment results are saved in the `results/[EXPERIMENT_ID]/` directory. For example, if you run the pLogP with Mol-AIR, the result is saved in `results/PLogP-MolAIR`. You can see the training plots of each experiment in the `results/[EXPERIMENT_ID]/plots/` directory. Also, you can see the training plots at once by TensorBoard: `$ tensorboard --logdir=results`. This makes you easily compare the results of different experiments.

### RL Training

The training plots are saved in the directories:

* `plots/Environment/`: metrics of the training environment.
* `plots/Inference/`: metrics of the inference environment.
* `plots/Training/`: losses of the RL agent.

The RL agents are saved as:

* `agent.pt`: final agent.
* `best_agent.pt`: best agent which has the highest score from the inference environments during training.
* `agent_ckpt/`: agent checkpoints directory at the every `agent_save_freq` frequency.

> Note: The pre-trained agent has only `agent.pt`.

The most important file is `episode_metric.csv` which contains the metrics of the generated molecules at each episode during training.

### Inference

The inference results are saved in the directory: `inference/`. You can see the three or four files in this directory:

* `metrics.csv`: summary of the metrics of the generated molecules.
* `molecules.csv`: generated molecules with the metrics.
* `best_molecule.csv`: the best molecule with the highest score.
* `top_50_unique_molecules.png` (optional): images of the top 50 unique molecules with the highest scores.

> Note: `top_50_unique_molecules.png` requires the `libXrender.so.1`. If you want to draw molecules, you should install it by `$ sudo apt-get install libxrender1` or `$ conda install -c conda-forge libxrender`.

### Pre-training

The pre-training plots are saved in the `plots/Pretrain/` directory. The pre-trained models are saved in the `pretrained_models/` directory:

* `best.pt`: best pre-trained model which has the lowest validation loss.
* `final.pt`: final pre-trained model at the end of the pre-training.

and the vocabulary used in the pre-training is saved as the `vocab.json` file.

## Configuration

If you want to customize the experiment configuration, you should modify the configuration file or create a new one. The configuration file is written in YAML format. The configuration file starts with experiment ID and can contain six sections: `Train`, `Inference`, `Pretrain`, `Env`, `Agent` and `CountIntReward`. For example, the pLogP with Mol-AIR configuration is as follows:

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

> Note: If the configuration variable has no default value, you must specify the value explicitly in the configuration file.

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

`Inference` section contains the parameters for the inference process after training.

|Parameters|Type|Default|Description|
|---|---|---|---|
|`num_envs`|`int`|`1`|The number of parallel environments for inference. It speeds up the inference process.|
|`n_episodes`|`int`|`1`|The total number of episodes (generated molecules). The greater `num_envs` is, the faster `n_episodes` is reached. We highly recommend that the value is larger.|
|`ckpt`|`str\|int`|`best`|Checkpoint name or step number. If the value is `best`, the best checkpoint `best_agent.pt` is used. If the value is `final`, `agent.pt` is used. Otherwise, the checkpoint with the step number is used.|
|`refset_path`|`str\|None`|`None`|Reference set path for calculating novelty metric. Typically, the reference set is the training set. Defaults to `Train.refset_path`.|
|`seed`|`int\|None`|`None`|Random seed for inference.|
|`device`|`str\|None`|`None`|Device on which the agent works. Defaults to `Train.device`.|

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
