import argparse
from dataclasses import dataclass
from typing import Optional

import torch.optim as optim

import drl
import drl.agent as agent
import envs
import util
from config_parser import ConfigParser
from envs.chem_env import (AsyncChemEnv, AsyncChemEnvDependentIntrinsicReward,
                           ChemExtrinsicRewardConfig,
                           ChemIntrinsicRewardConfig, InferenceChemEnv)
from inference import (Inference, InferenceCkpt, InferenceCkptConfig,
                       InferenceConfig)
from net import SelfiesRecurrentPPONet, SelfiesRecurrentPPORNDNet, SelfiesRecurrentPPONoisyRNDNet, SelfiesRecurrentPPOEpisodicRNDNet
from train import Train, TrainConfig


@dataclass(frozen=True)
class TrainSetup:
    num_envs: int
    seed: Optional[int] = None
    lr: float = 1e-3
    
@dataclass(frozen=True)
class InferenceSetup:
    seed: Optional[int] = None
    
@dataclass(frozen=True)
class AgentSetup:
    type: str
    rnd_hidden_state: bool = True
    
@dataclass(frozen=True)
class EnvSetup:
    max_str_len: int = 35
    intrinsic_reward_type: str = "independent"

LEARNING_RATE = 1e-3
GRAD_CLIP_MAX_NORM = 5.0

def make_recurrent_ppo_agent(env: envs.Env, config_parser: ConfigParser) -> agent.RecurrentPPO:
    config = config_parser.parse_agent_config(agent.RecurrentPPOConfig)
    
    network = SelfiesRecurrentPPONet(
        env.obs_shape[0], 
        env.num_actions
    )
    
    trainer = drl.Trainer(optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE
    )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
    
    return agent.RecurrentPPO(
        config,
        network,
        trainer,
        num_envs=env.num_envs
    )
    
def make_recurrent_ppo_rnd_agent(env: envs.Env, config_parser: ConfigParser, noisy: bool = False) -> agent.RecurrentPPORND:
    config = config_parser.parse_agent_config(agent.RecurrentPPORNDConfig)
    setup = config_parser.parse_agent_config(AgentSetup)
    
    if noisy:
        network = SelfiesRecurrentPPONoisyRNDNet(
            env.obs_shape[0],
            env.num_actions
        )
    else:
        network = SelfiesRecurrentPPORNDNet(
            env.obs_shape[0],
            env.num_actions,
            setup.rnd_hidden_state
        )
    
    trainer = drl.Trainer(optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE
    )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
    
    return agent.RecurrentPPORND(
        config,
        network,
        trainer,
        num_envs=env.num_envs
    )
    
def make_recurrent_ppo_episodic_rnd_agent(env: envs.Env, config_parser: ConfigParser) -> agent.RecurrentPPOEpisodicRND:
    config = config_parser.parse_agent_config(agent.RecurrentPPOEpisodicRNDConfig)
    
    network = SelfiesRecurrentPPOEpisodicRNDNet(
        env.obs_shape[0],
        env.num_actions
    )
    
    trainer = drl.Trainer(optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE
    )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
    
    return agent.RecurrentPPOEpisodicRND(
        config,
        network,
        trainer,
        num_envs=env.num_envs
    )
    
def make_recurrent_ppo_idrnd_agent(env: envs.Env, config_parser: ConfigParser) -> agent.RecurrentPPOIDRND:
    config = config_parser.parse_agent_config(agent.RecurrentPPORNDConfig)
    
    network = SelfiesRecurrentPPORNDNet(
        env.obs_shape[0],
        env.num_actions
    )
    
    trainer = drl.Trainer(optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE
    )).enable_grad_clip(network.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
    
    return agent.RecurrentPPOIDRND(
        config,
        network,
        trainer,
        num_envs=env.num_envs
    )

def make_agent(env: envs.Env, config_parser: ConfigParser) -> agent.Agent:
    agent_type = config_parser.parse_agent_type().lower()
    if agent_type == "recurrentppo":
        return make_recurrent_ppo_agent(env, config_parser)
    elif agent_type == "recurrentppornd":
        return make_recurrent_ppo_rnd_agent(env, config_parser)
    elif agent_type == "recurrentppoidrnd":
        return make_recurrent_ppo_idrnd_agent(env, config_parser)
    elif agent_type == "recurrentpponoisyrnd":
        return make_recurrent_ppo_rnd_agent(env, config_parser, noisy=True)
    elif agent_type == "recurrentppoepisodicrnd":
        return make_recurrent_ppo_episodic_rnd_agent(env, config_parser)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to config file")
    parser.add_argument("-i", "--inference", action="store_true", help="run inference")
    parser.add_argument("-ic", "--inference_ckpt", action="store_true", help="run inference on checkpoint")
    args = parser.parse_args()
    config_path = args.config_path
    is_inference = args.inference
    is_inference_ckpt = args.inference_ckpt
    
    config_parser = ConfigParser.from_yaml(config_path)
    env_setup = config_parser.parse_env_config(EnvSetup)
    ext_reward_config = config_parser.parse_env_config(ChemExtrinsicRewardConfig)
    int_reward_config = config_parser.parse_env_config(ChemIntrinsicRewardConfig)
    
    if is_inference:
        inference_setup = config_parser.parse_inference_config(InferenceSetup)
        inference_config = config_parser.parse_inference_config(InferenceConfig)
        
        # set seed
        if inference_setup.seed is not None:
            util.seed(inference_setup.seed)
            
        # make env
        inference_env = InferenceChemEnv(
            ext_reward_config=ext_reward_config,
            int_reward_config=int_reward_config,
            max_str_len=env_setup.max_str_len,
            record_data=True
        )
        
        # make agent
        inference_agent = make_agent(inference_env, config_parser)
        
        # inference start
        Inference(config_parser.id, inference_config, inference_agent, inference_env).inference().close()
    elif is_inference_ckpt:
        inference_setup = config_parser.parse_inference_config(InferenceSetup)
        inference_config = config_parser.parse_inference_config(InferenceCkptConfig)
        
        # set seed
        if inference_setup.seed is not None:
            util.seed(inference_setup.seed)
            
        # make env
        inference_env = InferenceChemEnv(
            ext_reward_config=ext_reward_config,
            int_reward_config=int_reward_config,
            max_str_len=env_setup.max_str_len,
            record_data=True,
            log_data_required_episodes=inference_config.episodes
        )
        
        # make agent
        inference_agent = make_agent(inference_env, config_parser)
        
        # inference start
        InferenceCkpt(config_parser.id, inference_config, inference_agent, inference_env).inference().close()
    else:
        util.logger.enable(config_parser.id, enable_log_file=False)
        util.try_create_dir(util.logger.dir())
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_parser.config_dict)
        util.logger.disable()
        
        train_setup = config_parser.parse_train_config(TrainSetup)
        train_config = config_parser.parse_train_config(TrainConfig)
        LEARNING_RATE = train_setup.lr
        
        # set seed
        if train_setup.seed is not None:
            util.seed(train_setup.seed)
        
        # make env
        intrinsic_reward_type = env_setup.intrinsic_reward_type.lower()
        if intrinsic_reward_type == "independent":
            env = AsyncChemEnv(
                ext_reward_config=ext_reward_config,
                int_reward_config=int_reward_config,
                num_envs=train_setup.num_envs,
                max_str_len=env_setup.max_str_len,
                seed=train_setup.seed
            )
        elif intrinsic_reward_type == "dependent":
            env = AsyncChemEnvDependentIntrinsicReward(
                ext_reward_config=ext_reward_config,
                int_reward_config=int_reward_config,
                num_envs=train_setup.num_envs,
                max_str_len=env_setup.max_str_len,
            )
        else:
            raise NotImplementedError
        
        # make agent
        train_agent = make_agent(env, config_parser)

        # train start
        Train(config_parser.id, train_config, train_agent, env).train().close()
