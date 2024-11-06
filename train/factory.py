from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim

import drl
import drl.agent as agent
import train.net as net
import util
from envs import Env
from envs.chem_env import make_async_chem_env
from train.train import Train
from util import instance_from_dict


class ConfigParsingError(Exception):
    pass

@dataclass(frozen=True)
class CommonConfig:
    num_envs: int = 1
    seed: Optional[int] = None
    device: Optional[str] = None
    lr: float = 1e-3
    grad_clip_max_norm: float = 5.0
    pretrained_path: Optional[str] = None
    num_inference_envs: int = 0

class MolRLTrainFactory:
    """
    Factory class creates a Train instance from a dictionary config.
    """
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLTrainFactory":
        """
        Create a MolRLTrainFactory from a YAML file.
        """
        try:
            config_dict = util.load_yaml(file_path)
        except FileNotFoundError:
            raise ConfigParsingError(f"Config file not found: {file_path}")
        
        try:
            config_id = tuple(config_dict.keys())[0]
        except:
            raise ConfigParsingError("YAML config file must start with the training ID.")
        config = config_dict[config_id]
        return MolRLTrainFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config = self._config.get("Agent", dict())
        self._env_config = self._config.get("Env", dict())
        self._train_config = self._config.get("Train", dict())
        self._count_int_reward_config = self._config.get("CountIntReward", dict())
        self._common_config = instance_from_dict(CommonConfig, self._train_config)
        self._pretrained = None
        
    def create_train(self) -> Train:
        self._train_setup()
        
        try:
            env = self._create_env()
            inference_env = self._create_inference_env()
        except TypeError:
            raise ConfigParsingError("Invalid Env config. Missing arguments or wrong type.")
        try:
            agent = self._create_agent(env)
        except TypeError:
            raise ConfigParsingError("Invalid Agent config. Missing arguments or wrong type.")
        try:
            train = instance_from_dict(Train, {
                "env": env,
                "agent": agent,
                "id": self._id,
                "inference_env": inference_env,
                **self._train_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Train config. Missing arguments or wrong type.")
        return train
    
    def _train_setup(self):
        util.logger.enable(self._id, enable_log_file=False)
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)
        util.logger.disable()
        
        if self._common_config.seed is not None:
            util.seed(self._common_config.seed)
            
        if self._common_config.pretrained_path is not None:
            self._pretrained = torch.load(self._common_config.pretrained_path)
        
    def _create_env(self) -> Env:
        env = make_async_chem_env(
            num_envs=self._common_config.num_envs,
            seed=self._common_config.seed,
            **{**self._env_config, **self._count_int_reward_config, "vocabulary": self._pretrained["vocabulary"] if self._pretrained is not None else None}
        )
        return env
    
    def _create_inference_env(self) -> Optional[Env]:
        if self._common_config.num_inference_envs == 0:
            return None
        
        env = make_async_chem_env(
            num_envs=self._common_config.num_inference_envs,
            seed=self._common_config.seed,
            **{**self._env_config, "vocabulary": self._pretrained["vocabulary"] if self._pretrained is not None else None}
        )
        return env
    
    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config["type"].lower()
        if agent_type == "ppo":
            return self._create_ppo_agent(env)
        elif agent_type == "rnd":
            return self._create_rnd_agent(env)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _create_ppo_agent(self, env: Env) -> agent.RecurrentPPO:
        config = instance_from_dict(agent.RecurrentPPOConfig, self._agent_config)
        network = net.SelfiesRecurrentPPONet(
            env.obs_shape[0],
            env.num_actions
        )
        if self._pretrained is not None:
            network.load_state_dict(self._pretrained["model"], strict=False)
        trainer = drl.Trainer(optim.Adam(
            network.parameters(),
            lr=self._common_config.lr
        )).enable_grad_clip(network.parameters(), max_norm=self._common_config.grad_clip_max_norm)
        
        return agent.RecurrentPPO(
            config=config,
            network=network,
            trainer=trainer,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )
    
    def _create_rnd_agent(self, env: Env) -> agent.RecurrentPPORND:
        config = instance_from_dict(agent.RecurrentPPORNDConfig, self._agent_config)
        network = net.SelfiesRecurrentPPORNDNet(
            env.obs_shape[0],
            env.num_actions
        )
        if self._pretrained is not None:
            network.load_state_dict(self._pretrained["model"], strict=False)
        trainer = drl.Trainer(optim.Adam(
            network.parameters(),
            lr=self._common_config.lr
        )).enable_grad_clip(network.parameters(), max_norm=self._common_config.grad_clip_max_norm)
        
        return agent.RecurrentPPORND(
            config=config,
            network=network,
            trainer=trainer,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )
        