from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.optim as optim

import drl
import drl.agent as agent
import train.net as net
import util
from envs import Env
from envs.chem_env import make_async_chem_env
from train.train import Train
from train.pretrain import Pretrain, SelfiesDataset
from util import instance_from_dict
from train.inference import Inference
from train.net import SelfiesPretrainedNet
import os

class ConfigParsingError(Exception):
    pass

def yaml_to_config_dict(file_path: str) -> Tuple[str, dict]:
    try:
        config_dict = util.load_yaml(file_path)
    except FileNotFoundError:
        raise ConfigParsingError(f"Config file not found: {file_path}")
    
    try:
        config_id = tuple(config_dict.keys())[0]
    except:
        raise ConfigParsingError("YAML config file must start with the training ID.")
    config = config_dict[config_id]
    return config_id, config

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
        Create a `MolRLTrainFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
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
            smiles_or_selfies_refset = util.load_smiles_or_selfies(self._train_config["refset_path"]) if "refset_path" in self._train_config else None
            train = instance_from_dict(Train, {
                "env": env,
                "agent": agent,
                "id": self._id,
                "inference_env": inference_env,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
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
        
        if self._common_config.seed is not None:
            util.seed(self._common_config.seed)
            
        if self._common_config.pretrained_path is not None:
            self._pretrained = torch.load(self._common_config.pretrained_path)
        else:
            if os.path.exists(f"{util.logger.dir()}/pretrained_models/best.pt"):
                self._pretrained = torch.load(f"{util.logger.dir()}/pretrained_models/best.pt")
    
        if "vocab_path" in self._env_config:
            vocab, max_str_len = util.load_vocab(self._env_config["vocab_path"])
            self._env_config["vocabulary"] = vocab
            self._env_config["max_str_len"] = max_str_len
        else:
            if os.path.exists(f"{util.logger.dir()}/vocab.json"):
                vocab, max_str_len = util.load_vocab(f"{util.logger.dir()}/vocab.json")
                self._env_config["vocabulary"] = vocab
                self._env_config["max_str_len"] = max_str_len
        util.logger.disable()
        
    def _create_env(self) -> Env:        
        env = make_async_chem_env(
            num_envs=self._common_config.num_envs,
            seed=self._common_config.seed,
            **{**self._env_config, **self._count_int_reward_config}
        )
        return env
    
    def _create_inference_env(self) -> Optional[Env]:
        if self._common_config.num_inference_envs == 0:
            return None
        
        env = make_async_chem_env(
            num_envs=self._common_config.num_inference_envs,
            seed=self._common_config.seed,
            **{**self._env_config}
        )
        return env
    
    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config["type"].lower()
        if agent_type == "ppo":
            return self._create_ppo_agent(env)
        elif agent_type == "rnd":
            return self._create_rnd_agent(env)
        elif agent_type == "pretrained":
            return self._create_pretrained_agent(env)
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
        
    def _create_pretrained_agent(self, env: Env) -> agent.PretrainedRecurrentAgent:
        assert env.obs_shape[0] == env.num_actions
        network = net.SelfiesPretrainedNet(
            env.num_actions,
        )
        if self._pretrained is not None:
            network.load_state_dict(self._pretrained["model"], strict=False)
        return agent.PretrainedRecurrentAgent(
            network=network,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )
        
class MolRLInferenceFactory:
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLInferenceFactory":
        """
        Create a `MolRLInferenceFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLInferenceFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config = self._config.get("Agent", dict())
        self._env_config = self._config.get("Env", dict())
        self._train_config = self._config.get("Train", dict())
        self._inference_config = self._config.get("Inference", dict())
        self._common_config = instance_from_dict(CommonConfig, self._train_config)
        self._pretrained = None
        
    def create_inference(self) -> Inference:
        self._inference_setup()
        
        try:
            env = self._create_env()
        except TypeError:
            raise ConfigParsingError("Invalid Env config. Missing arguments or wrong type.")
        try:
            agent = self._create_agent(env)
            agent = self._load_agent(agent)
            agent = agent.inference_agent(num_envs=env.num_envs, device=self._inference_config.get("device", self._common_config.device))
        except TypeError:
            raise ConfigParsingError("Invalid Agent config. Missing arguments or wrong type.")
        except FileNotFoundError as e:
            raise ConfigParsingError(str(e))
        try:    
            if "refset_path" in self._inference_config:
                smiles_or_selfies_refset = util.load_smiles_or_selfies(self._inference_config["refset_path"])
            elif "refset_path" in self._train_config:
                smiles_or_selfies_refset = util.load_smiles_or_selfies(self._train_config["refset_path"])
            else:
                smiles_or_selfies_refset = None
            inference = instance_from_dict(Inference, {
                "env": env,
                "agent": agent,
                "id": self._id,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
                **self._inference_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Train config. Missing arguments or wrong type.")
        return inference
        
    def _inference_setup(self):
        if "seed" in self._inference_config:
            util.seed(self._inference_config["seed"])
            
        if self._common_config.pretrained_path is not None:
            self._pretrained = torch.load(self._common_config.pretrained_path)
            
        util.logger.enable(self._id, enable_log_file=False)
        if "vocab_path" in self._env_config:
            vocab, max_str_len = util.load_vocab(self._env_config["vocab_path"])
            self._env_config["vocabulary"] = vocab
            self._env_config["max_str_len"] = max_str_len
        else:
            if os.path.exists(f"{util.logger.dir()}/vocab.json"):
                vocab, max_str_len = util.load_vocab(f"{util.logger.dir()}/vocab.json")
                self._env_config["vocabulary"] = vocab
                self._env_config["max_str_len"] = max_str_len
        util.logger.disable()
        
    def _create_env(self) -> Env:
        env = make_async_chem_env(
            num_envs=self._inference_config.get("num_envs", 1),
            seed=self._inference_config.get("seed", None),
            **{**self._env_config}
        )
        return env
    
    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config["type"].lower()
        if agent_type == "ppo":
            return self._create_ppo_agent(env)
        elif agent_type == "rnd":
            return self._create_rnd_agent(env)
        elif agent_type == "pretrained":
            return self._create_pretrained_agent(env)
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
        
    def _create_pretrained_agent(self, env: Env) -> agent.PretrainedRecurrentAgent:
        assert env.obs_shape[0] == env.num_actions
        network = net.SelfiesPretrainedNet(
            env.num_actions,
        )
        if self._pretrained is not None:
            network.load_state_dict(self._pretrained["model"], strict=False)
        return agent.PretrainedRecurrentAgent(
            network=network,
            num_envs=self._common_config.num_envs,
            device=self._common_config.device
        )
        
    def _load_agent(self, agent: agent.Agent) -> agent.Agent:
        ckpt = self._inference_config.get("ckpt", "best")
        util.logger.enable(self._id, enable_log_file=False)
        
        if ckpt == "best":
            ckpt_path = f"{util.logger.dir()}/best_agent.pt"
        elif ckpt == "final":
            ckpt_path = f"{util.logger.dir()}/agent.pt"
        elif isinstance(ckpt, int):
            ckpt_path = f"{util.logger.dir()}/agent_ckpt/agent_{ckpt}.pt"

        util.logger.disable()
        
        try:
            state_dict = torch.load(ckpt_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        agent.load_state_dict(state_dict["agent"])
        return agent
    
class MolRLPretrainFactory:
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLPretrainFactory":
        """
        Create a `MolRLPretrainFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLPretrainFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._pretrain_config = self._config.get("Pretrain", dict())

    def create_pretrain(self) -> Pretrain:
        self._pretrain_setup()
        
        dataset = SelfiesDataset.from_txt(self._pretrain_config["dataset_path"])
        net = SelfiesPretrainedNet(vocab_size=dataset.tokenizer.vocab_size)
        
        try:
            pretrain = instance_from_dict(Pretrain, {
                "id": self._id,
                "net": net,
                "dataset": dataset,
                **self._pretrain_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Pretrain config. Missing arguments or wrong type.")
        return pretrain
    
    def _pretrain_setup(self):
        util.logger.enable(self._id, enable_log_file=False)
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)
        self._log_dir = util.logger.dir()
        util.logger.disable()
        
        if "seed" in self._pretrain_config:
            util.seed(self._pretrain_config["seed"])