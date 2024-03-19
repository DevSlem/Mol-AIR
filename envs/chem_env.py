from dataclasses import dataclass, fields
from typing import Optional, Tuple, TypeVar, Generic, Callable, Union, List
from enum import Flag, auto
import warnings

import numpy as np
import rdkit
import selfies as sf
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tdc import Oracle

import envs.score_func as score_f
from envs.env import Env, EnvWrapper, env_config, AsyncEnv
from envs.selfies_util import is_finished
from envs.count_int_reward import CountIntReward
from envs.selfies_tokenizer import SelfiesTokenizer
from util import instance_from_dict

from drl.util import IncrementalMean


RDLogger.DisableLog('rdApp.*')

T = TypeVar("T")
    
@env_config(name="ChemEnv")
class ChemEnv(Env):
    @dataclass(frozen=True)
    class __Config:
        plogp_coef: float
        qed_coef: float
        similarity_coef: float
        gsk3b_coef: float
        jnk3_coef: float
        final_only: bool
        max_str_len: int
        
        @property
        def enabled_props(self) -> Tuple[str, ...]:
            props = []
            for field in fields(self):
                if field.name.endswith("_coef") and getattr(self, field.name) > 0.0:
                    props.append(field.name[:-5])
            return tuple(props)
    
    @property
    def enabled_props(self) -> Tuple[str, ...]:
        props = []
        for field in fields(self):
            if field.name.endswith("_coef") and getattr(self, field.name) > 0.0:
                props.append(field.name[:-5])
        return tuple(props)
    
    class TerminalCondition(Flag):
        NONE = 0
        INVALID = auto()
        STOP_TOKEN = auto()
        UNKNOWN = auto()
        MAX_LEN = auto()
        
    @dataclass
    class MoleculeProperty(Generic[T]):
        plogp: T
        qed: T
        similarity: T
        gsk3b: T
        jnk3: T
        
        @staticmethod
        def new(default: Union[T, Callable[[], T]]) -> "ChemEnv.MoleculeProperty[T]":
            field_dict = dict()
            for field in fields(ChemEnv.MoleculeProperty):
                field_dict[field.name] = default() if callable(default) else default
            return ChemEnv.MoleculeProperty(**field_dict)
        
        @staticmethod
        def none(default: Union[T, Callable[[], T], None] = None) -> "ChemEnv.MoleculeProperty[Optional[T]]":
            field_dict = dict()
            for field in fields(ChemEnv.MoleculeProperty):
                field_dict[field.name] = default() if callable(default) else default
            return ChemEnv.MoleculeProperty(**field_dict)
        
    _PROP_NAMES = MoleculeProperty(
        plogp="pLogP",
        qed="QED",
        similarity="Similarity",
        gsk3b="GSK3B",
        jnk3="JNK3",
    )
    
    @classmethod
    def format_prop_name(cls, prop_name: str) -> str:
        return getattr(cls._PROP_NAMES, prop_name.lower())
    
    @classmethod
    def prop_names(cls) -> Tuple[str, ...]:
        return tuple(getattr(cls._PROP_NAMES, field.name) for field in fields(cls._PROP_NAMES))
    
    def __init__(
        self, 
        plogp_coef: float = 0.0,
        qed_coef: float = 0.0,
        similarity_coef: float = 0.0,
        gsk3b_coef: float = 0.0,
        jnk3_coef: float = 0.0,
        final_only: bool = False,
        max_str_len: int = 35,
        seed: Optional[int] = None,
        env_id: Optional[int] = None
    ) -> None:
        self._config = ChemEnv.__Config(
            plogp_coef=plogp_coef,
            qed_coef=qed_coef,
            similarity_coef=similarity_coef,
            gsk3b_coef=gsk3b_coef,
            jnk3_coef=jnk3_coef,
            final_only=final_only,
            max_str_len=max_str_len
        )
        self._np_rng = np.random.default_rng(seed=seed)
        self._env_id = env_id
        
        self._selfies_list = []
        self._tokenizer = SelfiesTokenizer()
        
        self._episode = -1
        
        self._fp1 = AllChem.GetMorganFingerprint(AllChem.MolFromSmiles('CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'), 2)
        if self._config.gsk3b_coef > 0.0:
            self._calc_gsk3b = Oracle(name='GSK3B')
        if self._config.jnk3_coef > 0.0:
            self._calc_jnk3 = Oracle(name='JNK3')
            
        self._prop_keys = self._config.enabled_props
        
        self.obs_shape = (self._config.max_str_len,)
        self.num_actions = self._tokenizer.n_tokens
            
    def reset(self) -> np.ndarray:
        self._time_step = -1
        self._episode += 1
        
        self._init_selfies_idxes = self._make_init_selfies_idxes()
        # state initialization, (max_str_len,) shaped array
        self._encoded_selfies = self._tokenizer.encode("", seq_len=self._config.max_str_len)
        self._current_smiles = ""
        self._current_mol = None
        
        self._prev_score = 0.0
        self._current_prop = ChemEnv.MoleculeProperty[float].none()
        self._current_total_prop = None # score (weighted sum of properties)
        
        return self._encoded_selfies[np.newaxis, ...]
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        # increment time step
        self._time_step += 1
        
        # update state
        a = int(action.item())
        self._update_state(a)
        
        # check termination
        terminal_cond = self._check_terminal_condition(a)
        
        # calcaulte reward
        reward, take_penalty = self._calc_reward(terminal_cond)
        
        # convert to ndarray
        terminated = np.array([terminal_cond != ChemEnv.TerminalCondition.NONE])
        reward = np.array([reward], dtype=np.float64)
        
        next_obs = self._encoded_selfies[np.newaxis, ...]
        real_final_next_obs = next_obs[terminated]
        
        info = dict()
        
        # if terminated, next_obs is reset to the first observation of the next episode
        if terminated:
            info["metric"] = self._episode_metric_info_dict()
            info["valid_termination"] = not take_penalty
            next_obs = self.reset()
        
        return next_obs, reward, terminated, real_final_next_obs, info
    
    def close(self):
        pass
    
    def _make_init_selfies_idxes(self) -> List[int]:
        if (
            self._config.similarity_coef > 0.0 # Similarity
        ):
            return self._tokenizer.encode(["[C]"])[0].tolist()
        if (
            self._config.gsk3b_coef > 0.0 # GSK3B
            or self._config.jnk3_coef > 0.0 # JNK3
        ):
            selfies = self._np_rng.choice(['[C][C][C]', '[C][=C][C]', '[C][C][=N]', '[C][N][C]', '[C][O][C]'])
            return self._tokenizer.encode([selfies])[0].tolist()
        
        return list()
    
    def _update_state(self, action: int):
        if self._time_step < len(self._init_selfies_idxes):
            # append initial selfies_idxes regardless of the action
            self._encoded_selfies[self._time_step] = self._init_selfies_idxes[self._time_step]
        else:
            self._encoded_selfies[self._time_step] = action
        self._current_smiles = self._decode_smiles()
        self._current_mol = AllChem.MolFromSmiles(self._current_smiles)
        
    def _check_terminal_condition(self, action: int) -> TerminalCondition:
        # check termination
        terminal_cond = ChemEnv.TerminalCondition.NONE
        if action == self._tokenizer.stop_token_val:
            terminal_cond |= ChemEnv.TerminalCondition.STOP_TOKEN
        if self._current_mol is None:
            terminal_cond |= ChemEnv.TerminalCondition.INVALID
        if terminal_cond == is_finished(self._tokenizer.decode(self._encoded_selfies, include_stop_token=False)):
            terminal_cond |= ChemEnv.TerminalCondition.UNKNOWN
        if self._time_step == self._config.max_str_len - 1:
            terminal_cond |= ChemEnv.TerminalCondition.MAX_LEN
        return terminal_cond
    
    def _calc_reward(self, terminal_cond: TerminalCondition) -> Tuple[float, bool]:
        """Calculate a reward of the current state."""
        # if the terminal condition is one of the following cases,
        # case 1: the episode is terminated at the initial time step
        # case 2: the molecule is invalid
        # then take penalty
        take_penalty = (self._time_step == 0 and terminal_cond != ChemEnv.TerminalCondition.NONE) or (ChemEnv.TerminalCondition.INVALID in terminal_cond)   
        terminated = terminal_cond != ChemEnv.TerminalCondition.NONE
        
        if take_penalty:
            self._current_prop = ChemEnv.MoleculeProperty[float].none()
            self._current_total_prop = None
            return -1.0, True
        
        reward = 0.0
        
        # calculate the reward only if the episode is terminated
        if self._config.final_only and terminated:
            reward = self._calc_current_score()
        else:
            current_score = self._calc_current_score()
            reward = current_score - self._prev_score # delta reward
            self._prev_score = current_score
            
        return reward, False
    
    def _calc_current_score(self) -> float:
        score = 0.0
        
        # pLogP
        if self._config.plogp_coef > 0.0:
            self._current_prop.plogp = score_f.calculate_pLogP(self._current_smiles)
            score += self._config.plogp_coef * (max(self._current_prop.plogp, -10) / 10)
            
        # QED
        if self._config.qed_coef > 0.0:
            self._current_prop.qed = rdkit.Chem.QED.qed(self._current_mol) # type: ignore
            score += self._config.qed_coef * self._current_prop.qed
            
        # Similarity
        if self._config.similarity_coef > 0.0:
            fp2 = AllChem.GetMorganFingerprint(self._current_mol, 2) # type: ignore
            self._current_prop.similarity = TanimotoSimilarity(self._fp1, fp2)
            score += self._config.similarity_coef * self._current_prop.similarity
            
        warnings.filterwarnings("ignore")
        
        # GSK3B
        if self._config.gsk3b_coef > 0.0:            
            self._current_prop.gsk3b = self._calc_gsk3b([self._current_smiles,])[0] # type: ignore
            score += self._config.gsk3b_coef * self._current_prop.gsk3b
        
        # JNK3
        if self._config.jnk3_coef > 0.0:
            self._current_prop.jnk3 = self._calc_jnk3([self._current_smiles,])[0] # type: ignore
            score += self._config.jnk3_coef * self._current_prop.jnk3
            
        warnings.filterwarnings("default")
        
        self._current_total_prop = score
        
        return score
        
    def _decode_smiles(self) -> str:
        return sf.decoder(
            self._tokenizer.decode(self._encoded_selfies, include_stop_token=False)
        ) # type: ignore
        
    def _episode_metric_info_dict(self) -> dict:
        metric_info_dict = dict()
        # keys
        metric_info_dict["keys"] = dict()
        metric_info_dict["keys"]["episode"] = self._episode
        if self._env_id is not None:
            metric_info_dict["keys"]["env_id"] = self._env_id
        # values
        metric_info_dict["values"] = dict()
        metric_info_dict["values"]["score"] = self._current_total_prop
        for prop_name in self._prop_keys:
            metric_info_dict["values"][prop_name] = getattr(self._current_prop, prop_name)
        metric_info_dict["values"]["selfies"] = self._tokenizer.decode(self._encoded_selfies)
        return {"episode_metric": metric_info_dict}
    

class ChemEnvWrapper(EnvWrapper):
    def __init__(self, env: ChemEnv, count_int_reward: CountIntReward, crwd_coef: float = 0.0) -> None:
        super().__init__(env)
        
        self._count_int_reward = count_int_reward
        self._crwd_coef = crwd_coef
        self._tokenizer = SelfiesTokenizer()
        
        self._avg_count_reward = IncrementalMean()
        
    def reset(self) -> np.ndarray:
        obs = super().reset()
        self._avg_count_reward.reset()
        # in this case, the one-hot vector is a zero vector
        return self._last_token_one_hot(obs)
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        next_obs, reward, terminated, real_final_next_obs, info = super().step(action)
        
        if self._crwd_coef != 0.0:
            count_reward = 0.0
            # if either the episode is not terminated or the episode is terminated with a valid molecule,
            # then add the count-based intrinsic reward
            if info.get("valid_termination", True):
                encoded_sequences = next_obs[0] if not terminated else real_final_next_obs[0]
                smiles = sf.decoder(
                    self._tokenizer.decode(encoded_sequences, include_stop_token=False) # integer sequence -> selfies
                ) # selfies -> smiles
                count_reward = self._count_int_reward(smiles) # type: ignore
                
            reward += self._crwd_coef * count_reward
            self._avg_count_reward.update(count_reward)
                
            if "metric" in info and "episode_metric" in info["metric"]:
                info["metric"]["episode_metric"]["values"]["avg_count_int_reward"] = self._avg_count_reward.mean
                self._avg_count_reward.reset()
        
        # convert to one-hot vector of only the last token
        next_obs = self._last_token_one_hot(next_obs)
        real_final_next_obs = self._last_token_one_hot(real_final_next_obs)
        
        return next_obs, reward, terminated, real_final_next_obs, info
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return (self._tokenizer.n_tokens,)
    
    def _last_token_one_hot(self, obs: np.ndarray) -> np.ndarray:
        """from the integer sequence to the one-hot of the last token"""
        last_token_val = self._tokenizer.last_token_value(obs)
        return self._tokenizer.to_one_hot(last_token_val)
    
    
def make_async_chem_env(
    num_envs: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> AsyncEnv:
    def env_make_func(seed=None, env_id=None) -> Env:
        config = kwargs.copy()
        config["seed"] = seed
        config["env_id"] = env_id
        env = instance_from_dict(ChemEnv, config)
        
        config["np_rng"] = env._np_rng
        count_int_reward = instance_from_dict(CountIntReward, config)
        return ChemEnvWrapper(env, count_int_reward, config.get("crwd_coef", 0.0))
    
    if seed is not None:
        np_rng = np.random.default_rng(seed=seed)
        seeds = np_rng.integers(0, np.iinfo(np.int32).max, size=num_envs)
        env_make_func_list = [lambda seed=seed, env_id=i: env_make_func(seed=seed, env_id=env_id) for i, seed in enumerate(seeds)]
    else:
        env_make_func_list = [lambda env_id=i: env_make_func(env_id=i) for i in range(num_envs)]
    
    return AsyncEnv(env_make_func_list, trace_env=0)
    