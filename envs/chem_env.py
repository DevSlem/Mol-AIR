import csv
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from enum import Flag, auto
from queue import Queue
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import rdkit
import selfies as sf
import torch
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tdc import Oracle
from torch.nn.functional import one_hot

import envs.score_func as score_f
from drl.util import IncrementalMean
from envs.env import AsyncEnv, Env
from envs.selfies_util import is_finished
from envs.util import SizeLimitedDict

RDLogger.DisableLog('rdApp.*')

T = TypeVar("T")
    
@dataclass(frozen=True)
class ChemExtrinsicRewardConfig:
    plogp_coef: float = 0.0
    qed_coef: float = 0.0
    similarity_coef: float = 0.0
    gsk3b_coef: float = 0.0
    drd2_coef: float = 0.0
    jnk3_coef: float = 0.0
    final_only: bool = False
    sparse_reward_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    
    @property
    def enabled_props(self) -> Tuple[str, ...]:
        props = []
        for field in fields(self):
            if field.name.endswith("_coef") and getattr(self, field.name) > 0.0:
                props.append(field.name[:-5])
        return tuple(props)
    
    def to_sparse_reward(self, reward: float) -> float:
        if self.sparse_reward_range is None:
            return reward
        min_reward, max_reward = self.sparse_reward_range
        if min_reward is not None and reward < min_reward:
            return 0.0
        if max_reward is not None and reward > max_reward:
            return 0.0
        return reward
    
@dataclass(frozen=True)
class ChemIntrinsicRewardConfig:
    count_coef: float = 0.0
    memory_coef: float = 0.0
    memory_size: int = 1000
    fingerprint_bits: int = 256
    fingerprint_radius: int = 2
    lsh_bits: int = 32
    
class SelfiesActionSpace:
    def __init__(self) -> None:
        self._stop_symbol = "[STOP]"
        self._selfies_symbols = ["[#N]"] + sf.selfies_alphabet() + [self._stop_symbol]
        self._selfies_symbol_to_idx_dict = {symbole: i for i, symbole in enumerate(self._selfies_symbols)}
    
    def to_symbol(self, action: int) -> str:
        return self._selfies_symbols[action]
    
    def to_action(self, symbol: str) -> int:
        return self._selfies_symbol_to_idx_dict[symbol]
    
    def to_actions(self, selfies: str) -> Tuple[int, ...]:
        symbols = tuple('[' + character for character in selfies.split('['))[1:]
        return tuple(self.to_action(symbol) for symbol in symbols)
    
    def is_stop_action(self, action: int) -> bool:
        return self._selfies_symbols[action] == self._stop_symbol
    
    @property
    def num_actions(self) -> int:
        return len(self._selfies_symbols)

class ChemEnv(Env):
    """
    Single Chemical Environment. 
    The number of environments 1.
    
    `*obs_shape` = `(num_selfies_tokens,)`
    
    `info`: episode, score, property1, property2, ..., selfies
    """
    
    class TerminalCondition(Flag):
        NONE = 0
        INVALID = auto()
        STOP_TOKEN = auto()
        UNKNOWN = auto()
        MAX_LEN = auto()
        
    @dataclass(frozen=True)
    class BestMolecule:
        episode: int
        prop: float
        selfies: str
        
    @dataclass
    class MoleculeProperty(Generic[T]):
        plogp: T
        qed: T
        similarity: T
        gsk3b: T
        drd2: T
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
        drd2="DRD2",
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
        ext_reward_config: ChemExtrinsicRewardConfig,
        int_reward_config: ChemIntrinsicRewardConfig,
        max_str_len: int = 35, 
        record_data: bool = False,
        seed: Optional[int] = None,
        env_id: Optional[int] = None,
    ) -> None:        
        self._ext_reward_config = ext_reward_config
        self._int_reward_config = int_reward_config
        self._max_str_len = max_str_len
        self._is_record_data = record_data
        self._seed = seed
        self._np_rng = np.random.default_rng(seed=self._seed)
        self._env_id = env_id
        
        self._selifes_list = []
        self._action_space = SelfiesActionSpace()
        
        # self._molecule_prop_coef = ChemEnv.MoleculeProperty(**{prop_name.name: getattr(self._ext_reward_config, prop_name.name + "_coef") for prop_name in fields(ChemEnv.MoleculeProperty)})
        # self._final_molecule_prop_mean = ChemEnv.MoleculeProperty.new(lambda: IncrementalMean())
        self._final_molecule_total_prop_mean = IncrementalMean()
        
        self._best_final_molecule_prop = ChemEnv.MoleculeProperty[float].none()
        self._best_final_molecule_prop_selfies = ChemEnv.MoleculeProperty[str].none()
        self._episode = -1
        
        self._int_reward_enabled = self._int_reward_config.count_coef > 0.0 or self._int_reward_config.memory_coef > 0.0
                
        self._morgan_fp_counter = Counter()
        self._lsh_rand_matrix = self._np_rng.normal(size=(self._int_reward_config.lsh_bits, self._int_reward_config.fingerprint_bits))
        self._morgan_fp_unique_buffer = SizeLimitedDict(max_size=self._int_reward_config.memory_size)
        
        self._fp1 = AllChem.GetMorganFingerprint(AllChem.MolFromSmiles('CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'), 2)
        
        if self._ext_reward_config.drd2_coef > 0.0:
            self._calc_drd2 = Oracle(name='DRD2')
        if self._ext_reward_config.gsk3b_coef > 0.0:
            self._calc_gsk3b = Oracle(name='GSK3B')
        if self._ext_reward_config.jnk3_coef > 0.0:
            self._calc_jnk3 = Oracle(name='JNK3')
        
        # get what properties are used
        prop_keys = []
        for config_field in fields(self._ext_reward_config):
            config_name = config_field.name
            if config_name.endswith("_coef") and getattr(self._ext_reward_config, config_name) > 0.0:
                prop_name = config_name[:-5]
                prop_keys.append(prop_name)
        self._prop_keys = self._ext_reward_config.enabled_props
        
        self._avg_count_reward = IncrementalMean()
        self._avg_memory_reward = IncrementalMean()
            
    def reset(self) -> torch.Tensor:
        self._selfies_idxes = [] # state
        self._time_step = -1
        self._episode += 1
        self._prev_score = 0.0
        self._init_selfies_idxes = self._make_init_selfies_idxes()
        self._current_prop = ChemEnv.MoleculeProperty[float].none()
        self._current_total_prop = None # score (weighted sum of properties)
        self._current_smiles = ""
        self._current_mol = None
        self._avg_count_reward.reset()
        self._avg_memory_reward.reset()
        # initial observation is zero vector (1, *obs_shape)
        return torch.zeros(1, self.num_actions, dtype=torch.float32)
    
    def step(
        self, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # increment time step
        self._time_step += 1
        
        # update state
        a = int(action.item())
        self._update_state(a)
        
        # check termination
        terminal_cond = self._check_terminal_condition(a)
                
        # calcaulte reward
        reward = self._calc_reward(terminal_cond)
            
        # convert to tensor
        terminated = torch.tensor([terminal_cond != ChemEnv.TerminalCondition.NONE])
        reward = torch.tensor([[reward]])
        
        next_obs = self._state_to_obs()
        # if not terminated, the shape of real_final_next_obs is (0, *obs_shape)
        real_final_next_obs = next_obs[terminated]
        
        info = dict()
        
        # if terminated, next_obs is reset to the first observation of the next episode
        if terminated:
            # record
            info.update(self._final_info_dict())
            info["metric"] = self._episode_metric_info_dict()
            # if self._is_record_data:
            #     self._record_data()
            next_obs = self.reset()
        terminated = terminated.unsqueeze(dim=-1)
        
        return next_obs, reward, terminated, real_final_next_obs, info
    
    def close(self):
        pass
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return (self.num_actions,)
    
    @property
    def num_actions(self) -> int:
        return self._action_space.num_actions
    
    @property
    def num_envs(self) -> int:
        return 1        
    
    # @property
    # def log_data(self) -> Dict[str, Tuple[float, Optional[float]]]:
    #     ld = dict()
        
    #     if self._final_molecule_total_prop_mean.count > 0:
    #         ld["Environment/Final Molecule Property"] = (self._final_molecule_total_prop_mean.mean, None)
    #         self._final_molecule_total_prop_mean.reset()
        
    #     if self._best_final_molecule_follower is not None:
    #         ld["Environment/Best Final Molecule Property"] = (self._best_final_molecule_follower.item.prop, None)
        
    #     # for prop_name in fields(self._final_molecule_prop_mean):
    #     #     prop_name = prop_name.name
    #     #     prop_mean = getattr(self._final_molecule_prop_mean, prop_name)
    #     #     if prop_mean.count > 0:
    #     #         ld[f"Environment/Final Molecule {getattr(self._PROP_NAMES, prop_name)}"] = (prop_mean.mean, None)
    #     #         prop_mean.reset()
    #     for prop_name in fields(self._best_final_molecule_prop):
    #         prop_name = prop_name.name
    #         best_prop = getattr(self._best_final_molecule_prop, prop_name)
    #         if best_prop is not None:
    #             ld[f"Environment/Best Final Molecule {getattr(self._PROP_NAMES, prop_name)}"] = (best_prop, None)
    #     return ld
    
    @property
    def config_dict(self) -> dict:
        cd = {}
        cd["observation shape"] = self.obs_shape
        cd["number of actions"] = self.num_actions
        cd["max str len"] = self._max_str_len
        
        for prop_key in self._prop_keys:
            ext_reward_coef = getattr(self._ext_reward_config, f"{prop_key}_coef")
            if ext_reward_coef > 0.0:
                cd[f"{getattr(self._PROP_NAMES, prop_key)} coef"] = ext_reward_coef
        cd["final-only extrinsic reward"] = self._ext_reward_config.final_only
        if self._ext_reward_config.sparse_reward_range is not None:
            if self._ext_reward_config.sparse_reward_range[0] is not None:
                cd["sparse reward min"] = self._ext_reward_config.sparse_reward_range[0]
            if self._ext_reward_config.sparse_reward_range[1] is not None:
                cd["sparse reward max"] = self._ext_reward_config.sparse_reward_range[1]
        
        if self._int_reward_config.count_coef > 0.0:
            cd["count coef"] = self._int_reward_config.count_coef
        if self._int_reward_config.memory_coef > 0.0:
            cd["memory coef"] = self._int_reward_config.memory_coef
            cd["memory size"] = self._int_reward_config.memory_size
        if self._int_reward_config.count_coef > 0.0 or self._int_reward_config.memory_coef > 0.0:
            cd["intrinsic reward type"] = "independent"
        return cd
    
    # def save_data(self, base_dir: str):
    #     if not self._is_record_data:
    #         return
        
    #     # record generated molecules until now
    #     with open(f"{base_dir}/selfies.txt", "a") as f:
    #         for selfies in self._selifes_list:
    #             f.write(f"{selfies}\n")
    #     self._selifes_list.clear()
        
    #     if self._best_final_molecule_follower is None:
    #         return
        
    #     # record best molecule
    #     prop_names = []
    #     best_prop_values = []
    #     for prop_name in fields(self._best_final_molecule_prop):
    #         prop_name = prop_name.name
    #         best_prop = getattr(self._best_final_molecule_prop, prop_name)
    #         if best_prop is not None:
    #             prop_names.append(getattr(self._PROP_NAMES, prop_name))
    #             best_prop_values.append(best_prop)
                
        
    #         prop_names = "+".join(prop_names)
    #         best_prop_value = self._best_final_molecule_total_prop
    #         best_selfies = self._best_final_molecule_total_prop_selfies
    #         best_molecule_info = f"{prop_names}, "
        
    #     best_selfies = ""
    #     for prop_name in fields(self._best_final_molecule_prop):
    #         prop_name = prop_name.name
    #         best_prop = getattr(self._best_final_molecule_prop, prop_name)
    #         if best_prop is not None:
    #             best_selfies += f"{getattr(self._PROP_NAMES, prop_name)}, {best_prop}, {getattr(self._best_final_molecule_prop_selfies, prop_name)}\n"
    #     with open(f"{base_dir}/best_selfies.txt", "w") as f:
    #         f.write(best_selfies)
         
    def _make_init_selfies_idxes(self) -> Tuple[int, ...]:
        if (
            self._ext_reward_config.similarity_coef > 0.0 # Similarity
        ):
            return (self._action_space.to_action('[C]'),)
        if (
            self._ext_reward_config.drd2_coef > 0.0 # DRD2
            or self._ext_reward_config.gsk3b_coef > 0.0 # GSK3B
            or self._ext_reward_config.jnk3_coef > 0.0 # JNK3
        ):
            selfies = self._np_rng.choice(['[C][C][C]', '[C][=C][C]', '[C][C][=N]', '[C][N][C]', '[C][O][C]'])
            return self._action_space.to_actions(selfies)
        
        return tuple()
            
    def _update_state(self, action: int):
        if self._time_step < len(self._init_selfies_idxes):
            # initialize selfies_idxes
            self._selfies_idxes.append(self._init_selfies_idxes[self._time_step])
        else:
            self._selfies_idxes.append(action)
        self._current_smiles = self._make_smiles()
        self._current_mol = AllChem.MolFromSmiles(self._current_smiles)
        
    def _check_terminal_condition(self, action: int) -> TerminalCondition:
        # check termination
        terminal_cond = ChemEnv.TerminalCondition.NONE
        if self._action_space.is_stop_action(action):
            terminal_cond |= ChemEnv.TerminalCondition.STOP_TOKEN
        if self._current_mol is None:
            terminal_cond |= ChemEnv.TerminalCondition.INVALID
        if terminal_cond == is_finished(self._make_selfies()):
            terminal_cond |= ChemEnv.TerminalCondition.UNKNOWN
        if self._time_step == self._max_str_len - 1:
            terminal_cond |= ChemEnv.TerminalCondition.MAX_LEN
        return terminal_cond
        
    def _make_selfies(self, include_stop_token: bool = False) -> str:
        selfies = "".join([self._action_space.to_symbol(selfies_idx) for selfies_idx in self._selfies_idxes[:-1]])
        last_token_idx = self._selfies_idxes[-1]
        if include_stop_token or not self._action_space.is_stop_action(last_token_idx):
            selfies += self._action_space.to_symbol(last_token_idx)
        return selfies
    
    def _make_smiles(self) -> str:
        return sf.decoder(self._make_selfies()) # type: ignore
    
    def _state_to_obs(self) -> torch.Tensor:
        """Returns observation tensor `(1, *obs_shape)` from the current state."""
        return one_hot(
            torch.tensor([self._selfies_idxes[-1]]),
            num_classes=self.num_actions
        ).to(dtype=torch.float32)
    
    def _calc_reward(self, terminal_cond: TerminalCondition) -> float:
        """Calculate a reward of the current state."""
        # if the terminal condition is one of the following cases,
        # case 1: an episode is terminated at the initial time step
        # case 2: a molecule is invalid
        # then take penalty
        is_penalty = (self._time_step == 0 and terminal_cond != ChemEnv.TerminalCondition.NONE) or (ChemEnv.TerminalCondition.INVALID in terminal_cond)   
        terminated = terminal_cond != ChemEnv.TerminalCondition.NONE
        
        ext_reward = self._calc_ext_reward(terminated, is_penalty)
        int_reward = self._calc_int_reward(terminated, is_penalty)

        return ext_reward + int_reward
        
    def _calc_ext_reward(self, terminated: bool, is_penalty: bool) -> float:
        if is_penalty:
            self._current_prop = ChemEnv.MoleculeProperty[float].none()
            self._current_total_prop = None
            return -1.0
        
        reward = 0.0
        
        # calculate a reward only if an episode is terminated
        if self._ext_reward_config.final_only and terminated:
            reward = self._calc_current_score()
        # calculate a reward at every step
        else:
            current_score = self._calc_current_score() # real current reward
            reward = current_score - self._prev_score # delta reward
            self._prev_score = current_score
        
        return self._ext_reward_config.to_sparse_reward(reward)
            
    def _calc_current_score(self) -> float:
        current_reward = 0.0
        
        # pLogP
        if self._ext_reward_config.plogp_coef > 0.0:
            self._current_prop.plogp = score_f.calculate_pLogP(self._current_smiles)
            current_reward += self._ext_reward_config.plogp_coef * max(self._current_prop.plogp, -10) / 10
            
        # QED
        if self._ext_reward_config.qed_coef > 0.0:
            self._current_prop.qed = rdkit.Chem.QED.qed(self._current_mol) # type: ignore
            current_reward += self._ext_reward_config.qed_coef * self._current_prop.qed
            
        # Similarity
        if self._ext_reward_config.similarity_coef > 0.0:
            fp2 = AllChem.GetMorganFingerprint(self._current_mol, 2) # type: ignore
            self._current_prop.similarity = TanimotoSimilarity(self._fp1, fp2)
            current_reward += self._ext_reward_config.similarity_coef * self._current_prop.similarity
            
        warnings.filterwarnings("ignore")
        
        # GSK3B
        if self._ext_reward_config.gsk3b_coef > 0.0:            
            self._current_prop.gsk3b = self._calc_gsk3b([self._current_smiles,])[0] # type: ignore
            current_reward += self._ext_reward_config.gsk3b_coef * self._current_prop.gsk3b
            
        # DRD2
        if self._ext_reward_config.drd2_coef > 0.0:
            self._current_prop.drd2 = self._calc_drd2([self._current_smiles,])[0] # type: ignore
            current_reward += self._ext_reward_config.drd2_coef * self._current_prop.drd2    
            
        # JNK3
        if self._ext_reward_config.jnk3_coef > 0.0:
            self._current_prop.jnk3 = self._calc_jnk3([self._current_smiles,])[0] # type: ignore
            current_reward += self._ext_reward_config.jnk3_coef * self._current_prop.jnk3
            
        warnings.filterwarnings("default")
        
        self._current_total_prop = current_reward
                
        return current_reward
        
    def _calc_int_reward(self, terminated: bool, is_penalty: bool) -> float:
        reward = 0.0

        # count-based
        if self._int_reward_config.count_coef > 0.0:
            if is_penalty:
                count_reward = 0.0
            else:
                mol_count = self._calc_mol_count(self._current_smiles)
                # if self._int_reward_config.count_scale_type == "sqrt":
                #     count_reward = 1.0 / math.sqrt(mol_count)
                # elif self._int_reward_config.count_scale_type == "linear":
                #     count_reward = 1.0 / mol_count
                # elif self._int_reward_config.count_scale_type == "quadratic":
                #     count_reward = 1.0 / (mol_count ** 2)
                # else:
                #     raise ValueError(f"Invalid scale type: {self._int_reward_config.count_scale_type}")
                mol_count = min(mol_count - 1, 10)
                count_reward = math.exp(-mol_count)
                
            reward += self._int_reward_config.count_coef * count_reward
            self._avg_count_reward.update(count_reward)
        
        # memory-based
        if self._int_reward_config.memory_coef > 0.0:
            if is_penalty:
                memory_reward = -1.0
            else:
                max_ts = self._calc_max_tanimoto_similarity(self._current_smiles, terminated)
                memory_reward = -max_ts
            reward += self._int_reward_config.memory_coef * memory_reward
            self._avg_memory_reward.update(memory_reward)
        
        return reward
    
    def _lsh(self, array: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(self._lsh_rand_matrix, array))
    
    def _calc_mol_count(self, smiles: str) -> int:
        fp = self._get_fingerprint_bits(smiles)
        if type(fp) is not int:
            fp = self._lsh(fp).tobytes()
        self._morgan_fp_counter.update((fp,))
        return self._morgan_fp_counter[fp]
    
    def _calc_max_tanimoto_similarity(self, smiles: str, terminated: bool) -> float:
        new_mol = AllChem.MolFromSmiles(smiles)
        new_morgan_fp = AllChem.GetMorganFingerprint(new_mol, 2)
        ts_list = [0.0]
        for molgan_fp in self._morgan_fp_unique_buffer.values():
            ts = TanimotoSimilarity(new_morgan_fp, molgan_fp)
            ts_list.append(ts)
        
        if terminated:
            new_canonical_smiles = AllChem.MolToSmiles(new_mol)
            self._morgan_fp_unique_buffer[new_canonical_smiles] = new_morgan_fp
        return max(ts_list)
    
    def _get_fingerprint_bits(self, smiles: str) -> Union[np.ndarray, int]:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                self._int_reward_config.fingerprint_radius,
                nBits=self._int_reward_config.fingerprint_bits
            )
            x = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, x)
            return x
        else:
            return -1
                
    def _final_info_dict(self) -> dict:
        final_molecule_dict = dict()
        final_molecule_dict["episode"] = self._episode
        final_molecule_dict["score"] = self._current_total_prop
        for prop_name in self._prop_keys:
            final_molecule_dict[prop_name] = getattr(self._current_prop, prop_name)
        final_molecule_dict["selfies"] = self._make_selfies(include_stop_token=True)
        final_molecule_dict["intrinsic_reward"] = dict()
        if self._int_reward_config.count_coef > 0.0:
            final_molecule_dict["intrinsic_reward"]["count"] = self._avg_count_reward.mean
        if self._int_reward_config.memory_coef > 0.0:
            final_molecule_dict["intrinsic_reward"]["memory"] = self._avg_memory_reward.mean
        return {"final_molecule": final_molecule_dict}
    
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
        metric_info_dict["values"]["selfies"] = self._make_selfies(include_stop_token=True)
        if self._int_reward_config.count_coef > 0.0:
            metric_info_dict["values"]["avg_count_int_reward"] = self._avg_count_reward.mean
        if self._int_reward_config.memory_coef > 0.0:
            metric_info_dict["values"]["avg_memory_int_reward"] = self._avg_memory_reward.mean
        return {"episode_metric": metric_info_dict}
    
class AsyncChemEnv(AsyncEnv):
    def __init__(
        self,
        ext_reward_config: ChemExtrinsicRewardConfig,
        int_reward_config: ChemIntrinsicRewardConfig,
        num_envs: int = 1,
        max_str_len: int = 35, 
        seed: Optional[int] = None,
    ) -> None:
        def env_make_func(seed=None, env_id=None) -> ChemEnv:
            return ChemEnv(
                ext_reward_config=ext_reward_config,
                int_reward_config=int_reward_config,
                max_str_len=max_str_len, 
                seed=seed,
                env_id=env_id,
            )
            
        if seed is not None:
            np_rng = np.random.default_rng(seed=seed)
            seeds = np_rng.integers(0, np.iinfo(np.int32).max, size=num_envs)
            env_make_func_list = [lambda seed=seed, env_id=i: env_make_func(seed=seed, env_id=env_id) for i, seed in enumerate(seeds)]
        else:
            env_make_func_list = [lambda env_id=i: env_make_func(env_id=i) for i in range(num_envs)]
        
        super().__init__(env_make_func_list, trace_env=0)
        
        self._prop_keys = ext_reward_config.enabled_props
        
class AsyncChemEnvDependentIntrinsicReward(AsyncChemEnv):
    def __init__(
        self, 
        ext_reward_config: ChemExtrinsicRewardConfig,
        int_reward_config: ChemIntrinsicRewardConfig,
        num_envs: int = 1, 
        max_str_len: int = 35, 
    ) -> None:
        super().__init__(ext_reward_config, ChemIntrinsicRewardConfig(), num_envs, max_str_len)
        
        self._max_str_len = max_str_len
        self._action_space = SelfiesActionSpace()
        
        self._dependent_int_reward_config = int_reward_config
        self._morgan_fp_counter = Counter()
        self._lsh_rand_matrix = np.random.randn(self._dependent_int_reward_config.lsh_bits, self._dependent_int_reward_config.fingerprint_bits)
        
    def reset(self) -> torch.Tensor:
        self._selfies_idxes = self._init_selfies_idexes()
        self._steps = self._init_steps()
        return super().reset()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_obs, reward, terminated, real_final_next_obs = super().step(action)
        
        selfies_update_mask = np.eye(self._max_str_len, dtype=bool)[self._steps]
        self._selfies_idxes[selfies_update_mask] = action.squeeze(dim=-1).numpy()
        
        # update reward
        int_reward = self._calc_int_reward().unsqueeze(dim=-1)
        reward += int_reward
        
        # next step
        self._steps += 1
        
        # reset only terminated envs
        T = terminated.squeeze(dim=-1).numpy()
        self._selfies_idxes[T] = self._init_selfies_idexes()[T]
        self._steps[T] = self._init_steps()[T]
        
        return next_obs, reward, terminated, real_final_next_obs
    
    @property
    def config_dict(self) -> dict:
        cd = super().config_dict
        if self._dependent_int_reward_config.count_coef > 0.0:
            cd["count coef"] = self._dependent_int_reward_config.count_coef
        if self._dependent_int_reward_config.memory_coef > 0.0:
            cd["memory coef"] = self._dependent_int_reward_config.memory_coef
            cd["memory size"] = self._dependent_int_reward_config.memory_size
        if self._dependent_int_reward_config.count_coef > 0.0 or self._dependent_int_reward_config.memory_coef > 0.0:
            cd["intrinsic reward type"] = "dependent"
        return cd
    
    def _calc_int_reward(self) -> torch.Tensor:
        reward = torch.zeros((self.num_envs,))

        current_smiles = [self._make_smiles(env_id=i) for i in range(self.num_envs)]

        if self._dependent_int_reward_config.count_coef > 0.0:
            fp_list = []
            for i in range(self.num_envs):
                fp = self._get_fingerprint_bits(current_smiles[i])
                if type(fp) is not int:
                    fp = self._lsh(fp).tobytes()
                self._morgan_fp_counter.update((fp,))
                fp_list.append(fp)
            
            mol_counts = torch.empty((self.num_envs,))
            for i in range(self.num_envs):
                mol_counts[i] = self._morgan_fp_counter[fp_list[i]]
                
            reward += self._dependent_int_reward_config.count_coef * 1.0 / torch.sqrt(mol_counts)
        
        if self._dependent_int_reward_config.memory_coef > 0.0:
            raise NotImplementedError
        
        return reward
    
    def _make_selfies(self, env_id: int, include_stop_token: bool = False) -> str:
        selfies = "".join([self._action_space.to_symbol(self._selfies_idxes[env_id, i].item()) for i in range(self._steps[env_id])])
        last_token_idx = self._selfies_idxes[env_id, self._steps[env_id]].item()
        if include_stop_token or not self._action_space.is_stop_action(last_token_idx):
            selfies += self._action_space.to_symbol(last_token_idx)
        return selfies
    
    def _make_smiles(self, env_id: int) -> str:
        return sf.decoder(self._make_selfies(env_id)) # type: ignore
    
    def _lsh(self, array: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(self._lsh_rand_matrix, array))
    
    def _get_fingerprint_bits(self, smiles: str) -> Union[np.ndarray, int]:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                self._dependent_int_reward_config.fingerprint_radius,
                nBits=self._dependent_int_reward_config.fingerprint_bits
            )
            x = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, x)
            return x
        else:
            return -1
        
    def _init_selfies_idexes(self) -> np.ndarray:
        return np.zeros((self.num_envs, self._max_str_len), dtype=int)
    
    def _init_steps(self) -> np.ndarray:
        return np.zeros((self.num_envs,), dtype=int)
    
class InferenceChemEnv(ChemEnv):
    def __init__(
        self, 
        ext_reward_config: ChemExtrinsicRewardConfig, 
        int_reward_config: ChemIntrinsicRewardConfig, 
        max_str_len: int = 35, 
        record_data: bool = False,
        log_data_required_episodes: int = 1,
    ) -> None:
        super().__init__(ext_reward_config, int_reward_config, max_str_len, record_data)
        self._log_data_required_episodes = log_data_required_episodes
        self._episode = -1
        
    def reset(self) -> torch.Tensor:
        self._episode += 1
        return super().reset()
    
    @property
    def log_data(self) -> Dict[str, Tuple[float, Optional[float]]]:
        if self._episode >= self._log_data_required_episodes:
            self._episode = self._episode % self._log_data_required_episodes
            return super().log_data
        else:
            return dict()