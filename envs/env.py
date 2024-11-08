import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing.connection import Connection
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Generic, TypeVar

import numpy as np

T = TypeVar('T')

def env_config(
    name: str,
    obs_shape: Optional[Tuple[int, ...]] = None,
    num_actions: Optional[int] = None,
):
    def decorator(cls: T) -> T:
        if not issubclass(cls, Env):
            raise TypeError("Class must inherit from Env")
        cls.name = name
        if obs_shape is not None:
            cls.obs_shape = obs_shape
        if num_actions is not None:
            cls.num_actions = num_actions
        return cls
    return decorator

class Env(ABC):
    name: str
    obs_shape: Tuple[int, ...]
    num_actions: int
    
    @abstractmethod 
    def reset(self) -> np.ndarray:
        """
        Resets the environment to an initial state and returns the initial observation.

        Returns:
            obs (ndarray): `(num_envs, *obs_shape)`. Initial observation.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Takes a step in the environment using an action.

        Args:
            - action (ndarray): `(num_envs, num_actions)`. Action provided by the agent.

        Returns:
            - next_obs (ndarray): `(num_envs, *obs_shape)`. Next observation which is automatically reset to the first observation of the next episode. 
            - reward (ndarray): `(num_envs,)`. Scalar reward values.
            - terminated (ndarray): `(num_envs,)`. Whether the episode is terminated.
            - real_final_next_obs (ndarray): `(num_terminated_envs, *obs_shape)`. "Real" final next observation of the episode. You can access only if any environment is terminated. 
            - info (dict): Additional information.
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self):
        """Close the environment and release resources."""
        raise NotImplementedError
    
    @property
    def num_envs(self) -> int:
        return 1
    
    @property
    def config_dict(self) -> dict:
        """
        The configuration of the environment.
        """
        return dict()
    
    def save_data(self, base_dir: str):
        """
        Save the environment data. `base_dir` is different according to the configuration ID, training, inference, etc.
        """
        pass
    
EnvMakeFunc = Callable[[], Env]

class WorkerCommand(Enum):
    RESET = 0,
    STEP = 1,
    CLOSE = 2,
    SAVE_DATA = 3,
    
class AsyncEnv(Env):    
    def __init__(self, env_make_func_iter: Iterable[EnvMakeFunc], trace_env: int = 0) -> None:
        self._trace_env = trace_env
        # check the number of environments
        env_make_func_tuple = tuple(env_make_func_iter)
        self._num_envs = len(env_make_func_tuple)
        if self._num_envs == 0:
            raise ValueError('env_make_func must not be empty.')
        
        # set properties
        env_temp = env_make_func_tuple[0]()
        self._obs_shape = env_temp.obs_shape
        self._num_actions = env_temp.num_actions
        self._config_dict = env_temp.config_dict
        env_temp.close()
        
        # make workers
        self._parent_connections: List[Connection] = []
        self._workers: List[mp.Process] = []
        for env_make_func in env_make_func_tuple:
            parent_conn, child_conn = mp.Pipe()
            worker = mp.Process(
                target=AsyncEnv._worker,
                args=(
                    env_make_func,
                    child_conn,
                )
            )
            
            self._parent_connections.append(parent_conn)
            self._workers.append(worker)
            
            worker.start()
            child_conn.close()
        
    def reset(self) -> np.ndarray:
        for parent_conn in self._parent_connections:
            parent_conn.send((WorkerCommand.RESET, None))
        
        obs_tuple = tuple(parent_conn.recv() for parent_conn in self._parent_connections)
        return np.concatenate(obs_tuple, axis=0)
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        for i, parent_conn in enumerate(self._parent_connections):
            parent_conn.send((WorkerCommand.STEP, action[i:i+1]))
            
        unwrapped_results = tuple(parent_conn.recv() for parent_conn in self._parent_connections)
        results_tuple = tuple(zip(*unwrapped_results))
        concat_ndarrays = tuple(np.concatenate(result, axis=0) for result in results_tuple[:-1])
        info = self._merge_info(results_tuple[-1])
        
        return concat_ndarrays + (info,) # type: ignore
    
    def _merge_info(self, info_tuple: Tuple[dict, ...]) -> dict:
        num_envs = len(info_tuple)
        assert num_envs == self._num_envs
        # get all keys in info_tuple
        info_keys = set()
        for info in info_tuple:
            info_keys.update(info.keys())
        # merge info with mask
        merged_info = dict()
        for key in info_keys:
            items = np.full((num_envs,), None, dtype=object)
            mask = np.full((num_envs,), False, dtype=bool)
            for i, info in enumerate(info_tuple):
                if key in info:
                    items[i] = info[key]
                    mask[i] = True
            merged_info[key] = items
            merged_info[f"_{key}"] = mask
        return merged_info
    
    def close(self):
        for parent_conn in self._parent_connections:
            parent_conn.send((WorkerCommand.CLOSE, None))
        
        for parent_conn in self._parent_connections:
            parent_conn.recv()
            
        for parent_conn in self._parent_connections:
            parent_conn.close()
        
        for worker in self._workers:
            worker.join()
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return self._obs_shape
    
    @property
    def num_actions(self) -> int:
        return self._num_actions
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def config_dict(self) -> dict:
        return self._config_dict
    
    def _single_env_step(self, env_action: Tuple[Env, np.ndarray]):
        env = env_action[0]
        action = env_action[1]
        return env.step(action[np.newaxis, ...])
    
    @staticmethod
    def _worker(
        env_make_func: EnvMakeFunc,
        child: Connection,
    ):
        env = env_make_func()
        try:
            while True:
                command, data = child.recv()
                if command == WorkerCommand.RESET:
                    if data is not None:
                        raise ValueError('when you reset, data must be None.')
                    child.send(env.reset())
                elif command == WorkerCommand.STEP:
                    if not isinstance(data, np.ndarray):
                        raise ValueError(f'data must be np.ndarray, but got {type(data)}.')
                    child.send(env.step(data))
                elif command == WorkerCommand.CLOSE:
                    if data is not None:
                        raise ValueError('when you close, data must be None.')
                    env.close()
                    child.send(None)
                    break
                elif command == WorkerCommand.SAVE_DATA:
                    if not isinstance(data, str):
                        raise ValueError(f'since data is directory, it must be str, but got {type(data)}.')
                    child.send(env.save_data(data))
                else:
                    raise NotImplementedError(f'command {command} is not implemented.')
        except KeyboardInterrupt:
            command, data = child.recv()
            if command == WorkerCommand.CLOSE:
                if data is not None:
                    raise ValueError('when you close, data must be None.')
                child.send(None)
        except Exception as ex:
            raise ex
        finally:
            pass

                
class EnvWrapper(Env):
    def __init__(self, env: Env) -> None:
        self._env = env
    
    def reset(self) -> np.ndarray:
        return self._env.reset()
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        return self._env.step(action)
    
    def close(self):
        self._env.close()
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return self._env.obs_shape
    
    @property
    def num_actions(self) -> int:
        return self._env.num_actions
    
    @property
    def num_envs(self) -> int:
        return self._env.num_envs
    
    @property
    def config_dict(self) -> dict:
        return self._env.config_dict
    
    def save_data(self, base_dir: str):
        self._env.save_data(base_dir)
    
    def __getattr__(self, name):
        return getattr(self._env, name)