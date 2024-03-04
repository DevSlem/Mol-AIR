import multiprocessing as mp

import numpy as np
import torch

import envs
from envs.chem_env import (ChemEnv, ChemExtrinsicRewardConfig,
                           ChemIntrinsicRewardConfig)


def make_async_env(num_envs: int = 1):
    env_factories = tuple(lambda: ChemEnv(ChemExtrinsicRewardConfig(plogp_coef=1.0), ChemIntrinsicRewardConfig(), max_str_len=5) for _ in range(num_envs))
    return envs.AsyncEnv(env_factories)

def test_async_env_reset():
    print("=== test async env reset ===")
    env = make_async_env(num_envs=3)
    obs = env.reset()
    print(f"obs: {obs.shape}")
    env.close()
    
def test_async_env_step():
    print("=== test async env step ===")
    env = make_async_env(num_envs=3)
    obs = env.reset()
    for _ in range(10):
        obs, reward, terminated, real_final_next_obs, info = env.step(
            torch.randint(env.num_actions, (env.num_envs, 1))
        )
        # print(f"obs: {obs.shape}, reward: {reward.shape}, terminated: {terminated.shape}, real_final_next_obs: {real_final_next_obs.shape}, info: {info}")
        # print(f"terminated: {terminated}, info: {info}")
        if "metric" in info:
            print(info["metric"])
    env.close()
    
def test_tensorboard():
    import warnings
    warnings.filterwarnings(action="ignore")
    from torch.utils.tensorboard.writer import SummaryWriter

    warnings.filterwarnings(action="default")
    
    tb = SummaryWriter(log_dir="results/test")
    tb.add_scalar("Test", 0.1, 0)
    tb.add_scalar("Test", 1.0, 1)
    tb.flush()
    tb.close()
    
def test_shared_object_mp():
    class Foo:
        def __init__(self, shared_queue: mp.Queue) -> None:
            self._shared_queue = shared_queue
            
        def bar(self):
            arr = np.random.rand(3)
            print(arr)
            self._shared_queue.put(arr)
            
    def worker(foo_make_func):
        foo = foo_make_func()
        foo.bar()
            
    shared_queue = mp.Queue()
    foo_factories = tuple(lambda: Foo(shared_queue) for _ in range(3))
    workers = tuple(mp.Process(target=worker, args=(foo_make_func,)) for foo_make_func in foo_factories)
    for i, w in enumerate(workers):
        np.random.seed(i)
        w.start()
    import time
    time.sleep(1)
    print("queue test")
    for _ in range(3):
        print(shared_queue.get())
    for w in workers:
        w.join()
        
def test_env():
    env = ChemEnv(
        ChemExtrinsicRewardConfig(plogp_coef=1.0),
        ChemIntrinsicRewardConfig(),
        max_str_len=10,
        record_data=True
    )
    
    _ = env.reset()
    terminated = False
    while not terminated:
        action = torch.randint(env.num_actions, (1, 1))
        _, _, terminated, _ = env.step(action)
        
def test_prop():
    ## pip install PyTDC==0.4.0
    from tdc import Oracle

    ## Load oracles
    calculate_drd2 = Oracle(name='DRD2')
    calculate_gsk3b = Oracle(name='GSK3B')
    calculate_jnk3 = Oracle(name='JNK3')
    ## Example of SMILES
    smi = 'CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F'
    ## Property evaluation
    list_of_smi = [smi,]
    score_drd2 = calculate_drd2(list_of_smi)[0] # 0.001
    score_gsk3b = calculate_gsk3b(list_of_smi)[0] # 0.02
    score_jnk3 = calculate_jnk3(list_of_smi)[0] # 0.02
    print(score_drd2, score_gsk3b, score_jnk3)
    
def test_util():
    import util
    print("test1")
    with util.suppress_stdout():
        print("test2")
    print("test3")
    
if __name__ == '__main__':
    # test_async_env_reset()
    test_async_env_step()
    # test_shared_object_mp()
    # test_env()
    # test_prop()
    # test_util()