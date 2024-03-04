import torch
import torch.nn as nn

def get_model_device(model: nn.Module) -> torch.device:
    """Returns the device of the model."""
    return next(model.parameters()).device

def batch_to_perenv(batch: torch.Tensor, num_envs: int) -> torch.Tensor:
    """
    `(num_envs x n_steps, *shape)` -> `(num_envs, n_steps, *shape)`
    
    The input `batch` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
    
    Before::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
         
    After::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
    
    """
    shape = batch.shape
    # scalar data (num_envs * n,)
    if len(shape) < 2:
        return batch.reshape(-1, num_envs).T
    # non-scalar data (num_envs * n, *shape)
    else:
        shape = (-1, num_envs) + shape[1:]
        return batch.reshape(shape).transpose(0, 1)

def perenv_to_batch(per_env: torch.Tensor) -> torch.Tensor:
    """
    `(num_envs, n_steps, *shape)` -> `(num_envs x n_steps, *shape)`
    
    The input `per_env` must be like the following example `Before`:
    
    `num_envs` = 2, `n_steps` = 3
         
    Before::
    
        [[env1_step0, env1_step1, env1_step2],
         [env2_step0, env2_step1, env2_step2]]
         
    After::
    
        [env1_step0, 
         env2_step0, 
         env1_step1, 
         env2_step1, 
         env1_step2, 
         env2_step2]
    """
    shape = per_env.shape
    # scalar data (num_envs, n,)
    if len(shape) < 3:
        return per_env.T.reshape(-1)
    # non-scalar data (num_envs, n, *shape)
    else:
        shape = (-1,) + shape[2:]
        return per_env.transpose(0, 1).reshape(shape)
    