from typing import Optional, Union, Tuple
import torch

class IncrementalMean:
    """
    Incremental mean calculation. It uses only two memories: mean, n.
    
    See details in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm.
    """
    def __init__(self) -> None:
        self.reset()
        
    def reset(self):
        self._mean = 0.0
        self._n = 0
        
    def update(self, value):
        """Update current mean."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        return self._mean
        
    @property
    def mean(self):
        """Returns current mean."""
        return self._mean
    
    @property
    def count(self) -> int:
        return self._n
    
class IncrementalMeanVarianceFromBatch:
    """
    ## Summary
    
    Incremental mean and variance calculation from batch. 
    See details in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
    
    Args:
        unbiased (bool, optional): whether unbiased or biased variance. Defaults to unbiased variance.
        dim (int | None, optional): dimension along which mean and variance are computed. The default is to compute the values of the flattened array.
    """
    def __init__(self, unbiased: bool = True, dim: Optional[int] = None, dtype: torch.dtype = torch.float32) -> None:
        self._unbiased = unbiased
        self._ddof = 1 if unbiased else 0
        self._dim = dim
        self._dtype = dtype
        self.reset()
        
    @property
    def mean(self) -> torch.Tensor:
        return self._mean
    
    @property
    def variance(self) -> torch.Tensor:
        return self._var
    
    @property
    def batch_size(self) -> int:
        return self._n
    
    def reset(self):
        """Reset the mean and variance to zero."""
        self._mean = torch.tensor(0.0, dtype=self._dtype)
        self._var = torch.tensor(0.0, dtype=self._dtype)
        self._n = 0
    
    def update(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the mean and variance from a batch of data.
        
        Returns:
            mean (float | np.ndarray): updated mean
            variance (float | np.ndarray): updated variance
        """
        if self._dim is None:
            batch_mean = batch.mean()
            batch_var = batch.var(unbiased=self._unbiased)
            batch_size = batch.numel()
        else:
            batch_mean = batch.mean(dim=self._dim)
            batch_var = batch.var(unbiased=self._unbiased, dim=self._dim)
            batch_size = batch.shape[self._dim]
        return self._update_from_batch_mean_var(batch_mean, batch_var, batch_size)
        
    def _update_from_batch_mean_var(self, 
                                    batch_mean: Union[float, torch.Tensor], 
                                    batch_var: Union[float, torch.Tensor], 
                                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # n: batch size
        # M: sum of squares
        # a: old batch
        # b: new batch
        # d: DDOF
        d = self._ddof
        
        n_a = self._n
        n_b = batch_size
        n = n_a + n_b
        
        # update mean
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * n_b / n

        # update variance
        M_a = self._var * (n_a - d)
        M_b = batch_var * (n_b - d)
        M = M_a + M_b + delta**2 * n_a * n_b / n
        self._var = M / (n - d)
        
        # update batch size
        self._n = n
        
        return self._mean, self._var
    
class IncrementalEMA:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._ema = None

    def update(self, batch: torch.Tensor) -> torch.Tensor:
        # If this is the first batch, initialize ema_prev with the batch average
        if self._ema is None:
            self._ema = batch

        # Compute the EMA using the average of the current batch
        self._ema += self._alpha * (batch - self._ema)

        return self._ema

    @property
    def ema(self) -> torch.Tensor:
        if self._ema is None:
            raise ValueError("EMA is not initialized yet.")
        return self._ema