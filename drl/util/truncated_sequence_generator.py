from typing import Tuple, NamedTuple, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

class TruncatedSequenceGenerator:
    """
    ## Summary
    
    Truncated sequence generator from batch.
    
    Args:
        full_seq_len (int): full sequence length
        num_envs (int): number of environments
        n_steps (int): number of time steps
        padding_value (float, optional): padding value. Defaults to 0.
        
    ## Example
    
    ::
    
        full_seq_len, num_envs, n_steps = 4, 2, 7
        
        seq_generator = TruncatedSequenceGenerator(full_seq_len, num_envs, n_steps)
        
        hidden_state = torch.randn((num_envs, n_steps, 3, 8)) # (num_envs, n_steps, D x num_layers, H)
        obs = torch.randn((num_envs, n_steps, 3)) # (num_envs, n_steps, obs_features)
        terminated = torch.zeros((num_envs, n_steps))
        terminated[0, 2], terminated[0, 4] = 1, 1
        
        seq_generator.add(hidden_state, seq_len=1)
        seq_generator.add(obs)
        
        mask, seq_init_hidden_state, obs_seq = seq_generator.generate(terminated)
        seq_init_hidden_state.squeeze_(dim=1).swapaxes_(0, 1) # (D x num_layers, seq_batch_size, H)
        
        >>> mask.shape, seq_init_hidden_state.shape, obs_seq.shape
        (torch.Size([5, 4]), torch.Size([3, 5, 8]), torch.Size([5, 4, 3]))
    """
    
    class __SeqInfo(NamedTuple):
        batch: torch.Tensor
        start_idx: int
        seq_len: int
    
    def __init__(self, 
                 full_seq_len: int, 
                 num_envs: int, 
                 n_steps: int,
                 padding_value: float = 0.0) -> None:
        if full_seq_len <= 0:
            raise ValueError(f"full_seq_len must be greater than 0, but got {full_seq_len}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be greater than 0, but got {num_envs}")
        if n_steps <= 0: 
            raise ValueError(f"n_steps must be greater than 0, but got {n_steps}")
                
        self._seq_infos: List[TruncatedSequenceGenerator.__SeqInfo] = []
        self._full_seq_len = full_seq_len
        self._num_envs = num_envs
        self._n_steps = n_steps
        self._padding_value = padding_value
    
    def add(self, batch: torch.Tensor, start_idx: int = 0, seq_len: int = 0):
        """
        Add a batch to make a truncated sequence.
        
        Args:
            batch (Tensor): `(num_envs, n_steps, *shape)`
            start_idx (int, optional): start index of each sequence. Defaults to the sequence start point.
            seq_len (int, optional): the length of each sequence. Defaults to `full_seq_len` - `start_idx` which is from start to end of the sequence.
            
        Raises:
            ValueError: when `start_idx` or `seq_len` is out of sequence range
        """
        if len(batch.shape) < 2 or batch.shape[0] != self._num_envs or batch.shape[1] != self._n_steps:
            raise ValueError(f"batch must be (num_envs, n_steps, *shape), but got {batch.shape}")
        
        # pre-process start_idx
        if start_idx < -self._full_seq_len or start_idx >= self._full_seq_len:
            raise ValueError(f"start_idx={start_idx} is out of sequence range [{-self._full_seq_len}, {self._full_seq_len - 1}].")
        if start_idx < 0:
            start_idx += self._full_seq_len
            
        # pre-process seq_len
        len_to_end = self._full_seq_len - start_idx
        if seq_len <= -len_to_end or seq_len > len_to_end:
            raise ValueError(f"seq_len={seq_len} is out of sequence range [{-len_to_end + 1}, {len_to_end}].")
        if seq_len <= 0:
            seq_len += len_to_end
            
        # add sequence info
        self._seq_infos.append(TruncatedSequenceGenerator.__SeqInfo(batch, start_idx, seq_len))

    def generate(self, interrupted_binary_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Generate truncated sequences.
        
        Args:
            interrupted_binary_mask (Tensor | None, optional): interrupted binary mask `(num_envs, n_steps)`. 
            it's generally terminated tensor. Defaults to no interrupting.
            
        Raises:
            RuntimeError: No batch is added before calling this method.
            
        Returns:
            mask (Tensor): `(seq_batch_size, full_seq_len)`
            generated seuqences (Tensor, ...): `(seq_batch_size, seq_len, *shape)`
        """
        if len(self._seq_infos) == 0:
            raise RuntimeError("No batch is added before calling this method.")
        
        # field cashing
        seq_infos = self._seq_infos
        num_envs = self._num_envs
        n_steps = self._n_steps
        full_seq_len = self._full_seq_len
        
        # add the mask at the beginning of the list
        mask = torch.ones((num_envs, n_steps))
        seq_infos.insert(0, TruncatedSequenceGenerator.__SeqInfo(mask, 0, full_seq_len))
        
        stacked_batches = [[] for _ in range(len(seq_infos))]
        
        for env_id in range(num_envs):
            seq_start_t = 0
            interrupted_idx = 0
            
            interrupted_time_steps = torch.tensor([]) if interrupted_binary_mask is None else torch.where(interrupted_binary_mask[env_id] > 0.5)[0]
            
            while seq_start_t < n_steps:
                # when passing the last time step, it will be zero padded
                seq_end_t = min(seq_start_t + full_seq_len, n_steps)
                
                # if interupped in the middle of the sequence, it will be zero padded
                if interrupted_idx < len(interrupted_time_steps) and interrupted_time_steps[interrupted_idx] < seq_end_t:
                    seq_end_t = interrupted_time_steps[interrupted_idx].item() + 1
                    interrupted_idx += 1
                
                for i, seq_info in enumerate(seq_infos):
                    # determine the sequence range of the current sequence
                    current_seq_start_t = seq_start_t + seq_info.start_idx
                    current_seq_end_t = min(current_seq_start_t + seq_info.seq_len, seq_end_t)
                    current_seq_time_steps = torch.arange(current_seq_start_t, current_seq_end_t)
                    stacked_batches[i].append(seq_info.batch[env_id, current_seq_time_steps])
                    
                seq_start_t = seq_end_t
                
        # pad all sequences
        padded_sequences = []
        for stacked_batch in stacked_batches:
            padded_sequences.append(pad_sequence(stacked_batch, batch_first=True, padding_value=self._padding_value))

        # convert the float mask to boolean
        padded_sequences[0] = padded_sequences[0] > 0.5
        return tuple(padded_sequences)