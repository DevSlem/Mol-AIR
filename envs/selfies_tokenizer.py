from typing import Iterable, Optional, Union, List

import numpy as np
import selfies as sf


class SelfiesTokenizer:
    def __init__(self) -> None:
        self._stop_token = "[STOP]"
        self._selfies_tokens = ["[#N]"] + sf.selfies_alphabet() + [self._stop_token]
        self._token_to_value_dict = {token: i for i, token in enumerate(self._selfies_tokens)}
        self._padding_value = -1
        
    @property
    def n_tokens(self) -> int:
        return len(self._selfies_tokens)
    
    @property
    def stop_token(self) -> str:
        return self._stop_token
    
    @property
    def stop_token_val(self) -> int:
        return self._token_to_value_dict[self._stop_token]
        
    def encode(self, selfies_list: Union[str, List[str]], seq_len: Optional[int] = None) -> np.ndarray:
        """
        Encode the SELFIES strings to the integer sequences. 
        The integer sequences are padded with the padding value `-1`.

        Args:
            selfies_list (str | List[str]): one SELFIES string or `batch_size` SELFIES strings list.
            seq_len (Optional[int], optional): The length of the sequence. If None, the maximum length of the SELFIES strings is used. Defaults to None.

        Returns:
            encoded_sequences (ndarray): `(seq_len,)` or `(batch_size, seq_len)`
        """
        if isinstance(selfies_list, str):
            selfies_list = [selfies_list]
            return self._encode_batch(selfies_list, seq_len)[0]
        return self._encode_batch(selfies_list, seq_len)
    
    def decode(self, encoded_sequences: np.ndarray, include_stop_token: bool = True) -> Union[str, List[str]]:
        """
        Decode the integer sequences to the SELFIES strings. 
        Padding values are ignored.

        Args:
            encoded_sequences (ndarray): `(seq_len,)` or `(batch_size, seq_len)`

        Returns:
            selfies_list (str | List[str]): one SELFIES string or `batch_size` SELFIES strings list.
        """
        if len(encoded_sequences.shape) == 1:
            return self._decode(encoded_sequences, include_stop_token)
        return list(self._decode(encoded, include_stop_token) for encoded in encoded_sequences)
    
    def last_token_value(self, encoded_sequences: np.ndarray) -> np.ndarray:
        """
        Get the value of the last token in the sequence. 
        The last token is the one before the first occurrence of the padding token (-1).
        If the sequence is all padding tokens, the last token value will be -1.

        Args:
            encoded_sequences (ndarray): `(seq_len,)` or `(batch_size, seq_len)`

        Returns:
            last_token (int | ndarray): scalar integer or `(batch_size,)`
        """
        if len(encoded_sequences.shape) == 1:
            encoded_sequences = np.expand_dims(encoded_sequences, axis=0)
            return self._last_token_value_batch(encoded_sequences)[0]
        return self._last_token_value_batch(encoded_sequences)
    
    def to_one_hot(self, encoded_sequences: np.ndarray) -> np.ndarray:
        """
        Convert the encoded sequences to one-hot encoding. 
        Padding tokens are converted to zero vectors.

        Args:
            encoded_sequences (ndarray): `(seq_len,)` or `(batch_size, seq_len)`

        Returns:
            one_hot (ndarray): `(seq_len, n_tokens)` or `(batch_size, seq_len, n_tokens)`
        """
        if len(encoded_sequences.shape) == 1:
            encoded_sequences = np.expand_dims(encoded_sequences, axis=0)
            return self._to_one_hot_batch(encoded_sequences)[0]
        return self._to_one_hot_batch(encoded_sequences)
    
    def from_one_hot(self, one_hot: np.ndarray) -> np.ndarray:
        """
        Convert the one-hot encoded sequences back to the original encoded sequences.
        Zero vectors are converted to padding tokens.

        Args:
            one_hot (ndarray): `(seq_len, n_tokens)` or `(batch_size, seq_len, n_tokens)`

        Returns:
            encoded_sequences (ndarray): `(seq_len,)` or `(batch_size, seq_len)`
        """
        if len(one_hot.shape) == 2:
            one_hot = np.expand_dims(one_hot, axis=0)
            return self._from_one_hot_batch(one_hot)[0]
        return self._from_one_hot_batch(one_hot)
    
    def _encode_batch(self, selfies_list: Iterable[str], seq_len: Optional[int]) -> np.ndarray:
        selfies_tokens_list = [self._split_selfies(selfies) for selfies in selfies_list]
        if seq_len is None:
            seq_len = max(len(selfies_tokens) for selfies_tokens in selfies_tokens_list)
        encoded_list = [self._encode_from_tokens(selfies_tokens, seq_len) for selfies_tokens in selfies_tokens_list]
        return np.stack(encoded_list, axis=0)
    
    def _split_selfies(self, selfies: str) -> List[str]:
        return list('[' + character for character in selfies.split('['))[1:]
        
    def _encode_from_tokens(self, selfies_tokens: List[str], seq_len: int) -> np.ndarray:
        sequence = np.full((seq_len,), self._padding_value, dtype=np.int64)
        for i, token in enumerate(selfies_tokens):
            if seq_len <= i:
                break
            sequence[i] = self._token_to_value_dict[token]
        return sequence
    
    def _decode(self, encoded: np.ndarray, include_stop_token: bool) -> str:
        string = ""
        for idx in encoded:
            if idx == self._padding_value:
                break
            if not include_stop_token and idx == self._token_to_value_dict[self._stop_token]:
                break
            string += self._selfies_tokens[idx]
        return string
    
    def _last_token_value_batch(self, encoded_sequences: np.ndarray) -> np.ndarray:
        last_token = np.full((encoded_sequences.shape[0],), self._padding_value, dtype=np.int64)
        for i, encoded in enumerate(encoded_sequences):
            try:
                last_token[i] = encoded[encoded != self._padding_value][-1]
            except IndexError:
                last_token[i] = self._padding_value
        return last_token
    
    def _to_one_hot_batch(self, encoded_sequences: np.ndarray) -> np.ndarray:
        one_hot = np.zeros((encoded_sequences.shape[0], encoded_sequences.shape[1], self.n_tokens), dtype=np.int64)
        for i, encoded in enumerate(encoded_sequences):
            for j, idx in enumerate(encoded):
                if idx != self._padding_value:
                    one_hot[i, j, idx] = 1
        return one_hot
    
    def _from_one_hot_batch(self, one_hot: np.ndarray) -> np.ndarray:
        encoded_sequences = np.full((one_hot.shape[0], one_hot.shape[1]), self._padding_value, dtype=np.int64)
        for i, one_hot_seq in enumerate(one_hot):
            for j, token_one_hot in enumerate(one_hot_seq):
                if np.any(token_one_hot):
                    encoded_sequences[i, j] = np.argmax(token_one_hot)
        return encoded_sequences