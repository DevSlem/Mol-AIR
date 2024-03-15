import math
from collections import Counter
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from typing import Union, Optional

class CountIntReward:
    def __init__(
        self, 
        max_mol_count: int = 10,
        fingerprint_bits: int = 256,
        fingerprint_radius: int = 2,
        lsh_bits: int = 32,
        np_rng: Optional[np.random.Generator] = None
    ) -> None:
        self._fingerprint_bits = fingerprint_bits
        self._fingerprint_radius = fingerprint_radius
        self._lsh_bits = lsh_bits
        self._max_mol_count = max_mol_count
        
        if np_rng is None:
            np_rng = np.random.default_rng()
        self._lsh_rand_matrix = np_rng.normal(size=(self._lsh_bits, self._fingerprint_bits))
        self._counter = Counter()

    def calc_reward(self, smiles: str) -> float:
        new_mol_count = self._update_mol_count(smiles)
        x = min(new_mol_count - 1, self._max_mol_count)
        return math.exp(-x)
    
    def __call__(self, smiles: str) -> float:
        return self.calc_reward(smiles)
    
    def _update_mol_count(self, smiles: str) -> int:
        fp = self._to_fingerprint(smiles)
        if type(fp) is not int:
            fp = self._lsh(fp).tobytes()
        self._counter.update((fp,))
        return self._counter[fp]    
    
    def _to_fingerprint(self, smiles: str) -> Union[np.ndarray, int]:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                self._fingerprint_radius,
                nBits=self._fingerprint_bits
            )
            x = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, x)
            return x
        else:
            return -1
        
    def _lsh(self, fp: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(self._lsh_rand_matrix, fp))