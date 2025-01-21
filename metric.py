import itertools
from typing import List, Optional

from util import suppress_print

with suppress_print():
    from moleval.metrics.metrics import internal_diversity, novelty, preprocess_gen

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm


def canonicalize(smiles_list: List[str], pbar: bool = False) -> List[str]:
    """Returns a set of canonical smiles from a list of smiles."""
    canonical_smiles = []
    smiles_list_pbar = tqdm(smiles_list, desc="Canonicalize") if pbar else smiles_list
    for smi in smiles_list_pbar:
        mol = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        canonical_smiles.append(canonical_smi)
    return canonical_smiles

def calc_uniqueness(smiles_list: List[str], pbar: bool = False) -> float:
    canonical_smiles = set(canonicalize(smiles_list, pbar=pbar))
    return len(canonical_smiles) / len(smiles_list)

def calc_diversity(smiles_list: List[str], pbar: bool= False) -> float:
    # Generate fingerprints
    canonical_smiles = set(canonicalize(smiles_list, pbar=pbar))
    if len(canonical_smiles) <= 1:
        return 0.0
    canonical_smiles_pbar = tqdm(canonical_smiles, desc="Fingerprint") if pbar else canonical_smiles
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048) for smi in canonical_smiles_pbar]
    # Calculate pairwise similarities
    comb_pbar = tqdm(itertools.combinations(fingerprints, 2), desc="Similarity") if pbar else itertools.combinations(fingerprints, 2)
    similarities = [TanimotoSimilarity(fp1, fp2) for fp1, fp2 in comb_pbar]
    # Calculate diversity
    average_similarity = sum(similarities) / len(similarities)
    return 1.0 - average_similarity

def calc_novelty(smiles_list: List[str], smiles_list_ref: List[str], pbar: bool = False) -> float:
    canonical_smiles = set(canonicalize(smiles_list, pbar=pbar))
    canonical_smiles_ref = set(canonicalize(smiles_list_ref, pbar=pbar))
    novel_smiles = canonical_smiles - canonical_smiles_ref
    return len(novel_smiles) / len(canonical_smiles)

class MolMetric:
    def __init__(self) -> None:
        self.smiles_refset = None
        self.mols_refset = None
        self.smiles_generated = None
        self.mols_generated = None
        self.uniqueness = None
                
    def preprocess(self, smiles_refset: Optional[List[str]] = None, smiles_generated: Optional[List[str]] = None) -> "MolMetric":
        if smiles_refset is None and smiles_generated is None:
            raise ValueError("Please provide a reference set or generated smiles.")
        
        if smiles_refset is not None:
            self.smiles_refset, self.mols_refset, _, _, _ = preprocess_gen(smiles_refset, n_jobs=self._n_jobs(smiles_refset))
        
        if smiles_generated is not None:
            self.smiles_generated, self.mols_generated, _, _, self.uniqueness = preprocess_gen(smiles_generated, n_jobs=self._n_jobs(smiles_generated))
        
        return self
    
    def calc_uniqueness(self) -> float:
        if self.uniqueness is None: raise ValueError("Please preprocess the generated smiles first.")
        return self.uniqueness
    
    def calc_diversity(self) -> float:
        if self.mols_generated is None: raise ValueError("Please preprocess the generated smiles first.")
        return internal_diversity(self.mols_generated, n_jobs=self._n_jobs(self.mols_generated))
    
    def calc_novelty(self) -> float:
        if self.smiles_refset is None: raise ValueError("Please preprocess a reference set first.")
        if self.smiles_generated is None: raise ValueError("Please preprocess the generated smiles first.")
        return novelty(self.smiles_generated, self.smiles_refset, n_jobs=self._n_jobs(self.smiles_generated))
    
    def _n_jobs(self, smiles_or_mols_list) -> int:
        return min(len(smiles_or_mols_list) // 1000 + 1, 100)
