from typing import List, Set
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import itertools
from tqdm import tqdm

def canonicalize(smiles_list: List[str], pbar: bool = False) -> Set[str]:
    """Returns a set of canonical smiles from a list of smiles."""
    canonical_smiles = set()
    smiles_list_pbar = tqdm(smiles_list, desc="Canonicalize") if pbar else smiles_list
    for smi in smiles_list_pbar:
        mol = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        canonical_smiles.add(canonical_smi)
    return canonical_smiles

def calc_uniqueness(smiles_list: List[str], pbar: bool = False) -> float:
    canonical_smiles = canonicalize(smiles_list, pbar=pbar)
    return len(canonical_smiles) / len(smiles_list)

def calc_diversity(smiles_list: List[str], pbar: bool= False) -> float:
    # Generate fingerprints
    canonical_smiles = canonicalize(smiles_list, pbar=pbar)
    if len(canonical_smiles) <= 1:
        return 0.0
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048) for smi in canonical_smiles]
    # Calculate pairwise similarities
    similarities = [TanimotoSimilarity(fp1, fp2) for fp1, fp2 in itertools.combinations(fingerprints, 2)]
    # Calculate diversity
    average_similarity = sum(similarities) / len(similarities)
    return 1.0 - average_similarity

def calc_novelty(smiles_list: List[str], smiles_list_ref: List[str], pbar: bool = False) -> float:
    canonical_smiles = canonicalize(smiles_list, pbar=pbar)
    canonical_smiles_ref = canonicalize(smiles_list_ref, pbar=pbar)
    novel_smiles = canonical_smiles - canonical_smiles_ref
    return len(novel_smiles) / len(smiles_list)

# def calc_metrics