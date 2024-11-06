from typing import List, Set
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import itertools

def canonicalize(smiles_list: List[str]) -> Set[str]:
    """Returns a set of canonical smiles from a list of smiles."""
    canonical_smiles = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        canonical_smiles.add(canonical_smi)
    return canonical_smiles

def calc_uniqueness(smiles_list: List[str]) -> float:
    canonical_smiles = canonicalize(smiles_list)
    return len(canonical_smiles) / len(smiles_list)

def calc_diversity(smiles_list: List[str]) -> float:
    # Generate fingerprints
    canonical_smiles = canonicalize(smiles_list)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048) for smi in canonical_smiles]
    # Calculate pairwise similarities
    similarities = [TanimotoSimilarity(fp1, fp2) for fp1, fp2 in itertools.combinations(fingerprints, 2)]
    # Calculate diversity
    average_similarity = sum(similarities) / len(similarities)
    return 1.0 - average_similarity

def calc_novelty(smiles_list: List[str], smiles_list_ref: List[str]) -> float:
    canonical_smiles = canonicalize(smiles_list)
    canonical_smiles_ref = canonicalize(smiles_list_ref)
    novel_smiles = canonical_smiles - canonical_smiles_ref
    return len(novel_smiles) / len(smiles_list)

# def calc_metrics