import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(df: pd.DataFrame, smiles_col: str = 'SMILES', frac_train: float = 0.8, frac_valid: float = 0.1, frac_test: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset based on Bemis-Murcko scaffolds.
    This ensures that train, valid, and test sets contain structurally distinct molecules,
    providing a more rigorous evaluation of the model's generalization capabilities.
    
    Args:
        df: DataFrame containing the dataset.
        smiles_col: Name of the column containing SMILES strings.
        frac_train: Fraction of data for training.
        frac_valid: Fraction of data for validation.
        frac_test: Fraction of data for testing.
        random_state: Random seed.
        
    Returns:
        Tuple of DataFrames: (train_df, valid_df, test_df)
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    
    logger.info("Computing scaffolds for dataset...")
    scaffolds: Dict[str, List[int]] = {}
    for i, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)
        
    # Sort from largest to smallest scaffold size
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    
    train_cutoff = frac_train * len(df)
    valid_cutoff = (frac_train + frac_valid) * len(df)
    
    train_idx, valid_idx, test_idx = [], [], []
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
            
    logger.info(f"Split sizes -> Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    
    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    return train_df, valid_df, test_df
