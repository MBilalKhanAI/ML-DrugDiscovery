import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def compute_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """
    Computes Morgan Fingerprints (ECFP-like) for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings.
        radius: Radius for Morgan fingerprint (default: 2, equivalent to ECFP4).
        n_bits: Number of bits for the hashed fingerprint (default: 2048).
        
    Returns:
        DataFrame containing the fingerprints.
    """
    fps = []
    valid_indices = []
    
    logger.info(f"Computing Morgan Fingerprints (radius={radius}, n_bits={n_bits})...")
    for i, smiles in enumerate(tqdm(smiles_list, desc="Morgan FPs")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to process SMILES at index {i}: {smiles}. Error: {e}")
            
    df = pd.DataFrame(fps, columns=[f"MFP_{i}" for i in range(n_bits)], index=valid_indices)
    return df

def compute_rdkit_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """
    Computes all 200+ 2D descriptors available in RDKit.
    
    Args:
        smiles_list: List of SMILES strings.
        
    Returns:
        DataFrame containing RDKit 2D descriptors.
    """
    desc_names = [d[0] for d in Descriptors._descList]
    desc_calc = Descriptors.CalcMolDescriptors
    
    features = []
    valid_indices = []
    
    logger.info("Computing RDKit 2D descriptors...")
    for i, smiles in enumerate(tqdm(smiles_list, desc="RDKit Descs")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            descs = desc_calc(mol)
            features.append(descs)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to process SMILES at index {i}: {smiles}. Error: {e}")
            
    df = pd.DataFrame(features, index=valid_indices)
    # Remove features with zero variance or entirely NaN
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0) # Standard imputation for missing descriptor calculations
    return df

def generate_full_feature_set(smiles_list: List[str]) -> pd.DataFrame:
    """
    Generates a combined feature set (Morgan + RDKit 2D).
    
    Args:
        smiles_list: List of SMILES strings.
        
    Returns:
        Combined DataFrame of features.
    """
    mfp_df = compute_morgan_fingerprints(smiles_list)
    rdkit_df = compute_rdkit_descriptors(smiles_list)
    
    # Ensure indices match
    combined_df = mfp_df.join(rdkit_df, how='inner')
    logger.info(f"Generated combined feature matrix of shape: {combined_df.shape}")
    
    return combined_df
