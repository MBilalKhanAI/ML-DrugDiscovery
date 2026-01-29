"""
Step 2: Descriptor Calculation
Reproducing the paper's Dragon methodology using open-source Mordred library.
Maintains strict 3D structural information requirement.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_3d_conformer(smiles):
    """
    Generate 3D conformer from SMILES string using RDKit's ETKDG method.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        mol (Mol): RDKit molecule object with 3D coordinates, or None if failed
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Add hydrogens (critical for 3D structure)
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer using ETKDG method
        # ETKDG = Extended Topological Knowledge-based Distance Geometry
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if result != 0:
            logger.warning(f"Failed to embed 3D conformer for SMILES: {smiles}")
            return None
        
        # Optimize the 3D geometry using UFF force field
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            logger.warning(f"UFF optimization failed for {smiles}: {e}")
            # Continue anyway - the embedded structure is still valid
        
        return mol
        
    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return None


def calculate_descriptors(mol):
    """
    Calculate molecular descriptors using Mordred.
    Includes BOTH 2D and 3D descriptors to match Dragon methodology.
    
    Args:
        mol (Mol): RDKit molecule object with 3D coordinates
        
    Returns:
        dict: Dictionary of descriptor values, or None if failed
    """
    try:
        # Initialize Mordred calculator with 3D descriptors enabled
        # ignore_3D=False ensures 3D descriptors are calculated
        calc = Calculator(descriptors, ignore_3D=False)
        
        # Calculate all descriptors
        result = calc(mol)
        
        # Convert to dictionary, handling missing/error values
        desc_dict = {}
        for desc, value in zip(calc.descriptors, result):
            try:
                # Convert to float, skip if it's an error or missing
                if value is not None and not isinstance(value, Exception):
                    desc_dict[str(desc)] = float(value)
                else:
                    desc_dict[str(desc)] = np.nan
            except (ValueError, TypeError):
                desc_dict[str(desc)] = np.nan
        
        return desc_dict
        
    except Exception as e:
        logger.error(f"Error calculating descriptors: {e}")
        return None


def main():
    """Main execution function."""
    
    logger.info("="*80)
    logger.info("Step 2: Descriptor Calculation")
    logger.info("="*80)
    
    # 1. Load the dataset
    logger.info("Loading dataset.csv...")
    try:
        df = pd.read_csv('dataset.csv')
        logger.info(f"Loaded {len(df)} molecules from dataset.csv")
        logger.info(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logger.error("dataset.csv not found! Please ensure Step 1 is completed.")
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Validate required columns
    if 'SMILES' not in df.columns or 'pIC50' not in df.columns:
        logger.error("Dataset must contain 'SMILES' and 'pIC50' columns!")
        return
    
    # 2. Process each molecule
    logger.info("\nProcessing molecules...")
    logger.info("Generating 3D conformers and calculating descriptors...")
    
    successful_indices = []
    descriptor_list = []
    pic50_list = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        pic50 = row['pIC50']
        
        # Log progress every 10 molecules
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing molecule {idx + 1}/{len(df)}...")
        
        # Generate 3D conformer
        mol_3d = generate_3d_conformer(smiles)
        
        if mol_3d is None:
            failed_count += 1
            logger.warning(f"Skipping molecule {idx + 1} (failed 3D embedding)")
            continue
        
        # Calculate descriptors
        descriptors_dict = calculate_descriptors(mol_3d)
        
        if descriptors_dict is None:
            failed_count += 1
            logger.warning(f"Skipping molecule {idx + 1} (descriptor calculation failed)")
            continue
        
        # Store successful results
        successful_indices.append(idx)
        descriptor_list.append(descriptors_dict)
        pic50_list.append(pic50)
    
    # 3. Create DataFrames from results
    logger.info("\nCreating output DataFrames...")
    
    if len(descriptor_list) == 0:
        logger.error("No molecules were successfully processed! Check your input data.")
        return
    
    # Create descriptor DataFrame
    X_descriptors = pd.DataFrame(descriptor_list)
    logger.info(f"Calculated {len(X_descriptors.columns)} descriptors")
    logger.info(f"Descriptor columns include: {X_descriptors.columns[:5].tolist()}... (showing first 5)")
    
    # Create target DataFrame
    y_data = pd.DataFrame({'pIC50': pic50_list})
    
    # Ensure indices align
    X_descriptors.index = successful_indices
    y_data.index = successful_indices
    
    # 4. Save outputs
    logger.info("\nSaving results...")
    
    try:
        X_descriptors.to_csv('X_descriptors.csv', index=False)
        logger.info(f"✓ Saved X_descriptors.csv ({X_descriptors.shape[0]} molecules × {X_descriptors.shape[1]} descriptors)")
        
        y_data.to_csv('y_data.csv', index=False)
        logger.info(f"✓ Saved y_data.csv ({y_data.shape[0]} molecules)")
        
    except Exception as e:
        logger.error(f"Error saving output files: {e}")
        return
    
    # 5. Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total input molecules:        {len(df)}")
    logger.info(f"Successfully processed:       {len(descriptor_list)}")
    logger.info(f"Failed (discarded):           {failed_count}")
    logger.info(f"Success rate:                 {len(descriptor_list)/len(df)*100:.1f}%")
    logger.info(f"Total descriptors calculated: {X_descriptors.shape[1]}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - X_descriptors.csv: {X_descriptors.shape}")
    logger.info(f"  - y_data.csv: {y_data.shape}")
    logger.info(f"\n✓ Indices are perfectly aligned between X and y")
    
    # Check for 3D descriptors
    descriptor_names = X_descriptors.columns.tolist()
    three_d_descriptors = [d for d in descriptor_names if '3D' in d or 'GETAWAY' in d or 'WHIM' in d or 'RDF' in d]
    logger.info(f"\n3D descriptors detected: {len(three_d_descriptors)} (confirming 3D calculation)")
    
    logger.info("\n" + "="*80)
    logger.info("Step 2 completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
