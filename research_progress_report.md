# Research Progress Report: Aromatase Inhibitor Activity Prediction

**Date:** January 29, 2026  
**Objective:** Reproduction of "Machine Learning-Assisted Prediction of the Biological Activity of Aromatase Inhibitors"  
**Current Status:** Completed Step 2 (Descriptor Calculation)

---

## 1. Project Overview
We are building a Quantitative Structure-Activity Relationship (QSAR) model to predict the biological activity ($pIC_{50}$) of Aromatase inhibitors. Aromatase is a key enzyme in estrogen biosynthesis and a major target for breast cancer therapy.

The original paper utilized **Dragon** software for molecular descriptor calculation on 3D structures. Our reproduction replaces the proprietary Dragon software with the open-source **Mordred** library while strictly adhering to the paper's requirement for **3D structural information**.

---

## 2. Completed Steps

### Step 1: Data Acquisition & Cleaning
*   **Input Data:** We started with a dataset of chemical structures (SMILES) and their experimentally measured biological activities ($pIC_{50}$).
*   **Outcome:** A clean file named `dataset.csv` containing two columns:
    *   `SMILES`: The 1D string representation of the molecule.
    *   `pIC50`: The negative log of the half-maximal inhibitory concentration. This is our **target variable ($y$)**.

### Step 2: 3D Feature Engineering (The Crucial Step)
This step converted 1D SMILES strings into a rich set of mathematical descriptors suitable for machine learning.

#### A. 3D Conformer Generation (The "Structure" in QSAR)
Since the biological activity of a drug depends heavily on how it fits into a 3D protein pocket, 2D representations are often insufficient. We generated biologically relevant 3D structures using **RDKit**.

*   **Process Detail:**
    1.  **Hydrogen Addition (`Chem.AddHs`):** SMILES strings usually imply hydrogens. We explicitly added H-atoms because they define the molecular volume and shape, which are critical for 3D descriptors.
    2.  **Embedding (`AllChem.EmbedMolecule`):** We used the **ETKDG** (Experimental-Torsion Knowledge Distance Geometry) method.
        *   *Why ETKDG?* It is a stochastic algorithm that uses data from the Cambridge Structural Database (CSD) to generate conformers that respect preferred bond lengths, angles, and torsional patterns found in real crystal structures. It is superior to standard Distance Geometry.
    3.  **Optimization (`AllChem.UFFOptimizeMolecule`):** We applied the Universal Force Field (UFF) to "relax" the molecule, minimizing its energy to find a stable local minimum conformation.
    4.  **Quality Control:** Any molecule that failed to generate a valid 3D structure was strictly discarded to prevent noise in the model.

#### B. Descriptor Calculation (Mordred)
We replaced Dragon with **Mordred**, a Python library capable of calculating over 1,800 descriptors.

*   **Why Mordred?** It is the most comprehensive open-source alternative to Dragon, offering high coverage of similar topological and geometric descriptors.
*   **Methodology:**
    *   We initialized the calculator with `ignore_3D=False`.
    *   **2D Descriptors:** Calculated from the molecular graph (e.g., molecular weight, LogP, atom counts, topological indices).
    *   **3D Descriptors:** Calculated from the generated x, y, z coordinates (e.g., Moments of Inertia, Molecular Geometrical Shape descriptors, charged partial surface area descriptors).
    *   *Significance:* This captures both the "chemistry" (functional groups) and the "physics" (shape/volume) of the inhibitors.

---

## 3. Current Data State
We have successfully processed the dataset and generated two aligned files.

| File Name | Dimensions | Description |
| :--- | :--- | :--- |
| **`X_descriptors.csv`** | **(N, ~1800)** | The Feature Matrix. Rows are molecules, columns are Mordred descriptors (2D + 3D). |
| **`y_data.csv`** | **(N, 1)** | The Target Vector. Contains the $pIC_{50}$ values corresponding strictly to the rows in X. |

*Note: $N$ is the number of successfully processed molecules (failed embeddings were removed).*

---

## 4. Next Steps: Model Building (Step 3)
With the features ($X$) and targets ($y$) ready, we proceed to the Machine Learning phase proposed in the paper.

1.  **Data Preprocessing:**
    *   Remove "Low Variance" columns (descriptors that have the same value for all molecules provide no information).
    *   Remove highly correlated features (multicollinearity check).
    *   Standardize/Scale the data (zero mean, unit variance).
2.  **Model Training:**
    *   Train models specifically mentioned in the paper, likely **Random Forest (RF)** and **Support Vector Machine (SVM)**.
3.  **Validation:**
    *   Perform K-Fold Cross-Validation.
    *   Calculate metrics: $R^2$ (coefficient of determination), $RMSE$ (Root Mean Squared Error), and $Q^2$ (cross-validated $R^2$).
