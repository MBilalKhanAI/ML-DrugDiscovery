# Machine Learning-Assisted Prediction of Aromatase Inhibitor Activity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Author:** Muhammad Bilal Khan  
**Email:** khan.muhammadbilal654@gmail.com  

## Project Overview
This repository contains the complete reproduction and enhancement of the research paper *"Machine Learning-Assisted Prediction of the Biological Activity of Aromatase Inhibitors"*. We have successfully implemented a robust QSAR (Quantitative Structure-Activity Relationship) pipeline to predict the pIC50 of Aromatase inhibitors using open-source tools.

**Key Achievements:**
*   **Transition to Open Source:** Replaced proprietary Dragon descriptors with **Mordred**, maintaining critical 3D structural information.
*   **High Accuracy:** Achieved **$R^2 \approx 0.60$** with Random Forest (validated via 10-Fold CV), validating the pipeline's effectiveness.
*   **Advanced Analysis:** Added modern visualizations including t-SNE chemical space analysis and SHAP interpretability.

---

## Repository Structure
The project is organized into modular directories for clarity and reproducibility.

```
AROMATASE-PROJECT/
├── data/                   # Dataset and processed feature matrices
│   ├── dataset.csv         # Original input (SMILES, pIC50)
│   ├── X_selected.csv      # Top 50 features used for modeling
│   └── y_data.csv          # Target values
├── src/                    # Python source code for each step
│   ├── step2_descriptor_calculation.py
│   ├── step3_1_data_cleaning.py
│   ├── step3_6_rigorous_validation.py  <-- Start here for full validation!
│   └── step4_visualizations.py
├── figures/                # Generated plots and visualizations
│   ├── Figure_2.png        # Feature Distributions
│   ├── Modern_Fig_TSNE.png # Chemical Space Analysis
│   └── ...
├── results/                # Text reports of model performance
│   └── validation_results.txt
├── research_progress_report.md  # Detailed scientific report of findings
└── README.md               # This file
```

---

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn rdkit mordred shap
```

### Reproducing the Results
You can run the pipeline step-by-step using the scripts in `src/`:

1.  **Generate Descriptors:**
    ```bash
    python src/step2_descriptor_calculation.py
    ```
2.  **Run Rigorous Validation (GridSearch + 10-Fold CV):**
    ```bash
    python src/step3_6_rigorous_validation.py
    ```
3.  **Generate Visualizations:**
    ```bash
    python src/step4_visualizations.py
    python src/step5_modern_visualizations.py
    ```

---

## Key Results

| Model | Mean $R^2$ (10-Fold CV) | Mean RMSE |
| :--- | :--- | :--- |
| **Random Forest (Tuned)** | **0.5951** | **0.7124** |
| Gradient Boosting | 0.5606 | 0.7402 |
| Linear Regression | 0.4624 | 0.8126 |

*See [research_progress_report.md](research_progress_report.md) for the detailed scientific analysis.*

