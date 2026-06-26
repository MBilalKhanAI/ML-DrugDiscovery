# Machine Learning-Assisted Prediction of Aromatase Inhibitor Activity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Author:** Muhammad Bilal Khan  
**Email:** khan.muhammadbilal654@gmail.com  

## Abstract
Aromatase inhibitors are critical in the treatment of hormone-receptor-positive breast cancer. Predicting the inhibitory activity of novel compounds efficiently can significantly accelerate drug discovery pipelines. This repository introduces a state-of-the-art machine learning framework for predicting the $pIC_{50}$ of Aromatase inhibitors. 

Moving beyond traditional Quantitative Structure-Activity Relationship (QSAR) models, we implement a rigorous chemoinformatics pipeline featuring Bemis-Murcko scaffold splitting, Bayesian hyperparameter optimization, and a Stacking Regressor ensemble combining Extreme Gradient Boosting (XGBoost), LightGBM, and Random Forests.

## Methodological Advancements

This framework significantly improves upon standard QSAR methodologies through several key implementations:

1. **Bemis-Murcko Scaffold Splitting:** Rather than employing a naive random train-test split—which often leads to overestimated performance due to structural memorization—we partition the dataset based on molecular scaffolds. This strictly evaluates the model's capacity to generalize to novel chemical spaces, a prerequisite for real-world drug discovery.
2. **Feature Fusion:** We integrate extended RDKit 2D physicochemical descriptors with Morgan Fingerprints (radius 2, 2048 bits) to capture both global molecular properties and local structural topology.
3. **Bayesian Hyperparameter Optimization:** Employing `Optuna` for highly efficient, directed searches of hyperparameter spaces, surpassing the limitations of exhaustive Grid Search.
4. **Stacking Ensemble Meta-Learner:** A robust consensus model utilizing a Ridge regression meta-learner to aggregate predictions from tuned XGBoost, LightGBM, and Random Forest base estimators, effectively mitigating individual model biases.
5. **Comprehensive Interpretability:** Model decisions are completely transparent, utilizing SHAP (SHapley Additive exPlanations) values to identify specific structural motifs driving Aromatase inhibition.

---

## Repository Structure

```text
ML-DrugDiscovery/
├── data/
│   └── dataset.csv              # Annotated SMILES and pIC50 values
├── src/
│   ├── data/
│   │   └── splitting.py         # Bemis-Murcko scaffold partitioning
│   ├── features/
│   │   └── descriptors.py       # Morgan FPs & RDKit 2D descriptor generation
│   ├── models/
│   │   ├── optimize.py          # Optuna Bayesian optimization frameworks
│   │   └── train_ensemble.py    # Stacking Regressor architecture
│   └── visualization/
│       └── plots.py             # Publication-ready SHAP and Parity visualizations
├── main.py                      # Unified CLI execution pipeline
├── environment.yml              # Conda environment specification
└── README.md                    # Project documentation
```

---

## Installation

To ensure complete reproducibility, we recommend using `conda` to construct the isolated computational environment.

```bash
# Clone the repository
git clone https://github.com/MBilalKhanAI/ML-DrugDiscovery.git
cd ML-DrugDiscovery

# Create and activate the environment
conda env create -f environment.yml
conda activate ml-drugdiscovery
```

Alternatively, standard pip installation is supported:
```bash
pip install -r requirements.txt
```

---

## Usage

The entire computational pipeline—from descriptor calculation to model evaluation—can be executed via the unified CLI interface. 

To run the full pipeline (including Optuna tuning with 20 trials):
```bash
python main.py --dataset data/dataset.csv --n_trials 20
```

*Generated features will be automatically cached to disk (`results/features_cached.csv`) to accelerate subsequent executions.*

### Outputs
Upon completion, the pipeline generates:
* `results/academic_results.txt`: Comprehensive validation metrics.
* `figures/fig_parity_plot.png`: Publication-grade $pIC_{50}$ prediction parity plot.
* `figures/fig_shap_summary.png`: SHAP summary dot plot elucidating feature impact.

---

## Citing this Work
If you utilize this framework or code in your research, please consider attributing this repository.

> Khan, M. B. (2026). *State-of-the-Art Ensemble Machine Learning Framework for Aromatase Inhibitor Prediction.* GitHub Repository.

*For inquiries regarding PhD opportunities or research collaborations, please reach out via email.*
