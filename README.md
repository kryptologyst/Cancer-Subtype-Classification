# Cancer Subtype Classification - Research Demo

## DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION PROJECT ONLY**

- This project is for educational and research purposes
- NOT for clinical diagnosis or treatment decisions
- NOT medical advice
- Requires clinician supervision for any real-world applications
- Results should not be used for patient care without proper validation

## Overview

This project implements cancer subtype classification using gene expression data. It demonstrates modern machine learning approaches for bio/omics data analysis, including:

- Synthetic gene expression data generation
- Multiple classification algorithms (Random Forest, XGBoost, LightGBM, CatBoost)
- Deep tabular models for omics data
- Comprehensive evaluation metrics and calibration
- Explainability analysis using SHAP
- Interactive Streamlit demo

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main training script:
```bash
python src/train.py --config configs/default.yaml
```

3. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing and generation
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── configs/               # Configuration files
├── data/                  # Data directory
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated assets (plots, models)
├── demo/                  # Streamlit demo
└── docs/                  # Documentation
```

## Dataset Schema

The synthetic gene expression dataset includes:
- **Features**: 50 simulated gene expression values (normalized)
- **Labels**: 3 cancer subtypes (Luminal A, HER2+, Triple Negative)
- **Samples**: 300 synthetic patients
- **Split**: Patient-level train/validation/test splits

## Training Commands

```bash
# Basic training
python src/train.py

# With custom config
python src/train.py --config configs/xgboost.yaml

# Evaluation only
python src/eval.py --model-path assets/models/best_model.pkl
```

## Evaluation Metrics

- **Classification**: AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV
- **Calibration**: Brier Score, Expected Calibration Error (ECE)
- **Fairness**: Performance by simulated demographic groups
- **Explainability**: SHAP feature importance analysis

## Demo Features

The Streamlit demo provides:
- Interactive model comparison
- Feature importance visualization
- Prediction explanations
- Calibration plots
- Performance metrics dashboard

## Known Limitations

- Uses synthetic data only
- Limited to 3 cancer subtypes
- No real-world validation
- Simplified gene expression simulation
- No clinical feature integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.
# Cancer-Subtype-Classification
