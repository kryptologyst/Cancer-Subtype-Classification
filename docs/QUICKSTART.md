# Cancer Subtype Classification - Research Demo

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd cancer_subtype_classification

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### 2. Generate Synthetic Data

```bash
# Generate a synthetic dataset
python scripts/utils.py generate --n-samples 1000 --n-genes 100 --output-dir data
```

### 3. Train Models

```bash
# Train with default configuration
python src/train.py

# Train with specific model
python src/train.py --model xgboost --config configs/xgboost.yaml

# Train with custom parameters
python src/train.py --model random_forest --seed 123 --verbose
```

### 4. Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/app.py
```

### 5. Evaluate Trained Models

```bash
# Evaluate a specific model
python src/eval.py --model-path assets/models/random_forest_model.pkl \
                   --data-path data/synthetic_cancer_data.csv \
                   --processor-path assets/processors/data_processor.pkl
```

### 6. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
cancer_subtype_classification/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing and generation
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── explainability/    # Explainability analysis
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── configs/               # Configuration files
├── data/                  # Data directory
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated assets
│   ├── models/           # Trained models
│   ├── plots/            # Generated plots
│   └── results/          # Evaluation results
├── demo/                  # Streamlit demo
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── .github/workflows/     # CI/CD workflows
├── README.md             # Project overview
└── DISCLAIMER.md         # Important disclaimers
```

## Configuration

The project uses YAML configuration files in the `configs/` directory:

- `default.yaml`: Default configuration
- `xgboost.yaml`: XGBoost-specific configuration

### Key Configuration Parameters

```yaml
data:
  n_samples: 300          # Number of samples
  n_genes: 50            # Number of genes (features)
  n_subtypes: 3         # Number of cancer subtypes
  test_size: 0.2        # Test set proportion
  val_size: 0.2        # Validation set proportion

model:
  name: "random_forest"  # Model type
  params:               # Model-specific parameters
    random_forest:
      n_estimators: 100
      max_depth: 10

evaluation:
  metrics: ["auroc", "auprc", "accuracy", "precision", "recall", "f1"]
  calibration: true
  explainability: true
```

## Available Models

1. **Random Forest**: Traditional ensemble method
2. **XGBoost**: Gradient boosting with advanced features
3. **LightGBM**: Fast gradient boosting
4. **CatBoost**: Categorical boosting
5. **TabNet**: Neural network for tabular data

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **ROC Analysis**: AUROC (Area Under ROC Curve)
- **Precision-Recall**: AUPRC (Area Under PR Curve)
- **Calibration**: Brier Score, Expected Calibration Error (ECE)
- **Explainability**: SHAP values, Feature importance

## Demo Features

The Streamlit demo provides:

1. **Interactive Model Training**: Train models with custom parameters
2. **Performance Comparison**: Compare multiple models side-by-side
3. **Explainability Analysis**: SHAP-based feature importance
4. **Feature Analysis**: Correlation and distribution analysis
5. **Interactive Predictions**: Make predictions on custom inputs

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pre-commit**: Git hooks for quality checks
- **Pytest**: Unit testing

### Running Quality Checks

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest tests/ -v
```

### Adding New Models

1. Create a new model class inheriting from `BaseModel`
2. Implement required methods: `fit`, `predict`, `predict_proba`
3. Add model to the `create_model` function
4. Add configuration parameters
5. Write tests

### Adding New Metrics

1. Add metric calculation to `ModelEvaluator`
2. Add plotting functionality
3. Update configuration options
4. Write tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Use `--device cpu` for CPU-only training
3. **Memory Issues**: Reduce batch size or number of samples
4. **SHAP Errors**: Some models may not support SHAP analysis

### Getting Help

1. Check the logs in the `logs/` directory
2. Run tests to verify installation
3. Check configuration files for errors
4. Review the documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cancer_subtype_classification,
  title={Cancer Subtype Classification - Research Demo},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cancer_subtype_classification}
}
```
