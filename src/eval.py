"""Evaluation script for trained cancer subtype classification models."""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import DataProcessor
from src.metrics import ModelEvaluator, ModelComparison
from src.explainability import ModelExplainer
from src.utils import setup_logging, load_config

logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained cancer subtype classification models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--processor-path', type=str, required=True,
                       help='Path to data processor')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='assets/results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    
    # Load data processor
    logger.info("Loading data processor")
    data_processor = DataProcessor()
    data_processor.load_processor(args.processor_path)
    
    # Load test data
    logger.info("Loading test data")
    test_data = pd.read_csv(args.data_path)
    
    # Process test data
    logger.info("Processing test data")
    processed_data = data_processor.prepare_data(test_data)
    
    # Load model
    logger.info("Loading trained model")
    model = joblib.load(args.model_path)
    
    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(processed_data['X_test'])
    y_proba = model.predict_proba(processed_data['X_test'])
    
    # Evaluate model
    logger.info("Evaluating model")
    evaluator = ModelEvaluator(
        class_names=processed_data['class_names'].tolist(),
        output_dir=args.output_dir
    )
    
    metrics = evaluator.evaluate_model(
        processed_data['y_test'], 
        y_pred, 
        y_proba, 
        "evaluated_model"
    )
    
    # Generate plots
    logger.info("Generating evaluation plots")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], "evaluated_model")
    evaluator.plot_roc_curves(processed_data['y_test'], y_proba, "evaluated_model")
    evaluator.plot_precision_recall_curves(processed_data['y_test'], y_proba, "evaluated_model")
    evaluator.plot_calibration_curve(processed_data['y_test'], y_proba, "evaluated_model")
    
    # Explainability analysis
    logger.info("Performing explainability analysis")
    explainer = ModelExplainer(
        model=model,
        feature_names=processed_data['feature_names'],
        class_names=processed_data['class_names'].tolist(),
        output_dir=args.output_dir
    )
    
    importance_results = explainer.analyze_feature_importance(
        processed_data['X_test'], 
        processed_data['y_test']
    )
    
    # Generate explanation plots
    if importance_results.get('importance_df') is not None:
        explainer.plot_feature_importance(importance_results['importance_df'], "evaluated_model")
    
    if 'shap_values' in importance_results:
        explainer.plot_shap_summary(importance_results['shap_values'], 
                                  processed_data['X_test'][:100], "evaluated_model")
    
    # Save explanation results
    explainer.save_explanation_results(importance_results, "evaluated_model")
    
    # Print results
    logger.info("Evaluation completed successfully")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"F1-Score (Macro): {metrics.get('f1_macro', 0):.4f}")
    logger.info(f"ROC AUC (Macro): {metrics.get('roc_auc_macro', 0):.4f}")
    logger.info(f"Expected Calibration Error: {metrics.get('ece', 0):.4f}")
    
    # Print per-class metrics
    logger.info("Per-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {class_metrics.get('precision', 0):.4f}")
        logger.info(f"    Recall: {class_metrics.get('recall', 0):.4f}")
        logger.info(f"    F1-Score: {class_metrics.get('f1', 0):.4f}")


if __name__ == "__main__":
    main()
