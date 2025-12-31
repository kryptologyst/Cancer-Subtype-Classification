"""Main training script for cancer subtype classification."""

import os
import sys
import argparse
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import GeneExpressionDataGenerator, DataProcessor
from src.models import create_model
from src.metrics import ModelEvaluator, ModelComparison
from src.explainability import ModelExplainer, FeatureAnalyzer
from src.utils import setup_logging, set_deterministic_seed, load_config, create_directories

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train cancer subtype classification model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to train (overrides config)')
    parser.add_argument('--output-dir', type=str, default='assets',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model.name = args.model
    if args.seed:
        config.random_seed = args.seed
    
    # Set random seed
    set_deterministic_seed(config.random_seed)
    
    # Create output directories
    create_directories([
        args.output_dir,
        os.path.join(args.output_dir, 'models'),
        os.path.join(args.output_dir, 'plots'),
        os.path.join(args.output_dir, 'results')
    ])
    
    logger.info("Starting cancer subtype classification training")
    logger.info(f"Configuration: {config}")
    
    # Generate synthetic data
    logger.info("Generating synthetic gene expression data")
    data_generator = GeneExpressionDataGenerator(
        n_samples=config.data.n_samples,
        n_genes=config.data.n_genes,
        n_subtypes=config.data.n_subtypes,
        subtypes=config.data.subtypes,
        random_seed=config.random_seed
    )
    
    X, y = data_generator.generate_realistic_expression_data()
    df = data_generator.create_dataframe(X, y)
    
    # Save raw data
    df.to_csv(os.path.join(args.output_dir, 'raw_data.csv'), index=False)
    logger.info(f"Generated data shape: {X.shape}, labels: {len(np.unique(y))}")
    
    # Process data
    logger.info("Processing data")
    data_processor = DataProcessor(random_seed=config.random_seed)
    processed_data = data_processor.prepare_data(
        df, 
        test_size=config.data.test_size,
        val_size=config.data.val_size
    )
    
    # Save processed data
    processed_data['X_train'].to_csv(os.path.join(args.output_dir, 'X_train.csv'), index=False)
    processed_data['X_val'].to_csv(os.path.join(args.output_dir, 'X_val.csv'), index=False)
    processed_data['X_test'].to_csv(os.path.join(args.output_dir, 'X_test.csv'), index=False)
    
    # Initialize model evaluator and comparison
    evaluator = ModelEvaluator(
        class_names=processed_data['class_names'].tolist(),
        output_dir=os.path.join(args.output_dir, 'results')
    )
    
    model_comparison = ModelComparison(
        output_dir=os.path.join(args.output_dir, 'results')
    )
    
    # Train multiple models for comparison
    models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name} model")
        
        try:
            # Create model
            model_params = config.model.params.get(model_name, {})
            model = create_model(model_name, model_params)
            
            # Train model
            model.fit(
                processed_data['X_train'], 
                processed_data['y_train'],
                processed_data['X_val'], 
                processed_data['y_val']
            )
            
            # Make predictions
            y_pred = model.predict(processed_data['X_test'])
            y_proba = model.predict_proba(processed_data['X_test'])
            
            # Evaluate model
            metrics = evaluator.evaluate_model(
                processed_data['y_test'], 
                y_pred, 
                y_proba, 
                model_name
            )
            
            # Add to comparison
            model_comparison.add_model_results(model_name, metrics)
            
            # Generate plots
            evaluator.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
            evaluator.plot_roc_curves(processed_data['y_test'], y_proba, model_name)
            evaluator.plot_precision_recall_curves(processed_data['y_test'], y_proba, model_name)
            evaluator.plot_calibration_curve(processed_data['y_test'], y_proba, model_name)
            
            # Explainability analysis
            logger.info(f"Analyzing explainability for {model_name}")
            explainer = ModelExplainer(
                model=model,
                feature_names=processed_data['feature_names'],
                class_names=processed_data['class_names'].tolist(),
                output_dir=os.path.join(args.output_dir, 'results')
            )
            
            # Analyze feature importance
            importance_results = explainer.analyze_feature_importance(
                processed_data['X_test'], 
                processed_data['y_test']
            )
            
            # Generate explanation plots
            if importance_results.get('importance_df') is not None:
                explainer.plot_feature_importance(importance_results['importance_df'], model_name)
            
            if 'shap_values' in importance_results:
                explainer.plot_shap_summary(importance_results['shap_values'], 
                                          processed_data['X_test'][:100], model_name)
            
            # Save explanation results
            explainer.save_explanation_results(importance_results, model_name)
            
            # Save model
            model_path = os.path.join(args.output_dir, 'models', f'{model_name}_model.pkl')
            model.save_model(model_path)
            
            logger.info(f"Successfully trained and evaluated {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue
    
    # Feature analysis
    logger.info("Performing feature analysis")
    feature_analyzer = FeatureAnalyzer(
        feature_names=processed_data['feature_names'],
        output_dir=os.path.join(args.output_dir, 'results')
    )
    
    # Analyze feature correlations
    feature_analyzer.analyze_feature_correlations(processed_data['X_train'])
    
    # Analyze feature distributions
    feature_analyzer.analyze_feature_distributions(
        processed_data['X_train'], 
        processed_data['y_train'],
        processed_data['class_names'].tolist()
    )
    
    # Identify discriminative features
    discriminative_features = feature_analyzer.identify_discriminative_features(
        processed_data['X_train'], 
        processed_data['y_train']
    )
    discriminative_features.to_csv(
        os.path.join(args.output_dir, 'results', 'discriminative_features.csv'), 
        index=False
    )
    
    # Create and save leaderboard
    logger.info("Creating model leaderboard")
    leaderboard = model_comparison.create_leaderboard()
    model_comparison.save_leaderboard()
    
    # Plot model comparison
    model_comparison.plot_model_comparison('f1_macro')
    model_comparison.plot_model_comparison('roc_auc_macro')
    
    # Save final results
    results_summary = {
        'best_model': leaderboard.iloc[0]['model_name'],
        'best_f1_score': leaderboard.iloc[0]['f1_macro'],
        'best_roc_auc': leaderboard.iloc[0].get('roc_auc_macro', 'N/A'),
        'n_samples': config.data.n_samples,
        'n_features': config.data.n_genes,
        'n_classes': config.data.n_subtypes,
        'random_seed': config.random_seed
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv(os.path.join(args.output_dir, 'results', 'training_summary.csv'), index=False)
    
    logger.info("Training completed successfully")
    logger.info(f"Best model: {results_summary['best_model']}")
    logger.info(f"Best F1-score: {results_summary['best_f1_score']:.4f}")
    
    # Print leaderboard
    print("\n" + "="*50)
    print("MODEL LEADERBOARD")
    print("="*50)
    print(leaderboard[['model_name', 'accuracy', 'f1_macro', 'roc_auc_macro']].to_string(index=False))
    print("="*50)


if __name__ == "__main__":
    main()
