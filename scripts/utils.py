"""Utility scripts for cancer subtype classification project."""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import GeneExpressionDataGenerator, DataProcessor
from src.utils import setup_logging, set_deterministic_seed


def generate_synthetic_dataset():
    """Generate and save synthetic dataset."""
    parser = argparse.ArgumentParser(description='Generate synthetic cancer subtype dataset')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--n-genes', type=int, default=100,
                       help='Number of genes (features)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Generating synthetic dataset: {args.n_samples} samples, {args.n_genes} genes")
    
    # Generate data
    data_generator = GeneExpressionDataGenerator(
        n_samples=args.n_samples,
        n_genes=args.n_genes,
        random_seed=args.seed
    )
    
    X, y = data_generator.generate_realistic_expression_data()
    df = data_generator.create_dataframe(X, y)
    
    # Save data
    output_path = os.path.join(args.output_dir, 'synthetic_cancer_data.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset saved to {output_path}")
    
    # Generate summary statistics
    summary_stats = {
        'n_samples': args.n_samples,
        'n_genes': args.n_genes,
        'n_subtypes': len(np.unique(y)),
        'subtype_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'feature_stats': {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'dataset_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    logger.info(f"Summary statistics saved to {summary_path}")


def analyze_dataset():
    """Analyze dataset characteristics."""
    parser = argparse.ArgumentParser(description='Analyze cancer subtype dataset')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--output-dir', type=str, default='assets/analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Analyzing dataset: {args.data_path}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Basic info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Class distribution
    if 'cancer_subtype' in df.columns:
        class_counts = df['cancer_subtype'].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar')
        plt.title('Cancer Subtype Distribution')
        plt.xlabel('Subtype')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'class_distribution.png'), dpi=300)
        plt.close()
    
    # Feature analysis
    feature_cols = [col for col in df.columns if col.startswith('GENE_')]
    if feature_cols:
        X = df[feature_cols].values
        
        # Feature statistics
        feature_stats = pd.DataFrame({
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'q25': np.percentile(X, 25, axis=0),
            'q75': np.percentile(X, 75, axis=0)
        }, index=feature_cols)
        
        logger.info(f"Feature statistics:\n{feature_stats.head()}")
        
        # Save feature statistics
        feature_stats.to_csv(os.path.join(args.output_dir, 'feature_statistics.csv'))
        
        # Plot feature distributions
        plt.figure(figsize=(15, 10))
        
        # Select a few features for visualization
        selected_features = feature_cols[:12]
        
        for i, feature in enumerate(selected_features):
            plt.subplot(3, 4, i+1)
            plt.hist(X[:, i], bins=30, alpha=0.7)
            plt.title(feature)
            plt.xlabel('Expression Value')
            plt.ylabel('Frequency')
        
        plt.suptitle('Feature Distributions')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
        
        # Correlation analysis
        corr_matrix = np.corrcoef(X.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'feature_correlations.png'), dpi=300)
        plt.close()
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")


def compare_models():
    """Compare multiple trained models."""
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='assets/comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Comparing models from {args.results_dir}")
    
    # Find all model result files
    result_files = []
    for file in os.listdir(args.results_dir):
        if file.endswith('_metrics.csv'):
            result_files.append(file)
    
    if not result_files:
        logger.error("No model result files found")
        return
    
    # Load results
    results = {}
    for file in result_files:
        model_name = file.replace('_metrics.csv', '')
        results[model_name] = pd.read_csv(os.path.join(args.results_dir, file))
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, df in results.items():
        if not df.empty:
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': df.iloc[0].get('accuracy', 0),
                'F1-Score': df.iloc[0].get('f1_macro', 0),
                'ROC AUC': df.iloc[0].get('roc_auc_macro', 0),
                'PR AUC': df.iloc[0].get('pr_auc_macro', 0),
                'ECE': df.iloc[0].get('ece', 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    # Save comparison
    comparison_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
    
    # Plot comparison
    metrics = ['Accuracy', 'F1-Score', 'ROC AUC', 'PR AUC']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        axes[i].bar(comparison_df['Model'], comparison_df[metric])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    logger.info(f"Model comparison completed. Results saved to {args.output_dir}")
    logger.info(f"Best model: {comparison_df.iloc[0]['Model']}")
    logger.info(f"Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")


def main():
    """Main function for utility scripts."""
    parser = argparse.ArgumentParser(description='Utility scripts for cancer subtype classification')
    parser.add_argument('command', choices=['generate', 'analyze', 'compare'],
                       help='Command to execute')
    
    args, unknown_args = parser.parse_known_args()
    
    if args.command == 'generate':
        # Reset sys.argv for the subcommand
        sys.argv = ['generate_dataset'] + unknown_args
        generate_synthetic_dataset()
    elif args.command == 'analyze':
        sys.argv = ['analyze_dataset'] + unknown_args
        analyze_dataset()
    elif args.command == 'compare':
        sys.argv = ['compare_models'] + unknown_args
        compare_models()


if __name__ == "__main__":
    main()
