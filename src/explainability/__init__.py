"""Explainability and feature importance analysis for cancer subtype classification."""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Comprehensive model explainability analysis."""
    
    def __init__(self, model: Any, feature_names: List[str], class_names: List[str], 
                 output_dir: str = "assets/results"):
        """Initialize model explainer.
        
        Args:
            model: Trained model to explain
            feature_names: List of feature names
            class_names: List of class names
            output_dir: Directory to save explanation results
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = output_dir
        self.n_features = len(feature_names)
        self.n_classes = len(class_names)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized model explainer for {self.n_features} features and {self.n_classes} classes")
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                 method: str = "shap") -> Dict[str, Any]:
        """Analyze feature importance using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            method: Method to use ("shap", "permutation", "builtin")
            
        Returns:
            Dictionary with feature importance results
        """
        logger.info(f"Analyzing feature importance using {method}")
        
        if method == "shap":
            return self._analyze_shap_importance(X, y)
        elif method == "permutation":
            return self._analyze_permutation_importance(X, y)
        elif method == "builtin":
            return self._analyze_builtin_importance()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _analyze_shap_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance using SHAP."""
        try:
            # Create SHAP explainer
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.Explainer(self.model.predict_proba, X[:100])  # Use subset for speed
                shap_values = explainer(X[:100])
            else:
                explainer = shap.Explainer(self.model.predict, X[:100])
                shap_values = explainer(X[:100])
            
            # Calculate mean absolute SHAP values
            if hasattr(shap_values, 'values'):
                mean_shap_values = np.abs(shap_values.values).mean(axis=0)
            else:
                mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Handle multiclass case
            if len(mean_shap_values.shape) > 1:
                mean_shap_values = mean_shap_values.mean(axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap_values
            }).sort_values('importance', ascending=False)
            
            # Save SHAP values for later use
            self.shap_values = shap_values
            self.shap_explainer = explainer
            
            logger.info("SHAP analysis completed")
            
            return {
                'method': 'shap',
                'importance_df': importance_df,
                'shap_values': shap_values,
                'explainer': explainer
            }
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return self._analyze_builtin_importance()
    
    def _analyze_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance using permutation importance."""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            logger.info("Permutation importance analysis completed")
            
            return {
                'method': 'permutation',
                'importance_df': importance_df,
                'perm_importance': perm_importance
            }
            
        except Exception as e:
            logger.warning(f"Permutation importance analysis failed: {e}")
            return self._analyze_builtin_importance()
    
    def _analyze_builtin_importance(self) -> Dict[str, Any]:
        """Analyze feature importance using built-in methods."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_values = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_values = np.abs(self.model.coef_).mean(axis=0)
            else:
                logger.warning("No built-in feature importance method available")
                return {'method': 'none', 'importance_df': None}
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            logger.info("Built-in feature importance analysis completed")
            
            return {
                'method': 'builtin',
                'importance_df': importance_df
            }
            
        except Exception as e:
            logger.warning(f"Built-in feature importance analysis failed: {e}")
            return {'method': 'none', 'importance_df': None}
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, 
                              top_n: int = 20, save_plot: bool = True) -> None:
        """Plot feature importance."""
        if importance_df is None:
            logger.warning("No feature importance data to plot")
            return
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_feature_importance.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_shap_summary(self, shap_values: Any, X: np.ndarray, model_name: str, 
                         save_plot: bool = True) -> None:
        """Plot SHAP summary plot."""
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, f"{model_name}_shap_summary.png"), 
                           dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")
    
    def plot_shap_waterfall(self, shap_values: Any, X: np.ndarray, sample_idx: int, 
                           model_name: str, save_plot: bool = True) -> None:
        """Plot SHAP waterfall plot for a specific sample."""
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap_values[sample_idx], show=False)
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx} - {model_name}')
            
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, f"{model_name}_shap_waterfall_{sample_idx}.png"), 
                           dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"SHAP waterfall plot failed: {e}")
    
    def explain_prediction(self, X_sample: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain prediction for a specific sample."""
        try:
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(X_sample.reshape(1, -1))[0]
                prediction_class = np.argmax(prediction_proba)
            else:
                prediction_class = self.model.predict(X_sample.reshape(1, -1))[0]
                prediction_proba = None
            
            # Get SHAP values if available
            shap_explanation = None
            if hasattr(self, 'shap_explainer'):
                try:
                    shap_values_sample = self.shap_explainer(X_sample.reshape(1, -1))
                    shap_explanation = {
                        'values': shap_values_sample.values[0] if hasattr(shap_values_sample, 'values') else shap_values_sample[0],
                        'base_values': shap_values_sample.base_values[0] if hasattr(shap_values_sample, 'base_values') else None
                    }
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            explanation = {
                'sample_idx': sample_idx,
                'predicted_class': prediction_class,
                'predicted_class_name': self.class_names[prediction_class],
                'prediction_probability': prediction_proba,
                'shap_explanation': shap_explanation,
                'feature_values': dict(zip(self.feature_names, X_sample))
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            return {'error': str(e)}
    
    def create_explanation_report(self, importance_results: Dict[str, Any], 
                                model_name: str) -> str:
        """Create a comprehensive explanation report."""
        report = f"""
# Model Explanation Report - {model_name}

## Feature Importance Analysis
Method: {importance_results.get('method', 'Unknown')}

### Top 10 Most Important Features
"""
        
        if importance_results.get('importance_df') is not None:
            top_features = importance_results['importance_df'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report += f"{i}. **{row['feature']}**: {row['importance']:.4f}\n"
        
        report += """
## Interpretation Guidelines

### For Researchers:
- High importance features may indicate biological relevance
- Consider these features for further biological validation
- Cross-reference with known cancer biology literature

### For Clinicians:
- These features should be interpreted in clinical context
- Consider patient-specific factors and comorbidities
- Use as supplementary information, not diagnostic criteria

## Limitations
- Feature importance does not imply causation
- Results are based on synthetic data
- Real-world validation required for clinical use
"""
        
        return report
    
    def save_explanation_results(self, importance_results: Dict[str, Any], 
                              model_name: str) -> None:
        """Save explanation results to files."""
        # Save feature importance
        if importance_results.get('importance_df') is not None:
            importance_results['importance_df'].to_csv(
                os.path.join(self.output_dir, f"{model_name}_feature_importance.csv"), 
                index=False
            )
        
        # Save explanation report
        report = self.create_explanation_report(importance_results, model_name)
        with open(os.path.join(self.output_dir, f"{model_name}_explanation_report.md"), 'w') as f:
            f.write(report)
        
        logger.info(f"Explanation results saved for {model_name}")


class FeatureAnalyzer:
    """Analyze feature relationships and patterns."""
    
    def __init__(self, feature_names: List[str], output_dir: str = "assets/results"):
        """Initialize feature analyzer.
        
        Args:
            feature_names: List of feature names
            output_dir: Directory to save analysis results
        """
        self.feature_names = feature_names
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_feature_correlations(self, X: np.ndarray, save_plot: bool = True) -> pd.DataFrame:
        """Analyze feature correlations."""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Create DataFrame
        corr_df = pd.DataFrame(corr_matrix, 
                              index=self.feature_names, 
                              columns=self.feature_names)
        
        # Plot correlation heatmap
        if save_plot:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_df, annot=False, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_correlations.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return corr_df
    
    def analyze_feature_distributions(self, X: np.ndarray, y: np.ndarray, 
                                    class_names: List[str], save_plot: bool = True) -> None:
        """Analyze feature distributions by class."""
        n_features = min(12, len(self.feature_names))  # Limit to 12 features for readability
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            for j, class_name in enumerate(class_names):
                class_mask = y == j
                axes[i].hist(X[class_mask, i], alpha=0.7, label=class_name, bins=20)
            
            axes[i].set_title(self.feature_names[i])
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions by Class')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def identify_discriminative_features(self, X: np.ndarray, y: np.ndarray, 
                                       top_n: int = 10) -> pd.DataFrame:
        """Identify most discriminative features using statistical tests."""
        from scipy.stats import f_oneway
        
        f_scores = []
        p_values = []
        
        for i in range(X.shape[1]):
            # Group data by class
            groups = [X[y == j, i] for j in range(len(np.unique(y)))]
            
            # Perform ANOVA
            f_stat, p_val = f_oneway(*groups)
            f_scores.append(f_stat)
            p_values.append(p_val)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature': self.feature_names,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)
        
        return results_df.head(top_n)
