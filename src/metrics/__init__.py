"""Evaluation metrics and model assessment for cancer subtype classification."""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    calibration_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for cancer subtype classification."""
    
    def __init__(self, class_names: List[str], output_dir: str = "assets/results"):
        """Initialize model evaluator.
        
        Args:
            class_names: List of class names
            output_dir: Directory to save evaluation results
        """
        self.class_names = class_names
        self.output_dir = output_dir
        self.n_classes = len(class_names)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized model evaluator for {self.n_classes} classes")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None, model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Basic classification metrics
        metrics = self._calculate_classification_metrics(y_true, y_pred, y_proba)
        
        # Calibration metrics
        if y_proba is not None:
            calibration_metrics = self._calculate_calibration_metrics(y_true, y_proba)
            metrics.update(calibration_metrics)
        
        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(y_true, y_pred, y_proba)
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Save results
        self._save_evaluation_results(metrics, model_name)
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # ROC AUC and PR AUC
        if y_proba is not None:
            if self.n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                metrics['pr_auc_macro'] = average_precision_score(y_true, y_proba, average='macro')
                metrics['pr_auc_weighted'] = average_precision_score(y_true, y_proba, average='weighted')
        
        return metrics
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        if self.n_classes == 2:
            # Binary classification
            brier_score = brier_score_loss(y_true, y_proba[:, 1])
            metrics['brier_score'] = brier_score
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_true, y_proba[:, 1])
            metrics['ece'] = ece
            
        else:
            # Multiclass classification
            # Calculate Brier score for each class
            brier_scores = []
            for i in range(self.n_classes):
                class_brier = brier_score_loss((y_true == i).astype(int), y_proba[:, i])
                brier_scores.append(class_brier)
            
            metrics['brier_score_macro'] = np.mean(brier_scores)
            metrics['brier_score_weighted'] = np.average(brier_scores, weights=np.bincount(y_true))
            
            # ECE for multiclass
            ece = self._calculate_multiclass_ece(y_true, y_proba)
            metrics['ece'] = ece
        
        return metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error for binary classification."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_multiclass_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error for multiclass classification."""
        ece = 0
        for i in range(self.n_classes):
            class_ece = self._calculate_ece((y_true == i).astype(int), y_proba[:, i], n_bins)
            ece += class_ece
        
        return ece / self.n_classes
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            class_metrics = {
                'precision': precision_score(y_true, y_pred, labels=[i], average='micro'),
                'recall': recall_score(y_true, y_pred, labels=[i], average='micro'),
                'f1': f1_score(y_true, y_pred, labels=[i], average='micro')
            }
            
            if y_proba is not None:
                if self.n_classes == 2:
                    class_metrics['roc_auc'] = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                    class_metrics['pr_auc'] = average_precision_score((y_true == i).astype(int), y_proba[:, i])
                else:
                    class_metrics['roc_auc'] = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                    class_metrics['pr_auc'] = average_precision_score((y_true == i).astype(int), y_proba[:, i])
            
            per_class_metrics[class_name] = class_metrics
        
        return per_class_metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, Any], model_name: str) -> None:
        """Save evaluation results to files."""
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, f"{model_name}_metrics.csv"), index=False)
        
        # Save per-class metrics
        if 'per_class' in metrics:
            per_class_df = pd.DataFrame(metrics['per_class']).T
            per_class_df.to_csv(os.path.join(self.output_dir, f"{model_name}_per_class_metrics.csv"))
        
        logger.info(f"Evaluation results saved for {model_name}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save_plot: bool = True) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       model_name: str, save_plot: bool = True) -> None:
        """Plot ROC curves."""
        if self.n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
        else:
            # Multiclass classification
            plt.figure(figsize=(10, 8))
            
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                roc_auc = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {model_name}')
            plt.legend(loc="lower right")
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_roc_curves.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   model_name: str, save_plot: bool = True) -> None:
        """Plot precision-recall curves."""
        if self.n_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = average_precision_score(y_true, y_proba[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2, 
                    label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend(loc="lower left")
            
        else:
            # Multiclass classification
            plt.figure(figsize=(10, 8))
            
            for i, class_name in enumerate(self.class_names):
                precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_proba[:, i])
                pr_auc = average_precision_score((y_true == i).astype(int), y_proba[:, i])
                plt.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves - {model_name}')
            plt.legend(loc="lower left")
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_pr_curves.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                             model_name: str, save_plot: bool = True) -> None:
        """Plot calibration curve."""
        if self.n_classes == 2:
            # Binary classification
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba[:, 1], n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                    label=f'{model_name}')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Curve - {model_name}')
            plt.legend()
            
        else:
            # Multiclass classification
            fig, axes = plt.subplots(1, self.n_classes, figsize=(4*self.n_classes, 4))
            if self.n_classes == 1:
                axes = [axes]
            
            for i, class_name in enumerate(self.class_names):
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    (y_true == i).astype(int), y_proba[:, i], n_bins=10
                )
                
                axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", 
                           label=f'{class_name}')
                axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'{class_name}')
                axes[i].legend()
            
            plt.suptitle(f'Calibration Curves - {model_name}')
            plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"{model_name}_calibration.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_evaluation_report(self, metrics: Dict[str, Any], model_name: str) -> str:
        """Create a comprehensive evaluation report."""
        report = f"""
# Evaluation Report - {model_name}

## Overall Performance
- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}
- **Precision (Macro)**: {metrics.get('precision_macro', 'N/A'):.4f}
- **Recall (Macro)**: {metrics.get('recall_macro', 'N/A'):.4f}
- **F1-Score (Macro)**: {metrics.get('f1_macro', 'N/A'):.4f}

## ROC AUC Performance
- **ROC AUC (Macro)**: {metrics.get('roc_auc_macro', 'N/A'):.4f}
- **ROC AUC (Weighted)**: {metrics.get('roc_auc_weighted', 'N/A'):.4f}

## Precision-Recall AUC Performance
- **PR AUC (Macro)**: {metrics.get('pr_auc_macro', 'N/A'):.4f}
- **PR AUC (Weighted)**: {metrics.get('pr_auc_weighted', 'N/A'):.4f}

## Calibration Performance
- **Brier Score (Macro)**: {metrics.get('brier_score_macro', 'N/A'):.4f}
- **Expected Calibration Error**: {metrics.get('ece', 'N/A'):.4f}

## Per-Class Performance
"""
        
        if 'per_class' in metrics:
            for class_name, class_metrics in metrics['per_class'].items():
                report += f"""
### {class_name}
- **Precision**: {class_metrics.get('precision', 'N/A'):.4f}
- **Recall**: {class_metrics.get('recall', 'N/A'):.4f}
- **F1-Score**: {class_metrics.get('f1', 'N/A'):.4f}
- **ROC AUC**: {class_metrics.get('roc_auc', 'N/A'):.4f}
- **PR AUC**: {class_metrics.get('pr_auc', 'N/A'):.4f}
"""
        
        return report


class ModelComparison:
    """Compare multiple models and create leaderboard."""
    
    def __init__(self, output_dir: str = "assets/results"):
        """Initialize model comparison.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def add_model_results(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Add model results to comparison.
        
        Args:
            model_name: Name of the model
            metrics: Evaluation metrics
        """
        result = {'model_name': model_name}
        result.update(metrics)
        self.results.append(result)
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard DataFrame."""
        if not self.results:
            raise ValueError("No model results to compare")
        
        df = pd.DataFrame(self.results)
        
        # Sort by F1-score (macro)
        if 'f1_macro' in df.columns:
            df = df.sort_values('f1_macro', ascending=False)
        
        return df
    
    def save_leaderboard(self, filename: str = "model_leaderboard.csv") -> None:
        """Save leaderboard to CSV."""
        leaderboard = self.create_leaderboard()
        leaderboard.to_csv(os.path.join(self.output_dir, filename), index=False)
        logger.info(f"Leaderboard saved to {filename}")
    
    def plot_model_comparison(self, metric: str = 'f1_macro', save_plot: bool = True) -> None:
        """Plot model comparison for a specific metric."""
        if not self.results:
            raise ValueError("No model results to compare")
        
        df = pd.DataFrame(self.results)
        
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in results")
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['model_name'], df[metric])
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, f"model_comparison_{metric}.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
