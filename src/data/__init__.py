"""Data generation and processing for cancer subtype classification."""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


class GeneExpressionDataGenerator:
    """Generate synthetic gene expression data for cancer subtype classification."""
    
    def __init__(self, n_samples: int = 300, n_genes: int = 50, n_subtypes: int = 3, 
                 subtypes: Optional[list] = None, random_seed: int = 42):
        """Initialize data generator.
        
        Args:
            n_samples: Number of samples to generate
            n_genes: Number of genes (features)
            n_subtypes: Number of cancer subtypes
            subtypes: List of subtype names
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_subtypes = n_subtypes
        self.subtypes = subtypes or ["Luminal A", "HER2+", "Triple Negative"]
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Generate gene names
        self.gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
        
        logger.info(f"Initialized data generator: {n_samples} samples, {n_genes} genes, {n_subtypes} subtypes")
    
    def generate_realistic_expression_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic gene expression data with subtype-specific patterns.
        
        Returns:
            Tuple of (expression_matrix, labels)
        """
        # Initialize expression matrix
        X = np.zeros((self.n_samples, self.n_genes))
        y = np.zeros(self.n_samples, dtype=object)
        
        # Define subtype-specific gene expression patterns
        subtype_patterns = self._get_subtype_patterns()
        
        # Generate samples for each subtype
        samples_per_subtype = self.n_samples // self.n_subtypes
        remaining_samples = self.n_samples % self.n_subtypes
        
        start_idx = 0
        for i, subtype in enumerate(self.subtypes):
            # Calculate number of samples for this subtype
            n_subtype_samples = samples_per_subtype + (1 if i < remaining_samples else 0)
            end_idx = start_idx + n_subtype_samples
            
            # Generate expression data for this subtype
            X[start_idx:end_idx], y[start_idx:end_idx] = self._generate_subtype_data(
                n_subtype_samples, subtype, subtype_patterns[subtype]
            )
            
            start_idx = end_idx
        
        # Shuffle the data
        indices = np.random.permutation(self.n_samples)
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Generated expression data: {X.shape}, labels: {len(np.unique(y))} subtypes")
        return X, y
    
    def _get_subtype_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define subtype-specific gene expression patterns.
        
        Returns:
            Dictionary with subtype-specific patterns
        """
        patterns = {}
        
        # Luminal A: ER+, PR+, HER2-, generally good prognosis
        patterns["Luminal A"] = {
            "highly_expressed": list(range(0, 10)),  # First 10 genes
            "lowly_expressed": list(range(40, 50)),  # Last 10 genes
            "noise_level": 0.1,
            "baseline_expression": 0.3
        }
        
        # HER2+: HER2 overexpression, aggressive
        patterns["HER2+"] = {
            "highly_expressed": list(range(10, 20)),  # Genes 10-19
            "lowly_expressed": list(range(30, 40)),   # Genes 30-39
            "noise_level": 0.15,
            "baseline_expression": 0.4
        }
        
        # Triple Negative: ER-, PR-, HER2-, aggressive, poor prognosis
        patterns["Triple Negative"] = {
            "highly_expressed": list(range(20, 30)),  # Genes 20-29
            "lowly_expressed": list(range(0, 10)),    # Genes 0-9
            "noise_level": 0.2,
            "baseline_expression": 0.5
        }
        
        return patterns
    
    def _generate_subtype_data(self, n_samples: int, subtype: str, pattern: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate expression data for a specific subtype.
        
        Args:
            n_samples: Number of samples to generate
            subtype: Subtype name
            pattern: Subtype-specific pattern
            
        Returns:
            Tuple of (expression_data, labels)
        """
        X_subtype = np.zeros((n_samples, self.n_genes))
        
        # Generate baseline expression
        baseline = pattern["baseline_expression"]
        noise_level = pattern["noise_level"]
        
        # Add subtype-specific patterns
        for i in range(n_samples):
            # Start with baseline expression
            expression = np.full(self.n_genes, baseline)
            
            # Add high expression for subtype-specific genes
            for gene_idx in pattern["highly_expressed"]:
                expression[gene_idx] += np.random.normal(0.3, 0.1)
            
            # Add low expression for subtype-specific genes
            for gene_idx in pattern["lowly_expressed"]:
                expression[gene_idx] += np.random.normal(-0.2, 0.1)
            
            # Add noise
            expression += np.random.normal(0, noise_level, self.n_genes)
            
            # Ensure non-negative values (log-transformed data)
            expression = np.maximum(expression, 0)
            
            X_subtype[i] = expression
        
        # Create labels
        y_subtype = np.full(n_samples, subtype)
        
        return X_subtype, y_subtype
    
    def create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Create a pandas DataFrame from expression data.
        
        Args:
            X: Expression matrix
            y: Labels
            
        Returns:
            DataFrame with expression data and labels
        """
        # Create DataFrame with gene names as columns
        df = pd.DataFrame(X, columns=self.gene_names)
        
        # Add labels
        df['cancer_subtype'] = y
        
        # Add sample IDs
        df['sample_id'] = [f"SAMPLE_{i:04d}" for i in range(len(df))]
        
        # Reorder columns
        df = df[['sample_id', 'cancer_subtype'] + self.gene_names]
        
        return df


class DataProcessor:
    """Process and prepare gene expression data for machine learning."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize data processor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        logger.info("Initialized data processor")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """Prepare data for training, validation, and testing.
        
        Args:
            df: DataFrame with expression data and labels
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with prepared data splits
        """
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('GENE_')]
        X = df[feature_cols].values
        y = df['cancer_subtype'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_seed, stratify=y_encoded
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=self.random_seed, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.is_fitted = True
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        logger.info(f"Data prepared: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_cols,
            'class_names': self.label_encoder.classes_,
            'class_weights': class_weight_dict,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original labels.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Original label names
        """
        if not self.is_fitted:
            raise ValueError("Data processor must be fitted before inverse transform")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save_processor(self, save_path: str) -> None:
        """Save the fitted processor.
        
        Args:
            save_path: Path to save the processor
        """
        import joblib
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted
        }, save_path)
        
        logger.info(f"Data processor saved to {save_path}")
    
    def load_processor(self, load_path: str) -> None:
        """Load a fitted processor.
        
        Args:
            load_path: Path to load the processor from
        """
        import joblib
        
        processor_data = joblib.load(load_path)
        self.scaler = processor_data['scaler']
        self.label_encoder = processor_data['label_encoder']
        self.is_fitted = processor_data['is_fitted']
        
        logger.info(f"Data processor loaded from {load_path}")
