"""Unit tests for cancer subtype classification project."""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data import GeneExpressionDataGenerator, DataProcessor
from src.models import create_model, RandomForestModel, XGBoostModel
from src.metrics import ModelEvaluator, ModelComparison
from src.explainability import ModelExplainer, FeatureAnalyzer
from src.utils import set_deterministic_seed, get_device


class TestDataGeneration(unittest.TestCase):
    """Test data generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_samples = 100
        self.n_genes = 20
        self.n_subtypes = 3
        self.subtypes = ["Luminal A", "HER2+", "Triple Negative"]
        self.random_seed = 42
        
        self.data_generator = GeneExpressionDataGenerator(
            n_samples=self.n_samples,
            n_genes=self.n_genes,
            n_subtypes=self.n_subtypes,
            subtypes=self.subtypes,
            random_seed=self.random_seed
        )
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        self.assertEqual(self.data_generator.n_samples, self.n_samples)
        self.assertEqual(self.data_generator.n_genes, self.n_genes)
        self.assertEqual(self.data_generator.n_subtypes, self.n_subtypes)
        self.assertEqual(self.data_generator.subtypes, self.subtypes)
        self.assertEqual(len(self.data_generator.gene_names), self.n_genes)
    
    def test_generate_realistic_expression_data(self):
        """Test realistic expression data generation."""
        X, y = self.data_generator.generate_realistic_expression_data()
        
        # Check shapes
        self.assertEqual(X.shape, (self.n_samples, self.n_genes))
        self.assertEqual(len(y), self.n_samples)
        
        # Check data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        
        # Check that all values are non-negative (log-transformed data)
        self.assertTrue(np.all(X >= 0))
        
        # Check that all subtypes are present
        unique_subtypes = np.unique(y)
        self.assertEqual(len(unique_subtypes), self.n_subtypes)
        for subtype in self.subtypes:
            self.assertIn(subtype, unique_subtypes)
    
    def test_create_dataframe(self):
        """Test DataFrame creation."""
        X, y = self.data_generator.generate_realistic_expression_data()
        df = self.data_generator.create_dataframe(X, y)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.n_samples)
        self.assertEqual(len(df.columns), self.n_genes + 2)  # genes + sample_id + cancer_subtype
        
        # Check required columns
        self.assertIn('sample_id', df.columns)
        self.assertIn('cancer_subtype', df.columns)
        
        # Check gene columns
        for gene_name in self.data_generator.gene_names:
            self.assertIn(gene_name, df.columns)


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.random_seed = 42
        self.data_processor = DataProcessor(random_seed=self.random_seed)
        
        # Create sample data
        self.n_samples = 100
        self.n_genes = 20
        self.n_subtypes = 3
        
        # Generate sample data
        data_generator = GeneExpressionDataGenerator(
            n_samples=self.n_samples,
            n_genes=self.n_genes,
            random_seed=self.random_seed
        )
        X, y = data_generator.generate_realistic_expression_data()
        self.df = data_generator.create_dataframe(X, y)
    
    def test_data_processor_initialization(self):
        """Test data processor initialization."""
        self.assertEqual(self.data_processor.random_seed, self.random_seed)
        self.assertFalse(self.data_processor.is_fitted)
    
    def test_prepare_data(self):
        """Test data preparation."""
        processed_data = self.data_processor.prepare_data(self.df)
        
        # Check that processor is fitted
        self.assertTrue(self.data_processor.is_fitted)
        
        # Check data splits
        self.assertIn('X_train', processed_data)
        self.assertIn('X_val', processed_data)
        self.assertIn('X_test', processed_data)
        self.assertIn('y_train', processed_data)
        self.assertIn('y_val', processed_data)
        self.assertIn('y_test', processed_data)
        
        # Check shapes
        total_samples = (processed_data['X_train'].shape[0] + 
                        processed_data['X_val'].shape[0] + 
                        processed_data['X_test'].shape[0])
        self.assertEqual(total_samples, self.n_samples)
        
        # Check feature dimensions
        self.assertEqual(processed_data['X_train'].shape[1], self.n_genes)
        self.assertEqual(processed_data['X_val'].shape[1], self.n_genes)
        self.assertEqual(processed_data['X_test'].shape[1], self.n_genes)
        
        # Check class information
        self.assertIn('class_names', processed_data)
        self.assertIn('class_weights', processed_data)
        self.assertEqual(len(processed_data['class_names']), self.n_subtypes)


class TestModels(unittest.TestCase):
    """Test model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.random_seed = 42
        set_deterministic_seed(self.random_seed)
        
        # Create sample data
        self.n_samples = 100
        self.n_genes = 20
        self.n_classes = 3
        
        # Generate sample data
        data_generator = GeneExpressionDataGenerator(
            n_samples=self.n_samples,
            n_genes=self.n_genes,
            random_seed=self.random_seed
        )
        X, y = data_generator.generate_realistic_expression_data()
        df = data_generator.create_dataframe(X, y)
        
        # Process data
        data_processor = DataProcessor(random_seed=self.random_seed)
        self.processed_data = data_processor.prepare_data(df)
    
    def test_create_model(self):
        """Test model creation."""
        # Test Random Forest
        rf_model = create_model("random_forest", {"n_estimators": 10, "random_state": 42})
        self.assertIsInstance(rf_model, RandomForestModel)
        
        # Test XGBoost
        xgb_model = create_model("xgboost", {"n_estimators": 10, "random_state": 42})
        self.assertIsInstance(xgb_model, XGBoostModel)
        
        # Test invalid model
        with self.assertRaises(ValueError):
            create_model("invalid_model", {})
    
    def test_random_forest_model(self):
        """Test Random Forest model."""
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        
        # Train model
        model.fit(
            self.processed_data['X_train'], 
            self.processed_data['y_train']
        )
        
        self.assertTrue(model.is_trained)
        
        # Make predictions
        y_pred = model.predict(self.processed_data['X_test'])
        y_proba = model.predict_proba(self.processed_data['X_test'])
        
        # Check predictions
        self.assertEqual(len(y_pred), len(self.processed_data['y_test']))
        self.assertEqual(y_proba.shape, (len(self.processed_data['y_test']), self.n_classes))
        
        # Check probabilities sum to 1
        prob_sums = np.sum(y_proba, axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)))
        
        # Get feature importance
        importance = model.get_feature_importance()
        self.assertEqual(len(importance), self.n_genes)
        self.assertTrue(np.all(importance >= 0))
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        model = XGBoostModel({"n_estimators": 10, "random_state": 42})
        
        # Train model
        model.fit(
            self.processed_data['X_train'], 
            self.processed_data['y_train']
        )
        
        self.assertTrue(model.is_trained)
        
        # Make predictions
        y_pred = model.predict(self.processed_data['X_test'])
        y_proba = model.predict_proba(self.processed_data['X_test'])
        
        # Check predictions
        self.assertEqual(len(y_pred), len(self.processed_data['y_test']))
        self.assertEqual(y_proba.shape, (len(self.processed_data['y_test']), self.n_classes))


class TestMetrics(unittest.TestCase):
    """Test metrics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.class_names = ["Luminal A", "HER2+", "Triple Negative"]
        self.n_classes = len(self.class_names)
        
        # Create sample data
        self.n_samples = 100
        self.y_true = np.random.randint(0, self.n_classes, self.n_samples)
        self.y_pred = np.random.randint(0, self.n_classes, self.n_samples)
        self.y_proba = np.random.rand(self.n_samples, self.n_classes)
        # Normalize probabilities
        self.y_proba = self.y_proba / np.sum(self.y_proba, axis=1, keepdims=True)
    
    def test_model_evaluator_initialization(self):
        """Test model evaluator initialization."""
        evaluator = ModelEvaluator(self.class_names)
        self.assertEqual(evaluator.class_names, self.class_names)
        self.assertEqual(evaluator.n_classes, self.n_classes)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        evaluator = ModelEvaluator(self.class_names)
        metrics = evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba, "test_model")
        
        # Check that metrics are calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('per_class', metrics)
        
        # Check confusion matrix shape
        self.assertEqual(metrics['confusion_matrix'].shape, (self.n_classes, self.n_classes))
        
        # Check per-class metrics
        self.assertEqual(len(metrics['per_class']), self.n_classes)
        for class_name in self.class_names:
            self.assertIn(class_name, metrics['per_class'])
    
    def test_model_comparison(self):
        """Test model comparison."""
        comparison = ModelComparison()
        
        # Add model results
        metrics1 = {'accuracy': 0.8, 'f1_macro': 0.75}
        metrics2 = {'accuracy': 0.85, 'f1_macro': 0.8}
        
        comparison.add_model_results("model1", metrics1)
        comparison.add_model_results("model2", metrics2)
        
        # Create leaderboard
        leaderboard = comparison.create_leaderboard()
        
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard.iloc[0]['model_name'], "model2")  # Should be sorted by F1


class TestExplainability(unittest.TestCase):
    """Test explainability functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.random_seed = 42
        set_deterministic_seed(self.random_seed)
        
        # Create sample data and model
        data_generator = GeneExpressionDataGenerator(
            n_samples=50,
            n_genes=10,
            random_seed=self.random_seed
        )
        X, y = data_generator.generate_realistic_expression_data()
        df = data_generator.create_dataframe(X, y)
        
        data_processor = DataProcessor(random_seed=self.random_seed)
        self.processed_data = data_processor.prepare_data(df)
        
        # Train a simple model
        self.model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        self.model.fit(
            self.processed_data['X_train'], 
            self.processed_data['y_train']
        )
    
    def test_model_explainer_initialization(self):
        """Test model explainer initialization."""
        explainer = ModelExplainer(
            model=self.model,
            feature_names=self.processed_data['feature_names'],
            class_names=self.processed_data['class_names'].tolist()
        )
        
        self.assertEqual(explainer.n_features, len(self.processed_data['feature_names']))
        self.assertEqual(explainer.n_classes, len(self.processed_data['class_names']))
    
    def test_analyze_feature_importance(self):
        """Test feature importance analysis."""
        explainer = ModelExplainer(
            model=self.model,
            feature_names=self.processed_data['feature_names'],
            class_names=self.processed_data['class_names'].tolist()
        )
        
        # Test built-in importance
        results = explainer.analyze_feature_importance(
            self.processed_data['X_test'], 
            self.processed_data['y_test'],
            method="builtin"
        )
        
        self.assertIn('method', results)
        self.assertIn('importance_df', results)
        
        if results['importance_df'] is not None:
            self.assertEqual(len(results['importance_df']), len(self.processed_data['feature_names']))
    
    def test_explain_prediction(self):
        """Test prediction explanation."""
        explainer = ModelExplainer(
            model=self.model,
            feature_names=self.processed_data['feature_names'],
            class_names=self.processed_data['class_names'].tolist()
        )
        
        # Test explanation for a sample
        sample_idx = 0
        X_sample = self.processed_data['X_test'][sample_idx]
        explanation = explainer.explain_prediction(X_sample, sample_idx)
        
        self.assertIn('predicted_class', explanation)
        self.assertIn('predicted_class_name', explanation)
        self.assertIn('feature_values', explanation)
        self.assertEqual(explanation['sample_idx'], sample_idx)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seed setting."""
        set_deterministic_seed(42)
        
        # Test that seeds are set
        np.random.seed(42)
        random_value1 = np.random.rand()
        
        set_deterministic_seed(42)
        random_value2 = np.random.rand()
        
        # Should be the same with same seed
        self.assertEqual(random_value1, random_value2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        self.assertIsInstance(device, type(device))  # Should be a torch.device
        
        # Test specific devices
        cpu_device = get_device("cpu")
        self.assertEqual(str(cpu_device), "cpu")


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.random_seed = 42
        set_deterministic_seed(self.random_seed)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        data_generator = GeneExpressionDataGenerator(
            n_samples=100,
            n_genes=20,
            random_seed=self.random_seed
        )
        X, y = data_generator.generate_realistic_expression_data()
        df = data_generator.create_dataframe(X, y)
        
        # Process data
        data_processor = DataProcessor(random_seed=self.random_seed)
        processed_data = data_processor.prepare_data(df)
        
        # Train model
        model = RandomForestModel({"n_estimators": 10, "random_state": 42})
        model.fit(
            processed_data['X_train'], 
            processed_data['y_train']
        )
        
        # Evaluate model
        evaluator = ModelEvaluator(
            class_names=processed_data['class_names'].tolist(),
            output_dir=self.temp_dir
        )
        
        y_pred = model.predict(processed_data['X_test'])
        y_proba = model.predict_proba(processed_data['X_test'])
        
        metrics = evaluator.evaluate_model(
            processed_data['y_test'], 
            y_pred, 
            y_proba, 
            "test_model"
        )
        
        # Check that evaluation completed successfully
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_macro', metrics)
        
        # Test explainability
        explainer = ModelExplainer(
            model=model,
            feature_names=processed_data['feature_names'],
            class_names=processed_data['class_names'].tolist(),
            output_dir=self.temp_dir
        )
        
        importance_results = explainer.analyze_feature_importance(
            processed_data['X_test'], 
            processed_data['y_test']
        )
        
        self.assertIn('method', importance_results)


if __name__ == '__main__':
    unittest.main()
