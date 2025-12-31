"""Streamlit demo for cancer subtype classification."""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import GeneExpressionDataGenerator, DataProcessor
from src.models import create_model
from src.metrics import ModelEvaluator
from src.explainability import ModelExplainer

# Page configuration
st.set_page_config(
    page_title="Cancer Subtype Classification - Research Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6c3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧬 Cancer Subtype Classification</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>THIS IS A RESEARCH DEMONSTRATION ONLY</strong></p>
    <ul>
        <li>NOT for clinical diagnosis or treatment decisions</li>
        <li>NOT medical advice</li>
        <li>Results should NOT be used for patient care</li>
        <li>Requires proper clinical validation</li>
        <li>Use only under appropriate research supervision</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Configuration")

# Model selection
available_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
selected_model = st.sidebar.selectbox("Select Model", available_models)

# Data parameters
st.sidebar.subheader("📊 Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
n_genes = st.sidebar.slider("Number of Genes", 20, 100, 50)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=1000)

# Analysis options
st.sidebar.subheader("🔍 Analysis Options")
show_explainability = st.sidebar.checkbox("Show Explainability Analysis", value=True)
show_feature_analysis = st.sidebar.checkbox("Show Feature Analysis", value=True)
show_calibration = st.sidebar.checkbox("Show Calibration Analysis", value=True)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📈 Model Performance", "🔍 Explainability", "📊 Feature Analysis", "🎯 Predictions"])

with tab1:
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        This project demonstrates cancer subtype classification using gene expression data.
        The system can identify three common breast cancer subtypes:
        - **Luminal A**: ER+, PR+, HER2-, generally good prognosis
        - **HER2+**: HER2 overexpression, aggressive
        - **Triple Negative**: ER-, PR-, HER2-, aggressive, poor prognosis
        """)
        
        st.subheader("🔬 Methodology")
        st.write("""
        - **Data**: Synthetic gene expression data with realistic patterns
        - **Models**: Multiple ML algorithms (Random Forest, XGBoost, LightGBM, CatBoost)
        - **Evaluation**: Comprehensive metrics including calibration analysis
        - **Explainability**: SHAP analysis for feature importance
        """)
    
    with col2:
        st.subheader("📋 Dataset Information")
        st.write(f"""
        - **Samples**: {n_samples}
        - **Features**: {n_genes} genes
        - **Classes**: 3 cancer subtypes
        - **Split**: 60% train, 20% validation, 20% test
        - **Seed**: {random_seed}
        """)
        
        st.subheader("⚠️ Limitations")
        st.write("""
        - Uses synthetic data only
        - Limited to 3 cancer subtypes
        - No real-world validation
        - Simplified gene expression simulation
        - Not suitable for clinical use
        """)

with tab2:
    st.header("Model Performance Analysis")
    
    if st.button("🚀 Train and Evaluate Models", key="train_models"):
        with st.spinner("Training models and generating results..."):
            # Generate data
            data_generator = GeneExpressionDataGenerator(
                n_samples=n_samples,
                n_genes=n_genes,
                random_seed=random_seed
            )
            
            X, y = data_generator.generate_realistic_expression_data()
            df = data_generator.create_dataframe(X, y)
            
            # Process data
            data_processor = DataProcessor(random_seed=random_seed)
            processed_data = data_processor.prepare_data(df)
            
            # Train and evaluate models
            evaluator = ModelEvaluator(
                class_names=processed_data['class_names'].tolist(),
                output_dir="temp_results"
            )
            
            results = {}
            
            for model_name in available_models:
                try:
                    # Create and train model
                    model = create_model(model_name, {})
                    model.fit(
                        processed_data['X_train'], 
                        processed_data['y_train'],
                        processed_data['X_val'], 
                        processed_data['y_val']
                    )
                    
                    # Make predictions
                    y_pred = model.predict(processed_data['X_test'])
                    y_proba = model.predict_proba(processed_data['X_test'])
                    
                    # Evaluate
                    metrics = evaluator.evaluate_model(
                        processed_data['y_test'], 
                        y_pred, 
                        y_proba, 
                        model_name
                    )
                    
                    results[model_name] = {
                        'metrics': metrics,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'model': model
                    }
                    
                except Exception as e:
                    st.error(f"Failed to train {model_name}: {e}")
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['processed_data'] = processed_data
            st.session_state['data_generator'] = data_generator
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        processed_data = st.session_state['processed_data']
        
        # Model comparison
        st.subheader("📊 Model Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                'F1-Score': f"{metrics.get('f1_macro', 0):.3f}",
                'ROC AUC': f"{metrics.get('roc_auc_macro', 0):.3f}",
                'PR AUC': f"{metrics.get('pr_auc_macro', 0):.3f}",
                'ECE': f"{metrics.get('ece', 0):.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Performance plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 F1-Score Comparison")
            fig = px.bar(comparison_df, x='Model', y='F1-Score', 
                        title="F1-Score by Model", color='F1-Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 ROC AUC Comparison")
            fig = px.bar(comparison_df, x='Model', y='ROC AUC', 
                        title="ROC AUC by Model", color='ROC AUC')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics for selected model
        st.subheader(f"📋 Detailed Metrics - {selected_model.replace('_', ' ').title()}")
        
        if selected_model in results:
            metrics = results[selected_model]['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("F1-Score (Macro)", f"{metrics.get('f1_macro', 0):.3f}")
            with col3:
                st.metric("ROC AUC (Macro)", f"{metrics.get('roc_auc_macro', 0):.3f}")
            with col4:
                st.metric("Expected Calibration Error", f"{metrics.get('ece', 0):.3f}")
            
            # Confusion matrix
            st.subheader("🔍 Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=processed_data['class_names'], 
                       yticklabels=processed_data['class_names'], ax=ax)
            ax.set_title(f'Confusion Matrix - {selected_model.replace("_", " ").title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # ROC curves
            if show_calibration:
                st.subheader("📊 ROC Curves")
                y_proba = results[selected_model]['y_proba']
                y_test = processed_data['y_test']
                
                fig = go.Figure()
                
                for i, class_name in enumerate(processed_data['class_names']):
                    from sklearn.metrics import roc_curve, roc_auc_score
                    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
                    auc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{class_name} (AUC = {auc:.3f})',
                        line=dict(width=2)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    title='ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Explainability Analysis")
    
    if 'results' in st.session_state and show_explainability:
        results = st.session_state['results']
        processed_data = st.session_state['processed_data']
        
        if selected_model in results:
            model = results[selected_model]['model']
            
            # Feature importance analysis
            st.subheader("🔍 Feature Importance Analysis")
            
            explainer = ModelExplainer(
                model=model,
                feature_names=processed_data['feature_names'],
                class_names=processed_data['class_names'].tolist(),
                output_dir="temp_results"
            )
            
            # Analyze feature importance
            importance_results = explainer.analyze_feature_importance(
                processed_data['X_test'], 
                processed_data['y_test']
            )
            
            if importance_results.get('importance_df') is not None:
                importance_df = importance_results['importance_df']
                
                # Top 20 features
                top_features = importance_df.head(20)
                
                fig = px.bar(top_features, x='importance', y='feature', 
                           orientation='h', title="Top 20 Most Important Features")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.subheader("📋 Feature Importance Table")
                st.dataframe(importance_df.head(10), use_container_width=True)
            
            # Sample prediction explanation
            st.subheader("🎯 Sample Prediction Explanation")
            
            sample_idx = st.slider("Select Sample Index", 0, len(processed_data['X_test'])-1, 0)
            
            X_sample = processed_data['X_test'][sample_idx]
            explanation = explainer.explain_prediction(X_sample, sample_idx)
            
            if 'error' not in explanation:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Predicted Class**: {explanation['predicted_class_name']}")
                    st.write(f"**Sample Index**: {explanation['sample_idx']}")
                    
                    if explanation['prediction_probability'] is not None:
                        st.write("**Prediction Probabilities**:")
                        for i, prob in enumerate(explanation['prediction_probability']):
                            st.write(f"- {processed_data['class_names'][i]}: {prob:.3f}")
                
                with col2:
                    st.write("**Top 10 Feature Values**:")
                    feature_values = explanation['feature_values']
                    top_features_values = sorted(feature_values.items(), 
                                              key=lambda x: abs(x[1]), reverse=True)[:10]
                    
                    for feature, value in top_features_values:
                        st.write(f"- {feature}: {value:.3f}")
    
    else:
        st.info("Please train models first to see explainability analysis.")

with tab4:
    st.header("Feature Analysis")
    
    if 'results' in st.session_state and show_feature_analysis:
        processed_data = st.session_state['processed_data']
        
        # Feature correlation analysis
        st.subheader("🔗 Feature Correlation Analysis")
        
        # Calculate correlation matrix for top features
        X_train = processed_data['X_train']
        feature_names = processed_data['feature_names']
        
        # Select top 20 features by variance
        feature_vars = np.var(X_train, axis=0)
        top_feature_indices = np.argsort(feature_vars)[-20:]
        top_features = [feature_names[i] for i in top_feature_indices]
        
        corr_matrix = np.corrcoef(X_train[:, top_feature_indices].T)
        
        fig = px.imshow(corr_matrix, 
                       x=top_features, 
                       y=top_features,
                       title="Feature Correlation Matrix (Top 20 by Variance)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions by class
        st.subheader("📊 Feature Distributions by Class")
        
        # Select a few features to visualize
        selected_features = st.multiselect(
            "Select Features to Visualize", 
            feature_names, 
            default=feature_names[:5]
        )
        
        if selected_features:
            feature_indices = [feature_names.index(f) for f in selected_features]
            
            fig = make_subplots(
                rows=len(selected_features), 
                cols=1,
                subplot_titles=selected_features
            )
            
            for i, feature_idx in enumerate(feature_indices):
                for j, class_name in enumerate(processed_data['class_names']):
                    class_mask = processed_data['y_train'] == j
                    fig.add_trace(
                        go.Histogram(
                            x=X_train[class_mask, feature_idx],
                            name=class_name,
                            opacity=0.7
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(height=200*len(selected_features))
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please train models first to see feature analysis.")

with tab5:
    st.header("Interactive Predictions")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        processed_data = st.session_state['processed_data']
        
        if selected_model in results:
            model = results[selected_model]['model']
            
            st.subheader("🎯 Make Predictions")
            
            # Input method selection
            input_method = st.radio("Select Input Method", ["Random Sample", "Manual Input"])
            
            if input_method == "Random Sample":
                sample_idx = st.slider("Select Sample Index", 0, len(processed_data['X_test'])-1, 0)
                X_sample = processed_data['X_test'][sample_idx]
                
                # Show actual label
                actual_label = processed_data['class_names'][processed_data['y_test'][sample_idx]]
                st.write(f"**Actual Label**: {actual_label}")
                
            else:
                # Manual input
                st.write("Enter gene expression values (normalized):")
                
                X_sample = np.zeros(len(processed_data['feature_names']))
                
                # Create input fields for each feature
                cols = st.columns(5)
                for i, feature_name in enumerate(processed_data['feature_names']):
                    with cols[i % 5]:
                        X_sample[i] = st.number_input(
                            feature_name, 
                            value=0.0, 
                            min_value=-5.0, 
                            max_value=5.0, 
                            step=0.1,
                            key=f"feature_{i}"
                        )
            
            # Make prediction
            if st.button("🔮 Make Prediction"):
                y_pred = model.predict(X_sample.reshape(1, -1))
                y_proba = model.predict_proba(X_sample.reshape(1, -1))
                
                predicted_class = processed_data['class_names'][y_pred[0]]
                
                st.success(f"**Predicted Class**: {predicted_class}")
                
                # Show probabilities
                st.subheader("📊 Prediction Probabilities")
                
                prob_df = pd.DataFrame({
                    'Class': processed_data['class_names'],
                    'Probability': y_proba[0]
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Class', y='Probability', 
                           title="Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show probability table
                st.dataframe(prob_df, use_container_width=True)
    
    else:
        st.info("Please train models first to make predictions.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Cancer Subtype Classification - Research Demo</strong></p>
    <p>This is a research demonstration project only. Not for clinical use.</p>
    <p>For questions about this research tool, please contact the development team.</p>
</div>
""", unsafe_allow_html=True)
