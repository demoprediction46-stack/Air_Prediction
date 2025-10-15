import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class AirQualityPredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # You can load your pre-trained models here
            # For demonstration, we'll create placeholder models
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'XGBoost': xgb.XGBRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'KNN': KNeighborsRegressor(n_neighbors=3),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0)
            }
        except Exception as e:
            st.warning(f"Some models could not be loaded: {e}")

    def train_models(self, X, y):
        """Train all models on the provided data"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = metrics.mean_absolute_error(y_test, y_pred)
                mse = metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = metrics.r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
            except Exception as e:
                st.error(f"Error training {name}: {e}")
        
        return results, X_test, y_test

def analyze_dataset(df, dataset_name):
    """Perform comprehensive analysis on a dataset"""
    st.markdown(f"### 📊 Analysis for {dataset_name}")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Samples", len(df))
    with col2:
        st.metric("Number of Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data[missing_data > 0].plot(kind='bar', ax=ax)
        ax.set_title('Missing Values by Feature')
        ax.set_ylabel('Number of Missing Values')
        st.pyplot(fig)
    else:
        st.success("No missing values found in the dataset!")
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    # Feature importance
    if len(df.columns) > 1:
        st.subheader("Feature Importance")
        X_temp = df.iloc[:, :-1]
        y_temp = df.iloc[:, -1]
        
        model = ExtraTreesRegressor()
        model.fit(X_temp, y_temp)
        
        feat_importances = pd.Series(model.feature_importances_, index=X_temp.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feat_importances.nlargest(min(10, len(feat_importances))).plot(kind='barh', ax=ax)
        ax.set_title('Top Feature Importances')
        st.pyplot(fig)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌫️ Air Quality Prediction System</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = AirQualityPredictor()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Dataset Analysis", "Model Training", "Prediction", "Model Comparison"])
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Your CSV Files")
    
    uploaded_files = st.file_uploader(
        "Upload up to 4 CSV files for air quality prediction",
        type=['csv'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    datasets = {}
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files[:4]):  # Limit to 4 files
            try:
                df = pd.read_csv(uploaded_file)
                datasets[uploaded_file.name] = df
                st.success(f"✅ Successfully loaded {uploaded_file.name} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {e}")
    
    if not datasets:
        st.info("👆 Please upload CSV files to get started. You can upload up to 4 files simultaneously.")
        return
    
    # Main content based on selected mode
    if app_mode == "Dataset Analysis":
        show_dataset_analysis(datasets)
    
    elif app_mode == "Model Training":
        show_model_training(datasets, predictor)
    
    elif app_mode == "Prediction":
        show_prediction(datasets, predictor)
    
    elif app_mode == "Model Comparison":
        show_model_comparison(datasets, predictor)

def show_dataset_analysis(datasets):
    st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for each dataset
    tabs = st.tabs(list(datasets.keys()))
    
    for tab, (name, df) in zip(tabs, datasets.items()):
        with tab:
            analyze_dataset(df, name)

def show_model_training(datasets, predictor):
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset for Training", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    # Feature selection
    st.subheader("Feature Selection")
    
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns (features + target)")
        return
    
    # Let user select features and target
    all_columns = df.columns.tolist()
    feature_cols = st.multiselect("Select Feature Columns", all_columns, default=all_columns[:-1])
    target_col = st.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)
    
    if not feature_cols:
        st.error("Please select at least one feature column")
        return
    
    if target_col in feature_cols:
        st.error("Target column cannot be in feature columns")
        return
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Model selection
    st.subheader("Model Selection")
    available_models = list(predictor.models.keys())
    selected_models = st.multiselect("Select Models to Train", available_models, default=available_models[:3])
    
    if st.button("🚀 Train Models", use_container_width=True):
        if not selected_models:
            st.error("Please select at least one model")
            return
        
        # Filter models
        models_to_train = {name: predictor.models[name] for name in selected_models}
        predictor.models = models_to_train
        
        # Train models
        with st.spinner("Training models... This may take a few minutes."):
            results, X_test, y_test = predictor.train_models(X, y)
        
        # Display results
        st.subheader("📈 Training Results")
        
        # Metrics comparison
        metrics_data = []
        for model_name, result in results.items():
            metrics_data.append({
                'Model': model_name,
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'R² Score': result['r2']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'), use_container_width=True)
        
        # Best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        st.success(f"🎯 Best Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted for best model
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, best_model[1]['predictions'], alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Actual vs Predicted - {best_model[0]}')
            st.pyplot(fig)
        
        with col2:
            # Residuals plot
            fig, ax = plt.subplots(figsize=(10, 6))
            residuals = y_test - best_model[1]['predictions']
            ax.scatter(best_model[1]['predictions'], residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals Plot - {best_model[0]}')
            st.pyplot(fig)

def show_prediction(datasets, predictor):
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    # Model selection (in a real scenario, you'd use pre-trained models)
    st.info("🔧 This section would use your pre-trained models for predictions. For now, we'll demonstrate with sample data.")
    
    # Sample prediction interface
    st.subheader("Manual Prediction Input")
    
    if len(df.columns) > 1:
        # Create input fields based on dataset features (excluding target)
        feature_cols = df.columns[:-1] if len(df.columns) > 1 else df.columns
        input_data = {}
        
        cols = st.columns(2)
        for i, col_name in enumerate(feature_cols):
            with cols[i % 2]:
                input_data[col_name] = st.number_input(
                    f"{col_name}",
                    value=float(df[col_name].mean()),
                    step=0.1
                )
        
        if st.button("🔮 Predict Air Quality", use_container_width=True):
            # Create input array
            input_array = np.array([list(input_data.values())])
            
            # Make prediction (this is a simplified version)
            st.success("🎯 Prediction functionality would be implemented with your trained models")
            st.write("Input features:", input_data)
            
            # You would replace this with actual model prediction
            # prediction = predictor.models['Random Forest'].predict(input_array)[0]
            # st.metric("Predicted Air Quality", f"{prediction:.2f}")

def show_model_comparison(datasets, predictor):
    st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset for Comparison", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    if len(df.columns) < 2:
        st.error("Dataset must have at least 2 columns for comparison")
        return
    
    # Use all columns except last as features, last as target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    if st.button("📊 Compare All Models", use_container_width=True):
        with st.spinner("Training and comparing all models..."):
            results, X_test, y_test = predictor.train_models(X, y)
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE Comparison
            rmse_values = {name: result['rmse'] for name, result in results.items()}
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(rmse_values.keys())
            values = list(rmse_values.values())
            bars = ax.bar(models, values, color='skyblue')
            ax.set_title('RMSE Comparison Across Models')
            ax.set_ylabel('RMSE')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        with col2:
            # R² Score Comparison
            r2_values = {name: result['r2'] for name, result in results.items()}
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(r2_values.keys())
            values = list(r2_values.values())
            bars = ax.bar(models, values, color='lightgreen')
            ax.set_title('R² Score Comparison Across Models')
            ax.set_ylabel('R² Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics Comparison")
        metrics_data = []
        for model_name, result in results.items():
            metrics_data.append({
                'Model': model_name,
                'MAE': f"{result['mae']:.4f}",
                'MSE': f"{result['mse']:.4f}",
                'RMSE': f"{result['rmse']:.4f}",
                'R² Score': f"{result['r2']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

if __name__ == "__main__":
    main()