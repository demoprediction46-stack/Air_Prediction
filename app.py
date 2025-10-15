import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2e86ab;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2e86ab;
    }
    .upload-section {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px dashed #2e86ab;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class AirQualityPredictor:
    def __init__(self):
        self.models = self.initialize_models()
    
    def initialize_models(self):
        """Initialize machine learning models"""
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }

    def train_models(self, X, y):
        """Train all models on the provided data"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            try:
                status_text.text(f"Training {name}...")
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
                st.error(f"Error training {name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(self.models))
        
        status_text.text("Training completed!")
        return results, X_test, y_test

def analyze_dataset(df, dataset_name):
    """Perform comprehensive analysis on a dataset"""
    st.markdown(f'<div class="sub-header">📊 Analysis for {dataset_name}</div>', unsafe_allow_html=True)
    
    # Basic information in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Samples", len(df))
    with col2:
        st.metric("Number of Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Dataset preview
    with st.expander("🔍 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Data information
    with st.expander("ℹ️ Data Information"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    # Missing values analysis
    with st.expander("🔎 Missing Values Analysis"):
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(missing_data[missing_data > 0].rename('Missing Count'))
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                missing_data[missing_data > 0].plot(kind='bar', ax=ax, color='red')
                ax.set_title('Missing Values by Feature')
                ax.set_ylabel('Number of Missing Values')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.success("✅ No missing values found in the dataset!")
    
    # Correlation heatmap
    with st.expander("🎨 Correlation Heatmap"):
        if len(df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 features for correlation analysis")
    
    # Feature importance
    with st.expander("🎯 Feature Importance"):
        if len(df.columns) > 1:
            try:
                X_temp = df.iloc[:, :-1]
                y_temp = df.iloc[:, -1]
                
                model = ExtraTreesRegressor(random_state=42)
                model.fit(X_temp, y_temp)
                
                feat_importances = pd.Series(model.feature_importances_, index=X_temp.columns)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                feat_importances.nlargest(min(10, len(feat_importances))).plot(kind='barh', ax=ax, color='green')
                ax.set_title('Top Feature Importances')
                ax.set_xlabel('Importance Score')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")

def main():
    # Header
    st.markdown('<div class="main-header">🌫️ Air Quality Prediction System</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = AirQualityPredictor()
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["🏠 Dataset Analysis", "🤖 Model Training", "🔮 Prediction", "📊 Model Comparison"])
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Your CSV Files")
    
    uploaded_files = st.file_uploader(
        "Upload up to 4 CSV files for air quality prediction",
        type=['csv'],
        accept_multiple_files=True,
        help="Each file should contain features and a target variable for air quality prediction"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    datasets = {}
    if uploaded_files:
        for uploaded_file in uploaded_files[:4]:  # Limit to 4 files
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
                datasets[dataset_name] = df
                st.markdown(f'<div class="success-box">✅ Successfully loaded {dataset_name} ({len(df)} rows, {len(df.columns)} columns)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error loading {uploaded_file.name}: {str(e)}</div>', unsafe_allow_html=True)
    
    if not datasets:
        st.info("""
        👆 **Please upload CSV files to get started!** 
        
        You can upload up to 4 files simultaneously. Each file should contain:
        - Multiple feature columns (temperature, humidity, pressure, etc.)
        - One target column (PM2.5, AQI, or other air quality metric)
        """)
        return
    
    # Remove mode prefix for display
    current_mode = app_mode.replace("🏠 ", "").replace("🤖 ", "").replace("🔮 ", "").replace("📊 ", "")
    
    # Main content based on selected mode
    if "Dataset Analysis" in app_mode:
        show_dataset_analysis(datasets)
    
    elif "Model Training" in app_mode:
        show_model_training(datasets, predictor)
    
    elif "Prediction" in app_mode:
        show_prediction(datasets, predictor)
    
    elif "Model Comparison" in app_mode:
        show_model_comparison(datasets, predictor)

def show_dataset_analysis(datasets):
    st.markdown('<div class="sub-header">📊 Dataset Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for each dataset
    tabs = st.tabs([f"📁 {name}" for name in datasets.keys()])
    
    for tab, (name, df) in zip(tabs, datasets.items()):
        with tab:
            analyze_dataset(df, name)

def show_model_training(datasets, predictor):
    st.markdown('<div class="sub-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    # Dataset selection
    selected_dataset = st.selectbox("Select Dataset for Training", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    if len(df.columns) < 2:
        st.error("❌ Dataset must have at least 2 columns (features + target)")
        return
    
    # Feature selection
    st.subheader("🎯 Feature Selection")
    
    all_columns = df.columns.tolist()
    default_features = all_columns[:-1] if len(all_columns) > 1 else []
    feature_cols = st.multiselect("Select Feature Columns", all_columns, default=default_features)
    target_col = st.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)
    
    if not feature_cols:
        st.error("❌ Please select at least one feature column")
        return
    
    if target_col in feature_cols:
        st.error("❌ Target column cannot be in feature columns")
        return
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Features", len(feature_cols))
    with col2:
        st.metric("Target Variable", target_col)
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Model selection
    st.subheader("🛠️ Model Selection")
    available_models = list(predictor.models.keys())
    selected_models = st.multiselect("Select Models to Train", available_models, default=available_models[:3])
    
    if st.button("🚀 Train Selected Models", type="primary", use_container_width=True):
        if not selected_models:
            st.error("❌ Please select at least one model")
            return
        
        # Filter models
        models_to_train = {name: predictor.models[name] for name in selected_models}
        predictor.models = models_to_train
        
        # Train models
        with st.spinner("🔄 Training models... This may take a few moments."):
            results, X_test, y_test = predictor.train_models(X, y)
        
        if not results:
            st.error("❌ No models were successfully trained")
            return
        
        # Display results
        st.markdown('<div class="sub-header">📈 Training Results</div>', unsafe_allow_html=True)
        
        # Metrics comparison
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
        
        # Highlight best model for each metric
        def highlight_min(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        
        st.dataframe(metrics_df.style.apply(highlight_min, subset=['MAE', 'MSE', 'RMSE']), 
                    use_container_width=True)
        
        # Best model
        best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
        best_rmse = results[best_model_name]['rmse']
        st.success(f"🎯 **Best Performing Model**: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        # Visualization
        st.subheader("📊 Model Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted for best model
            fig, ax = plt.subplots(figsize=(8, 6))
            predictions = results[best_model_name]['predictions']
            ax.scatter(y_test, predictions, alpha=0.6, color='blue')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Actual vs Predicted - {best_model_name}')
            st.pyplot(fig)
        
        with col2:
            # Residuals plot
            fig, ax = plt.subplots(figsize=(8, 6))
            residuals = y_test - predictions
            ax.scatter(predictions, residuals, alpha=0.6, color='orange')
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals Plot - {best_model_name}')
            st.pyplot(fig)

def show_prediction(datasets, predictor):
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    st.info("""
    🎯 **Prediction Feature**
    
    This feature allows you to make air quality predictions using trained models. 
    Upload your data and train models first to enable predictions.
    """)
    
    if not datasets:
        st.warning("Please upload datasets and train models first.")
        return
    
    # Simple prediction interface
    st.subheader("Manual Prediction Input")
    
    selected_dataset = st.selectbox("Select Reference Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    if len(df.columns) > 1:
        st.write("Enter feature values for prediction:")
        
        feature_cols = df.columns[:-1]
        input_data = {}
        
        cols = st.columns(2)
        for i, col_name in enumerate(feature_cols):
            with cols[i % 2]:
                col_mean = df[col_name].mean()
                col_std = df[col_name].std()
                input_data[col_name] = st.number_input(
                    f"{col_name}",
                    value=float(col_mean),
                    min_value=float(col_mean - 3*col_std),
                    max_value=float(col_mean + 3*col_std),
                    step=float(col_std/10),
                    help=f"Range: {col_mean - 3*col_std:.2f} to {col_mean + 3*col_std:.2f}"
                )
        
        if st.button("🎯 Predict Air Quality", type="primary"):
            st.success("🚀 Prediction functionality ready!")
            st.write("**Input Features:**", input_data)
            st.info("Train models in the 'Model Training' section to get actual predictions.")

def show_model_comparison(datasets, predictor):
    st.markdown('<div class="sub-header">📊 Model Comparison</div>', unsafe_allow_html=True)
    
    selected_dataset = st.selectbox("Select Dataset for Comparison", list(datasets.keys()))
    df = datasets[selected_dataset]
    
    if len(df.columns) < 2:
        st.error("❌ Dataset must have at least 2 columns for comparison")
        return
    
    # Use all columns except last as features, last as target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    if st.button("📈 Compare All Models", type="primary", use_container_width=True):
        with st.spinner("🔄 Training and comparing all models..."):
            results, X_test, y_test = predictor.train_models(X, y)
        
        if not results:
            st.error("❌ No models were successfully trained")
            return
        
        # Create comparison charts
        st.subheader("Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE Comparison
            rmse_values = {name: result['rmse'] for name, result in results.items()}
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(rmse_values.keys())
            values = list(rmse_values.values())
            bars = ax.bar(models, values, color='lightcoral')
            ax.set_title('RMSE Comparison (Lower is Better)')
            ax.set_ylabel('RMSE')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            # R² Score Comparison
            r2_values = {name: result['r2'] for name, result in results.items()}
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(r2_values.keys())
            values = list(r2_values.values())
            bars = ax.bar(models, values, color='lightgreen')
            ax.set_title('R² Score Comparison (Higher is Better)')
            ax.set_ylabel('R² Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        # Best model overall
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        st.success(f"🏆 **Overall Best Model**: {best_model[0]} with RMSE: {best_model[1]['rmse']:.4f} and R²: {best_model[1]['r2']:.4f}")

# Add missing import at the top
import io

if __name__ == "__main__":
    main()