import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Solar Power Generation Analysis", layout="wide", page_icon="☀️")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1e40af; text-align: center; margin-bottom: 1.5rem;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #374151; margin-top: 1rem;}
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {font-size: 2rem; font-weight: 700; margin: 0;}
    .metric-label {font-size: 0.9rem; margin: 0.5rem 0 0 0; opacity: 0.9;}
    .info-box {
        background: #f3f4f6;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">☀️ Solar Power Generation Prediction System</h1>', unsafe_allow_html=True)

@st.cache_data
def load_csv_data(file):
    if isinstance(file, str):
        return pd.read_csv(file)
    else:
        return pd.read_csv(file)

@st.cache_resource
def train_models_from_data(df):
    """Train all models and return best one with results"""
    
    with st.spinner('🔄 Training models... This may take 1-2 minutes'):
        df_clean = df.fillna(df.median())
        
        feature_cols = [col for col in df_clean.columns if col != 'power-generated']
        X = df_clean[feature_cols]
        y = df_clean['power-generated']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'is_scaled': True
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42, max_depth=15),
                'X_train': X_train,
                'X_test': X_test,
                'is_scaled': False
            },
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
                'X_train': X_train,
                'X_test': X_test,
                'is_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
                'X_train': X_train,
                'X_test': X_test,
                'is_scaled': False
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=100, gamma='auto'),
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'is_scaled': True
            }
        }
        
        results = []
        best_model = None
        best_r2 = -np.inf
        best_model_name = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, config) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            model = config['model']
            X_tr = config['X_train']
            X_te = config['X_test']
            
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'R²': round(r2, 4),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'CV Mean': round(cv_mean, 4),
                'CV Std': round(cv_std, 4)
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = name
                best_is_scaled = config['is_scaled']
                best_rmse = rmse
                best_mae = mae
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.empty()
        progress_bar.empty()
        
        results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
        
        model_data = {
            'model': best_model,
            'scaler': scaler if best_is_scaled else None,
            'is_scaled': best_is_scaled,
            'best_model_name': best_model_name,
            'best_r2': best_r2,
            'best_rmse': best_rmse,
            'best_mae': best_mae,
            'feature_columns': feature_cols,
            'results': results_df
        }
        
        with open('solar_power_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success('✓ Models trained successfully!')
        
        return model_data

csv_exists = os.path.exists('solarpowergeneration-1.csv')

if csv_exists:
    df = load_csv_data('solarpowergeneration-1.csv')
    st.sidebar.success("✓ Dataset loaded from local file")
else:
    st.sidebar.warning("📁 No local CSV found")
    uploaded_file = st.sidebar.file_uploader("Upload solarpowergeneration-1.csv", type=['csv'])
    
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        st.sidebar.success("✓ Dataset loaded from upload")
    else:
        st.info("""
        ### Welcome to Solar Power Generation Prediction System
        
        **Getting Started:**
        
        **Option 1:** Place `solarpowergeneration-1.csv` in the app directory and refresh
        
        **Option 2:** Upload the CSV file using the sidebar uploader ⬅️
        
        The system will automatically train models when you upload the data.
        """)
        st.stop()

if os.path.exists('solar_power_model.pkl'):
    try:
        with open('solar_power_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        st.sidebar.success("✓ Pre-trained model loaded")
    except:
        st.sidebar.warning("⚠️ Model file corrupted, retraining...")
        model_data = train_models_from_data(df)
else:
    st.sidebar.info("🔄 Training models (first time only)...")
    model_data = train_models_from_data(df)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{df.shape[0]:,}</p>
        <p class="metric-label">Records</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{df.shape[1]}</p>
        <p class="metric-label">Features</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{df.isnull().sum().sum()}</p>
        <p class="metric-label">Missing Values</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{df['power-generated'].min()}</p>
        <p class="metric-label">Min Power (J)</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{df['power-generated'].max():,}</p>
        <p class="metric-label">Max Power (J)</p>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(["📊 EDA", "📈 Feature Analysis", "🔍 Data Quality", "📊 Correlation & VIF", "🤖 Model Performance", "🔮 Live Prediction"])

with tabs[0]:
    st.markdown('<p class="sub-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Power Generation Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['power-generated'], bins=50, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Power Generated (Joules)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Power Generation')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**Power Generation Boxplot**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, y='power-generated', color='#3b82f6', ax=ax)
        ax.set_ylabel('Power Generated (Joules)')
        ax.set_title('Boxplot - Power Generation')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("**Dataset Overview**")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("**Statistical Summary**")
    st.dataframe(df.describe(), use_container_width=True)

with tabs[1]:
    st.markdown('<p class="sub-header">Feature Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("**Feature Distributions**")
    
    feature_cols = [col for col in df.columns if col != 'power-generated']
    
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='#10b981', alpha=0.7, edgecolor='black')
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    for idx in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("**Bivariate Analysis - Feature vs Target**")
    
    selected_feature = st.selectbox("Select Feature", feature_cols)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[selected_feature], df['power-generated'], alpha=0.5, color='#3b82f6', s=20)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Power Generated (Joules)')
    ax.set_title(f'{selected_feature} vs Power Generated')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tabs[2]:
    st.markdown('<p class="sub-header">Data Quality Assessment</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Missing Values Analysis**")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        st.markdown("**Outlier Detection (IQR Method)**")
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outliers[col] = outlier_count
        
        outlier_df = pd.DataFrame(list(outliers.items()), columns=['Feature', 'Outlier Count'])
        outlier_df = outlier_df.sort_values('Outlier Count', ascending=False)
        st.dataframe(outlier_df, use_container_width=True)
    
    st.markdown("**Skewness Analysis**")
    skew_data = df.select_dtypes(include=[np.number]).skew().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    skew_data.plot(kind='bar', ax=ax, color='#f59e0b', alpha=0.7)
    ax.set_xlabel('Features')
    ax.set_ylabel('Skewness')
    ax.set_title('Feature Skewness')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tabs[3]:
    st.markdown('<p class="sub-header">Correlation and Multicollinearity Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("**Correlation Heatmap**")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("**Feature Correlation with Target**")
    target_corr = df.corr()['power-generated'].drop('power-generated').sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    target_corr.plot(kind='barh', ax=ax, color='#3b82f6', alpha=0.7)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Features')
    ax.set_title('Feature Correlation with Power Generated')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.dataframe(pd.DataFrame({'Feature': target_corr.index, 'Correlation': target_corr.values}), 
                 use_container_width=True)
    
    st.markdown("**VIF (Variance Inflation Factor) Analysis**")
    
    feature_cols = [col for col in df.columns if col != 'power-generated']
    X_vif = df[feature_cols].fillna(df[feature_cols].median())
    
    vif_data = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X_vif.values, i)
            vif_data.append({'Feature': col, 'VIF': round(vif, 2)})
        except:
            vif_data.append({'Feature': col, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    st.dataframe(vif_df, use_container_width=True)
    
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        st.warning(f"⚠️ {len(high_vif)} feature(s) have VIF > 10, indicating high multicollinearity")

with tabs[4]:
    st.markdown('<p class="sub-header">Model Performance Evaluation</p>', unsafe_allow_html=True)
    
    st.markdown("**Model Comparison Results**")
    st.dataframe(model_data['results'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model R² Score Comparison**")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
        model_data['results'].plot(x='Model', y='R²', kind='bar', ax=ax, 
                                  color=colors[:len(model_data['results'])], alpha=0.8, legend=False)
        ax.set_ylabel('R² Score')
        ax.set_xlabel('Model')
        ax.set_title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**Model Metrics Comparison**")
        fig, ax = plt.subplots(figsize=(8, 5))
        model_data['results'].plot(x='Model', y=['RMSE', 'MAE'], kind='bar', ax=ax, alpha=0.7)
        ax.set_ylabel('Error Value')
        ax.set_xlabel('Model')
        ax.set_title('RMSE and MAE Comparison')
        plt.xticks(rotation=45, ha='right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown(f"""
    <div class="info-box">
        <h3>🏆 Best Model: {model_data['best_model_name']}</h3>
        <p><strong>R² Score:</strong> {model_data['best_r2']:.4f}</p>
        <p><strong>RMSE:</strong> {model_data['best_rmse']:.2f}</p>
        <p><strong>MAE:</strong> {model_data['best_mae']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with tabs[5]:
    st.markdown('<p class="sub-header">Live Prediction</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'input_values_stored' not in st.session_state:
        st.session_state.input_values_stored = None
    
    st.markdown("**Enter Environmental Conditions**")
    
    feature_cols = [col for col in df.columns if col != 'power-generated']
    
    input_values = {}
    
    n_cols = 3
    for i in range(0, len(feature_cols), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i + j < len(feature_cols):
                feature = feature_cols[i + j]
                with cols[j]:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    median_val = float(df[feature].median())
                    
                    step = (max_val - min_val) / 100 if max_val != min_val else 0.01
                    
                    input_values[feature] = st.number_input(
                        label=feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=step,
                        format="%.4f" if step < 0.1 else "%.2f",
                        key=f"input_{feature}"
                    )
    
    if st.button("🚀 Predict Power Generation", type="primary", use_container_width=True):
        input_array = np.array([input_values[feature] for feature in feature_cols]).reshape(1, -1)
        
        if model_data['is_scaled']:
            input_scaled = model_data['scaler'].transform(input_array)
            prediction = model_data['model'].predict(input_scaled)[0]
        else:
            prediction = model_data['model'].predict(input_array)[0]
        
        st.session_state.prediction_result = prediction
        st.session_state.input_values_stored = input_values.copy()
    
    # Display results if prediction exists
    if st.session_state.prediction_result is not None:
        prediction = st.session_state.prediction_result
        stored_inputs = st.session_state.input_values_stored
        
        st.markdown("---")
        
        # Prediction Result Card
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 2rem; border-radius: 10px; text-align: center; 
                        margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='margin: 0; font-size: 3rem; font-weight: 700;'>{prediction:,.2f}</h2>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Joules per 3 hours</p>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Model: {model_data['best_model_name']} | R²: {model_data['best_r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison with Dataset Statistics
        st.markdown("**Prediction Analysis**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        dataset_mean = df['power-generated'].mean()
        dataset_median = df['power-generated'].median()
        dataset_min = df['power-generated'].min()
        dataset_max = df['power-generated'].max()
        
        with col1:
            st.metric("Dataset Mean", f"{dataset_mean:,.0f} J", 
                     f"{prediction - dataset_mean:+,.0f} J")
        with col2:
            st.metric("Dataset Median", f"{dataset_median:,.0f} J", 
                     f"{prediction - dataset_median:+,.0f} J")
        with col3:
            percentile = (df['power-generated'] < prediction).sum() / len(df) * 100
            st.metric("Percentile", f"{percentile:.1f}%")
        with col4:
            range_pct = (prediction - dataset_min) / (dataset_max - dataset_min) * 100
            st.metric("Range Position", f"{range_pct:.1f}%")
        
        # Visualizations
        st.markdown("**Prediction Visualization**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['power-generated'], bins=50, color='#3b82f6', alpha=0.6, edgecolor='black', label='Dataset Distribution')
            ax.axvline(prediction, color='#10b981', linewidth=3, linestyle='--', label=f'Your Prediction: {prediction:,.0f} J')
            ax.axvline(dataset_mean, color='#f59e0b', linewidth=2, linestyle=':', label=f'Dataset Mean: {dataset_mean:,.0f} J')
            ax.set_xlabel('Power Generated (Joules)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Prediction vs Dataset Distribution', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Prediction']
            values = [
                dataset_min,
                df['power-generated'].quantile(0.25),
                dataset_median,
                df['power-generated'].quantile(0.75),
                dataset_max,
                prediction
            ]
            colors = ['#ef4444', '#f59e0b', '#3b82f6', '#f59e0b', '#ef4444', '#10b981']
            
            bars = ax.barh(categories, values, color=colors, alpha=0.7, edgecolor='black')
            bars[-1].set_linewidth(3)
            bars[-1].set_edgecolor('#059669')
            
            ax.set_xlabel('Power Generated (Joules)', fontsize=12)
            ax.set_title('Prediction Position in Dataset Range', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, (cat, val) in enumerate(zip(categories, values)):
                ax.text(val, i, f' {val:,.0f}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Input Values Comparison
        st.markdown("**Your Input Values vs Dataset Statistics**")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_cols):
            ax = axes[idx]
            
            ax.hist(df[feature].dropna(), bins=30, color='#3b82f6', alpha=0.5, edgecolor='black', label='Dataset')
            
            input_val = stored_inputs[feature]
            ax.axvline(input_val, color='#10b981', linewidth=3, linestyle='--', label=f'Your Input: {input_val:.2f}')
            
            mean_val = df[feature].mean()
            ax.axvline(mean_val, color='#f59e0b', linewidth=2, linestyle=':', label=f'Mean: {mean_val:.2f}')
            
            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Frequency', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Input Summary Table
        st.markdown("**Detailed Input Summary**")
        
        comparison_data = []
        for feature in feature_cols:
            input_val = stored_inputs[feature]
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            comparison_data.append({
                'Feature': feature,
                'Your Input': f"{input_val:.2f}",
                'Dataset Mean': f"{mean_val:.2f}",
                'Dataset Median': f"{median_val:.2f}",
                'Min': f"{min_val:.2f}",
                'Max': f"{max_val:.2f}",
                'Percentile': f"{(df[feature] < input_val).sum() / len(df) * 100:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Reset button
        if st.button("🔄 Reset and Make New Prediction", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.input_values_stored = None
            st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #f3f4f6; border-radius: 8px;'>
    <p style='margin: 0; color: #6b7280;'>Solar Power Generation Prediction System | Model Evaluation Phase</p>
</div>
""", unsafe_allow_html=True)
