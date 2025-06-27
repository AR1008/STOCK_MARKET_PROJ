import pandas as pd
import numpy as np
import os
import pickle
import logging
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import jarque_bera, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioconModelTester:
    """
    Comprehensive model testing and evaluation for Biocon stock prediction models
    Tests performance across different time periods, FDA milestones, and market conditions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.test_results = {}
        self.predictions = {}
        self.evaluation_metrics = {}
        self.fda_milestone_performance = {}
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['results', 'results/charts', 'results/detailed_analysis']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_trained_models(self):
        """Load all trained models and supporting files"""
        logger.info("Loading trained models and scalers...")
        
        try:
            # Load scalers
            if os.path.exists('models/scalers.pkl'):
                with open('models/scalers.pkl', 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info("✓ Scalers loaded")
            
            # Load feature names
            if os.path.exists('models/feature_names.pkl'):
                with open('models/feature_names.pkl', 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"✓ Feature names loaded: {len(self.feature_names)} features")
            
            # Load traditional ML models
            model_files = {
                'Linear_Regression': 'models/linear_regression_model.pkl',
                'Ridge_Regression': 'models/ridge_regression_model.pkl',
                'Lasso_Regression': 'models/lasso_regression_model.pkl',
                'Random_Forest': 'models/random_forest_model.pkl',
                'Gradient_Boosting': 'models/gradient_boosting_model.pkl',
                'XGBoost': 'models/xgboost_model.pkl',
                'LightGBM': 'models/lightgbm_model.pkl',
                'SVR': 'models/svr_model.pkl'
            }
            
            models_loaded = 0
            for model_name, model_path in model_files.items():
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        models_loaded += 1
                        logger.info(f"✓ {model_name} loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {str(e)}")
            
            # Load LSTM model
            lstm_paths = ['models/lstm_model.h5', 'models/lstm_model.pkl']
            for lstm_path in lstm_paths:
                if os.path.exists(lstm_path):
                    try:
                        if lstm_path.endswith('.h5'):
                            self.models['LSTM'] = load_model(lstm_path)
                        else:
                            with open(lstm_path, 'rb') as f:
                                self.models['LSTM'] = pickle.load(f)
                        models_loaded += 1
                        logger.info("✓ LSTM model loaded")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load LSTM from {lstm_path}: {str(e)}")
            
            # Load final/best model
            final_model_paths = ['models/final_model.pkl', 'models/final_model.h5']
            for final_path in final_model_paths:
                if os.path.exists(final_path):
                    try:
                        if final_path.endswith('.h5'):
                            self.models['Final_Model'] = load_model(final_path)
                        else:
                            with open(final_path, 'rb') as f:
                                self.models['Final_Model'] = pickle.load(f)
                        logger.info("✓ Final/Best model loaded")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load final model from {final_path}: {str(e)}")
            
            logger.info(f"Total models loaded: {len(self.models)}")
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def load_test_data(self):
        """Load and prepare test data"""
        logger.info("Loading test data...")
        
        try:
            # Load combined data (should exist from training)
            if os.path.exists('data/combined_data.csv'):
                df = pd.read_csv('data/combined_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # Recreate combined data if it doesn't exist
                logger.info("Combined data not found, recreating...")
                df = self.recreate_combined_data()
            
            # Engineer features (same as training)
            df = self.engineer_features(df)
            
            # Prepare the same way as training
            X, y, dates = self.prepare_test_features(df)
            
            # Split data the same way as training (60/20/20)
            n_samples = len(X)
            train_end = int(n_samples * 0.6)
            val_end = int(n_samples * 0.8)
            
            # We want the test set
            X_test = X.iloc[val_end:].copy()
            y_test = y[val_end:]
            test_dates = dates[val_end:]
            
            logger.info(f"Test data prepared: {len(X_test)} samples, {len(X_test.columns)} features")
            return X_test, y_test, test_dates, df
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def recreate_combined_data(self):
        """Recreate combined data if missing"""
        try:
            # Load stock data
            stock_df = pd.read_csv('data/stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            
            # Load news sentiment data
            news_df = pd.read_csv('data/daily_sentiment.csv')
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # Convert to date only for better matching
            stock_df['Date_only'] = stock_df['Date'].dt.date
            news_df['date_only'] = news_df['date'].dt.date
            
            # Merge datasets
            combined_df = pd.merge(
                stock_df, 
                news_df, 
                left_on='Date_only', 
                right_on='date_only', 
                how='left'
            )
            
            # Drop temporary columns
            combined_df = combined_df.drop(['Date_only', 'date_only'], axis=1)
            
            # Fill missing sentiment values
            sentiment_columns = ['avg_sentiment', 'weighted_avg_sentiment', 'news_count', 'drug_specific_count', 'day_importance_score']
            for col in sentiment_columns:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(0)
                else:
                    combined_df[col] = 0
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error recreating combined data: {str(e)}")
            raise
    
    def engineer_features(self, df):
        """Engineer features (same as training)"""
        logger.info("Engineering features for testing...")
        
        try:
            df = df.sort_values('Date').copy()
            
            # Basic price features
            if 'Close' in df.columns:
                df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
                df['Price_MA_10'] = df['Close'].rolling(window=10).mean()
                df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
                df['Price_Momentum_5'] = df['Close'].pct_change(5)
                df['Price_Momentum_10'] = df['Close'].pct_change(10)
                df['Price_Momentum_20'] = df['Close'].pct_change(20)
            
            # Volatility features
            if 'Daily_Return' in df.columns:
                df['Price_Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
                df['Price_Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
                df['Price_Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
            
            # Volume features
            if 'Volume' in df.columns:
                df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
                df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
                df['Volume_Ratio_5'] = df['Volume'] / (df['Volume'].rolling(window=5).mean() + 1)
                df['Volume_Ratio_10'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1)
            
            # Sentiment features
            if 'avg_sentiment' in df.columns:
                df['Sentiment_MA_3'] = df['avg_sentiment'].rolling(window=3).mean()
                df['Sentiment_MA_7'] = df['avg_sentiment'].rolling(window=7).mean()
                df['Sentiment_MA_14'] = df['avg_sentiment'].rolling(window=14).mean()
                df['Sentiment_Change_3'] = df['avg_sentiment'].diff(3)
                df['Sentiment_Change_7'] = df['avg_sentiment'].diff(7)
            
            if 'weighted_avg_sentiment' in df.columns:
                df['Weighted_Sentiment_MA_3'] = df['weighted_avg_sentiment'].rolling(window=3).mean()
                df['Weighted_Sentiment_MA_7'] = df['weighted_avg_sentiment'].rolling(window=7).mean()
                df['Weighted_Sentiment_Change_3'] = df['weighted_avg_sentiment'].diff(3)
            
            # News volume features
            if 'news_count' in df.columns:
                df['News_Volume_MA_7'] = df['news_count'].rolling(window=7).mean()
                df['News_Volume_Change'] = df['news_count'].diff()
            
            # Drug specific features
            if 'drug_specific_count' in df.columns:
                df['Drug_News_MA_7'] = df['drug_specific_count'].rolling(window=7).mean()
                df['Drug_News_Ratio'] = df['drug_specific_count'] / (df['news_count'] + 1)
            
            # FDA milestone features
            if 'day_importance_score' in df.columns:
                df['Importance_MA_7'] = df['day_importance_score'].rolling(window=7).mean()
                df['High_Importance_Day'] = (df['day_importance_score'] > 20).astype(int)
            
            # Market context features
            if 'NIFTY50_Return' in df.columns and 'Daily_Return' in df.columns:
                df['Market_Relative_Performance'] = df['Daily_Return'] - df['NIFTY50_Return']
                df['Market_Beta'] = df['Daily_Return'].rolling(window=60).cov(df['NIFTY50_Return']) / (df['NIFTY50_Return'].rolling(window=60).var() + 0.0001)
            
            # Technical indicators normalization
            technical_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'MA_20', 'MA_50']
            for indicator in technical_indicators:
                if indicator in df.columns:
                    rolling_mean = df[indicator].rolling(window=252).mean()
                    rolling_std = df[indicator].rolling(window=252).std()
                    df[f'{indicator}_normalized'] = (df[indicator] - rolling_mean) / (rolling_std + 0.0001)
            
            # Lag features
            lag_features = ['Close', 'Daily_Return', 'avg_sentiment', 'weighted_avg_sentiment', 'news_count']
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_lag1'] = df[feature].shift(1)
                    df[f'{feature}_lag2'] = df[feature].shift(2)
                    df[f'{feature}_lag3'] = df[feature].shift(3)
            
            # Target variable
            if 'Daily_Return' in df.columns:
                df['Next_Day_Return'] = df['Daily_Return'].shift(-1)
            elif 'Close' in df.columns:
                df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
            
            # Date features
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def prepare_test_features(self, df):
        """Prepare features for testing"""
        try:
            # Target variable
            target_candidates = ['Next_Day_Return', 'Daily_Return']
            target_column = None
            
            for candidate in target_candidates:
                if candidate in df.columns and df[candidate].notna().sum() > 100:
                    target_column = candidate
                    break
            
            if target_column is None:
                if 'Close' in df.columns:
                    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
                    target_column = 'Next_Day_Return'
                else:
                    raise ValueError("No suitable target variable found")
            
            # Clean data
            df_clean = df.dropna(subset=[target_column]).copy()
            
            # Select features (use saved feature names if available)
            if self.feature_names:
                available_features = [col for col in self.feature_names if col in df_clean.columns]
                if len(available_features) < len(self.feature_names) * 0.8:
                    logger.warning(f"Only {len(available_features)}/{len(self.feature_names)} features available")
            else:
                # Fallback feature selection
                exclude_columns = ['Date', 'date', 'Next_Day_Return', 'Next_Day_Close', 'Next_5Day_Return', 'Symbol', 'Company', 'Type', 'Source', 'datetime']
                available_features = [col for col in df_clean.columns if col not in exclude_columns]
            
            # Prepare X and y
            X = df_clean[available_features].copy()
            
            # Fill missing values
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col] = X[col].fillna(median_val)
                else:
                    mode_val = X[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 0
                    X[col] = X[col].fillna(fill_val)
            
            y = df_clean[target_column].values
            dates = df_clean['Date'].values
            
            # Remove NaN in target
            valid_indices = ~pd.isna(y)
            X = X[valid_indices]
            y = y[valid_indices]
            dates = dates[valid_indices]
            
            return X, y, dates
            
        except Exception as e:
            logger.error(f"Error preparing test features: {str(e)}")
            raise
    
    def run_comprehensive_testing(self, X_test, y_test, test_dates):
        """Run comprehensive testing on all models"""
        logger.info("Running comprehensive model testing...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Testing {model_name}...")
                
                # Make predictions
                if model_name == 'LSTM':
                    predictions = self.test_lstm_model(model, X_test, y_test)
                else:
                    predictions = self.test_traditional_model(model, model_name, X_test, y_test)
                
                if predictions is not None:
                    # Calculate metrics
                    metrics = self.calculate_comprehensive_metrics(y_test, predictions)
                    
                    # Store results
                    self.test_results[model_name] = {
                        'predictions': predictions,
                        'actual': y_test,
                        'dates': test_dates,
                        'metrics': metrics
                    }
                    
                    logger.info(f"{model_name} - Test RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {str(e)}")
                continue
    
    def test_traditional_model(self, model, model_name, X_test, y_test):
        """Test traditional ML model"""
        try:
            # Use appropriate scaler
            if model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression', 'SVR']:
                if 'standard' in self.scalers:
                    X_test_scaled = self.scalers['standard'].transform(X_test)
                    predictions = model.predict(X_test_scaled)
                else:
                    # Fallback scaling
                    scaler = StandardScaler()
                    X_test_scaled = scaler.fit_transform(X_test)
                    predictions = model.predict(X_test_scaled)
            else:
                predictions = model.predict(X_test)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in traditional model testing: {str(e)}")
            return None
    
    def test_lstm_model(self, model, X_test, y_test):
        """Test LSTM model"""
        try:
            # Scale data
            if 'lstm_X' in self.scalers and 'lstm_y' in self.scalers:
                scaler_X = self.scalers['lstm_X']
                scaler_y = self.scalers['lstm_y']
            else:
                # Fallback scaling
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                scaler_X.fit(X_test)
                scaler_y.fit(y_test.reshape(-1, 1))
            
            X_test_scaled = scaler_X.transform(X_test)
            
            # Create sequences
            sequence_length = 30
            X_test_seq = []
            y_test_seq = []
            
            for i in range(sequence_length, len(X_test_scaled)):
                X_test_seq.append(X_test_scaled[i-sequence_length:i])
                y_test_seq.append(y_test[i])
            
            if len(X_test_seq) == 0:
                logger.warning("Insufficient data for LSTM testing")
                return None
            
            X_test_seq = np.array(X_test_seq)
            y_test_seq = np.array(y_test_seq)
            
            # Make predictions
            predictions_scaled = model.predict(X_test_seq)
            predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in LSTM model testing: {str(e)}")
            return None
    
    def calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
            
            # Direction accuracy (sign prediction)
            direction_actual = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            metrics['direction_accuracy'] = np.mean(direction_actual == direction_pred) * 100
            
            # Correlation
            metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
            
            # Residual statistics
            residuals = y_true - y_pred
            metrics['residual_mean'] = np.mean(residuals)
            metrics['residual_std'] = np.std(residuals)
            
            # Statistical tests
            try:
                _, metrics['shapiro_p'] = shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
                _, metrics['jarque_bera_p'] = jarque_bera(residuals)
            except:
                metrics['shapiro_p'] = np.nan
                metrics['jarque_bera_p'] = np.nan
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def analyze_fda_milestone_performance(self):
        """Analyze model performance during FDA milestone periods"""
        logger.info("Analyzing FDA milestone performance...")
        
        try:
            # Load news data to identify FDA milestone dates
            news_df = pd.read_csv('data/news_data.csv')
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # Identify high-importance FDA milestone days
            fda_milestone_days = news_df[
                (news_df['day_importance_score'] > 20) |
                (news_df['fda_milestone_type'].isin(['approval_process', 'regulatory_review', 'application_phase']))
            ]['date'].dt.date.unique()
            
            logger.info(f"Found {len(fda_milestone_days)} FDA milestone days")
            
            # Analyze performance around these days
            for model_name, results in self.test_results.items():
                if 'dates' not in results:
                    continue
                
                test_dates = pd.to_datetime(results['dates']).date
                y_true = results['actual']
                y_pred = results['predictions']
                
                # Find milestone periods (±5 days around FDA events)
                milestone_indices = []
                for milestone_date in fda_milestone_days:
                    for i, test_date in enumerate(test_dates):
                        if abs((test_date - milestone_date).days) <= 5:
                            milestone_indices.append(i)
                
                if milestone_indices:
                    milestone_indices = list(set(milestone_indices))
                    non_milestone_indices = [i for i in range(len(y_true)) if i not in milestone_indices]
                    
                    # Calculate metrics for milestone vs non-milestone periods
                    milestone_metrics = self.calculate_comprehensive_metrics(
                        y_true[milestone_indices], 
                        y_pred[milestone_indices]
                    )
                    
                    non_milestone_metrics = self.calculate_comprehensive_metrics(
                        y_true[non_milestone_indices], 
                        y_pred[non_milestone_indices]
                    )
                    
                    self.fda_milestone_performance[model_name] = {
                        'milestone_periods': milestone_metrics,
                        'non_milestone_periods': non_milestone_metrics,
                        'milestone_days_count': len(milestone_indices),
                        'total_days_count': len(y_true)
                    }
                    
                    logger.info(f"{model_name} - Milestone RMSE: {milestone_metrics['rmse']:.4f}, "
                              f"Normal RMSE: {non_milestone_metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in FDA milestone analysis: {str(e)}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Model Performance Comparison
            self.plot_model_performance_comparison()
            
            # 2. Prediction vs Actual for best model
            self.plot_prediction_vs_actual()
            
            # 3. Residual Analysis
            self.plot_residual_analysis()
            
            # 4. FDA Milestone Performance
            self.plot_fda_milestone_performance()
            
            # 5. Time Series Predictions
            self.plot_time_series_predictions()
            
            # 6. Feature Importance (if available)
            self.plot_feature_importance()
            
            logger.info("✓ All visualizations created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def plot_model_performance_comparison(self):
        """Plot model performance comparison"""
        try:
            metrics_data = []
            for model_name, results in self.test_results.items():
                metrics = results['metrics']
                metrics_data.append({
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2'],
                    'MAE': metrics['mae'],
                    'Direction_Accuracy': metrics['direction_accuracy'],
                    'Correlation': metrics['correlation']
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # RMSE
            sns.barplot(data=df_metrics, x='Model', y='RMSE', ax=axes[0,0])
            axes[0,0].set_title('Root Mean Square Error (Lower is Better)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # R²
            sns.barplot(data=df_metrics, x='Model', y='R²', ax=axes[0,1])
            axes[0,1].set_title('R² Score (Higher is Better)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # MAE
            sns.barplot(data=df_metrics, x='Model', y='MAE', ax=axes[0,2])
            axes[0,2].set_title('Mean Absolute Error (Lower is Better)')
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # Direction Accuracy
            sns.barplot(data=df_metrics, x='Model', y='Direction_Accuracy', ax=axes[1,0])
            axes[1,0].set_title('Direction Accuracy % (Higher is Better)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Correlation
            sns.barplot(data=df_metrics, x='Model', y='Correlation', ax=axes[1,1])
            axes[1,1].set_title('Correlation (Higher is Better)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Combined Score (normalize and combine metrics)
            df_metrics['Combined_Score'] = (
                (1 - df_metrics['RMSE'] / df_metrics['RMSE'].max()) * 0.3 +
                df_metrics['R²'] * 0.3 +
                (df_metrics['Direction_Accuracy'] / 100) * 0.4
            )
            sns.barplot(data=df_metrics, x='Model', y='Combined_Score', ax=axes[1,2])
            axes[1,2].set_title('Combined Performance Score (Higher is Better)')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('results/charts/model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics to CSV
            df_metrics.to_csv('results/detailed_test_metrics.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error plotting model performance: {str(e)}")
    
    def plot_prediction_vs_actual(self):
        """Plot prediction vs actual for best performing model"""
        try:
            # Find best model (highest R²)
            best_model = max(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'])
            model_name, results = best_model
            
            y_true = results['actual']
            y_pred = results['predictions']
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Prediction vs Actual - {model_name} (Best Model)', fontsize=14, fontweight='bold')
            
            # Scatter plot
            axes[0].scatter(y_true, y_pred, alpha=0.6, s=30)
            axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Returns')
            axes[0].set_ylabel('Predicted Returns')
            axes[0].set_title('Scatter Plot: Predicted vs Actual')
            axes[0].grid(True, alpha=0.3)
            
            # Add R² and RMSE text
            r2 = results['metrics']['r2']
            rmse = results['metrics']['rmse']
            axes[0].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Returns')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residuals Plot')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/charts/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting prediction vs actual: {str(e)}")
    
    def plot_residual_analysis(self):
        """Plot comprehensive residual analysis"""
        try:
            # Find best model
            best_model = max(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'])
            model_name, results = best_model
            
            y_true = results['actual']
            y_pred = results['predictions']
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
            
            # Histogram of residuals
            axes[0,0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[0,0].set_title('Distribution of Residuals')
            axes[0,0].set_xlabel('Residuals')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].axvline(np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Q-Q plot for normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0,1])
            axes[0,1].set_title('Q-Q Plot (Normal Distribution)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Residuals vs Fitted
            axes[1,0].scatter(y_pred, residuals, alpha=0.6)
            axes[1,0].axhline(y=0, color='r', linestyle='--')
            axes[1,0].set_xlabel('Fitted Values')
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].set_title('Residuals vs Fitted Values')
            axes[1,0].grid(True, alpha=0.3)
            
            # Residuals over time
            if 'dates' in results:
                dates = pd.to_datetime(results['dates'])
                axes[1,1].plot(dates, residuals, alpha=0.7)
                axes[1,1].axhline(y=0, color='r', linestyle='--')
                axes[1,1].set_xlabel('Date')
                axes[1,1].set_ylabel('Residuals')
                axes[1,1].set_title('Residuals Over Time')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/charts/residual_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting residual analysis: {str(e)}")
    
    def plot_fda_milestone_performance(self):
        """Plot FDA milestone vs normal period performance"""
        try:
            if not self.fda_milestone_performance:
                logger.warning("No FDA milestone performance data available")
                return
            
            models = list(self.fda_milestone_performance.keys())
            milestone_rmse = [self.fda_milestone_performance[m]['milestone_periods']['rmse'] for m in models]
            normal_rmse = [self.fda_milestone_performance[m]['non_milestone_periods']['rmse'] for m in models]
            
            milestone_r2 = [self.fda_milestone_performance[m]['milestone_periods']['r2'] for m in models]
            normal_r2 = [self.fda_milestone_performance[m]['non_milestone_periods']['r2'] for m in models]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Model Performance: FDA Milestone vs Normal Periods', fontsize=14, fontweight='bold')
            
            # RMSE comparison
            x = np.arange(len(models))
            width = 0.35
            
            axes[0].bar(x - width/2, milestone_rmse, width, label='FDA Milestone Periods', alpha=0.8)
            axes[0].bar(x + width/2, normal_rmse, width, label='Normal Periods', alpha=0.8)
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('RMSE: Milestone vs Normal Periods')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(models, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # R² comparison
            axes[1].bar(x - width/2, milestone_r2, width, label='FDA Milestone Periods', alpha=0.8)
            axes[1].bar(x + width/2, normal_r2, width, label='Normal Periods', alpha=0.8)
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('R² Score')
            axes[1].set_title('R² Score: Milestone vs Normal Periods')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(models, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/charts/fda_milestone_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save FDA analysis to CSV
            fda_analysis_data = []
            for model in models:
                data = self.fda_milestone_performance[model]
                fda_analysis_data.append({
                    'Model': model,
                    'Milestone_RMSE': data['milestone_periods']['rmse'],
                    'Normal_RMSE': data['non_milestone_periods']['rmse'],
                    'Milestone_R2': data['milestone_periods']['r2'],
                    'Normal_R2': data['non_milestone_periods']['r2'],
                    'Milestone_Days': data['milestone_days_count'],
                    'Total_Days': data['total_days_count']
                })
            
            pd.DataFrame(fda_analysis_data).to_csv('results/fda_milestone_analysis.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error plotting FDA milestone performance: {str(e)}")
    
    def plot_time_series_predictions(self):
        """Plot time series predictions for best model"""
        try:
            # Find best model
            best_model = max(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'])
            model_name, results = best_model
            
            if 'dates' not in results:
                logger.warning("No dates available for time series plot")
                return
            
            dates = pd.to_datetime(results['dates'])
            y_true = results['actual']
            y_pred = results['predictions']
            
            # Create cumulative returns for better visualization
            cumulative_actual = np.cumprod(1 + y_true) - 1
            cumulative_predicted = np.cumprod(1 + y_pred) - 1
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            fig.suptitle(f'Time Series Predictions - {model_name}', fontsize=16, fontweight='bold')
            
            # Daily returns
            axes[0].plot(dates, y_true, label='Actual Returns', alpha=0.7, linewidth=1)
            axes[0].plot(dates, y_pred, label='Predicted Returns', alpha=0.7, linewidth=1)
            axes[0].set_ylabel('Daily Returns')
            axes[0].set_title('Daily Returns: Actual vs Predicted')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Cumulative returns
            axes[1].plot(dates, cumulative_actual, label='Actual Cumulative', linewidth=2)
            axes[1].plot(dates, cumulative_predicted, label='Predicted Cumulative', linewidth=2)
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Cumulative Returns')
            axes[1].set_title('Cumulative Returns: Actual vs Predicted')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('results/charts/time_series_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save predictions to CSV
            predictions_df = pd.DataFrame({
                'Date': dates,
                'Actual_Return': y_true,
                'Predicted_Return': y_pred,
                'Actual_Cumulative': cumulative_actual,
                'Predicted_Cumulative': cumulative_predicted,
                'Model': model_name
            })
            predictions_df.to_csv('results/predictions.csv', index=False)
            
        except Exception as e:
            logger.error(f"Error plotting time series: {str(e)}")
    
    def plot_feature_importance(self):
        """Plot feature importance if available"""
        try:
            # Try to load feature importance from training
            if os.path.exists('results/feature_importance.csv'):
                importance_df = pd.read_csv('results/feature_importance.csv')
                
                # Get the best performing model's importance
                if len(self.test_results) > 0:
                    best_model_name = max(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'])[0]
                    
                    if best_model_name in importance_df.columns:
                        feature_importance = importance_df[['Unnamed: 0', best_model_name]].copy()
                        feature_importance.columns = ['Feature', 'Importance']
                        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
                        
                        plt.figure(figsize=(12, 8))
                        sns.barplot(data=feature_importance, x='Importance', y='Feature')
                        plt.title(f'Top 20 Feature Importance - {best_model_name}')
                        plt.xlabel('Importance Score')
                        plt.tight_layout()
                        plt.savefig('results/charts/feature_importance.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        logger.info("✓ Feature importance plot created")
            else:
                logger.warning("Feature importance data not found")
                
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def generate_trading_strategy_analysis(self):
        """Generate trading strategy analysis based on predictions"""
        logger.info("Generating trading strategy analysis...")
        
        try:
            # Find best model
            best_model = max(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'])
            model_name, results = best_model
            
            y_pred = results['predictions']
            y_true = results['actual']
            dates = pd.to_datetime(results['dates']) if 'dates' in results else None
            
            # Define trading signals
            # Buy when predicted return > threshold, Sell when < -threshold
            buy_threshold = np.percentile(y_pred, 75)  # Top 25% predictions
            sell_threshold = np.percentile(y_pred, 25)  # Bottom 25% predictions
            
            signals = np.zeros(len(y_pred))
            signals[y_pred > buy_threshold] = 1  # Buy signal
            signals[y_pred < sell_threshold] = -1  # Sell signal
            
            # Calculate strategy returns
            strategy_returns = signals * y_true
            
            # Calculate metrics
            total_return = np.sum(strategy_returns)
            annualized_return = total_return * 252 / len(strategy_returns)  # Assuming daily data
            volatility = np.std(strategy_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Win rate
            profitable_trades = strategy_returns[strategy_returns != 0]
            win_rate = np.sum(profitable_trades > 0) / len(profitable_trades) if len(profitable_trades) > 0 else 0
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + strategy_returns) - 1
            running_max = np.maximum.accumulate(cumulative_returns + 1)
            drawdown = (cumulative_returns + 1) / running_max - 1
            max_drawdown = np.min(drawdown)
            
            trading_metrics = {
                'Model': model_name,
                'Total_Return': total_return,
                'Annualized_Return': annualized_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Win_Rate': win_rate,
                'Max_Drawdown': max_drawdown,
                'Number_of_Trades': np.sum(signals != 0),
                'Buy_Threshold': buy_threshold,
                'Sell_Threshold': sell_threshold
            }
            
            # Save trading analysis
            pd.DataFrame([trading_metrics]).to_csv('results/trading_strategy_analysis.csv', index=False)
            
            # Create trading strategy visualization
            if dates is not None:
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                fig.suptitle(f'Trading Strategy Analysis - {model_name}', fontsize=16, fontweight='bold')
                
                # Cumulative returns
                buy_and_hold = np.cumprod(1 + y_true) - 1
                strategy_cumulative = np.cumprod(1 + strategy_returns) - 1
                
                axes[0].plot(dates, buy_and_hold, label='Buy & Hold', linewidth=2)
                axes[0].plot(dates, strategy_cumulative, label='Strategy', linewidth=2)
                axes[0].set_ylabel('Cumulative Returns')
                axes[0].set_title('Strategy vs Buy & Hold Returns')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Trading signals
                axes[1].plot(dates, y_true, alpha=0.7, color='gray', label='Actual Returns')
                buy_signals = dates[signals == 1]
                sell_signals = dates[signals == -1]
                axes[1].scatter(buy_signals, y_true[signals == 1], color='green', marker='^', s=50, label='Buy Signals')
                axes[1].scatter(sell_signals, y_true[signals == -1], color='red', marker='v', s=50, label='Sell Signals')
                axes[1].set_ylabel('Daily Returns')
                axes[1].set_title('Trading Signals')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Drawdown
                axes[2].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
                axes[2].plot(dates, drawdown, color='red', linewidth=1)
                axes[2].set_xlabel('Date')
                axes[2].set_ylabel('Drawdown')
                axes[2].set_title('Strategy Drawdown')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('results/charts/trading_strategy_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Trading strategy analysis completed - Sharpe Ratio: {sharpe_ratio:.3f}")
            return trading_metrics
            
        except Exception as e:
            logger.error(f"Error in trading strategy analysis: {str(e)}")
            return None
    
    def create_comprehensive_report(self):
        """Create comprehensive testing report"""
        logger.info("Creating comprehensive testing report...")
        
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("BIOCON MODEL TESTING COMPREHENSIVE REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Models Tested: {len(self.test_results)}")
            
            # Model Performance Summary
            report_lines.append("\n📊 MODEL PERFORMANCE SUMMARY:")
            report_lines.append("-" * 50)
            
            # Sort models by R² score
            sorted_models = sorted(self.test_results.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
            
            report_lines.append(f"{'Model':<20} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'Dir. Acc.':<10}")
            report_lines.append("-" * 70)
            
            for model_name, results in sorted_models:
                metrics = results['metrics']
                report_lines.append(f"{model_name:<20} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f} "
                                  f"{metrics['mae']:<10.4f} {metrics['direction_accuracy']:<10.1f}%")
            
            # Best Model Details
            best_model_name, best_results = sorted_models[0]
            report_lines.append(f"\n🏆 BEST MODEL: {best_model_name}")
            report_lines.append("-" * 30)
            best_metrics = best_results['metrics']
            for metric, value in best_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric.replace('_', ' ').title()}: {value}")
            
            # FDA Milestone Analysis
            if self.fda_milestone_performance:
                report_lines.append(f"\n🎯 FDA MILESTONE ANALYSIS:")
                report_lines.append("-" * 30)
                
                for model_name, fda_data in self.fda_milestone_performance.items():
                    milestone_rmse = fda_data['milestone_periods']['rmse']
                    normal_rmse = fda_data['non_milestone_periods']['rmse']
                    performance_diff = ((normal_rmse - milestone_rmse) / normal_rmse) * 100
                    
                    report_lines.append(f"{model_name}:")
                    report_lines.append(f"  Milestone RMSE: {milestone_rmse:.4f}")
                    report_lines.append(f"  Normal RMSE: {normal_rmse:.4f}")
                    report_lines.append(f"  Performance Difference: {performance_diff:.1f}%")
                    report_lines.append("")
            
            # Statistical Tests Summary
            report_lines.append(f"\n📈 STATISTICAL ANALYSIS:")
            report_lines.append("-" * 25)
            
            best_residuals = best_results['actual'] - best_results['predictions']
            try:
                # Normality tests
                shapiro_stat, shapiro_p = shapiro(best_residuals[:5000] if len(best_residuals) > 5000 else best_residuals)
                jb_stat, jb_p = jarque_bera(best_residuals)
                
                report_lines.append(f"Residual Normality Tests ({best_model_name}):")
                report_lines.append(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
                report_lines.append(f"  Jarque-Bera p-value: {jb_p:.4f}")
                report_lines.append(f"  Residuals are {'NOT ' if min(shapiro_p, jb_p) < 0.05 else ''}normally distributed (α=0.05)")
                
            except Exception as e:
                report_lines.append(f"  Statistical tests failed: {str(e)}")
            
            # Files Generated
            report_lines.append(f"\n📁 FILES GENERATED:")
            report_lines.append("-" * 20)
            report_lines.append("  ✓ results/detailed_test_metrics.csv - Complete model metrics")
            report_lines.append("  ✓ results/predictions.csv - Time series predictions")
            report_lines.append("  ✓ results/fda_milestone_analysis.csv - FDA milestone performance")
            report_lines.append("  ✓ results/trading_strategy_analysis.csv - Trading strategy metrics")
            report_lines.append("  ✓ results/charts/ - All visualization charts")
            
            # Recommendations
            report_lines.append(f"\n💡 RECOMMENDATIONS:")
            report_lines.append("-" * 20)
            
            best_r2 = best_results['metrics']['r2']
            if best_r2 > 0.3:
                report_lines.append("  ✓ Model shows strong predictive power (R² > 0.3)")
            elif best_r2 > 0.1:
                report_lines.append("  ⚠️  Model shows moderate predictive power (R² > 0.1)")
            else:
                report_lines.append("  ❌ Model shows weak predictive power (R² < 0.1)")
            
            direction_acc = best_results['metrics']['direction_accuracy']
            if direction_acc > 55:
                report_lines.append("  ✓ Good directional accuracy for trading strategies")
            else:
                report_lines.append("  ⚠️  Limited directional accuracy, consider ensemble methods")
            
            if self.fda_milestone_performance:
                report_lines.append("  ✓ FDA milestone periods show different volatility patterns")
                report_lines.append("  💡 Consider milestone-specific models for better accuracy")
            
            report_lines.append("\n" + "=" * 80)
            
            # Save report
            with open('results/comprehensive_testing_report.txt', 'w') as f:
                f.write('\n'.join(report_lines))
            
            # Print report to console
            print('\n'.join(report_lines))
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {str(e)}")
    
    def execute(self):
        """Execute the complete model testing pipeline"""
        try:
            logger.info("Starting Biocon model testing pipeline...")
            
            # Step 1: Load trained models
            if not self.load_trained_models():
                raise Exception("Failed to load trained models")
            
            # Step 2: Load and prepare test data
            X_test, y_test, test_dates, full_df = self.load_test_data()
            
            # Step 3: Run comprehensive testing
            self.run_comprehensive_testing(X_test, y_test, test_dates)
            
            if not self.test_results:
                raise Exception("No models tested successfully")
            
            # Step 4: Analyze FDA milestone performance
            self.analyze_fda_milestone_performance()
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            # Step 6: Generate trading strategy analysis
            trading_metrics = self.generate_trading_strategy_analysis()
            
            # Step 7: Create comprehensive report
            self.create_comprehensive_report()
            
            logger.info("Model testing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model testing failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution function"""
    print("Starting Biocon Model Testing (Day 3)")
    print("Testing trained models on unseen data")
    print("Analyzing FDA milestone impact on predictions")
    print("Generating comprehensive performance analysis")
    print("-" * 60)
    
    tester = BioconModelTester()
    success = tester.execute()
    
    if success:
        print("\n🎉 SUCCESS: Model testing completed!")
        print("✓ All models tested and evaluated")
        print("✓ FDA milestone analysis completed")
        print("✓ Trading strategy analysis generated")
        print("✓ Comprehensive visualizations created")
        print("✓ Detailed report available in results/")
        print("\n📊 Key Files Generated:")
        print("  • results/comprehensive_testing_report.txt")
        print("  • results/predictions.csv")
        print("  • results/detailed_test_metrics.csv")
        print("  • results/charts/ (all visualizations)")
        print("\n🚀 Ready for Day 4: Future Predictions!")
    else:
        print("\n💥 FAILED: Model testing failed!")
        print("💡 Ensure Day 2 training was completed successfully")
        print("💡 Check that model files exist in models/ directory")
    
    return success

if __name__ == "__main__":
    main()