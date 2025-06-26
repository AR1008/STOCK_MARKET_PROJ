import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioconModelTrainer:
    """
    Comprehensive model trainer for Biocon stock prediction using news sentiment and FDA milestones
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.best_model_name = None
        self.best_model = None
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'results/charts']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_and_prepare_data(self):
        """Load stock and news data, combine and prepare features"""
        logger.info("Loading and preparing data...")
        
        try:
            # Load stock data
            stock_df = pd.read_csv('data/stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            
            # Remove timezone info if present
            if hasattr(stock_df['Date'].dtype, 'tz') and stock_df['Date'].dtype.tz is not None:
                stock_df['Date'] = stock_df['Date'].dt.tz_localize(None)
            
            logger.info(f"Loaded stock data: {len(stock_df)} records")
            
            # Load news sentiment data
            news_df = pd.read_csv('data/daily_sentiment.csv')
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # Remove timezone info if present
            if hasattr(news_df['date'].dtype, 'tz') and news_df['date'].dtype.tz is not None:
                news_df['date'] = news_df['date'].dt.tz_localize(None)
            
            logger.info(f"Loaded news sentiment data: {len(news_df)} records")
            
            # Convert both to date only (remove time component) for better matching
            stock_df['Date_only'] = stock_df['Date'].dt.date
            news_df['date_only'] = news_df['date'].dt.date
            
            # Merge datasets using date only
            combined_df = pd.merge(
                stock_df, 
                news_df, 
                left_on='Date_only', 
                right_on='date_only', 
                how='left'
            )
            
            # Drop the temporary date columns
            combined_df = combined_df.drop(['Date_only', 'date_only'], axis=1)
            
            # Fill missing sentiment values with neutral
            sentiment_columns = [
                'avg_sentiment', 'weighted_avg_sentiment', 'news_count',
                'drug_specific_count', 'day_importance_score'
            ]
            
            for col in sentiment_columns:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(0)
                else:
                    # Create the column with default values if it doesn't exist
                    combined_df[col] = 0
            
            # Add FDA milestone flags if they exist
            milestone_columns = [col for col in combined_df.columns if col.startswith('has_')]
            for col in milestone_columns:
                combined_df[col] = combined_df[col].fillna(0)
            
            # If no milestone columns exist, create basic ones
            if not milestone_columns:
                combined_df['has_fda_news'] = 0
                combined_df['has_drug_news'] = 0
            
            logger.info(f"Combined dataset: {len(combined_df)} records")
            logger.info(f"Available columns: {list(combined_df.columns)}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def engineer_features(self, df):
        """Create comprehensive feature set for model training"""
        logger.info("Engineering features...")
        
        try:
            # Sort by date
            df = df.sort_values('Date').copy()
            
            # Basic price-based features (only if Close column exists)
            if 'Close' in df.columns:
                df['Price_MA_5'] = df['Close'].rolling(window=5).mean()
                df['Price_MA_10'] = df['Close'].rolling(window=10).mean()
                df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
                
                # Price momentum
                df['Price_Momentum_5'] = df['Close'].pct_change(5)
                df['Price_Momentum_10'] = df['Close'].pct_change(10)
                df['Price_Momentum_20'] = df['Close'].pct_change(20)
            
            # Daily return volatility (only if Daily_Return exists)
            if 'Daily_Return' in df.columns:
                df['Price_Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
                df['Price_Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
                df['Price_Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
            
            # Volume features (only if Volume column exists)
            if 'Volume' in df.columns:
                df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
                df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
                df['Volume_Ratio_5'] = df['Volume'] / (df['Volume'].rolling(window=5).mean() + 1)
                df['Volume_Ratio_10'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1)
            
            # News sentiment features (only if columns exist)
            if 'avg_sentiment' in df.columns:
                df['Sentiment_MA_3'] = df['avg_sentiment'].rolling(window=3).mean()
                df['Sentiment_MA_7'] = df['avg_sentiment'].rolling(window=7).mean()
                df['Sentiment_MA_14'] = df['avg_sentiment'].rolling(window=14).mean()
                
                # Sentiment momentum
                df['Sentiment_Change_3'] = df['avg_sentiment'].diff(3)
                df['Sentiment_Change_7'] = df['avg_sentiment'].diff(7)
            
            # Weighted sentiment features
            if 'weighted_avg_sentiment' in df.columns:
                df['Weighted_Sentiment_MA_3'] = df['weighted_avg_sentiment'].rolling(window=3).mean()
                df['Weighted_Sentiment_MA_7'] = df['weighted_avg_sentiment'].rolling(window=7).mean()
                df['Weighted_Sentiment_Change_3'] = df['weighted_avg_sentiment'].diff(3)
            
            # News volume features
            if 'news_count' in df.columns:
                df['News_Volume_MA_7'] = df['news_count'].rolling(window=7).mean()
                df['News_Volume_Change'] = df['news_count'].diff()
            
            # Drug-specific news features
            if 'drug_specific_count' in df.columns:
                df['Drug_News_MA_7'] = df['drug_specific_count'].rolling(window=7).mean()
                df['Drug_News_Ratio'] = df['drug_specific_count'] / (df['news_count'] + 1)
            
            # FDA milestone impact features
            if 'day_importance_score' in df.columns:
                df['Importance_MA_7'] = df['day_importance_score'].rolling(window=7).mean()
                df['High_Importance_Day'] = (df['day_importance_score'] > 20).astype(int)
            
            # Market context features (only if available)
            if 'NIFTY50_Return' in df.columns and 'Daily_Return' in df.columns:
                df['Market_Relative_Performance'] = df['Daily_Return'] - df['NIFTY50_Return']
                df['Market_Beta'] = df['Daily_Return'].rolling(window=60).cov(df['NIFTY50_Return']) / (df['NIFTY50_Return'].rolling(window=60).var() + 0.0001)
            
            # Technical indicators (if available)
            technical_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'MA_20', 'MA_50']
            for indicator in technical_indicators:
                if indicator in df.columns:
                    rolling_mean = df[indicator].rolling(window=252).mean()
                    rolling_std = df[indicator].rolling(window=252).std()
                    df[f'{indicator}_normalized'] = (df[indicator] - rolling_mean) / (rolling_std + 0.0001)
            
            # Lag features (previous day values) - only for key columns that exist
            lag_features = ['Close', 'Daily_Return', 'avg_sentiment', 'weighted_avg_sentiment', 'news_count']
            for feature in lag_features:
                if feature in df.columns:
                    df[f'{feature}_lag1'] = df[feature].shift(1)
                    df[f'{feature}_lag2'] = df[feature].shift(2)
                    df[f'{feature}_lag3'] = df[feature].shift(3)
            
            # Target variables (what we want to predict) - only if Daily_Return and Close exist
            if 'Daily_Return' in df.columns:
                df['Next_Day_Return'] = df['Daily_Return'].shift(-1)  # Next day return
            if 'Close' in df.columns:
                df['Next_Day_Close'] = df['Close'].shift(-1)  # Next day closing price
                df['Next_5Day_Return'] = df['Close'].pct_change(-5)  # 5-day forward return
            
            # Date-based features
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
            
            logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def prepare_training_data(self, df):
        """Prepare data for training with proper train/validation/test splits"""
        logger.info("Preparing training data...")
        
        try:
            # Check if we have the target variable
            target_candidates = ['Next_Day_Return', 'Daily_Return']
            target_column = None
            
            for candidate in target_candidates:
                if candidate in df.columns and df[candidate].notna().sum() > 100:
                    target_column = candidate
                    break
            
            if target_column is None:
                # Create a simple target if none exists
                if 'Close' in df.columns:
                    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
                    target_column = 'Next_Day_Return'
                else:
                    raise ValueError("No suitable target variable found and cannot create one")
            
            logger.info(f"Using target variable: {target_column}")
            
            # Remove rows with NaN in target variable
            df_clean = df.dropna(subset=[target_column]).copy()
            logger.info(f"Clean dataset size: {len(df_clean)} records")
            
            if len(df_clean) < 50:
                raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} records")
            
            # Select feature columns (exclude target, date, and identifier columns)
            exclude_columns = [
                'Date', 'date', 'Next_Day_Return', 'Next_Day_Close', 'Next_5Day_Return',
                'Symbol', 'Company', 'Type', 'Source', 'datetime'
            ]
            
            feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
            
            # Remove any remaining columns with all NaN values
            for col in feature_columns[:]:
                if df_clean[col].isna().all():
                    feature_columns.remove(col)
                    logger.warning(f"Removed column {col} - all NaN values")
            
            logger.info(f"Selected {len(feature_columns)} feature columns")
            
            # Fill remaining NaN values with median/mode
            X = df_clean[feature_columns].copy()
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
            
            # Target variable
            y = df_clean[target_column].values
            
            # Remove any remaining NaN values in target
            valid_indices = ~pd.isna(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            logger.info(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
            
            # Time-aware split (important for time series data)
            # Use last 20% for testing, middle 20% for validation, first 60% for training
            n_samples = len(X)
            train_end = int(n_samples * 0.6)
            val_end = int(n_samples * 0.8)
            
            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            logger.info(f"Training data prepared:")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Validation samples: {len(X_val)}")
            logger.info(f"  Test samples: {len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train_traditional_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train traditional machine learning models"""
        logger.info("Training traditional ML models...")
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Define models
        models_to_train = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Lasso_Regression': Lasso(alpha=0.1),
            'Random_Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Use scaled data for linear models and SVR, original data for tree-based models
                if model_name in ['Linear_Regression', 'Ridge_Regression', 'Lasso_Regression', 'SVR']:
                    model.fit(X_train_scaled, y_train)
                    y_val_pred = model.predict(X_val_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_val_pred = model.predict(X_val)
                    y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Store model and metrics
                self.models[model_name] = model
                self.performance_metrics[model_name] = {
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_r2': val_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'val_rmse': np.sqrt(val_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(
                        self.feature_names, 
                        model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[model_name] = dict(zip(
                        self.feature_names, 
                        np.abs(model.coef_)
                    ))
                
                logger.info(f"{model_name} - Val RMSE: {np.sqrt(val_mse):.4f}, Val R²: {val_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
    
    def create_lstm_sequences(self, X, y, sequence_length=30):
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train LSTM model for time series prediction"""
        logger.info("Training LSTM model...")
        
        try:
            # Scale data for LSTM
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            X_test_scaled = scaler_X.transform(X_test)
            
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            self.scalers['lstm_X'] = scaler_X
            self.scalers['lstm_y'] = scaler_y
            
            # Create sequences
            sequence_length = 30
            X_train_seq, y_train_seq = self.create_lstm_sequences(X_train_scaled, y_train_scaled, sequence_length)
            X_val_seq, y_val_seq = self.create_lstm_sequences(X_val_scaled, y_val_scaled, sequence_length)
            X_test_seq, y_test_seq = self.create_lstm_sequences(X_test_scaled, y_test_scaled, sequence_length)
            
            if len(X_train_seq) == 0:
                logger.warning("Insufficient data for LSTM training")
                return
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            y_val_pred_scaled = model.predict(X_val_seq)
            y_test_pred_scaled = model.predict(X_test_seq)
            
            # Inverse transform predictions
            y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled).flatten()
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
            
            # Calculate metrics
            val_mse = mean_squared_error(y_val_seq, y_val_pred_scaled.flatten())
            test_mse = mean_squared_error(y_test_seq, y_test_pred_scaled.flatten())
            
            # Calculate metrics on original scale
            y_val_actual = y_val[sequence_length:]
            y_test_actual = y_test[sequence_length:]
            
            val_mse_orig = mean_squared_error(y_val_actual, y_val_pred)
            val_mae_orig = mean_absolute_error(y_val_actual, y_val_pred)
            val_r2_orig = r2_score(y_val_actual, y_val_pred)
            
            test_mse_orig = mean_squared_error(y_test_actual, y_test_pred)
            test_mae_orig = mean_absolute_error(y_test_actual, y_test_pred)
            test_r2_orig = r2_score(y_test_actual, y_test_pred)
            
            # Store model and metrics
            self.models['LSTM'] = model
            self.performance_metrics['LSTM'] = {
                'val_mse': val_mse_orig,
                'val_mae': val_mae_orig,
                'val_r2': val_r2_orig,
                'test_mse': test_mse_orig,
                'test_mae': test_mae_orig,
                'test_r2': test_r2_orig,
                'val_rmse': np.sqrt(val_mse_orig),
                'test_rmse': np.sqrt(test_mse_orig)
            }
            
            logger.info(f"LSTM - Val RMSE: {np.sqrt(val_mse_orig):.4f}, Val R²: {val_r2_orig:.4f}")
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
    
    def select_best_model(self):
        """Select the best performing model based on validation metrics"""
        logger.info("Selecting best model...")
        
        if not self.performance_metrics:
            logger.error("No models trained successfully")
            return
        
        # Compare models based on validation R² score
        best_r2 = -float('inf')
        best_model_name = None
        
        for model_name, metrics in self.performance_metrics.items():
            val_r2 = metrics['val_r2']
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} (Val R²: {best_r2:.4f})")
    
    def save_models(self):
        """Save trained models and scalers"""
        logger.info("Saving models...")
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                if model_name == 'LSTM':
                    # Save Keras model
                    model.save(f'models/{model_name.lower()}_model.h5')
                else:
                    # Save sklearn model
                    with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
                        pickle.dump(model, f)
            
            # Save the best model as final_model.pkl
            if self.best_model is not None:
                if self.best_model_name == 'LSTM':
                    self.best_model.save('models/final_model.h5')
                else:
                    with open('models/final_model.pkl', 'wb') as f:
                        pickle.dump(self.best_model, f)
            
            # Save scalers
            with open('models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save feature names
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            # Save performance metrics
            performance_df = pd.DataFrame(self.performance_metrics).T
            performance_df.to_csv('results/model_performance.csv')
            
            # Save feature importance
            if self.feature_importance:
                importance_df = pd.DataFrame(self.feature_importance)
                importance_df.to_csv('results/feature_importance.csv')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*80)
        print("BIOCON MODEL TRAINING SUMMARY")
        print("="*80)
        
        if not self.performance_metrics:
            print("❌ No models trained successfully")
            return
        
        print(f"✓ Models Trained: {len(self.models)}")
        print(f"✓ Features Used: {len(self.feature_names)}")
        print(f"✓ Best Model: {self.best_model_name}")
        
        print(f"\n📊 MODEL PERFORMANCE COMPARISON:")
        print(f"{'Model':<20} {'Val RMSE':<12} {'Val R²':<10} {'Test RMSE':<12} {'Test R²':<10}")
        print("-" * 70)
        
        # Sort by validation R²
        sorted_models = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1]['val_r2'],
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} {metrics['val_rmse']:<12.4f} {metrics['val_r2']:<10.4f} "
                  f"{metrics['test_rmse']:<12.4f} {metrics['test_r2']:<10.4f}")
        
        # Feature importance for best model
        if self.best_model_name in self.feature_importance:
            print(f"\n🎯 TOP 10 FEATURES ({self.best_model_name}):")
            importance = self.feature_importance[self.best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance_score in top_features:
                print(f"  {feature}: {importance_score:.4f}")
        
        print(f"\n📁 SAVED FILES:")
        print(f"  ✓ models/final_model.pkl - Best performing model")
        print(f"  ✓ models/scalers.pkl - Data scalers")
        print(f"  ✓ models/feature_names.pkl - Feature names")
        print(f"  ✓ results/model_performance.csv - Performance metrics")
        print(f"  ✓ results/feature_importance.csv - Feature importance")
        
        print("="*80)
    
    def execute(self):
        """Execute the complete model training pipeline"""
        try:
            logger.info("Starting Biocon model training pipeline...")
            
            # Step 1: Load and prepare data
            combined_df = self.load_and_prepare_data()
            
            # Step 2: Engineer features
            df_with_features = self.engineer_features(combined_df)
            
            # Step 3: Prepare training data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(df_with_features)
            
            # Step 4: Train traditional models
            self.train_traditional_models(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 5: Train LSTM model
            self.train_lstm_model(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 6: Select best model
            self.select_best_model()
            
            # Step 7: Save models
            self.save_models()
            
            # Step 8: Print summary
            self.print_training_summary()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution function"""
    print("Starting Biocon Model Training (Day 2)")
    print("Combining stock data with news sentiment for prediction")
    print("Training multiple models: Linear, Tree-based, Neural Networks")
    print("-" * 60)
    
    trainer = BioconModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 SUCCESS: Model training completed!")
        print("✓ Multiple models trained and evaluated")
        print("✓ Best model selected and saved")
        print("✓ Ready for model testing (Day 3)")
    else:
        print("\n💥 FAILED: Model training failed!")
        print("💡 Check data files exist: data/stock_data.csv, data/daily_sentiment.csv")
    
    return success

if __name__ == "__main__":
    main()