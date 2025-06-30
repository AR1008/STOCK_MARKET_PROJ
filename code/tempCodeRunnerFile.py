"""
Complete Machine Learning Training Pipeline for Biocon FDA Project
Day 3: Advanced Model Training with Multiple Algorithms and Comprehensive Evaluation

Features:
- Multiple ML algorithms (Ridge, Random Forest, XGBoost, LightGBM, LSTM)
- Advanced feature engineering and selection
- Comprehensive model evaluation and comparison
- Ensemble modeling
- Cross-validation and hyperparameter tuning
- Model persistence and metadata tracking
- Extensive visualization and reporting
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, GridSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFE, SelectFromModel
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Import configuration
from config import (
    COMPANY_INFO, PATHS, DATA_FILES, MODEL_CONFIG,
    FEATURE_CONFIG, create_directories, validate_config
)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PATHS['logs'] / 'model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    """
    Advanced ML model trainer with comprehensive evaluation and ensemble methods
    """
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.best_features = None
        
        # Create directories
        create_directories()
        validate_config()
        
        # Set up results directories
        self.charts_dir = PATHS['results'] / 'charts'
        self.charts_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Advanced Model Trainer Initialized")
        logger.info(f"🏢 Company: {COMPANY_INFO['name']}")
    
    def load_and_prepare_data(self):
        """
        Load and prepare the dataset for training
        """
        logger.info("📊 Loading and preparing data...")
        
        try:
            # Load main dataset
            data_path = PATHS['data'] / DATA_FILES['stock_data']
            if not data_path.exists():
                raise FileNotFoundError(f"Stock data not found: {data_path}")
            
            self.data = pd.read_csv(data_path)
            logger.info(f"✅ Loaded dataset: {len(self.data)} records, {len(self.data.columns)} features")
            
            # Convert Date column
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            # Remove rows with missing target values
            if 'Forward_Return_1D' in self.data.columns:
                self.target_column = 'Forward_Return_1D'
            elif 'Close' in self.data.columns:
                # Create next day return as target
                self.data['Forward_Return_1D'] = self.data['Close'].pct_change(-1)
                self.target_column = 'Forward_Return_1D'
            else:
                raise ValueError("No suitable target variable found")
            
            # Remove last row (no forward return)
            self.data = self.data[:-1]
            
            # Remove rows with missing target
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=[self.target_column])
            logger.info(f"📊 Removed {initial_len - len(self.data)} rows with missing target")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def engineer_features(self):
        """
        Advanced feature engineering and selection
        """
        logger.info("🔧 Engineering and selecting features...")
        
        try:
            # Exclude non-feature columns
            exclude_columns = [
                'Date', 'Symbol', 'Company', 'Source', self.target_column,
                'Forward_Return_3D', 'Forward_Return_5D', 'Forward_Return_10D'
            ]
            
            # Get feature columns
            feature_columns = [col for col in self.data.columns 
                             if col not in exclude_columns and not col.startswith('Forward_')]
            
            # Handle missing values
            for col in feature_columns:
                if self.data[col].dtype in ['float64', 'int64']:
                    # Fill with median for numeric columns
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    # Fill with mode for categorical columns
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else 0)
            
            # Remove highly correlated features
            numeric_features = [col for col in feature_columns 
                              if self.data[col].dtype in ['float64', 'int64']]
            
            if len(numeric_features) > 1:
                corr_matrix = self.data[numeric_features].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                )
                
                # Find features with correlation > 0.95
                high_corr_features = [column for column in upper_triangle.columns 
                                    if any(upper_triangle[column] > 0.95)]
                
                feature_columns = [col for col in feature_columns if col not in high_corr_features]
                logger.info(f"🔧 Removed {len(high_corr_features)} highly correlated features")
            
            # Remove features with low variance
            from sklearn.feature_selection import VarianceThreshold
            
            numeric_data = self.data[numeric_features].select_dtypes(include=[np.number])
            if not numeric_data.empty:
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(numeric_data)
                low_var_features = [col for col, var in zip(numeric_features, selector.variances_) if var <= 0.01]
                feature_columns = [col for col in feature_columns if col not in low_var_features]
                logger.info(f"🔧 Removed {len(low_var_features)} low variance features")
            
            # Create additional features
            if 'Close' in self.data.columns and 'Volume' in self.data.columns:
                self.data['Price_Volume_Trend'] = self.data['Close'] * self.data['Volume']
                feature_columns.append('Price_Volume_Trend')
            
            if 'High' in self.data.columns and 'Low' in self.data.columns:
                self.data['Daily_Range'] = (self.data['High'] - self.data['Low']) / self.data['Low']
                feature_columns.append('Daily_Range')
            
            # Interaction features
            if 'RSI_14' in self.data.columns and 'MACD' in self.data.columns:
                self.data['RSI_MACD_Interaction'] = self.data['RSI_14'] * self.data['MACD']
                feature_columns.append('RSI_MACD_Interaction')
            
            self.features = feature_columns
            self.target = self.target_column
            
            logger.info(f"✅ Feature engineering completed: {len(self.features)} features selected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {str(e)}")
            return False
    
    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets using time-based split
        """
        logger.info("📊 Splitting data into train/test sets...")
        
        try:
            # Prepare feature matrix and target
            X = self.data[self.features].copy()
            y = self.data[self.target].copy()
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            # Time-based split (important for financial data)
            split_date = self.data['Date'].quantile(1 - test_size)
            train_mask = self.data['Date'] < split_date
            test_mask = self.data['Date'] >= split_date
            
            # Apply the same mask to remove NaN rows
            train_mask = train_mask[mask]
            test_mask = test_mask[mask]
            
            self.X_train = X[train_mask]
            self.X_test = X[test_mask]
            self.y_train = y[train_mask]
            self.y_test = y[test_mask]
            
            logger.info(f"✅ Data split completed:")
            logger.info(f"   Training set: {len(self.X_train)} samples")
            logger.info(f"   Testing set: {len(self.X_test)} samples")
            logger.info(f"   Features: {len(self.features)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error splitting data: {str(e)}")
            return False
    
    def scale_features(self):
        """
        Scale features using multiple scalers
        """
        logger.info("📏 Scaling features...")
        
        try:
            # Standard Scaler
            self.scalers['standard'] = StandardScaler()
            X_train_std = self.scalers['standard'].fit_transform(self.X_train)
            X_test_std = self.scalers['standard'].transform(self.X_test)
            
            # Robust Scaler
            self.scalers['robust'] = RobustScaler()
            X_train_robust = self.scalers['robust'].fit_transform(self.X_train)
            X_test_robust = self.scalers['robust'].transform(self.X_test)
            
            # MinMax Scaler
            self.scalers['minmax'] = MinMaxScaler()
            X_train_minmax = self.scalers['minmax'].fit_transform(self.X_train)
            X_test_minmax = self.scalers['minmax'].transform(self.X_test)
            
            # Store scaled data
            self.scaled_data = {
                'standard': (X_train_std, X_test_std),
                'robust': (X_train_robust, X_test_robust),
                'minmax': (X_train_minmax, X_test_minmax)
            }
            
            logger.info("✅ Feature scaling completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error scaling features: {str(e)}")
            return False
    
    def select_best_features(self, max_features=50):
        """
        Select best features using multiple methods
        """
        logger.info("🎯 Selecting best features...")
        
        try:
            # Use standard scaled data for feature selection
            X_train_scaled, _ = self.scaled_data['standard']
            
            # Method 1: Statistical feature selection
            selector_stats = SelectKBest(score_func=f_regression, k=min(max_features, len(self.features)))
            selector_stats.fit(X_train_scaled, self.y_train)
            
            # Method 2: Random Forest feature importance
            rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_selector.fit(X_train_scaled, self.y_train)
            
            # Get feature importance scores
            feature_scores = pd.DataFrame({
                'feature': self.features,
                'statistical_score': selector_stats.scores_,
                'rf_importance': rf_selector.feature_importances_
            })
            
            # Combined ranking
            feature_scores['statistical_rank'] = feature_scores['statistical_score'].rank(ascending=False)
            feature_scores['rf_rank'] = feature_scores['rf_importance'].rank(ascending=False)
            feature_scores['combined_rank'] = (feature_scores['statistical_rank'] + feature_scores['rf_rank']) / 2
            
            # Select top features
            top_features = feature_scores.nsmallest(max_features, 'combined_rank')['feature'].tolist()
            
            # Update features and data
            self.best_features = top_features
            self.X_train = self.X_train[self.best_features]
            self.X_test = self.X_test[self.best_features]
            
            # Update scaled data
            feature_indices = [self.features.index(f) for f in self.best_features]
            for scaler_name, (X_train_scaled, X_test_scaled) in self.scaled_data.items():
                self.scaled_data[scaler_name] = (
                    X_train_scaled[:, feature_indices],
                    X_test_scaled[:, feature_indices]
                )
            
            # Save feature importance
            feature_scores.to_csv(PATHS['results'] / 'feature_importance.csv', index=False)
            
            logger.info(f"✅ Feature selection completed: {len(self.best_features)} features selected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in feature selection: {str(e)}")
            return False
    
    def train_ridge_regression(self):
        """
        Train Ridge Regression with hyperparameter tuning
        """
        logger.info("🤖 Training Ridge Regression...")
        
        try:
            X_train_scaled, X_test_scaled = self.scaled_data['standard']
            
            # Hyperparameter tuning
            param_grid = {'alpha': MODEL_CONFIG['ridge_regression']['alpha']}
            
            ridge = Ridge(random_state=42)
            grid_search = GridSearchCV(
                ridge, param_grid, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, self.y_train)
            
            # Best model
            best_ridge = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_ridge.predict(X_train_scaled)
            y_test_pred = best_ridge.predict(X_test_scaled)
            
            # Store model and predictions
            self.models['ridge'] = best_ridge
            self.predictions['ridge'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['ridge'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"✅ Ridge Regression: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training Ridge Regression: {str(e)}")
            return False
    
    def train_random_forest(self):
        """
        Train Random Forest with hyperparameter tuning
        """
        logger.info("🌲 Training Random Forest...")
        
        try:
            # Use original scaled data (Random Forest handles scaling well)
            X_train_scaled, X_test_scaled = self.scaled_data['standard']
            
            # Simplified hyperparameter grid for faster training
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3,  # Reduced CV for speed
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, self.y_train)
            
            # Best model
            best_rf = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_rf.predict(X_train_scaled)
            y_test_pred = best_rf.predict(X_test_scaled)
            
            # Store model and predictions
            self.models['random_forest'] = best_rf
            self.predictions['random_forest'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Feature importance
            self.feature_importance['random_forest'] = dict(zip(
                self.best_features, best_rf.feature_importances_
            ))
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['random_forest'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"✅ Random Forest: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training Random Forest: {str(e)}")
            return False
    
    def train_xgboost(self):
        """
        Train XGBoost with hyperparameter tuning
        """
        logger.info("🚀 Training XGBoost...")
        
        try:
            X_train_scaled, X_test_scaled = self.scaled_data['standard']
            
            # Simplified hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, self.y_train)
            
            # Best model
            best_xgb = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_xgb.predict(X_train_scaled)
            y_test_pred = best_xgb.predict(X_test_scaled)
            
            # Store model and predictions
            self.models['xgboost'] = best_xgb
            self.predictions['xgboost'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Feature importance
            self.feature_importance['xgboost'] = dict(zip(
                self.best_features, best_xgb.feature_importances_
            ))
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['xgboost'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"✅ XGBoost: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training XGBoost: {str(e)}")
            return False
    
    def train_lightgbm(self):
        """
        Train LightGBM with hyperparameter tuning
        """
        logger.info("💡 Training LightGBM...")
        
        try:
            X_train_scaled, X_test_scaled = self.scaled_data['standard']
            
            # Simplified hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50]
            }
            
            lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=3,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, self.y_train)
            
            # Best model
            best_lgb = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_lgb.predict(X_train_scaled)
            y_test_pred = best_lgb.predict(X_test_scaled)
            
            # Store model and predictions
            self.models['lightgbm'] = best_lgb
            self.predictions['lightgbm'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Feature importance
            self.feature_importance['lightgbm'] = dict(zip(
                self.best_features, best_lgb.feature_importances_
            ))
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['lightgbm'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"✅ LightGBM: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training LightGBM: {str(e)}")
            return False
    
    def train_lstm(self):
        """
        Train LSTM neural network
        """
        logger.info("🧠 Training LSTM Neural Network...")
        
        try:
            # Use MinMax scaled data for LSTM
            X_train_scaled, X_test_scaled = self.scaled_data['minmax']
            
            # Reshape data for LSTM (samples, timesteps, features)
            # For simplicity, we'll use 1 timestep
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1]),
                     kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(25, kernel_regularizer=l2(0.01)),
                Dropout(0.2),
                BatchNormalization(),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
            )
            
            # Train model
            history = model.fit(
                X_train_reshaped, self.y_train,
                epochs=50,  # Reduced for faster training
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_train_pred = model.predict(X_train_reshaped).flatten()
            y_test_pred = model.predict(X_test_reshaped).flatten()
            
            # Store model and predictions
            self.models['lstm'] = model
            self.predictions['lstm'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['lstm'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'epochs_trained': len(history.history['loss'])
            }
            
            logger.info(f"✅ LSTM: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training LSTM: {str(e)}")
            return False
    
    def create_ensemble_model(self):
        """
        Create ensemble model combining best performers
        """
        logger.info("🎭 Creating ensemble model...")
        
        try:
            # Get models that trained successfully
            available_models = []
            available_names = []
            
            for name, model in self.models.items():
                if name != 'lstm':  # Exclude LSTM from ensemble for simplicity
                    available_models.append((name, model))
                    available_names.append(name)
            
            if len(available_models) < 2:
                logger.warning("⚠️ Not enough models for ensemble")
                return False
            
            # Create voting ensemble
            estimators = available_models
            ensemble = VotingRegressor(estimators=estimators)
            
            # Fit ensemble
            X_train_scaled, X_test_scaled = self.scaled_data['standard']
            ensemble.fit(X_train_scaled, self.y_train)
            
            # Predictions
            y_train_pred = ensemble.predict(X_train_scaled)
            y_test_pred = ensemble.predict(X_test_scaled)
            
            # Store ensemble
            self.models['ensemble'] = ensemble
            self.predictions['ensemble'] = {
                'train': y_train_pred,
                'test': y_test_pred
            }
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            self.model_metrics['ensemble'] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'component_models': available_names
            }
            
            logger.info(f"✅ Ensemble: Test R² = {test_r2:.4f}, Test MSE = {test_mse:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble: {str(e)}")
            return False
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation and comparison
        """
        logger.info("📊 Evaluating all models...")
        
        try:
            # Create comprehensive metrics dataframe
            results = []
            
            for model_name in self.models.keys():
                if model_name in self.predictions:
                    y_test_pred = self.predictions[model_name]['test']
                    y_train_pred = self.predictions[model_name]['train']
                    
                    # Calculate comprehensive metrics
                    test_mse = mean_squared_error(self.y_test, y_test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_mae = mean_absolute_error(self.y_test, y_test_pred)
                    test_r2 = r2_score(self.y_test, y_test_pred)
                    test_mape = mean_absolute_percentage_error(self.y_test, y_test_pred) * 100
                    
                    train_mse = mean_squared_error(self.y_train, y_train_pred)
                    train_r2 = r2_score(self.y_train, y_train_pred)
                    
                    # Overfitting check
                    overfitting = train_r2 - test_r2
                    
                    results.append({
                        'Model': model_name,
                        'Test_R2': test_r2,
                        'Test_MSE': test_mse,
                        'Test_RMSE': test_rmse,
                        'Test_MAE': test_mae,
                        'Test_MAPE': test_mape,
                        'Train_R2': train_r2,
                        'Overfitting': overfitting
                    })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Test_R2', ascending=False)
            
            # Save results
            results_df.to_csv(PATHS['results'] / 'model_comparison.csv', index=False)
            
            # Print results
            print("\n" + "="*80)
            print("🏆 MODEL PERFORMANCE COMPARISON")
            print("="*80)
            print(results_df.round(4).to_string(index=False))
            print("="*80)
            
            # Identify best model
            best_model_name = results_df.iloc[0]['Model']
            best_r2 = results_df.iloc[0]['Test_R2']
            
            logger.info(f"🥇 Best Model: {best_model_name} with Test R² = {best_r2:.4f}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"❌ Error evaluating models: {str(e)}")
            return None
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations
        """
        logger.info("📈 Creating visualizations...")
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # R² scores
            models = list(self.model_metrics.keys())
            test_r2_scores = [self.model_metrics[m]['test_r2'] for m in models]
            train_r2_scores = [self.model_metrics[m]['train_r2'] for m in models]
            
            axes[0, 0].bar(models, test_r2_scores, alpha=0.7, label='Test R²')
            axes[0, 0].bar(models, train_r2_scores, alpha=0.5, label='Train R²')
            axes[0, 0].set_title('Model R² Scores Comparison')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # MSE scores
            test_mse_scores = [self.model_metrics[m]['test_mse'] for m in models]
            axes[0, 1].bar(models, test_mse_scores, alpha=0.7, color='orange')
            axes[0, 1].set_title('Model MSE Scores (Lower is Better)')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 2. Predictions vs Actual (Best Model)
            best_model = max(self.models.keys(), key=lambda x: self.model_metrics[x]['test_r2'])
            y_test_pred = self.predictions[best_model]['test']
            
            axes[1, 0].scatter(self.y_test, y_test_pred, alpha=0.6)
            axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Returns')
            axes[1, 0].set_ylabel('Predicted Returns')
            axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
            
            # 3. Residuals plot
            residuals = self.y_test - y_test_pred
            axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Predicted Returns')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Plot')
            
            plt.tight_layout()
            plt.savefig(self.charts_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Feature Importance (if available)
            if self.feature_importance:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                for i, (model_name, importance) in enumerate(self.feature_importance.items()):
                    if i >= 4:  # Only plot first 4 models
                        break
                    
                    # Sort features by importance
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    features, importances = zip(*sorted_importance[:15])  # Top 15 features
                    
                    axes[i].barh(range(len(features)), importances)
                    axes[i].set_yticks(range(len(features)))
                    axes[i].set_yticklabels(features)
                    axes[i].set_title(f'Feature Importance - {model_name}')
                    axes[i].set_xlabel('Importance')
                
                plt.tight_layout()
                plt.savefig(self.charts_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Time Series of Predictions
            if 'Date' in self.data.columns:
                # Get test dates
                test_dates = self.data.iloc[-len(self.y_test):]['Date']
                
                plt.figure(figsize=(15, 8))
                plt.plot(test_dates, self.y_test.values, label='Actual', alpha=0.7)
                
                for model_name in ['ridge', 'random_forest', 'xgboost', 'ensemble']:
                    if model_name in self.predictions:
                        y_pred = self.predictions[model_name]['test']
                        plt.plot(test_dates, y_pred, label=f'{model_name} Prediction', alpha=0.7)
                
                plt.title('Prediction Time Series Comparison')
                plt.xlabel('Date')
                plt.ylabel('Returns')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.charts_dir / 'predictions_timeseries.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("✅ Visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {str(e)}")
            return False
    
    def save_models(self):
        """
        Save all trained models and metadata
        """
        logger.info("💾 Saving models and metadata...")
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                if model_name == 'lstm':
                    # Save Keras model
                    model.save(PATHS['models'] / f'{model_name}_model.h5')
                else:
                    # Save sklearn models
                    joblib.dump(model, PATHS['models'] / f'{model_name}_model.pkl')
            
            # Save scalers
            joblib.dump(self.scalers, PATHS['models'] / 'scalers.pkl')
            
            # Save feature names
            with open(PATHS['models'] / 'feature_names.json', 'w') as f:
                json.dump({
                    'all_features': self.features,
                    'best_features': self.best_features,
                    'target': self.target
                }, f, indent=2)
            
            # Save model metrics
            with open(PATHS['models'] / 'model_metrics.json', 'w') as f:
                json.dump(self.model_metrics, f, indent=2, default=str)
            
            # Save feature importance
            with open(PATHS['models'] / 'feature_importance.json', 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            # Save predictions
            predictions_df = pd.DataFrame()
            predictions_df['actual'] = self.y_test.values
            
            for model_name, preds in self.predictions.items():
                predictions_df[f'{model_name}_pred'] = preds['test']
            
            predictions_df.to_csv(PATHS['results'] / 'predictions.csv', index=False)
            
            # Create model summary
            summary = {
                'training_date': datetime.now().isoformat(),
                'company': COMPANY_INFO['name'],
                'target_variable': self.target,
                'total_features': len(self.features),
                'selected_features': len(self.best_features),
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'models_trained': list(self.models.keys()),
                'best_model': max(self.models.keys(), key=lambda x: self.model_metrics[x]['test_r2']),
                'best_test_r2': max(self.model_metrics[m]['test_r2'] for m in self.models.keys())
            }
            
            with open(PATHS['models'] / 'training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("✅ All models and metadata saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {str(e)}")
            return False
    
    def print_comprehensive_summary(self):
        """
        Print detailed training summary
        """
        if not self.models:
            logger.error("❌ No models trained")
            return
        
        print("\n" + "="*80)
        print("🚀 BIOCON FDA PROJECT - COMPREHENSIVE TRAINING SUMMARY")
        print("="*80)
        
        print(f"🏢 Company: {COMPANY_INFO['name']}")
        print(f"🎯 Target Variable: {self.target}")
        print(f"📊 Dataset: {len(self.data)} total records")
        print(f"🔧 Features: {len(self.best_features)} selected from {len(self.features)} total")
        print(f"📚 Training Set: {len(self.X_train)} samples")
        print(f"🧪 Test Set: {len(self.X_test)} samples")
        
        print(f"\n🤖 Models Trained: {len(self.models)}")
        for model_name in self.models.keys():
            print(f"   ✅ {model_name.title()}")
        
        print(f"\n🏆 Performance Summary:")
        best_model = max(self.models.keys(), key=lambda x: self.model_metrics[x]['test_r2'])
        best_r2 = self.model_metrics[best_model]['test_r2']
        best_mse = self.model_metrics[best_model]['test_mse']
        
        print(f"   🥇 Best Model: {best_model}")
        print(f"   📈 Best Test R²: {best_r2:.4f}")
        print(f"   📉 Best Test MSE: {best_mse:.6f}")
        
        if 'ensemble' in self.models:
            ensemble_r2 = self.model_metrics['ensemble']['test_r2']
            print(f"   🎭 Ensemble R²: {ensemble_r2:.4f}")
        
        print(f"\n📁 Outputs Generated:")
        print(f"   ✅ Models saved to: {PATHS['models']}")
        print(f"   ✅ Results saved to: {PATHS['results']}")
        print(f"   ✅ Charts saved to: {self.charts_dir}")
        print(f"   ✅ Predictions: {PATHS['results'] / 'predictions.csv'}")
        print(f"   ✅ Model comparison: {PATHS['results'] / 'model_comparison.csv'}")
        
        print(f"\n🎯 Top 10 Most Important Features:")
        if self.feature_importance and 'random_forest' in self.feature_importance:
            rf_importance = self.feature_importance['random_forest']
            top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        print(f"\n🔄 Ready for Day 4: Model Testing & Validation")
        print("="*80)
    
    def execute(self):
        """
        Execute the complete training pipeline
        """
        try:
            logger.info("🚀 Starting Comprehensive Model Training Pipeline...")
            
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data():
                raise Exception("Failed to load data")
            
            # Step 2: Feature engineering
            if not self.engineer_features():
                raise Exception("Failed to engineer features")
            
            # Step 3: Split data
            if not self.split_data():
                raise Exception("Failed to split data")
            
            # Step 4: Scale features
            if not self.scale_features():
                raise Exception("Failed to scale features")
            
            # Step 5: Feature selection
            if not self.select_best_features():
                raise Exception("Failed to select features")
            
            # Step 6: Train models
            models_to_train = [
                ('Ridge Regression', self.train_ridge_regression),
                ('Random Forest', self.train_random_forest),
                ('XGBoost', self.train_xgboost),
                ('LightGBM', self.train_lightgbm),
                ('LSTM', self.train_lstm)
            ]
            
            successful_models = 0
            for model_name, train_func in models_to_train:
                try:
                    if train_func():
                        successful_models += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to train {model_name}: {str(e)}")
            
            if successful_models == 0:
                raise Exception("No models trained successfully")
            
            logger.info(f"✅ Successfully trained {successful_models}/{len(models_to_train)} models")
            
            # Step 7: Create ensemble
            self.create_ensemble_model()
            
            # Step 8: Evaluate models
            self.evaluate_models()
            
            # Step 9: Create visualizations
            self.create_visualizations()
            
            # Step 10: Save models
            self.save_models()
            
            # Step 11: Print summary
            self.print_comprehensive_summary()
            
            logger.info("🎉 Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            logger.info("💡 Troubleshooting steps:")
            logger.info("  1. Verify stock data exists: data/stock_data.csv")
            logger.info("  2. Check data quality and completeness")
            logger.info("  3. Ensure required libraries are installed")
            logger.info("  4. Check logs for detailed error information")
            return False

def main():
    """
    Main execution function for Day 3 - Model Training
    """
    print("🚀 BIOCON FDA PROJECT - DAY 3")
    print("Comprehensive Machine Learning Model Training")
    print("="*60)
    print(f"🏢 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"🎯 Objective: Train multiple ML models to predict stock returns")
    print(f"🤖 Models: Ridge, Random Forest, XGBoost, LightGBM, LSTM, Ensemble")
    print("-" * 60)
    
    trainer = AdvancedModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 SUCCESS: Model training completed!")
        print("✅ Multiple ML models trained and evaluated")
        print("✅ Feature importance analysis completed")
        print("✅ Ensemble model created")
        print("✅ Comprehensive visualizations generated")
        print("✅ Models and metadata saved")
        print("✅ Performance metrics calculated")
        print("🔄 Ready for Day 4: Model Testing & Deployment")
    else:
        print("\n❌ FAILED: Model training failed")
        print("💡 Check logs for details: logs/model_training.log")
        print("🔧 Troubleshooting:")
        print("  - Ensure stock data collection (Day 1) completed successfully")
        print("  - Verify data quality in data/stock_data.csv")
        print("  - Check system resources (RAM/CPU) for large models")
        print("  - Install missing dependencies: pip install -r requirements.txt")
    
    return success

if __name__ == "__main__":
    main()