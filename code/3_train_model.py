"""
Advanced Model Training for Biocon FDA Project
Day 2 - Step 3: Train multiple machine learning models for stock price prediction

Features:
- Combines stock data with news sentiment and FDA events
- Enhanced feature selection with correlation and RFE
- Multiple model architectures (Linear Regression, Random Forest, Gradient Boosting, LSTM)
- Cross-validation and hyperparameter tuning
- Model performance evaluation and visualization
- Stacking ensemble with fallback to weighted average
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
try:
    from config import (
        COMPANY_INFO, DATA_START_DATE, DATA_END_DATE,
        PATHS, DATA_FILES, MODEL_CONFIG, create_directories, validate_config
    )
except ImportError:
    from config import (
        COMPANY_INFO, DATA_START_DATE, DATA_END_DATE,
        PATHS, DATA_FILES, create_directories, validate_config
    )
    MODEL_CONFIG = {
        'ridge_regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'lasso_regression': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [10, 20],
            'min_split_gain': [0.0]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'svr': {
            'kernel': ['rbf'],
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.5]
        },
        'lstm': {
            'epochs': 100,
            'batch_size': 32,
            'units': [50, 25]
        }
    }
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    def __init__(self):
        self.start_date = DATA_START_DATE
        self.end_date = DATA_END_DATE
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scalers = {}
        self.models = {}
        self.feature_names = []
        self.random_state = 42
        
        create_directories()
        validate_config()
        
        logger.info("🚀 Advanced Model Trainer Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"🏢 Target Company: {COMPANY_INFO['name']}")

    def load_and_prepare_data(self):
        """
        Load and combine stock, sentiment, and FDA event data with enhanced feature selection
        """
        logger.info("📊 Loading and preparing data...")
        
        try:
            # Check for saved features and scalers
            feature_file = PATHS['models'] / 'feature_names.pkl'
            scaler_file = PATHS['models'] / 'scalers.pkl'
            if feature_file.exists() and scaler_file.exists():
                with open(feature_file, 'rb') as f:
                    self.feature_names = pickle.load(f)
                with open(scaler_file, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info("✅ Loaded saved features and scalers for consistency")
            
            # Load stock data (assumes benchmark indices are included)
            stock_file = PATHS['data'] / DATA_FILES['stock_data']
            if not stock_file.exists():
                raise FileNotFoundError(f"Stock data file not found: {stock_file}")
            stock_df = pd.read_csv(stock_file)
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            
            # Load daily sentiment data
            sentiment_file = PATHS['data'] / DATA_FILES['daily_sentiment']
            if sentiment_file.exists():
                sentiment_df = pd.read_csv(sentiment_file)
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
                sentiment_df = sentiment_df.rename(columns={'avg_sentiment': 'sentiment_score'})
                if 'sentiment_score' not in sentiment_df.columns:
                    logger.warning("⚠️ 'sentiment_score' column missing, using zeros")
                    sentiment_df['sentiment_score'] = 0.0
            else:
                logger.warning(f"Sentiment data file not found: {sentiment_file}")
                sentiment_df = pd.DataFrame({
                    'date': stock_df['Date'],
                    'sentiment_score': [0.0] * len(stock_df)
                })
            
            # Load FDA events
            fda_file = PATHS['data'] / DATA_FILES['fda_events']
            if fda_file.exists():
                fda_df = pd.read_csv(fda_file)
                fda_df['date'] = pd.to_datetime(fda_df['date']).dt.tz_localize(None)
            else:
                logger.warning(f"FDA events file not found: {fda_file}")
                fda_df = pd.DataFrame({
                    'date': stock_df['Date'],
                    'fda_milestone_type': ['other'] * len(stock_df)
                })
            
            # Merge datasets
            df = pd.merge(stock_df, sentiment_df, left_on='Date', right_on='date', how='left')
            df = df.drop(columns=['date'], errors='ignore')
            if not fda_df.empty:
                fda_df = fda_df.groupby('date').agg({
                    'fda_milestone_type': lambda x: '|'.join(set(x)),
                    'milestone_confidence': 'mean',
                    'importance_score': 'sum'
                }).reset_index()
                df = pd.merge(df, fda_df, left_on='Date', right_on='date', how='left')
                df = df.drop(columns=['date'], errors='ignore')
            
            # Add advanced features
            df = self.add_advanced_features(df)
            
            # Clip outliers in target variable
            target_col = 'Forward_Return_5D'
            if target_col in df.columns:
                q1, q3 = df[target_col].quantile([0.01, 0.99])
                df[target_col] = df[target_col].clip(q1, q3)
            else:
                raise ValueError(f"Target column {target_col} not found in dataset")
            
            # Filter date range
            df = df[(df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)]
            
            # Fill missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('none')
            
            # Select features
            feature_cols = [
                col for col in df.columns 
                if col not in ['Date', 'date', 'Symbol', 'Company', 'Source', 
                              'Forward_Return_1D', 'Forward_Return_3D', 'Forward_Return_5D', 
                              'Forward_Return_10D', 'Forward_Direction_1D', 
                              'Forward_Direction_3D', 'Forward_Direction_5D', 
                              'Forward_Direction_10D']
            ]
            target_col = 'Forward_Return_5D'
            
            self.data = df
            self.feature_names = feature_cols if not self.feature_names else self.feature_names
            
            # Prepare X and y
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
            
            # Use saved features if available
            if self.feature_names:
                missing_features = [f for f in self.feature_names if f not in X.columns]
                if missing_features:
                    logger.warning(f"⚠️ Missing features from saved set: {missing_features}, adding zeros")
                    for f in missing_features:
                        X[f] = 0.0
                X = X[self.feature_names]
            else:
                # Enhanced feature selection
                corr_matrix = X.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
                X = X.drop(columns=to_drop)
                logger.info(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
                
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
                mi_df = pd.Series(mi_scores, index=X.columns).nlargest(50)
                X = X[mi_df.index]
                
                rf = RandomForestRegressor(random_state=self.random_state)
                rfe = RFE(estimator=rf, n_features_to_select=30)
                rfe.fit(X, y)
                self.feature_names = X.columns[rfe.support_].tolist()
                X = X[self.feature_names]
                logger.info(f"Selected Features: {self.feature_names}")
                
                # Save feature names
                with open(PATHS['models'] / 'feature_names.pkl', 'wb') as f:
                    pickle.dump(self.feature_names, f)
            
            # Split data
            train_end = pd.to_datetime('2021-12-31')
            val_end = pd.to_datetime('2023-12-31')
            train_idx = df['Date'] <= train_end
            val_idx = (df['Date'] > train_end) & (df['Date'] <= val_end)
            test_idx = df['Date'] > val_end
            
            self.X_train = X[train_idx]
            self.y_train = y[train_idx]
            self.X_val = X[val_idx]
            self.y_val = y[val_idx]
            self.X_test = X[test_idx]
            self.y_test = y[test_idx]
            
            # Check sample sizes
            if len(self.X_train) != len(self.y_train) or len(self.X_val) != len(self.y_val) or len(self.X_test) != len(self.y_test):
                raise ValueError(f"Inconsistent sample sizes: X_train={len(self.X_train)}, y_train={len(self.y_train)}, "
                               f"X_val={len(self.X_val)}, y_val={len(self.y_val)}, "
                               f"X_test={len(self.X_test)}, y_test={len(self.y_test)}")
            
            # Combine train and validation for final training
            self.X_train = pd.concat([self.X_train, self.X_val])
            self.y_train = pd.concat([self.y_train, self.y_val])
            
            # Verify combined sample sizes
            if len(self.X_train) != len(self.y_train):
                raise ValueError(f"Inconsistent combined sample sizes: X_train={len(self.X_train)}, y_train={len(self.y_train)}")
            
            # Scale features
            if 'X' not in self.scalers:
                self.scalers['X'] = StandardScaler()
                self.X_train = self.scalers['X'].fit_transform(self.X_train)
            else:
                self.X_train = self.scalers['X'].transform(self.X_train)
            self.X_test = self.scalers['X'].transform(self.X_test)
            if self.X_val.shape[0] > 0:
                self.X_val = self.scalers['X'].transform(self.X_val)
            
            # Scale target variable
            if 'y' not in self.scalers:
                self.scalers['y'] = MinMaxScaler(feature_range=(-1, 1))
                self.y_train = self.scalers['y'].fit_transform(self.y_train.values.reshape(-1, 1)).flatten()
            else:
                self.y_train = self.scalers['y'].transform(self.y_train.values.reshape(-1, 1)).flatten()
            self.y_test = self.scalers['y'].transform(self.y_test.values.reshape(-1, 1)).flatten()
            if self.y_val.shape[0] > 0:
                self.y_val = self.scalers['y'].transform(self.y_val.values.reshape(-1, 1)).flatten()
            
            # Save scalers
            with open(PATHS['models'] / 'scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"✅ Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} test samples, {len(self.X_val)} validation samples")
            logger.info(f"📈 Features selected: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {str(e)}")
            raise

    def add_advanced_features(self, df):
        """
        Add advanced features with leakage prevention
        """
        logger.info("🧠 Adding advanced features...")
        
        try:
            if 'sentiment_score' in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'sentiment_lag_{lag}'] = df['sentiment_score'].shift(lag)
                df['sentiment_3d_roll_mean'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()
                if 'Volatility_20' in df.columns:
                    df['sentiment_volatility_interaction'] = df['sentiment_score'] * df['Volatility_20']
            else:
                logger.warning("⚠️ 'sentiment_score' column missing, setting sentiment features to 0")
                for lag in [1, 2, 3, 5]:
                    df[f'sentiment_lag_{lag}'] = 0.0
                df['sentiment_3d_roll_mean'] = 0.0
                df['sentiment_volatility_interaction'] = 0.0
            
            if 'fda_milestone_type' in df.columns:
                for milestone in ['application_phase', 'clinical_trials', 'regulatory_review', 'approval_process', 'post_approval', 'regulatory_issues']:
                    df[f'has_{milestone}'] = df['fda_milestone_type'].str.contains(milestone, na=False).astype(int)
            else:
                logger.warning("⚠️ 'fda_milestone_type' column missing, setting milestone flags to 0")
                for milestone in ['application_phase', 'clinical_trials', 'regulatory_review', 'approval_process', 'post_approval', 'regulatory_issues']:
                    df[f'has_{milestone}'] = 0
            
            if 'news_headline' in df.columns:
                df['semglee_mention'] = df['news_headline'].str.lower().str.contains('semglee', na=False).astype(int)
            else:
                df['semglee_mention'] = 0
            
            if 'Volume' in df.columns and 'Volume_SMA_20' in df.columns:
                df['insider_signal'] = (df['Volume'] > df['Volume_SMA_20'] * 2).astype(int)
            else:
                df['insider_signal'] = 0
            
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
            df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
            df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding advanced features: {str(e)}")
            raise

    def train_linear_models(self):
        """
        Train linear regression models with regularization
        """
        logger.info("🧠 Training linear models...")
        
        try:
            self.models['linear_regression'] = LinearRegression().fit(self.X_train, self.y_train)
            logger.info("✅ Linear Regression trained")
            
            ridge = GridSearchCV(Ridge(random_state=self.random_state), MODEL_CONFIG['ridge_regression'], cv=TimeSeriesSplit(n_splits=3))
            ridge.fit(self.X_train, self.y_train)
            self.models['ridge_regression'] = ridge.best_estimator_
            logger.info(f"✅ Ridge Regression trained (best alpha: {ridge.best_params_['alpha']})")
            
            lasso = GridSearchCV(Lasso(random_state=self.random_state), MODEL_CONFIG['lasso_regression'], cv=TimeSeriesSplit(n_splits=3))
            lasso.fit(self.X_train, self.y_train)
            self.models['lasso_regression'] = lasso.best_estimator_
            logger.info(f"✅ Lasso Regression trained (best alpha: {lasso.best_params_['alpha']})")
            
        except Exception as e:
            logger.error(f"❌ Error training linear models: {str(e)}")
            raise

    def train_tree_models(self):
        """
        Train tree-based ensemble models
        """
        logger.info("🌳 Training tree-based models...")
        
        try:
            rf = GridSearchCV(RandomForestRegressor(random_state=self.random_state), MODEL_CONFIG['random_forest'], cv=TimeSeriesSplit(n_splits=3))
            rf.fit(self.X_train, self.y_train)
            self.models['random_forest'] = rf.best_estimator_
            logger.info(f"✅ Random Forest trained (best params: {rf.best_params_})")
            
            # Log feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Random Forest Feature Importance:\n{importance.head(10)}")
            
            gb = GridSearchCV(GradientBoostingRegressor(random_state=self.random_state), MODEL_CONFIG['gradient_boosting'], cv=TimeSeriesSplit(n_splits=3))
            gb.fit(self.X_train, self.y_train)
            self.models['gradient_boosting'] = gb.best_estimator_
            logger.info(f"✅ Gradient Boosting trained (best params: {gb.best_params_})")
            
            lgb_model = GridSearchCV(lgb.LGBMRegressor(random_state=self.random_state, force_col_wise=True), MODEL_CONFIG['lightgbm'], cv=TimeSeriesSplit(n_splits=3))
            lgb_model.fit(self.X_train, self.y_train)
            self.models['lightgbm'] = lgb_model.best_estimator_
            logger.info(f"✅ LightGBM trained (best params: {lgb_model.best_params_})")
            
            xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=self.random_state), MODEL_CONFIG['xgboost'], cv=TimeSeriesSplit(n_splits=3))
            xgb_model.fit(self.X_train, self.y_train)
            self.models['xgboost'] = xgb_model.best_estimator_
            logger.info(f"✅ XGBoost trained (best params: {xgb_model.best_params_})")
            
        except Exception as e:
            logger.error(f"❌ Error training tree models: {str(e)}")
            raise

    def train_svr(self):
        """
        Train Support Vector Regression model
        """
        logger.info("🧠 Training SVR model...")
        
        try:
            svr = GridSearchCV(SVR(), MODEL_CONFIG['svr'], cv=TimeSeriesSplit(n_splits=3))
            svr.fit(self.X_train, self.y_train)
            self.models['svr'] = svr.best_estimator_
            logger.info(f"✅ SVR trained (best params: {svr.best_params_})")
            
        except Exception as e:
            logger.error(f"❌ Error training SVR model: {str(e)}")
            raise

    def train_lstm(self):
        """
        Train LSTM model with early stopping
        """
        logger.info("🧠 Training LSTM model...")
        
        try:
            lstm_params = MODEL_CONFIG['lstm']
            X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
            model = Sequential([
                LSTM(lstm_params['units'][0], activation='relu', input_shape=(1, self.X_train.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(lstm_params['units'][1], activation='relu'),
                Dropout(0.2),
                Dense(lstm_params['units'][1] // 2, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(
                X_train_reshaped, self.y_train,
                epochs=lstm_params['epochs'],
                batch_size=lstm_params['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            self.models['lstm'] = model
            logger.info("✅ LSTM model trained")
            
        except Exception as e:
            logger.error(f"❌ Error training LSTM model: {str(e)}")
            raise

    def train_bert_sentiment(self):
        """
        Placeholder for BERT sentiment model
        """
        logger.info("🧠 Note: BERT sentiment model assumes precomputed scores in daily_sentiment.csv")

    def create_ensemble(self):
        """
        Create stacking ensemble with fallback to weighted average
        """
        logger.info("🤝 Creating ensemble model...")
        
        try:
            estimators = [
                (name, model) for name, model in self.models.items() if name != 'lstm' and name != 'ensemble'
            ]
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(random_state=self.random_state),
                cv=TimeSeriesSplit(n_splits=3)  # Reduced splits
            )
            stacking.fit(self.X_train, self.y_train)
            self.models['ensemble'] = stacking
            logger.info("✅ Stacking ensemble model created")
            
        except Exception as e:
            logger.warning(f"⚠️ Stacking ensemble failed: {str(e)}, falling back to weighted average")
            try:
                predictions = {}
                for name, model in self.models.items():
                    if name == 'lstm':
                        X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
                        predictions[name] = model.predict(X_train_reshaped, verbose=0).flatten()
                    else:
                        predictions[name] = model.predict(self.X_train)
                
                weights = {
                    'linear_regression': 0.05,
                    'ridge_regression': 0.15,
                    'lasso_regression': 0.15,
                    'random_forest': 0.2,
                    'gradient_boosting': 0.2,
                    'lightgbm': 0.15,
                    'xgboost': 0.15,
                    'svr': 0.1,
                    'lstm': 0.1
                }
                
                ensemble_predictions = np.zeros_like(predictions['ridge_regression'])
                for name, pred in predictions.items():
                    ensemble_predictions += weights[name] * pred
                
                self.models['ensemble'] = {'predictions': ensemble_predictions, 'weights': weights}
                logger.info("✅ Fallback weighted ensemble created")
                
            except Exception as e:
                logger.error(f"❌ Error creating fallback ensemble: {str(e)}")
                raise

    def evaluate_models(self):
        """
        Evaluate all models and generate performance metrics
        """
        logger.info("📈 Evaluating model performance...")
        
        try:
            performance_metrics = {}
            predictions_df = pd.DataFrame()
            y_test_orig = self.scalers['y'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
            if self.y_val.shape[0] > 0:
                y_val_orig = self.scalers['y'].inverse_transform(self.y_val.reshape(-1, 1)).flatten()
            
            for name, model in self.models.items():
                if name == 'ensemble' and isinstance(model, dict):
                    y_pred = np.zeros_like(self.y_test)
                    for m_name, m in self.models.items():
                        if m_name != 'ensemble':
                            weight = model['weights'].get(m_name, 0)
                            if m_name == 'lstm':
                                X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
                                y_pred += weight * m.predict(X_test_reshaped, verbose=0).flatten()
                            else:
                                y_pred += weight * m.predict(self.X_test)
                else:
                    if name == 'lstm':
                        X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
                        y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
                    else:
                        y_pred = model.predict(self.X_test)
                
                y_pred = self.scalers['y'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                mse = mean_squared_error(y_test_orig, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_orig, y_pred)
                r2 = r2_score(y_test_orig, y_pred)
                
                performance_metrics[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
                
                # Validation set evaluation
                if self.y_val.shape[0] > 0:
                    if name == 'ensemble' and isinstance(model, dict):
                        y_pred_val = np.zeros_like(self.y_val)
                        for m_name, m in self.models.items():
                            if m_name != 'ensemble':
                                weight = model['weights'].get(m_name, 0)
                                if m_name == 'lstm':
                                    X_val_reshaped = self.X_val.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
                                    y_pred_val += weight * m.predict(X_val_reshaped, verbose=0).flatten()
                                else:
                                    y_pred_val += weight * m.predict(self.X_val)
                    else:
                        if name == 'lstm':
                            X_val_reshaped = self.X_val.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
                            y_pred_val = model.predict(X_val_reshaped, verbose=0).flatten()
                        else:
                            y_pred_val = model.predict(self.X_val)
                    
                    y_pred_val = self.scalers['y'].inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
                    performance_metrics[name]['Val_R2'] = r2_score(y_val_orig, y_pred_val)
                
                predictions_df[name] = y_pred
            
            predictions_df['Actual'] = y_test_orig
            predictions_df['Date'] = self.data.iloc[-len(self.y_test):]['Date']
            predictions_df.to_csv(PATHS['results'] / 'predictions.csv')
            
            plt.figure(figsize=(12, 6))
            for name in predictions_df.columns:
                if name != 'Date':
                    plt.plot(predictions_df['Date'], predictions_df[name], label=name)
            plt.title('Model Predictions vs Actual (Original Scale)')
            plt.xlabel('Date')
            plt.ylabel('Forward 5-Day Return')
            plt.legend()
            plt.savefig(PATHS['results'] / 'prediction_vs_actual.png')
            plt.close()
            
            logger.info("✅ Model evaluation completed")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating models: {str(e)}")
            raise

    def save_models(self):
        """
        Save all trained models and metadata
        """
        logger.info("💾 Saving models and metadata...")
        
        try:
            for name, model in self.models.items():
                if name != 'ensemble' or not isinstance(model, dict):
                    with open(PATHS['models'] / f"{name}_model.pkl", 'wb') as f:
                        pickle.dump(model, f)
            
            if 'ensemble' in self.models and isinstance(self.models['ensemble'], dict):
                with open(PATHS['models'] / 'ensemble_model.pkl', 'wb') as f:
                    pickle.dump(self.models['ensemble']['weights'], f)
            
            metadata = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': len(self.feature_names),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'val_samples': len(self.X_val),
                'models_trained': list(self.models.keys())
            }
            with open(PATHS['models'] / 'model_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("✅ Models and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {str(e)}")
            raise

    def print_comprehensive_summary(self, performance_metrics):
        """
        Print detailed training summary
        """
        print("\n" + "="*80)
        print("🚀 ADVANCED MODEL TRAINING SUMMARY")
        print("="*80)
        
        print(f"📊 Dataset Overview:")
        print(f"   Training Samples: {len(self.X_train):,}")
        print(f"   Validation Samples: {len(self.X_val):,}")
        print(f"   Test Samples: {len(self.X_test):,}")
        print(f"   Features Used: {len(self.feature_names):,}")
        print(f"   Date Range: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        
        print(f"\n🧠 Models Trained:")
        for model_name in self.models.keys():
            print(f"   ✅ {model_name}")
        
        print(f"\n📈 Model Performance (Original Scale):")
        for model_name, metrics in performance_metrics.items():
            print(f"   {model_name}:")
            print(f"      MSE: {metrics['MSE']:.6f}")
            print(f"      RMSE: {metrics['RMSE']:.6f}")
            print(f"      MAE: {metrics['MAE']:.6f}")
            print(f"      R2: {metrics['R2']:.4f}")
            if 'Val_R2' in metrics:
                print(f"      Validation R2: {metrics['Val_R2']:.4f}")
        
        print(f"\n📁 Output Files:")
        print(f"   ✅ models/*_model.pkl - Trained model files")
        print(f"   ✅ models/feature_names.pkl - Feature names")
        print(f"   ✅ models/scalers.pkl - Data scalers")
        print(f"   ✅ models/model_metadata.pkl - Model metadata")
        print(f"   ✅ results/predictions.csv - Model predictions")
        print(f"   ✅ results/prediction_vs_actual.png - Visualization")
        
        print(f"\n🎉 Ready for Day 2 Step 4: Model Testing")
        print("="*80)

    def execute(self):
        """
        Execute the complete model training pipeline
        """
        try:
            logger.info("🚀 Starting Advanced Model Training Pipeline...")
            
            self.load_and_prepare_data()
            self.train_linear_models()
            self.train_tree_models()
            self.train_svr()
            self.train_lstm()
            self.train_bert_sentiment()
            self.create_ensemble()
            performance_metrics = self.evaluate_models()
            self.save_models()
            self.print_comprehensive_summary(performance_metrics)
            
            logger.info("🎉 Model training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            return False

def main():
    """
    Main execution function for Day 2 - Step 3
    """
    print("🚀 BIOCON FDA PROJECT - DAY 2 STEP 3")
    print("Advanced Model Training for Stock Price Prediction")
    print("="*60)
    print(f"🏢 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"📅 Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"🎯 Models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, LightGBM, XGBoost, SVR, LSTM, Ensemble")
    print("-" * 60)
    
    trainer = AdvancedModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 SUCCESS: Advanced model training completed!")
        print("✅ Multiple models trained and evaluated")
        print("✅ Ensemble model created")
        print("✅ Performance metrics and visualizations generated")
        print("✅ Models saved to: models/")
        print("🔄 Ready for Day 2 Step 4: Model Testing")
    else:
        print("\n❌ FAILED: Model training failed")
        print("💡 Check logs for details: logs/model_training.log")
        print("🔧 Troubleshooting: Verify data files and dependencies")
    
    return success

if __name__ == "__main__":
    main()