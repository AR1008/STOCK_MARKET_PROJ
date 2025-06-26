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
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioconModelTrainer:
    """
    Day 2: Corrected Model Training for Biocon Stock Prediction
    Based on actual data structure from Day 1 collection scripts
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = []
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'results/charts']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_and_validate_data(self):
        """Load and validate the data files from Day 1"""
        logger.info("Loading and validating data from Day 1...")
        
        try:
            # Check if required files exist
            required_files = {
                'stock_data.csv': 'data/stock_data.csv',
                'daily_sentiment.csv': 'data/daily_sentiment.csv'
            }
            
            for file_name, file_path in required_files.items():
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_name} not found at {file_path}. Please run Day 1 data collection first.")
            
            # Load stock data
            stock_df = pd.read_csv('data/stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            logger.info(f"✓ Loaded stock data: {len(stock_df)} records")
            
            # Load daily sentiment data
            sentiment_df = pd.read_csv('data/daily_sentiment.csv')
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
            logger.info(f"✓ Loaded sentiment data: {len(sentiment_df)} records")
            
            # Display available columns for debugging
            logger.info(f"Stock data columns: {list(stock_df.columns)}")
            logger.info(f"Sentiment data columns: {list(sentiment_df.columns)}")
            
            return stock_df, sentiment_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def combine_and_clean_data(self, stock_df, sentiment_df):
        """Combine stock and sentiment data with proper cleaning"""
        logger.info("Combining and cleaning datasets...")
        
        try:
            # Merge datasets on date
            combined_df = pd.merge(
                stock_df, 
                sentiment_df, 
                left_on='Date', 
                right_on='date', 
                how='left'
            )
            
            # Drop duplicate date column
            if 'date' in combined_df.columns:
                combined_df = combined_df.drop(['date'], axis=1)
            
            logger.info(f"Combined dataset: {len(combined_df)} records")
            
            # Fill missing sentiment values based on actual column structure
            # From your news collection script, these are the actual columns created:
            sentiment_columns = [
                'avg_sentiment', 'sentiment_std', 'news_count',
                'weighted_avg_sentiment', 'weighted_sentiment_std', 
                'avg_priority', 'max_priority', 'total_priority',
                'drug_specific_count', 'company_news_count',
                'drug_news_ratio', 'day_importance_score'
            ]
            
            for col in sentiment_columns:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(0)
                    logger.info(f"✓ Filled missing values for {col}")
            
            # Handle FDA milestone flags (has_approval_process, has_regulatory_review, etc.)
            milestone_columns = [col for col in combined_df.columns if col.startswith('has_')]
            for col in milestone_columns:
                combined_df[col] = combined_df[col].fillna(0).astype(int)
                logger.info(f"✓ Processed FDA milestone flag: {col}")
            
            logger.info(f"Data cleaning completed. Final shape: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise
    
    def engineer_comprehensive_features(self, df):
        """Create comprehensive features based on your actual data structure"""
        logger.info("Engineering comprehensive features...")
        
        try:
            # Sort by date to ensure proper time series order
            df = df.sort_values('Date').copy()
            df = df.reset_index(drop=True)
            
            # === PRICE-BASED FEATURES ===
            if 'Close' in df.columns:
                # Price momentum features
                for days in [1, 3, 5, 10, 20]:
                    df[f'Price_Return_{days}D'] = df['Close'].pct_change(days)
                
                # Price moving averages and ratios
                for window in [5, 10, 20, 50]:
                    if len(df) > window:
                        df[f'Price_MA_{window}'] = df['Close'].rolling(window=window).mean()
                        df[f'Price_Ratio_MA_{window}'] = df['Close'] / (df[f'Price_MA_{window}'] + 0.0001)
                
                # Price volatility
                for window in [5, 10, 20]:
                    if len(df) > window:
                        df[f'Price_Volatility_{window}D'] = df['Close'].pct_change().rolling(window=window).std()
            
            # === EXISTING TECHNICAL INDICATORS ===
            # Use technical indicators already calculated in Day 1
            existing_technical = ['RSI', 'MACD', 'MACD_Signal', 'MA_20', 'MA_50', 'MA_200', 
                                'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volatility_20D']
            
            for indicator in existing_technical:
                if indicator in df.columns:
                    # Create normalized versions
                    rolling_mean = df[indicator].rolling(window=60, min_periods=10).mean()
                    rolling_std = df[indicator].rolling(window=60, min_periods=10).std()
                    df[f'{indicator}_normalized'] = (df[indicator] - rolling_mean) / (rolling_std + 0.0001)
            
            # === VOLUME FEATURES ===
            if 'Volume' in df.columns:
                # Volume moving averages and ratios
                for window in [5, 10, 20]:
                    if len(df) > window:
                        df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                        df[f'Volume_Ratio_{window}'] = df['Volume'] / (df[f'Volume_MA_{window}'] + 1)
                
                # Volume surge detection
                if 'Volume_MA_20' in df.columns:
                    df['Volume_Surge'] = (df['Volume'] > 2 * df['Volume_MA_20']).astype(int)
            
            # === SENTIMENT FEATURES ===
            if 'avg_sentiment' in df.columns:
                # Sentiment moving averages
                for window in [3, 7, 14, 21]:
                    if len(df) > window:
                        df[f'Sentiment_MA_{window}'] = df['avg_sentiment'].rolling(window=window).mean()
                
                # Sentiment momentum and changes
                for days in [1, 3, 7]:
                    df[f'Sentiment_Change_{days}D'] = df['avg_sentiment'].diff(days)
                
                # Sentiment volatility
                df['Sentiment_Volatility_7D'] = df['avg_sentiment'].rolling(window=7).std()
                
                # Sentiment extreme flags
                sentiment_q75 = df['avg_sentiment'].quantile(0.75)
                sentiment_q25 = df['avg_sentiment'].quantile(0.25)
                df['High_Sentiment'] = (df['avg_sentiment'] > sentiment_q75).astype(int)
                df['Low_Sentiment'] = (df['avg_sentiment'] < sentiment_q25).astype(int)
            
            # === WEIGHTED SENTIMENT FEATURES ===
            if 'weighted_avg_sentiment' in df.columns:
                # Weighted sentiment moving averages
                for window in [3, 7, 14]:
                    if len(df) > window:
                        df[f'Weighted_Sentiment_MA_{window}'] = df['weighted_avg_sentiment'].rolling(window=window).mean()
                
                # Weighted sentiment changes
                for days in [1, 3, 7]:
                    df[f'Weighted_Sentiment_Change_{days}D'] = df['weighted_avg_sentiment'].diff(days)
            
            # === NEWS VOLUME FEATURES ===
            if 'news_count' in df.columns:
                # News volume moving averages
                for window in [3, 7, 14]:
                    if len(df) > window:
                        df[f'News_Volume_MA_{window}'] = df['news_count'].rolling(window=window).mean()
                
                # News volume changes and flags
                df['News_Volume_Change'] = df['news_count'].diff()
                news_q80 = df['news_count'].quantile(0.8)
                df['High_News_Day'] = (df['news_count'] > news_q80).astype(int)
            
            # === DRUG-SPECIFIC NEWS FEATURES ===
            if 'drug_specific_count' in df.columns:
                # Drug news moving averages
                for window in [3, 7, 14]:
                    if len(df) > window:
                        df[f'Drug_News_MA_{window}'] = df['drug_specific_count'].rolling(window=window).mean()
                
                # Drug news ratio and flags
                df['Drug_News_Change'] = df['drug_specific_count'].diff()
                df['Drug_News_Day'] = (df['drug_specific_count'] > 0).astype(int)
            
            # === FDA MILESTONE FEATURES ===
            if 'day_importance_score' in df.columns:
                # Importance score features
                for window in [3, 7, 14]:
                    if len(df) > window:
                        df[f'Importance_MA_{window}'] = df['day_importance_score'].rolling(window=window).mean()
                
                # FDA event flags
                df['High_Importance_Day'] = (df['day_importance_score'] > 20).astype(int)
                df['Medium_Importance_Day'] = ((df['day_importance_score'] > 10) & (df['day_importance_score'] <= 20)).astype(int)
                df['FDA_Event_Week'] = df['day_importance_score'].rolling(window=7).max()
            
            # === INTERACTION FEATURES ===
            # Sentiment-Price interactions
            if 'avg_sentiment' in df.columns and 'Close' in df.columns:
                df['Sentiment_Price_Interaction'] = df['avg_sentiment'] * df['Price_Return_1D']
            
            # Volume-Sentiment interactions
            if 'Volume' in df.columns and 'avg_sentiment' in df.columns:
                df['Volume_Sentiment_Interaction'] = df['Volume_Ratio_5'] * df['avg_sentiment']
            
            # === LAG FEATURES ===
            # Important lag features based on your data
            lag_features = []
            if 'Close' in df.columns:
                lag_features.append('Close')
            if 'avg_sentiment' in df.columns:
                lag_features.append('avg_sentiment')
            if 'weighted_avg_sentiment' in df.columns:
                lag_features.append('weighted_avg_sentiment')
            if 'news_count' in df.columns:
                lag_features.append('news_count')
            if 'day_importance_score' in df.columns:
                lag_features.append('day_importance_score')
            
            for feature in lag_features:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
            
            # === DATE FEATURES ===
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Monday'] = (df['Date'].dt.dayofweek == 0).astype(int)
            df['Is_Friday'] = (df['Date'].dt.dayofweek == 4).astype(int)
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
            
            # === MARKET CONTEXT FEATURES ===
            # Use benchmark data if available
            if 'NIFTY50_Return' in df.columns and 'Daily_Return' in df.columns:
                df['Market_Relative_Performance'] = df['Daily_Return'] - df['NIFTY50_Return']
                # Rolling beta calculation
                window = 60
                if len(df) > window:
                    df['Market_Beta'] = df['Daily_Return'].rolling(window=window).cov(df['NIFTY50_Return']) / (df['NIFTY50_Return'].rolling(window=window).var() + 0.0001)
            
            # === TARGET VARIABLES ===
            # Create multiple target options
            if 'Close' in df.columns:
                # Next day return (main target)
                df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
                # Next day close price
                df['Next_Day_Close'] = df['Close'].shift(-1)
                # Future returns
                df['Next_3Day_Return'] = df['Close'].pct_change(3).shift(-3)
                df['Next_5Day_Return'] = df['Close'].pct_change(5).shift(-5)
            elif 'Daily_Return' in df.columns:
                # Fallback to daily return
                df['Next_Day_Return'] = df['Daily_Return'].shift(-1)
            
            logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
            
            # Log feature categories
            feature_categories = {
                'Price': [col for col in df.columns if 'Price' in col or 'Close' in col],
                'Technical': [col for col in df.columns if any(tech in col for tech in ['RSI', 'MACD', 'MA_', 'BB_'])],
                'Volume': [col for col in df.columns if 'Volume' in col],
                'Sentiment': [col for col in df.columns if 'Sentiment' in col],
                'News': [col for col in df.columns if 'News' in col or 'Drug' in col],
                'FDA': [col for col in df.columns if 'Importance' in col or col.startswith('has_')],
                'Date': [col for col in df.columns if any(date_feat in col for date_feat in ['Day_', 'Month', 'Quarter', 'Is_'])],
                'Lag': [col for col in df.columns if 'lag' in col]
            }
            
            for category, features in feature_categories.items():
                if features:
                    logger.info(f"{category} features ({len(features)}): {features[:3]}...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def prepare_model_data(self, df):
        """Prepare data for model training with proper validation"""
        logger.info("Preparing data for model training...")
        
        try:
            # Select target variable
            target_candidates = ['Next_Day_Return', 'Daily_Return']
            target_column = None
            
            for candidate in target_candidates:
                if candidate in df.columns:
                    valid_targets = df[candidate].dropna()
                    if len(valid_targets) > 100:  # Need sufficient data
                        target_column = candidate
                        break
            
            if target_column is None:
                raise ValueError("No suitable target variable found with sufficient data")
            
            logger.info(f"Using target variable: {target_column}")
            
            # Remove rows where target is NaN
            df_clean = df.dropna(subset=[target_column]).copy()
            logger.info(f"Clean dataset after removing NaN targets: {len(df_clean)} records")
            
            if len(df_clean) < 100:
                raise ValueError(f"Insufficient clean data: {len(df_clean)} records")
            
            # Define columns to exclude from features
            exclude_columns = {
                'Date', 'date', 'datetime',  # Date columns
                'Next_Day_Return', 'Next_Day_Close', 'Next_3Day_Return', 'Next_5Day_Return',  # Target variables
                'Symbol', 'Company', 'Type', 'Source',  # Metadata
                'fda_milestones', 'market_impacts',  # Text columns
            }
            
            # Select feature columns
            all_columns = set(df_clean.columns)
            feature_columns = list(all_columns - exclude_columns)
            
            # Remove any columns that are all NaN or have too many NaN values
            final_features = []
            for col in feature_columns:
                if col in df_clean.columns:
                    non_nan_count = df_clean[col].count()
                    if non_nan_count > len(df_clean) * 0.5:  # At least 50% non-NaN
                        final_features.append(col)
                    else:
                        logger.warning(f"Excluding {col} - too many NaN values ({non_nan_count}/{len(df_clean)})")
            
            logger.info(f"Selected {len(final_features)} features for training")
            
            # Prepare feature matrix X
            X = df_clean[final_features].copy()
            
            # Fill remaining NaN values
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col] = X[col].fillna(median_val)
                else:
                    # Fill non-numeric columns with mode or 0
                    mode_vals = X[col].mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else 0
                    X[col] = X[col].fillna(fill_val)
            
            # Prepare target vector y
            y = df_clean[target_column].values
            
            # Final validation
            assert len(X) == len(y), "Feature matrix and target vector length mismatch"
            assert not np.any(pd.isna(y)), "Target vector contains NaN values"
            assert not X.isnull().any().any(), "Feature matrix contains NaN values"
            
            logger.info(f"Model data prepared successfully:")
            logger.info(f"  Samples: {len(X)}")
            logger.info(f"  Features: {len(X.columns)}")
            logger.info(f"  Target: {target_column}")
            logger.info(f"  Target range: {y.min():.4f} to {y.max():.4f}")
            
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            return X, y, target_column
            
        except Exception as e:
            logger.error(f"Error preparing model data: {str(e)}")
            raise
    
    def create_train_val_test_splits(self, X, y):
        """Create time-aware train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        try:
            # Time series split (important for financial data)
            n_samples = len(X)
            
            # Use 60% for training, 20% for validation, 20% for testing
            train_end = int(n_samples * 0.6)
            val_end = int(n_samples * 0.8)
            
            X_train = X.iloc[:train_end].copy()
            X_val = X.iloc[train_end:val_end].copy()
            X_test = X.iloc[val_end:].copy()
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Data splits created:")
            logger.info(f"  Training: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
            logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error creating splits: {str(e)}")
            raise
    
    def train_ml_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train multiple machine learning models"""
        logger.info("Training machine learning models...")
        
        # Prepare scaled data for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Define models to train
        models_config = {
            'Linear_Regression': {
                'model': LinearRegression(),
                'use_scaled': True
            },
            'Ridge_Regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'use_scaled': True
            },
            'Lasso_Regression': {
                'model': Lasso(alpha=0.1, random_state=42, max_iter=2000),
                'use_scaled': True
            },
            'Random_Forest': {
                'model': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'use_scaled': False
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'use_scaled': False
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'use_scaled': True
            }
        }
        
        # Train each model
        for model_name, config in models_config.items():
            try:
                logger.info(f"Training {model_name}...")
                
                model = config['model']
                use_scaled = config['use_scaled']
                
                # Select appropriate data
                if use_scaled:
                    X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
                else:
                    X_tr, X_v, X_te = X_train.values, X_val.values, X_test.values
                
                # Train model
                model.fit(X_tr, y_train)
                
                # Make predictions
                y_val_pred = model.predict(X_v)
                y_test_pred = model.predict(X_te)
                
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
                    'val_rmse': np.sqrt(val_mse),
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_r2': test_r2,
                    'test_rmse': np.sqrt(test_mse),
                    'use_scaled': use_scaled
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
                
                logger.info(f"✓ {model_name} - Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
    
    def select_best_model(self):
        """Select the best performing model"""
        logger.info("Selecting best model...")
        
        if not self.performance_metrics:
            logger.error("No models trained successfully")
            return
        
        # Select best model based on validation R² score
        best_r2 = -float('inf')
        best_model_name = None
        
        for model_name, metrics in self.performance_metrics.items():
            val_r2 = metrics['val_r2']
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name} (Validation R²: {best_r2:.4f})")
    
    def save_models_and_results(self):
        """Save trained models and results"""
        logger.info("Saving models and results...")
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                model_path = f'models/{model_name.lower()}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"✓ Saved {model_name} to {model_path}")
            
            # Save the best model as final_model.pkl
            if self.best_model is not None:
                with open('models/final_model.pkl', 'wb') as f:
                    pickle.dump(self.best_model, f)
                logger.info(f"✓ Saved best model ({self.best_model_name}) as final_model.pkl")
            
            # Save scalers
            with open('models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            logger.info("✓ Saved scalers")
            
            # Save feature names
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            logger.info("✓ Saved feature names")
            
            # Save model metadata
            model_metadata = {
                'best_model_name': self.best_model_name,
                'n_features': len(self.feature_names),
                'training_date': datetime.now().isoformat()
            }
            with open('models/model_metadata.pkl', 'wb') as f:
                pickle.dump(model_metadata, f)
            
            # Save performance metrics
            performance_df = pd.DataFrame(self.performance_metrics).T
            performance_df.to_csv('results/model_performance.csv')
            logger.info("✓ Saved model performance metrics")
            
            # Save feature importance
            if self.feature_importance:
                importance_data = []
                for model_name, importances in self.feature_importance.items():
                    for feature, importance in importances.items():
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': importance
                        })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df.to_csv('results/feature_importance.csv', index=False)
                logger.info("✓ Saved feature importance")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*80)
        print("BIOCON MODEL TRAINING SUMMARY (DAY 2)")
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
        
        # Sort models by validation R²
        sorted_models = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1]['val_r2'],
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} {metrics['val_rmse']:<12.4f} {metrics['val_r2']:<10.4f} "
                  f"{metrics['test_rmse']:<12.4f} {metrics['test_r2']:<10.4f}")
        
        # Show top features for best model
        if self.best_model_name in self.feature_importance:
            print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES ({self.best_model_name}):")
            importance = self.feature_importance[self.best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"  {i:2d}. {feature:<30} {score:.4f}")
        
        print(f"\n📁 FILES CREATED:")
        print(f"  ✓ models/final_model.pkl           - Best performing model")
        print(f"  ✓ models/scalers.pkl              - Data preprocessing scalers")
        print(f"  ✓ models/feature_names.pkl        - Feature names list")
        print(f"  ✓ models/model_metadata.pkl       - Training metadata")
        print(f"  ✓ results/model_performance.csv   - All model metrics")
        print(f"  ✓ results/feature_importance.csv  - Feature importance scores")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  📈 Day 3: Run model testing and backtesting")
        print(f"  🔮 Day 4: Generate future predictions")
        print(f"  📊 Analysis: Review feature importance and model performance")
        
        print("="*80)
    
    def execute(self):
        """Execute the complete Day 2 training pipeline"""
        try:
            logger.info("="*60)
            logger.info("STARTING DAY 2: BIOCON MODEL TRAINING")
            logger.info("="*60)
            
            # Step 1: Load and validate data
            stock_df, sentiment_df = self.load_and_validate_data()
            
            # Step 2: Combine and clean data
            combined_df = self.combine_and_clean_data(stock_df, sentiment_df)
            
            # Step 3: Engineer comprehensive features
            df_with_features = self.engineer_comprehensive_features(combined_df)
            
            # Step 4: Prepare model data
            X, y, target_column = self.prepare_model_data(df_with_features)
            
            # Step 5: Create train/val/test splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_splits(X, y)
            
            # Step 6: Train ML models
            self.train_ml_models(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 7: Select best model
            self.select_best_model()
            
            # Step 8: Save models and results
            success = self.save_models_and_results()
            
            # Step 9: Print summary
            self.print_training_summary()
            
            if success:
                logger.info("Day 2 model training completed successfully!")
                return True
            else:
                logger.error("Day 2 model training completed with errors!")
                return False
                
        except Exception as e:
            logger.error(f"Day 2 training pipeline failed: {str(e)}")
            print(f"\n💥 ERROR: {str(e)}")
            print("\n🔧 TROUBLESHOOTING:")
            print("  1. Ensure Day 1 data collection completed successfully")
            print("  2. Check that data/stock_data.csv and data/daily_sentiment.csv exist")
            print("  3. Verify data files have sufficient records (>100 rows)")
            print("  4. Check logs for detailed error information")
            return False

def main():
    """Main execution function for Day 2"""
    print("🚀 BIOCON FDA PROJECT - DAY 2: MODEL TRAINING")
    print("Combining stock data with news sentiment for ML prediction models")
    print("Training multiple algorithms and selecting the best performer")
    print("-" * 70)
    
    trainer = BioconModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 DAY 2 SUCCESS!")
        print("✅ Multiple ML models trained and evaluated")
        print("✅ Best model selected based on validation performance")
        print("✅ Models and results saved for Day 3 testing")
        print("\n📋 READY FOR DAY 3:")
        print("  • Run: python code/4_test_model.py")
        print("  • This will perform backtesting and generate predictions")
    else:
        print("\n💥 DAY 2 FAILED!")
        print("❌ Check the error messages above")
        print("❌ Ensure Day 1 data collection completed first")
        print("\n🔧 TROUBLESHOOTING:")
        print("  • Run: python code/1_collect_stock_data.py")
        print("  • Run: python code/2_collect_news_data.py")
        print("  • Check data/ folder contains CSV files")
    
    return success

if __name__ == "__main__":
    main()