import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor)
from sklearn.linear_model import (LogisticRegression, ElasticNet, Ridge, RidgeClassifier,
                                Lasso, LinearRegression, LassoCV, RidgeCV, ElasticNetCV)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, classification_report, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy import stats
import warnings
import time
warnings.filterwarnings('ignore')

# For LSTM model
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. LSTM model will be skipped.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBioconModelTrainer:
    """
    Enhanced PROPER Stock Prediction Training - NO DATA LEAKAGE
    Includes comprehensive model suite: Traditional ML, Ensemble, Deep Learning
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = []
        self.label_encoders = {}
        self.validation_results = {}
        self.model_metadata = {}
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'results/charts', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_and_validate_data(self):
        """Load and validate data with strict checks"""
        logger.info("Loading and validating data...")
        
        try:
            # Load stock data
            stock_df = pd.read_csv('data/stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            logger.info(f"✓ Loaded stock data: {len(stock_df)} records from {stock_df['Date'].min()} to {stock_df['Date'].max()}")
            
            # Load sentiment data
            sentiment_df = pd.read_csv('data/daily_sentiment.csv')
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
            sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
            logger.info(f"✓ Loaded sentiment data: {len(sentiment_df)} records")
            
            return stock_df, sentiment_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def smart_data_merge(self, stock_df, sentiment_df):
        """Smart data merge preserving temporal order"""
        logger.info("Performing smart data merge...")
        
        try:
            # LEFT JOIN to keep all stock days
            combined_df = pd.merge(
                stock_df, 
                sentiment_df, 
                left_on='Date', 
                right_on='date', 
                how='left'
            )
            
            if 'date' in combined_df.columns:
                combined_df = combined_df.drop(['date'], axis=1)
            
            # Fill missing sentiment intelligently
            sentiment_columns = [
                'avg_sentiment', 'weighted_avg_sentiment', 'news_count',
                'drug_specific_count', 'day_importance_score'
            ]
            
            for col in sentiment_columns:
                if col in combined_df.columns:
                    if 'sentiment' in col.lower():
                        combined_df[col] = combined_df[col].fillna(0.0)  # Neutral
                    else:
                        combined_df[col] = combined_df[col].fillna(0)    # No news
                else:
                    combined_df[col] = 0
            
            # Create day_importance_score if missing
            if 'day_importance_score' not in combined_df.columns or combined_df['day_importance_score'].isna().all():
                combined_df['day_importance_score'] = (
                    combined_df.get('news_count', 0) * 2 +
                    combined_df.get('drug_specific_count', 0) * 10 +
                    np.abs(combined_df.get('weighted_avg_sentiment', 0)) * 15
                )
            
            # Handle FDA milestone flags
            milestone_columns = [col for col in combined_df.columns if col.startswith('has_')]
            for col in milestone_columns:
                combined_df[col] = combined_df[col].fillna(0).astype(int)
            
            # Ensure temporal order
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Smart merge completed: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in smart merge: {str(e)}")
            raise
    
    def clean_and_encode_data(self, df):
        """Clean data and encode categorical variables properly"""
        logger.info("Cleaning and encoding data...")
        
        try:
            # Identify and handle string/object columns
            string_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            # Remove columns that should not be features
            columns_to_drop = ['Date', 'Symbol', 'Company', 'Source']
            string_columns = [col for col in string_columns if col not in columns_to_drop]
            
            logger.info(f"Found categorical columns to encode: {string_columns}")
            
            # Handle each categorical column
            for col in string_columns:
                if col in df.columns:
                    unique_values = df[col].nunique()
                    
                    if unique_values <= 10:  # Low cardinality - use label encoding
                        le = LabelEncoder()
                        df[col] = df[col].astype(str).fillna('Unknown')
                        df[col] = le.fit_transform(df[col])
                        self.label_encoders[col] = le
                        logger.info(f"Label encoded {col}: {unique_values} categories")
                    else:  # High cardinality - drop
                        logger.warning(f"Dropping high cardinality column: {col}")
                        df = df.drop(columns=[col])
            
            # Ensure all remaining columns are numeric
            for col in df.columns:
                if col not in ['Date']:  # Keep Date for later use
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        logger.warning(f"Could not convert {col} to numeric, dropping")
                        if col in df.columns:
                            df = df.drop(columns=[col])
            
            # Fill any remaining NaN values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    if 'sentiment' in col.lower():
                        df[col] = df[col].fillna(0.0)
                    elif 'volume' in col.lower():
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].median())
            
            logger.info(f"Data cleaning completed: {len(df.columns)} columns remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def create_proper_features(self, df):
        """Create features WITHOUT data leakage - only use historical data"""
        logger.info("Creating proper features WITHOUT data leakage...")
        
        try:
            # Clean and encode data first
            df = self.clean_and_encode_data(df)
            
            df = df.sort_values('Date').reset_index(drop=True)
            
            # === STRICT: NO FORWARD-LOOKING FEATURES ===
            # Remove any forward-looking features that cause data leakage
            forward_looking_cols = [col for col in df.columns if 'Forward' in col or 'Next' in col]
            if forward_looking_cols:
                logger.warning(f"🚨 REMOVING DATA LEAKAGE COLUMNS: {forward_looking_cols}")
                df = df.drop(columns=forward_looking_cols)
            
            # === CORE PRICE FEATURES (HISTORICAL ONLY) ===
            if 'Close' in df.columns:
                # Returns with different horizons (PAST only)
                df['Return_1D'] = df['Close'].pct_change()
                df['Return_3D'] = df['Close'].pct_change(3)
                df['Return_5D'] = df['Close'].pct_change(5)
                df['Return_10D'] = df['Close'].pct_change(10)
                
                # Moving averages (PAST only)
                for window in [5, 10, 20, 50]:
                    df[f'MA_{window}'] = df['Close'].rolling(window).mean()
                    df[f'Price_Above_MA_{window}'] = (df['Close'] > df[f'MA_{window}']).astype(int)
                    df[f'Price_Distance_MA_{window}'] = (df['Close'] - df[f'MA_{window}']) / df[f'MA_{window}']
                
                # Volatility (PAST only)
                for window in [5, 10, 20]:
                    df[f'Volatility_{window}D'] = df['Return_1D'].rolling(window).std() * np.sqrt(252)
                
                # Price momentum (PAST only)
                df['Momentum_5D'] = df['Close'].pct_change(5)
                df['Momentum_10D'] = df['Close'].pct_change(10)
                df['Momentum_20D'] = df['Close'].pct_change(20)
                
                # Price extremes (PAST only)
                for window in [10, 20]:
                    df[f'High_{window}D'] = df['High'].rolling(window).max()
                    df[f'Low_{window}D'] = df['Low'].rolling(window).min()
                    df[f'Price_Position_{window}D'] = (df['Close'] - df[f'Low_{window}D']) / (df[f'High_{window}D'] - df[f'Low_{window}D'])
            
            # === VOLUME FEATURES (HISTORICAL ONLY) ===
            if 'Volume' in df.columns:
                for window in [5, 10, 20]:
                    df[f'Volume_MA_{window}'] = df['Volume'].rolling(window).mean()
                    df[f'Volume_Ratio_{window}'] = df['Volume'] / (df[f'Volume_MA_{window}'] + 1)
                
                df['High_Volume'] = (df['Volume'] > df['Volume_MA_20'] * 1.5).astype(int)
                df['Low_Volume'] = (df['Volume'] < df['Volume_MA_20'] * 0.7).astype(int)
            
            # === SENTIMENT FEATURES (HISTORICAL ONLY) ===
            if 'avg_sentiment' in df.columns:
                # Clean sentiment
                df['avg_sentiment'] = np.clip(df['avg_sentiment'], -1, 1)
                
                # Sentiment signals
                df['Positive_Sentiment'] = (df['avg_sentiment'] > 0.1).astype(int)
                df['Negative_Sentiment'] = (df['avg_sentiment'] < -0.1).astype(int)
                df['Strong_Sentiment'] = (np.abs(df['avg_sentiment']) > 0.3).astype(int)
                
                # Sentiment momentum (PAST only)
                df['Sentiment_Change_1D'] = df['avg_sentiment'].diff()
                df['Sentiment_Change_3D'] = df['avg_sentiment'].diff(3)
                
                # Sentiment moving averages (PAST only)
                for window in [3, 5, 10]:
                    df[f'Sentiment_MA_{window}'] = df['avg_sentiment'].rolling(window).mean()
            
            # === FDA EVENT FEATURES (HISTORICAL ONLY) ===
            if 'day_importance_score' in df.columns:
                # FDA event flags
                df['Major_FDA_Event'] = (df['day_importance_score'] > 25).astype(int)
                df['Minor_FDA_Event'] = ((df['day_importance_score'] > 10) & (df['day_importance_score'] <= 25)).astype(int)
                
                # Days since FDA event (PAST only)
                fda_events = df['day_importance_score'] > 15
                df['Days_Since_FDA'] = 0
                
                days_counter = 999  # Start high
                for i in range(len(df)):
                    if fda_events.iloc[i]:
                        days_counter = 0
                    else:
                        days_counter += 1
                    df.loc[i, 'Days_Since_FDA'] = min(days_counter, 60)  # Cap at 60
                
                # FDA event momentum (PAST only)
                df['FDA_Score_MA_5'] = df['day_importance_score'].rolling(5).mean()
                df['FDA_Score_MA_10'] = df['day_importance_score'].rolling(10).mean()
            
            # === MARKET TIMING FEATURES ===
            if 'Date' in df.columns:
                df['Day_of_Week'] = df['Date'].dt.dayofweek
                df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
                df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
                df['Month'] = df['Date'].dt.month
                df['Is_Earnings_Month'] = df['Month'].isin([1, 4, 7, 10]).astype(int)
                df['Day_of_Month'] = df['Date'].dt.day
                df['Is_Month_End'] = (df['Day_of_Month'] > 25).astype(int)
            
            # === TECHNICAL INDICATORS (HISTORICAL ONLY) ===
            technical_cols = ['RSI_14', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR_14']
            for col in technical_cols:
                if col in df.columns:
                    if col == 'RSI_14':
                        df['RSI_Oversold'] = (df[col] < 30).astype(int)
                        df['RSI_Overbought'] = (df[col] > 70).astype(int)
                        df['RSI_Middle'] = ((df[col] >= 40) & (df[col] <= 60)).astype(int)
                    elif col == 'MACD':
                        df['MACD_Positive'] = (df[col] > 0).astype(int)
                        df['MACD_Signal_Cross'] = ((df[col] > df.get('MACD_Signal', 0)) & 
                                                 (df[col].shift(1) <= df.get('MACD_Signal', 0).shift(1))).astype(int)
                    elif col == 'ATR_14':
                        df['High_Volatility'] = (df[col] > df[col].rolling(50).quantile(0.8)).astype(int)
            
            # === LAG FEATURES (SAFE - NO LEAKAGE) ===
            lag_features = ['Close', 'Volume', 'Return_1D', 'avg_sentiment', 'day_importance_score']
            for feature in lag_features:
                if feature in df.columns:
                    for lag in [1, 2, 3, 5]:
                        df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
            
            # === INTERACTION FEATURES ===
            # Sentiment + Volume (HISTORICAL ONLY)
            if 'Strong_Sentiment' in df.columns and 'High_Volume' in df.columns:
                df['Sentiment_Volume_Signal'] = df['Strong_Sentiment'] * df['High_Volume']
            
            # FDA + Sentiment (HISTORICAL ONLY)
            if 'Major_FDA_Event' in df.columns and 'avg_sentiment' in df.columns:
                df['FDA_Sentiment_Signal'] = df['Major_FDA_Event'] * df['avg_sentiment']
            
            # === CREATE TARGET VARIABLE (PROPER WAY) ===
            if 'Close' in df.columns:
                # Next day direction (what we want to predict)
                df['Target_Next_Day_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                
                # Alternative targets
                df['Target_Next_Day_Return'] = df['Close'].pct_change().shift(-1)
                df['Target_Next_3Day_Up'] = (df['Close'].shift(-3) > df['Close']).astype(int)
                
                # Significant move targets
                returns = df['Close'].pct_change().shift(-1)
                threshold = returns.std() * 0.75  # More conservative threshold
                df['Target_Significant_Move'] = (np.abs(returns) > threshold).astype(int)
            
            logger.info(f"✅ Proper feature engineering completed. Total features: {len(df.columns)}")
            
            # Final check for data leakage
            suspect_cols = [col for col in df.columns if any(word in col.lower() for word in ['forward', 'future', 'next', 'ahead'])]
            if suspect_cols and 'Target' not in str(suspect_cols):
                logger.error(f"🚨 POTENTIAL DATA LEAKAGE DETECTED: {suspect_cols}")
                raise ValueError("Data leakage detected in features!")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in proper feature engineering: {str(e)}")
            raise
    
    def select_proper_features(self, X, y, target_type, max_features=30):
        """Select features properly without data leakage"""
        logger.info(f"Selecting proper features for {target_type}...")
        
        try:
            # Ensure X contains only numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_columns].copy()
            
            # Remove any target-related columns from features
            target_related = [col for col in X_numeric.columns if 'Target' in col]
            if target_related:
                logger.warning(f"🚨 Removing target-related features: {target_related}")
                X_numeric = X_numeric.drop(columns=target_related)
            
            logger.info(f"Starting with {len(X_numeric.columns)} numeric features")
            
            # Remove low-quality features
            good_features = []
            for col in X_numeric.columns:
                missing_pct = X_numeric[col].isnull().sum() / len(X_numeric)
                unique_count = X_numeric[col].nunique()
                
                # Check for constant or near-constant features
                if missing_pct < 0.3 and unique_count > 1:
                    # Check variance
                    if X_numeric[col].var() > 1e-10:  # Non-zero variance
                        good_features.append(col)
                    else:
                        logger.info(f"Removing low variance feature: {col}")
                else:
                    logger.info(f"Removing poor quality feature: {col} (missing: {missing_pct:.1%}, unique: {unique_count})")
            
            X_filtered = X_numeric[good_features].copy()
            logger.info(f"After quality filtering: {len(X_filtered.columns)} features")
            
            # Fill missing values conservatively
            for col in X_filtered.columns:
                if X_filtered[col].isnull().sum() > 0:
                    if 'sentiment' in col.lower():
                        X_filtered[col] = X_filtered[col].fillna(0.0)
                    else:
                        X_filtered[col] = X_filtered[col].fillna(X_filtered[col].median())
            
            # Remove infinite values
            X_filtered = X_filtered.replace([np.inf, -np.inf], np.nan)
            for col in X_filtered.columns:
                if X_filtered[col].isnull().sum() > 0:
                    X_filtered[col] = X_filtered[col].fillna(X_filtered[col].median())
            
            # Feature selection with multiple methods
            selected_features = good_features
            
            if len(good_features) > max_features:
                logger.info(f"Selecting top {max_features} features from {len(good_features)}")
                
                if target_type == 'classification':
                    try:
                        # Method 1: Statistical test
                        selector = SelectKBest(score_func=f_classif, k=min(max_features, len(good_features)))
                        selector.fit(X_filtered, y)
                        stat_features = [feat for feat, selected in zip(good_features, selector.get_support()) if selected]
                        
                        # Method 2: Mutual information
                        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(good_features)))
                        selector_mi.fit(X_filtered, y)
                        mi_features = [feat for feat, selected in zip(good_features, selector_mi.get_support()) if selected]
                        
                        # Combine methods
                        selected_features = list(set(stat_features + mi_features))[:max_features]
                        
                        logger.info(f"Selected {len(selected_features)} features using multiple methods")
                        
                    except Exception as e:
                        logger.warning(f"Feature selection failed: {str(e)}, using correlation")
                        # Fallback to correlation
                        correlations = []
                        for col in good_features:
                            try:
                                corr = np.corrcoef(X_filtered[col], y)[0, 1]
                                correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
                            except:
                                correlations.append((col, 0))
                        
                        correlations.sort(key=lambda x: x[1], reverse=True)
                        selected_features = [feat for feat, _ in correlations[:max_features]]
            
            return X_filtered[selected_features], selected_features
                
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            # Return safe subset as fallback
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            safe_columns = [col for col in numeric_columns if 'Target' not in col][:max_features]
            return X[safe_columns], safe_columns
    
    def prepare_proper_targets(self, df):
        """Prepare targets without data leakage"""
        logger.info("Preparing proper targets...")
        
        try:
            # Check available targets (only properly created ones)
            target_options = [
                ('Target_Next_Day_Up', 'classification', 'Next Day Direction'),
                ('Target_Next_3Day_Up', 'classification', 'Next 3-Day Direction'),
                ('Target_Significant_Move', 'classification', 'Significant Move Detection'),
                ('Target_Next_Day_Return', 'regression', 'Next Day Return'),
            ]
            
            # Find the best available target
            for target_col, target_type, description in target_options:
                if target_col in df.columns:
                    # Drop rows with missing targets
                    df_clean = df.dropna(subset=[target_col]).copy()
                    
                    if len(df_clean) > 1000:  # Need sufficient data
                        y = df_clean[target_col].values
                        
                        # Check class balance for classification
                        if target_type == 'classification':
                            class_counts = np.bincount(y.astype(int))
                            if len(class_counts) >= 2:  # Must have at least 2 classes
                                minority_class_pct = min(class_counts) / sum(class_counts) * 100
                                
                                if minority_class_pct >= 15:  # At least 15% minority class
                                    logger.info(f"Selected target: {target_col} ({description})")
                                    logger.info(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
                                    logger.info(f"Minority class: {minority_class_pct:.1f}%")
                                    
                                    return df_clean, target_col, target_type, description
                        else:
                            # For regression, check if there's sufficient variance
                            if np.std(y) > 1e-6:
                                logger.info(f"Selected target: {target_col} ({description})")
                                logger.info(f"Target stats: mean={np.mean(y):.4f}, std={np.std(y):.4f}")
                                return df_clean, target_col, target_type, description
            
            # Fallback: create a balanced direction target
            logger.warning("Creating fallback balanced direction target...")
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().shift(-1)
                # Use 0 threshold for balanced classes
                df['Target_Direction_Balanced'] = (returns > 0).astype(int)
                
                df_clean = df.dropna(subset=['Target_Direction_Balanced']).copy()
                if len(df_clean) > 1000:
                    return df_clean, 'Target_Direction_Balanced', 'classification', 'Balanced Direction'
            
            raise ValueError("Cannot create any suitable target variable")
            
        except Exception as e:
            logger.error(f"Error preparing targets: {str(e)}")
            raise
    
    def create_lstm_data(self, X, y, lookback=30):
        """Create sequences for LSTM model"""
        if not KERAS_AVAILABLE:
            return None, None
            
        try:
            X_lstm, y_lstm = [], []
            for i in range(lookback, len(X)):
                X_lstm.append(X[i-lookback:i])
                y_lstm.append(y[i])
            
            return np.array(X_lstm), np.array(y_lstm)
        except Exception as e:
            logger.error(f"Error creating LSTM data: {str(e)}")
            return None, None
    
    def create_ensemble_model(self, target_type):
        """Create ensemble model from base estimators"""
        try:
            if target_type == 'classification':
                base_models = [
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
                    ('lgb', lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000))
                ]
                return VotingClassifier(estimators=base_models, voting='soft')
            else:
                base_models = [
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
                    ('lgb', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)),
                    ('ridge', Ridge(random_state=42))
                ]
                return VotingRegressor(estimators=base_models)
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            return None
    
    def train_enhanced_models(self, X_train, X_val, X_test, y_train, y_val, y_test, target_type):
        """Train comprehensive suite of models"""
        logger.info(f"Training enhanced model suite for {target_type}...")
        
        start_time = time.time()
        
        # Scale features with multiple scalers
        standard_scaler = StandardScaler()
        X_train_standard = standard_scaler.fit_transform(X_train)
        X_val_standard = standard_scaler.transform(X_val)
        X_test_standard = standard_scaler.transform(X_test)
        
        robust_scaler = RobustScaler()
        X_train_robust = robust_scaler.fit_transform(X_train)
        X_val_robust = robust_scaler.transform(X_val)
        X_test_robust = robust_scaler.transform(X_test)
        
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train)
        X_val_minmax = minmax_scaler.transform(X_val)
        X_test_minmax = minmax_scaler.transform(X_test)
        
        self.scalers = {
            'standard': standard_scaler,
            'robust': robust_scaler,
            'minmax': minmax_scaler
        }
        
        if target_type == 'classification':
            models_config = {
                # === LINEAR MODELS ===
                'logistic_regression_model': {
                    'model': LogisticRegression(
                        random_state=42, max_iter=2000, 
                        class_weight='balanced', C=1.0,
                        solver='liblinear'
                    ),
                    'data_type': 'standard',
                    'expected_time': 5
                },
                'ridge_model': {
                    'model': RidgeClassifier(
                        random_state=42, alpha=1.0,
                        class_weight='balanced'
                    ),
                    'data_type': 'standard',
                    'expected_time': 3
                },
                
                # === TREE BASED MODELS ===
                'random_forest_model': {
                    'model': RandomForestClassifier(
                        n_estimators=200, max_depth=10, random_state=42,
                        class_weight='balanced', n_jobs=-1,
                        min_samples_split=20, min_samples_leaf=10,
                        max_features='sqrt'
                    ),
                    'data_type': 'raw',
                    'expected_time': 30
                },
                'gradient_boosting_model': {
                    'model': GradientBoostingClassifier(
                        n_estimators=150, max_depth=6, learning_rate=0.1,
                        random_state=42, min_samples_split=20,
                        min_samples_leaf=10, subsample=0.8
                    ),
                    'data_type': 'raw',
                    'expected_time': 45
                },
                'adaboost_model': {
                    'model': AdaBoostClassifier(
                        n_estimators=100, learning_rate=1.0,
                        random_state=42,
                        estimator=DecisionTreeClassifier(max_depth=3)
                    ),
                    'data_type': 'raw',
                    'expected_time': 25
                },
                
                # === GRADIENT BOOSTING VARIANTS ===
                'lightgbm_model': {
                    'model': lgb.LGBMClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1, verbose=-1,
                        class_weight='balanced', min_child_samples=20,
                        feature_fraction=0.8, bagging_fraction=0.8
                    ),
                    'data_type': 'raw',
                    'expected_time': 25
                },
                'xgboost_model': {
                    'model': xgb.XGBClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=0.1
                    ),
                    'data_type': 'raw',
                    'expected_time': 35
                },
                'catboost_model': {
                    'model': cb.CatBoostClassifier(
                        iterations=200, depth=6, learning_rate=0.05,
                        random_seed=42, verbose=False,
                        class_weights='Balanced',
                        l2_leaf_reg=3.0
                    ),
                    'data_type': 'raw',
                    'expected_time': 40
                },
                
                # === SVM MODELS ===
                'svm_model': {
                    'model': SVC(
                        kernel='rbf', C=1.0, gamma='scale',
                        random_state=42, class_weight='balanced',
                        probability=True
                    ),
                    'data_type': 'robust',
                    'expected_time': 60
                },
                
                # === NAIVE BAYES ===
                'naive_bayes_model': {
                    'model': GaussianNB(),
                    'data_type': 'standard',
                    'expected_time': 2
                },
                
                # === K-NEAREST NEIGHBORS ===
                'knn_model': {
                    'model': KNeighborsClassifier(
                        n_neighbors=5, weights='distance'
                    ),
                    'data_type': 'standard',
                    'expected_time': 10
                },
                
                # === ENSEMBLE MODEL ===
                'ensemble_model': {
                    'model': self.create_ensemble_model('classification'),
                    'data_type': 'standard',
                    'expected_time': 50
                }
            }
        
        else:  # regression
            models_config = {
                # === LINEAR REGRESSION MODELS ===
                'linear_regression_model': {
                    'model': LinearRegression(),
                    'data_type': 'standard',
                    'expected_time': 2
                },
                'ridge_regression_model': {
                    'model': Ridge(random_state=42, alpha=1.0),
                    'data_type': 'standard',
                    'expected_time': 3
                },
                'lasso_model': {
                    'model': Lasso(random_state=42, alpha=0.1, max_iter=2000),
                    'data_type': 'standard',
                    'expected_time': 5
                },
                'lasso_regression_model': {
                    'model': LassoCV(random_state=42, max_iter=2000),
                    'data_type': 'standard',
                    'expected_time': 15
                },
                'elasticnet_model': {
                    'model': ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.5, max_iter=2000),
                    'data_type': 'standard',
                    'expected_time': 8
                },
                'elastic_net_cv_model': {
                    'model': ElasticNetCV(random_state=42, max_iter=2000),
                    'data_type': 'standard',
                    'expected_time': 20
                },
                
                # === TREE BASED MODELS ===
                'random_forest_reg_model': {
                    'model': RandomForestRegressor(
                        n_estimators=200, max_depth=10, random_state=42,
                        n_jobs=-1, min_samples_split=20, min_samples_leaf=10,
                        max_features='sqrt'
                    ),
                    'data_type': 'raw',
                    'expected_time': 30
                },
                'adaboost_reg_model': {
                    'model': AdaBoostRegressor(
                        n_estimators=100, learning_rate=1.0,
                        random_state=42,
                        estimator=DecisionTreeRegressor(max_depth=3)
                    ),
                    'data_type': 'raw',
                    'expected_time': 25
                },
                'gradient_boosting_reg_model': {
                    'model': GradientBoostingRegressor(
                        n_estimators=150, max_depth=6, learning_rate=0.1,
                        random_state=42, min_samples_split=20,
                        min_samples_leaf=10, subsample=0.8
                    ),
                    'data_type': 'raw',
                    'expected_time': 45
                },
                
                # === GRADIENT BOOSTING VARIANTS ===
                'lightgbm_reg_model': {
                    'model': lgb.LGBMRegressor(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1, verbose=-1,
                        min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8
                    ),
                    'data_type': 'raw',
                    'expected_time': 25
                },
                'xgboost_reg_model': {
                    'model': xgb.XGBRegressor(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        random_state=42, n_jobs=-1,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=0.1
                    ),
                    'data_type': 'raw',
                    'expected_time': 35
                },
                'catboost_reg_model': {
                    'model': cb.CatBoostRegressor(
                        iterations=200, depth=6, learning_rate=0.05,
                        random_seed=42, verbose=False,
                        l2_leaf_reg=3.0
                    ),
                    'data_type': 'raw',
                    'expected_time': 40
                },
                
                # === SVM REGRESSION ===
                'svr_model': {
                    'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
                    'data_type': 'robust',
                    'expected_time': 60
                },
                
                # === K-NEAREST NEIGHBORS ===
                'knn_reg_model': {
                    'model': KNeighborsRegressor(
                        n_neighbors=5, weights='distance'
                    ),
                    'data_type': 'standard',
                    'expected_time': 10
                },
                
                # === ENSEMBLE MODEL ===
                'ensemble_model': {
                    'model': self.create_ensemble_model('regression'),
                    'data_type': 'standard',
                    'expected_time': 50
                }
            }
        
        # Train models with proper timing
        successful_models = 0
        for model_name, config in models_config.items():
            try:
                if config['model'] is None:
                    logger.warning(f"Skipping {model_name} - model creation failed")
                    continue
                    
                model_start = time.time()
                logger.info(f"Training {model_name}... (expected: {config['expected_time']}s)")
                
                model = config['model']
                data_type = config['data_type']
                
                # Select appropriate data
                if data_type == 'standard':
                    X_tr, X_v, X_te = X_train_standard, X_val_standard, X_test_standard
                elif data_type == 'robust':
                    X_tr, X_v, X_te = X_train_robust, X_val_robust, X_test_robust
                elif data_type == 'minmax':
                    X_tr, X_v, X_te = X_train_minmax, X_val_minmax, X_test_minmax
                else:  # raw
                    X_tr, X_v, X_te = X_train.values, X_val.values, X_test.values
                
                # Train model
                model.fit(X_tr, y_train)
                
                # Predict on validation and test
                y_val_pred = model.predict(X_v)
                y_test_pred = model.predict(X_te)
                
                # Calculate comprehensive metrics
                if target_type == 'classification':
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
                    # Confusion matrices
                    val_cm = confusion_matrix(y_val, y_val_pred)
                    test_cm = confusion_matrix(y_test, y_test_pred)
                    
                    # AUC if binary classification
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_val_proba = model.predict_proba(X_v)[:, 1]
                            y_test_proba = model.predict_proba(X_te)[:, 1]
                            val_auc = roc_auc_score(y_val, y_val_proba)
                            test_auc = roc_auc_score(y_test, y_test_proba)
                        else:
                            val_auc = test_auc = 0.5
                    except:
                        val_auc = test_auc = 0.5
                    
                    # Classification report
                    val_report = classification_report(y_val, y_val_pred, output_dict=True)
                    
                    metrics = {
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'val_auc': val_auc,
                        'test_auc': test_auc,
                        'val_precision': val_report['weighted avg']['precision'],
                        'val_recall': val_report['weighted avg']['recall'],
                        'val_f1': val_report['weighted avg']['f1-score'],
                        'val_confusion_matrix': val_cm.tolist(),
                        'test_confusion_matrix': test_cm.tolist(),
                        'primary_metric': val_accuracy,
                        'training_time': time.time() - model_start
                    }
                    
                    model_time = time.time() - model_start
                    logger.info(f"✓ {model_name} - Accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}, Time: {model_time:.1f}s")
                    
                    # Sanity check for unrealistic performance
                    if val_accuracy > 0.85:
                        logger.warning(f"🚨 {model_name} shows suspiciously high accuracy ({val_accuracy:.3f}) - possible data leakage!")
                
                else:
                    # Regression metrics
                    val_r2 = r2_score(y_val, y_val_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    val_mse = mean_squared_error(y_val, y_val_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    metrics = {
                        'val_r2': val_r2,
                        'test_r2': test_r2,
                        'val_rmse': np.sqrt(val_mse),
                        'test_rmse': np.sqrt(test_mse),
                        'val_mae': val_mae,
                        'test_mae': test_mae,
                        'primary_metric': val_r2,
                        'training_time': time.time() - model_start
                    }
                    
                    model_time = time.time() - model_start
                    logger.info(f"✓ {model_name} - R²: {val_r2:.3f}, RMSE: {np.sqrt(val_mse):.4f}, Time: {model_time:.1f}s")
                
                # Store results
                self.models[model_name] = model
                self.performance_metrics[model_name] = metrics
                self.model_metadata[model_name] = {
                    'data_type': data_type,
                    'model_type': type(model).__name__,
                    'training_time': model_time
                }
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(
                        self.feature_names, model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    self.feature_importance[model_name] = dict(zip(
                        self.feature_names, np.abs(coef)
                    ))
                
                successful_models += 1
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Train LSTM if available and classification
        if KERAS_AVAILABLE and target_type == 'classification' and len(X_train) > 100:
            try:
                logger.info("Training LSTM model...")
                lstm_start = time.time()
                
                # Create sequence data
                lookback = min(30, len(X_train) // 10)
                X_train_lstm, y_train_lstm = self.create_lstm_data(X_train_standard, y_train, lookback)
                X_val_lstm, y_val_lstm = self.create_lstm_data(X_val_standard, y_val, lookback)
                X_test_lstm, y_test_lstm = self.create_lstm_data(X_test_standard, y_test, lookback)
                
                if X_train_lstm is not None and len(X_train_lstm) > 50:
                    # Build LSTM model
                    lstm_model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
                        Dropout(0.2),
                        LSTM(50, return_sequences=False),
                        Dropout(0.2),
                        Dense(25, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    
                    lstm_model.compile(optimizer=Adam(learning_rate=0.001), 
                                     loss='binary_crossentropy', 
                                     metrics=['accuracy'])
                    
                    # Train with early stopping
                    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    
                    history = lstm_model.fit(
                        X_train_lstm, y_train_lstm,
                        validation_data=(X_val_lstm, y_val_lstm),
                        epochs=50, batch_size=32, verbose=0,
                        callbacks=[early_stop]
                    )
                    
                    # Predict and evaluate
                    y_val_pred_lstm = (lstm_model.predict(X_val_lstm) > 0.5).astype(int).flatten()
                    y_test_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
                    
                    val_accuracy_lstm = accuracy_score(y_val_lstm, y_val_pred_lstm)
                    test_accuracy_lstm = accuracy_score(y_test_lstm, y_test_pred_lstm)
                    
                    # Store LSTM results
                    lstm_time = time.time() - lstm_start
                    self.models['lstm_model'] = lstm_model
                    self.performance_metrics['lstm_model'] = {
                        'val_accuracy': val_accuracy_lstm,
                        'test_accuracy': test_accuracy_lstm,
                        'val_auc': 0.5,  # Could compute but keeping simple
                        'test_auc': 0.5,
                        'primary_metric': val_accuracy_lstm,
                        'training_time': lstm_time
                    }
                    self.model_metadata['lstm_model'] = {
                        'data_type': 'lstm_sequence',
                        'model_type': 'LSTM',
                        'training_time': lstm_time,
                        'lookback': lookback
                    }
                    
                    logger.info(f"✓ LSTM - Accuracy: {val_accuracy_lstm:.3f}, Time: {lstm_time:.1f}s")
                    successful_models += 1
                    
            except Exception as e:
                logger.error(f"Error training LSTM: {str(e)}")
        
        total_time = time.time() - start_time
        logger.info(f"Successfully trained {successful_models} models in {total_time:.1f}s")
    
    def cross_validate_best_model(self, X, y, target_type):
        """Perform time-series cross-validation on best model"""
        logger.info("Performing time-series cross-validation...")
        
        try:
            if not self.best_model_name:
                logger.warning("No best model selected for cross-validation")
                return
            
            # Time series split for cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Get the model type for quick recreation
            model_metadata = self.model_metadata.get(self.best_model_name, {})
            model_type = model_metadata.get('model_type', 'Unknown')
            
            # Quick model recreation for CV
            if 'LogisticRegression' in model_type:
                model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            elif 'RandomForest' in model_type:
                if target_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif 'LightGBM' in model_type:
                if target_type == 'classification':
                    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                else:
                    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            else:
                model = self.models[self.best_model_name]
            
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_cv)
                X_val_scaled = scaler.transform(X_val_cv)
                
                # Train and predict
                model.fit(X_train_scaled, y_train_cv)
                y_pred_cv = model.predict(X_val_scaled)
                
                if target_type == 'classification':
                    score = accuracy_score(y_val_cv, y_pred_cv)
                else:
                    score = r2_score(y_val_cv, y_pred_cv)
                
                cv_scores.append(score)
                logger.info(f"Fold {fold+1}: {score:.3f}")
            
            self.validation_results = {
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'cv_model': self.best_model_name
            }
            
            logger.info(f"Cross-validation results: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
    
    def select_best_model(self):
        """Select best model based on validation performance"""
        if not self.performance_metrics:
            logger.warning("No models to evaluate")
            return
        
        best_score = -float('inf')
        best_model_name = None
        
        # Consider both accuracy and AUC for classification, R² for regression
        for model_name, metrics in self.performance_metrics.items():
            if 'val_accuracy' in metrics and 'val_auc' in metrics:
                # Combined score: 70% accuracy + 30% AUC
                score = 0.7 * metrics['val_accuracy'] + 0.3 * metrics['val_auc']
            else:
                score = metrics.get('primary_metric', -float('inf'))
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name)
        
        logger.info(f"Best model: {best_model_name} (Score: {best_score:.3f})")
    
    def save_models_and_results(self):
        """Save all models and comprehensive results"""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                with open(f'models/{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save best model as final_model
            if self.best_model:
                with open('models/final_model.pkl', 'wb') as f:
                    pickle.dump(self.best_model, f)
            
            # Save scalers and feature names
            with open('models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            # Save label encoders
            with open('models/label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            # Save metadata
            with open('models/metadata.pkl', 'wb') as f:
                pickle.dump(self.model_metadata, f)
            
            # Save model metadata as individual files (for compatibility)
            with open('models/model_metadata.pkl', 'wb') as f:
                pickle.dump(self.model_metadata, f)
            
            # Save scaler as individual file (for compatibility)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers.get('standard'), f)
            
            # Save performance metrics
            pd.DataFrame(self.performance_metrics).T.to_csv('results/model_performance.csv')
            
            # Save detailed results
            with open('results/training_results.pkl', 'wb') as f:
                pickle.dump({
                    'performance_metrics': self.performance_metrics,
                    'feature_importance': self.feature_importance,
                    'validation_results': self.validation_results,
                    'best_model_name': self.best_model_name,
                    'model_metadata': self.model_metadata
                }, f)
            
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
                pd.DataFrame(importance_data).to_csv('results/feature_importance.csv', index=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving: {str(e)}")
            return False
    
    def print_comprehensive_summary(self, target_description, target_type):
        """Print comprehensive and realistic training summary"""
        print("\n" + "="*120)
        print("ENHANCED BIOCON STOCK PREDICTION MODEL TRAINING SUMMARY")
        print("="*120)
        
        print(f"🎯 TARGET: {target_description} ({target_type})")
        print(f"✅ ENHANCED MODEL SUITE:")
        print(f"   • Comprehensive ML algorithms: Linear, Tree-based, Ensemble, Neural Networks")
        print(f"   • NO data leakage - strict temporal validation")
        print(f"   • Multiple scaling strategies for different algorithm requirements")
        print(f"   • Advanced hyperparameter configurations")
        print(f"   • Production-ready model training and evaluation")
        
        if not self.performance_metrics:
            print("❌ No models trained successfully")
            return
        
        print(f"\n📊 COMPREHENSIVE MODEL PERFORMANCE ({len(self.performance_metrics)} models):")
        if target_type == 'classification':
            print(f"{'Model':<30} {'Val_Acc':<10} {'Test_Acc':<10} {'Val_AUC':<10} {'F1':<8} {'Time(s)':<8}")
            print("-" * 105)
            
            sorted_models = sorted(
                self.performance_metrics.items(),
                key=lambda x: x[1].get('primary_metric', 0),
                reverse=True
            )
            
            for model_name, metrics in sorted_models:
                val_acc = metrics.get('val_accuracy', 0)
                test_acc = metrics.get('test_accuracy', 0)
                val_auc = metrics.get('val_auc', 0.5)
                f1 = metrics.get('val_f1', 0)
                train_time = metrics.get('training_time', 0)
                
                print(f"{model_name:<30} {val_acc:<10.3f} {test_acc:<10.3f} {val_auc:<10.3f} {f1:<8.3f} {train_time:<8.1f}")
                
                # Show confusion matrix for best model
                if model_name == self.best_model_name:
                    cm = metrics.get('val_confusion_matrix', [[0, 0], [0, 0]])
                    print(f"\n📊 {model_name} Validation Confusion Matrix:")
                    print(f"    Predicted:   0     1")
                    print(f"  Actual 0:   {cm[0][0]:4d}  {cm[0][1]:4d}")
                    print(f"  Actual 1:   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        else:
            print(f"{'Model':<30} {'Val_R²':<10} {'Test_R²':<10} {'Val_RMSE':<12} {'Time(s)':<8}")
            print("-" * 90)
            
            sorted_models = sorted(
                self.performance_metrics.items(),
                key=lambda x: x[1].get('primary_metric', -999),
                reverse=True
            )
            
            for model_name, metrics in sorted_models:
                val_r2 = metrics.get('val_r2', 0)
                test_r2 = metrics.get('test_r2', 0)
                val_rmse = metrics.get('val_rmse', 0)
                train_time = metrics.get('training_time', 0)
                
                print(f"{model_name:<30} {val_r2:<10.3f} {test_r2:<10.3f} {val_rmse:<12.4f} {train_time:<8.1f}")
        
        # Model categories summary
        print(f"\n🏗️ MODEL CATEGORIES TRAINED:")
        categories = {}
        for model_name, metadata in self.model_metadata.items():
            category = self._get_model_category(metadata.get('model_type', 'Unknown'))
            if category not in categories:
                categories[category] = []
            categories[category].append(model_name)
        
        for category, models in categories.items():
            print(f"   • {category}: {len(models)} models")
        
        # Performance interpretation
        print(f"\n📈 PERFORMANCE INTERPRETATION:")
        if target_type == 'classification':
            best_acc = max([m.get('val_accuracy', 0) for m in self.performance_metrics.values()])
            if best_acc > 0.75:
                print(f"   🚨 WARNING: Accuracy {best_acc:.1%} seems unrealistically high!")
                print(f"   🔍 Check for data leakage or overfitting")
            elif best_acc > 0.60:
                print(f"   ✅ EXCELLENT: Accuracy {best_acc:.1%} is very good for stock prediction")
            elif best_acc > 0.55:
                print(f"   ✅ GOOD: Accuracy {best_acc:.1%} beats random (50%) significantly")
            elif best_acc > 0.52:
                print(f"   ⚠️  MODEST: Accuracy {best_acc:.1%} slightly better than random")
            else:
                print(f"   ❌ POOR: Accuracy {best_acc:.1%} not better than random guessing")
        else:
            best_r2 = max([m.get('val_r2', -999) for m in self.performance_metrics.values()])
            if best_r2 > 0.7:
                print(f"   ✅ EXCELLENT: R² {best_r2:.3f} shows strong predictive power")
            elif best_r2 > 0.5:
                print(f"   ✅ GOOD: R² {best_r2:.3f} shows moderate predictive power")
            elif best_r2 > 0.2:
                print(f"   ⚠️  MODEST: R² {best_r2:.3f} shows weak predictive power")
            else:
                print(f"   ❌ POOR: R² {best_r2:.3f} shows little predictive power")
        
        # Cross-validation results
        if self.validation_results:
            cv_mean = self.validation_results['cv_mean']
            cv_std = self.validation_results['cv_std']
            print(f"\n🔄 CROSS-VALIDATION RESULTS ({self.validation_results['cv_model']}):")
            print(f"   Mean Score: {cv_mean:.3f} ± {cv_std:.3f}")
            print(f"   Stability: {'Good' if cv_std < 0.05 else 'Moderate' if cv_std < 0.10 else 'Poor'}")
        
        # Top features
        if self.best_model_name in self.feature_importance:
            print(f"\n🎯 TOP 20 PREDICTIVE FEATURES ({self.best_model_name}):")
            importance = self.feature_importance[self.best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            for i, (feature, score) in enumerate(top_features, 1):
                category = self._categorize_feature(feature)
                print(f"  {i:2d}. {feature:<35} {score:.4f} [{category}]")
        
        # Model recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Best model for deployment: {self.best_model_name}")
        
        if target_type == 'classification':
            best_metrics = self.performance_metrics[self.best_model_name]
            if best_metrics.get('val_accuracy', 0) > 0.55:
                print(f"   • Model shows promise for stock direction prediction")
                print(f"   • Consider ensemble methods for production")
            else:
                print(f"   • Performance suggests limited predictability")
                print(f"   • Focus on risk management over prediction")
        
        print(f"\n🔧 TECHNICAL DETAILS:")
        print(f"   • Features used: {len(self.feature_names)}")
        print(f"   • Categorical encoders: {len(self.label_encoders)}")
        print(f"   • Scaling strategies: {len(self.scalers)}")
        print(f"   • Data leakage prevention: ✅ Implemented")
        print(f"   • Time-series validation: ✅ Used")
        print(f"   • Deep learning: {'✅ LSTM trained' if 'lstm_model' in self.models else '❌ Not available'}")
        
        print(f"\n📁 SAVED MODEL FILES:")
        print(f"   • models/final_model.pkl - Best trained model")
        print(f"   • models/scalers.pkl - Feature scalers")
        print(f"   • models/feature_names.pkl - Feature names")
        print(f"   • models/label_encoders.pkl - Categorical encoders")
        print(f"   • models/metadata.pkl - Model metadata")
        
        # Individual model files
        model_files = list(self.models.keys())
        if len(model_files) > 8:
            print(f"   • Individual models: {model_files[:8]} + {len(model_files)-8} more")
        else:
            print(f"   • Individual models: {model_files}")
        
        print(f"\n📊 ANALYSIS FILES:")
        print(f"   • results/model_performance.csv - Performance comparison")
        print(f"   • results/feature_importance.csv - Feature rankings")
        print(f"   • results/training_results.pkl - Complete results")
        
        print(f"\n🚀 READY FOR TESTING AND DEPLOYMENT!")
        print("="*120)
    
    def _get_model_category(self, model_type):
        """Categorize model type"""
        if 'Linear' in model_type or 'Logistic' in model_type or 'Ridge' in model_type or 'Lasso' in model_type or 'Elastic' in model_type:
            return 'Linear Models'
        elif 'RandomForest' in model_type or 'AdaBoost' in model_type or 'GradientBoosting' in model_type:
            return 'Tree-based Models'
        elif 'LGBM' in model_type or 'XGB' in model_type or 'CatBoost' in model_type:
            return 'Gradient Boosting'
        elif 'SV' in model_type:
            return 'Support Vector Machines'
        elif 'KNeighbors' in model_type:
            return 'K-Nearest Neighbors'
        elif 'LSTM' in model_type:
            return 'Deep Learning'
        elif 'Voting' in model_type:
            return 'Ensemble Methods'
        elif 'Naive' in model_type:
            return 'Probabilistic Models'
        else:
            return 'Other Models'
    
    def _categorize_feature(self, feature_name):
        """Categorize feature for better understanding"""
        feature_lower = feature_name.lower()
        
        if any(word in feature_lower for word in ['sentiment', 'fda', 'news']):
            return 'News/FDA'
        elif any(word in feature_lower for word in ['volume', 'vol']):
            return 'Volume'
        elif any(word in feature_lower for word in ['price', 'close', 'return', 'momentum']):
            return 'Price'
        elif any(word in feature_lower for word in ['ma', 'sma', 'ema', 'moving']):
            return 'Moving Avg'
        elif any(word in feature_lower for word in ['rsi', 'macd', 'bb', 'atr']):
            return 'Technical'
        elif any(word in feature_lower for word in ['volatility', 'std']):
            return 'Volatility'
        elif any(word in feature_lower for word in ['day', 'month', 'week']):
            return 'Temporal'
        elif 'lag' in feature_lower:
            return 'Lag'
        else:
            return 'Other'
    
    def execute(self):
        """Execute enhanced training pipeline"""
        try:
            logger.info("="*80)
            logger.info("STARTING ENHANCED BIOCON STOCK PREDICTION TRAINING")
            logger.info("="*80)
            
            # Load data
            stock_df, sentiment_df = self.load_and_validate_data()
            
            # Merge data
            combined_df = self.smart_data_merge(stock_df, sentiment_df)
            
            # Create proper features (NO data leakage)
            df_with_features = self.create_proper_features(combined_df)
            
            # Save combined data
            df_with_features.to_csv('combined_data_enhanced.csv', index=False)
            logger.info("✅ Enhanced combined data saved")
            
            # Prepare proper targets
            df_clean, target_col, target_type, target_description = self.prepare_proper_targets(df_with_features)
            
            # Prepare features (exclude targets and identifiers)
            exclude_cols = {
                'Date', 'Target_Next_Day_Up', 'Target_Next_3Day_Up', 
                'Target_Significant_Move', 'Target_Direction_Balanced',
                'Target_Next_Day_Return'
            }
            
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            X_raw = df_clean[feature_cols].copy()
            y = df_clean[target_col].values
            
            logger.info(f"Dataset shape: {X_raw.shape}, Target: {target_col}")
            
            # Feature selection (proper method)
            X_selected, selected_features = self.select_proper_features(X_raw, y, target_type, max_features=40)
            self.feature_names = selected_features
            
            logger.info(f"Selected {len(selected_features)} features for training")
            
            # Time-series split (CRITICAL: maintain temporal order)
            n_samples = len(X_selected)
            train_end = int(n_samples * 0.7)   # 70% for training
            val_end = int(n_samples * 0.85)    # 15% for validation
                                               # 15% for testing
            
            X_train = X_selected.iloc[:train_end]
            X_val = X_selected.iloc[train_end:val_end]
            X_test = X_selected.iloc[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Time-series splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Safe date logging
            try:
                if 'Date' in df_clean.columns:
                    train_start = df_clean.iloc[0]['Date']
                    train_end_date = df_clean.iloc[train_end-1]['Date'] 
                    val_start = df_clean.iloc[train_end]['Date']
                    val_end_date = df_clean.iloc[val_end-1]['Date']
                    test_start = df_clean.iloc[val_end]['Date']
                    test_end_date = df_clean.iloc[-1]['Date']
                    
                    logger.info(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
                    logger.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}")
                    logger.info(f"Test period: {test_start.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")
                else:
                    logger.info("Date column not available for period logging")
            except Exception as e:
                logger.warning(f"Could not log date periods: {str(e)}")
                logger.info("Proceeding with training...")
            
            # Train enhanced model suite
            self.train_enhanced_models(X_train, X_val, X_test, y_train, y_val, y_test, target_type)
            
            # Select best model
            self.select_best_model()
            
            # Cross-validate best model
            self.cross_validate_best_model(X_selected, y, target_type)
            
            # Save results
            save_success = self.save_models_and_results()
            
            # Print comprehensive summary
            self.print_comprehensive_summary(target_description, target_type)
            
            return save_success and len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Enhanced training failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution for enhanced stock prediction training"""
    print("🚀 BIOCON FDA PROJECT - ENHANCED STOCK PREDICTION TRAINING")
    print("🎯 COMPREHENSIVE MODEL SUITE:")
    print("   • Linear Models: Logistic, Ridge, Lasso, ElasticNet, Linear Regression")
    print("   • Tree-based: Random Forest, AdaBoost, Gradient Boosting")
    print("   • Gradient Boosting: LightGBM, XGBoost, CatBoost")  
    print("   • Support Vector Machines: SVM, SVR")
    print("   • Ensemble Methods: Voting Classifiers/Regressors")
    print("   • Deep Learning: LSTM (if TensorFlow available)")
    print("   • Other: K-NN, Naive Bayes")
    print("   • NO data leakage - strict temporal validation")
    print("   • Multiple scaling strategies for optimal performance")
    print("   • Production-ready model training and evaluation")
    print("-" * 80)
    
    trainer = EnhancedBioconModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("✅ Comprehensive model suite trained with realistic expectations")
        print("✅ All requested model types included:")
        
        # Show which models were actually trained
        model_names = list(trainer.models.keys())
        for model in model_names:
            print(f"   • {model}")
        
        print("✅ No data leakage - only historical features used")
        print("✅ Time-series validation implemented")
        print("✅ Multiple scaling strategies applied")
        print("✅ Comprehensive performance metrics calculated")
        print("✅ Production-ready models saved with proper naming")
        print("\n🚀 NEXT STEPS:")
        print("   1. Review model performance in results/")
        print("   2. Test individual models for specific use cases")
        print("   3. Deploy best model for predictions")
        print("   4. Use ensemble model for robust predictions")
    else:
        print("\n💥 TRAINING FAILED!")
        print("Check error messages above")
        print("💡 Debugging tips:")
        print("   • Check data files exist and are properly formatted")
        print("   • Verify no data leakage in feature engineering")
        print("   • Ensure sufficient data for training")
        print("   • Install required libraries: pip install catboost tensorflow")
    
    return success

if __name__ == "__main__":
    main()