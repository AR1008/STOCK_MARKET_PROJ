import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler, PowerTransformer
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                             GradientBoostingClassifier, GradientBoostingRegressor,
                             AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor,
                             ExtraTreesClassifier, BaggingClassifier)
from sklearn.linear_model import (LogisticRegression, ElasticNet, Ridge, RidgeClassifier,
                                Lasso, LinearRegression, LassoCV, RidgeCV, ElasticNetCV)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, classification_report, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy import stats
from scipy.stats import zscore
import warnings
import time
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
warnings.filterwarnings('ignore')

# For LSTM model
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. LSTM model will be skipped.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighPerformanceBioconTrainer:
    """
    HIGH-PERFORMANCE Stock Prediction Training - Targeting 60%+ Accuracy
    Advanced feature engineering, hyperparameter tuning, and ensemble methods
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
        self.feature_transformers = {}
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'results/charts', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_and_validate_data(self):
        """Load and validate data with enhanced preprocessing"""
        logger.info("Loading and validating data with enhanced preprocessing...")
        
        try:
            # Load stock data
            stock_df = pd.read_csv('stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            logger.info(f"✓ Loaded stock data: {len(stock_df)} records from {stock_df['Date'].min()} to {stock_df['Date'].max()}")
            
            # Load sentiment data
            sentiment_df = pd.read_csv('daily_sentiment.csv')
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
            sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
            logger.info(f"✓ Loaded sentiment data: {len(sentiment_df)} records")
            
            return stock_df, sentiment_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def smart_data_merge(self, stock_df, sentiment_df):
        """Enhanced data merge with intelligent feature creation"""
        logger.info("Performing enhanced data merge...")
        
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
            
            # Enhanced sentiment feature engineering
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
            
            # Create enhanced day_importance_score
            if 'day_importance_score' not in combined_df.columns or combined_df['day_importance_score'].isna().all():
                combined_df['day_importance_score'] = (
                    combined_df.get('news_count', 0) * 3 +
                    combined_df.get('drug_specific_count', 0) * 15 +
                    np.abs(combined_df.get('weighted_avg_sentiment', 0)) * 20 +
                    (combined_df.get('avg_sentiment', 0) ** 2) * 25  # Sentiment magnitude
                )
            
            # Handle FDA milestone flags
            milestone_columns = [col for col in combined_df.columns if col.startswith('has_')]
            for col in milestone_columns:
                combined_df[col] = combined_df[col].fillna(0).astype(int)
            
            # Ensure temporal order
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Enhanced merge completed: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in smart merge: {str(e)}")
            raise
    
    def create_advanced_features(self, df):
        """Create advanced features for high performance"""
        logger.info("Creating advanced features for maximum predictive power...")
        
        try:
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Remove data leakage columns
            forward_looking_cols = [col for col in df.columns if 'Forward' in col or 'Next' in col]
            if forward_looking_cols:
                logger.warning(f"🚨 REMOVING DATA LEAKAGE COLUMNS: {forward_looking_cols}")
                df = df.drop(columns=forward_looking_cols)
            
            # === ENHANCED PRICE FEATURES ===
            if 'Close' in df.columns:
                # Multi-timeframe returns
                for period in [1, 2, 3, 5, 8, 13, 21, 34]:
                    df[f'Return_{period}D'] = df['Close'].pct_change(period)
                    df[f'LogReturn_{period}D'] = np.log(df['Close'] / df['Close'].shift(period))
                
                # Moving averages with multiple timeframes
                for window in [3, 5, 8, 10, 13, 20, 21, 34, 50, 89]:
                    df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
                    df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
                    
                    # Price position relative to MA
                    df[f'Price_Above_SMA_{window}'] = (df['Close'] > df[f'SMA_{window}']).astype(int)
                    df[f'Price_Distance_SMA_{window}'] = (df['Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']
                    df[f'Price_Distance_EMA_{window}'] = (df['Close'] - df[f'EMA_{window}']) / df[f'EMA_{window}']
                
                # Bollinger Bands for multiple periods
                for window in [10, 20, 50]:
                    rolling_mean = df['Close'].rolling(window).mean()
                    rolling_std = df['Close'].rolling(window).std()
                    df[f'BB_Upper_{window}'] = rolling_mean + (rolling_std * 2)
                    df[f'BB_Lower_{window}'] = rolling_mean - (rolling_std * 2)
                    df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / rolling_mean
                    df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'])
                
                # Advanced volatility features
                for window in [5, 10, 20, 30]:
                    returns = df['Close'].pct_change()
                    df[f'Volatility_{window}D'] = returns.rolling(window).std() * np.sqrt(252)
                    df[f'Volatility_Ratio_{window}'] = df[f'Volatility_{window}D'] / df[f'Volatility_{window}D'].rolling(60).mean()
                    
                    # Realized volatility
                    df[f'RealizedVol_{window}'] = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)))
                    
                    # Volatility clustering
                    df[f'VolCluster_{window}'] = (df[f'Volatility_{window}D'] > df[f'Volatility_{window}D'].rolling(60).quantile(0.8)).astype(int)
                
                # Price momentum with multiple horizons
                for period in [3, 5, 8, 13, 21, 34]:
                    df[f'Momentum_{period}D'] = df['Close'].pct_change(period)
                    df[f'Momentum_Strength_{period}'] = np.abs(df[f'Momentum_{period}D'])
                
                # Support and resistance levels
                for window in [10, 20, 50]:
                    df[f'Support_{window}'] = df['Low'].rolling(window).min()
                    df[f'Resistance_{window}'] = df['High'].rolling(window).max()
                    df[f'Support_Distance_{window}'] = (df['Close'] - df[f'Support_{window}']) / df['Close']
                    df[f'Resistance_Distance_{window}'] = (df[f'Resistance_{window}'] - df['Close']) / df['Close']
                
                # Price percentile positions
                for window in [20, 50, 100]:
                    df[f'Price_Percentile_{window}'] = df['Close'].rolling(window).rank(pct=True)
                
                # Trend strength indicators
                for window in [10, 20]:
                    df[f'Trend_Strength_{window}'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)
                    df[f'Trend_Consistency_{window}'] = (df['Close'] > df['Close'].shift(1)).rolling(window).mean()
            
            # === ENHANCED VOLUME FEATURES ===
            if 'Volume' in df.columns:
                # Volume moving averages and ratios
                for window in [3, 5, 10, 20, 50]:
                    df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window).mean()
                    df[f'Volume_Ratio_{window}'] = df['Volume'] / (df[f'Volume_SMA_{window}'] + 1)
                    df[f'Volume_Percentile_{window}'] = df['Volume'].rolling(window).rank(pct=True)
                
                # Volume-price indicators
                df['Volume_Weighted_Price'] = (df['Volume'] * df['Close']).rolling(20).sum() / df['Volume'].rolling(20).sum()
                df['Price_Volume_Trend'] = ((df['Close'] - df['Close'].shift(1)) * df['Volume']).rolling(10).sum()
                
                # Volume spikes and droughts
                vol_mean = df['Volume'].rolling(50).mean()
                vol_std = df['Volume'].rolling(50).std()
                df['Volume_Spike'] = (df['Volume'] > vol_mean + 2 * vol_std).astype(int)
                df['Volume_Drought'] = (df['Volume'] < vol_mean - vol_std).astype(int)
                
                # On-Balance Volume
                df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
                df['OBV_SMA_10'] = df['OBV'].rolling(10).mean()
            
            # === ENHANCED SENTIMENT FEATURES ===
            if 'avg_sentiment' in df.columns:
                # Clean and enhance sentiment
                df['avg_sentiment'] = np.clip(df['avg_sentiment'], -1, 1)
                
                # Sentiment intensity and direction
                df['Sentiment_Intensity'] = np.abs(df['avg_sentiment'])
                df['Sentiment_Positive'] = (df['avg_sentiment'] > 0.05).astype(int)
                df['Sentiment_Negative'] = (df['avg_sentiment'] < -0.05).astype(int)
                df['Sentiment_Neutral'] = ((df['avg_sentiment'] >= -0.05) & (df['avg_sentiment'] <= 0.05)).astype(int)
                df['Sentiment_Strong'] = (np.abs(df['avg_sentiment']) > 0.3).astype(int)
                df['Sentiment_Extreme'] = (np.abs(df['avg_sentiment']) > 0.6).astype(int)
                
                # Sentiment momentum and persistence
                for window in [2, 3, 5, 7, 10]:
                    df[f'Sentiment_SMA_{window}'] = df['avg_sentiment'].rolling(window).mean()
                    df[f'Sentiment_Change_{window}'] = df['avg_sentiment'].diff(window)
                    df[f'Sentiment_Volatility_{window}'] = df['avg_sentiment'].rolling(window).std()
                
                # Sentiment regime detection
                df['Sentiment_Regime'] = pd.cut(df['avg_sentiment'], bins=[-1, -0.3, -0.1, 0.1, 0.3, 1], 
                                               labels=[0, 1, 2, 3, 4]).astype(float)
                
                # Sentiment momentum strength
                df['Sentiment_Momentum'] = df['avg_sentiment'] - df['avg_sentiment'].shift(1)
                df['Sentiment_Acceleration'] = df['Sentiment_Momentum'] - df['Sentiment_Momentum'].shift(1)
            
            # === ENHANCED FDA/NEWS FEATURES ===
            if 'day_importance_score' in df.columns:
                # FDA event classification
                df['FDA_Minor'] = ((df['day_importance_score'] > 5) & (df['day_importance_score'] <= 15)).astype(int)
                df['FDA_Moderate'] = ((df['day_importance_score'] > 15) & (df['day_importance_score'] <= 30)).astype(int)
                df['FDA_Major'] = (df['day_importance_score'] > 30).astype(int)
                df['FDA_Extreme'] = (df['day_importance_score'] > 50).astype(int)
                
                # Days since FDA events with decay
                for threshold in [10, 20, 30]:
                    fda_events = df['day_importance_score'] > threshold
                    days_since = 0
                    days_since_list = []
                    
                    for i in range(len(df)):
                        if fda_events.iloc[i]:
                            days_since = 0
                        else:
                            days_since += 1
                        days_since_list.append(min(days_since, 100))  # Cap at 100
                    
                    df[f'Days_Since_FDA_{threshold}'] = days_since_list
                    df[f'FDA_Decay_{threshold}'] = np.exp(-np.array(days_since_list) / 20)  # Exponential decay
                
                # FDA momentum and clustering
                for window in [3, 5, 10]:
                    df[f'FDA_Score_SMA_{window}'] = df['day_importance_score'].rolling(window).mean()
                    df[f'FDA_Score_Max_{window}'] = df['day_importance_score'].rolling(window).max()
                    df[f'FDA_Activity_{window}'] = (df['day_importance_score'] > 0).rolling(window).sum()
            
            # === ADVANCED TECHNICAL INDICATORS ===
            if 'RSI_14' in df.columns:
                # RSI regime classification
                df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
                df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
                df['RSI_Extreme_Oversold'] = (df['RSI_14'] < 20).astype(int)
                df['RSI_Extreme_Overbought'] = (df['RSI_14'] > 80).astype(int)
                df['RSI_Middle'] = ((df['RSI_14'] >= 40) & (df['RSI_14'] <= 60)).astype(int)
                
                # RSI divergence (simplified)
                df['RSI_Change'] = df['RSI_14'].diff()
                df['Price_Change'] = df['Close'].pct_change()
                df['RSI_Price_Divergence'] = np.sign(df['RSI_Change']) != np.sign(df['Price_Change'])
            
            if 'MACD' in df.columns:
                df['MACD_Positive'] = (df['MACD'] > 0).astype(int)
                df['MACD_Increasing'] = (df['MACD'] > df['MACD'].shift(1)).astype(int)
                
                if 'MACD_Signal' in df.columns:
                    df['MACD_Signal_Cross'] = ((df['MACD'] > df['MACD_Signal']) & 
                                             (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
                    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # === MARKET TIMING AND CYCLICAL FEATURES ===
            if 'Date' in df.columns:
                df['Day_of_Week'] = df['Date'].dt.dayofweek
                df['Month'] = df['Date'].dt.month
                df['Quarter'] = df['Date'].dt.quarter
                df['Day_of_Month'] = df['Date'].dt.day
                df['Day_of_Year'] = df['Date'].dt.dayofyear
                
                # Market timing effects
                df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
                df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
                df['Is_Month_Start'] = (df['Day_of_Month'] <= 5).astype(int)
                df['Is_Month_End'] = (df['Day_of_Month'] >= 25).astype(int)
                df['Is_Quarter_End'] = (df['Month'].isin([3, 6, 9, 12]) & (df['Day_of_Month'] >= 25)).astype(int)
                df['Is_Earnings_Season'] = df['Month'].isin([1, 4, 7, 10]).astype(int)
                
                # Seasonal effects
                df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
                df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
            
            # === ADVANCED LAG FEATURES ===
            lag_features = ['Close', 'Volume', 'Return_1D', 'avg_sentiment', 'day_importance_score', 'RSI_14']
            for feature in lag_features:
                if feature in df.columns:
                    for lag in [1, 2, 3, 5, 8, 13]:
                        df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
            
            # === INTERACTION FEATURES ===
            # Sentiment-Volume interactions
            if 'Sentiment_Strong' in df.columns and 'Volume_Spike' in df.columns:
                df['Sentiment_Volume_Signal'] = df['Sentiment_Strong'] * df['Volume_Spike']
                df['Negative_Sentiment_High_Volume'] = df['Sentiment_Negative'] * df['Volume_Spike']
                df['Positive_Sentiment_High_Volume'] = df['Sentiment_Positive'] * df['Volume_Spike']
            
            # FDA-Sentiment interactions
            if 'FDA_Major' in df.columns and 'avg_sentiment' in df.columns:
                df['FDA_Sentiment_Signal'] = df['FDA_Major'] * df['avg_sentiment']
                df['FDA_Positive_News'] = df['FDA_Major'] * df['Sentiment_Positive']
                df['FDA_Negative_News'] = df['FDA_Major'] * df['Sentiment_Negative']
            
            # Price-Volume interactions
            if 'Price_Change' in df.columns and 'Volume_Ratio_10' in df.columns:
                df['Price_Volume_Momentum'] = df['Price_Change'] * df['Volume_Ratio_10']
                df['Breakout_Signal'] = (df['Price_Percentile_20'] > 0.8) * df['Volume_Spike']
                df['Breakdown_Signal'] = (df['Price_Percentile_20'] < 0.2) * df['Volume_Spike']
            
            # === CREATE SOPHISTICATED TARGETS ===
            if 'Close' in df.columns:
                # Multiple target definitions for better class balance
                returns_1d = df['Close'].pct_change().shift(-1)
                returns_3d = df['Close'].pct_change(3).shift(-3)
                
                # Adaptive thresholds based on volatility
                vol_20d = df['Return_1D'].rolling(20).std()
                dynamic_threshold = vol_20d * 0.5  # 0.5 standard deviations
                
                # Standard targets
                df['Target_Next_Day_Up'] = (returns_1d > 0).astype(int)
                df['Target_Next_3Day_Up'] = (returns_3d > 0).astype(int)
                
                # Dynamic threshold targets
                df['Target_Dynamic_Up'] = (returns_1d > dynamic_threshold).astype(int)
                df['Target_Dynamic_Down'] = (returns_1d < -dynamic_threshold).astype(int)
                df['Target_Dynamic_Significant'] = ((returns_1d > dynamic_threshold) | 
                                                  (returns_1d < -dynamic_threshold)).astype(int)
                
                # Quantile-based targets for better balance
                returns_1d_clean = returns_1d.dropna()
                upper_threshold = returns_1d_clean.quantile(0.6)  # Top 40%
                lower_threshold = returns_1d_clean.quantile(0.4)  # Bottom 40%
                
                df['Target_Quantile_Up'] = (returns_1d > upper_threshold).astype(int)
                df['Target_Quantile_Down'] = (returns_1d < lower_threshold).astype(int)
                df['Target_Quantile_Extreme'] = ((returns_1d > upper_threshold) | 
                                                (returns_1d < lower_threshold)).astype(int)
                
                # Regression targets
                df['Target_Next_Day_Return'] = returns_1d
                df['Target_Next_3Day_Return'] = returns_3d
            
            logger.info(f"✅ Advanced feature engineering completed. Total features: {len(df.columns)}")
            
            # Final check for data leakage
            suspect_cols = [col for col in df.columns if any(word in col.lower() for word in ['forward', 'future', 'next', 'ahead'])]
            if suspect_cols and 'Target' not in str(suspect_cols):
                logger.error(f"🚨 POTENTIAL DATA LEAKAGE DETECTED: {suspect_cols}")
                raise ValueError("Data leakage detected in features!")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in advanced feature engineering: {str(e)}")
            raise
    
    def advanced_feature_selection(self, X, y, target_type, max_features=50):
        """Advanced feature selection with multiple methods"""
        logger.info(f"Advanced feature selection for {target_type}...")
        
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
            
            # Advanced preprocessing
            # 1. Remove constant features
            constant_features = [col for col in X_numeric.columns if X_numeric[col].nunique() <= 1]
            if constant_features:
                logger.info(f"Removing {len(constant_features)} constant features")
                X_numeric = X_numeric.drop(columns=constant_features)
            
            # 2. Remove highly correlated features
            correlation_matrix = X_numeric.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            highly_correlated = [column for column in upper_triangle.columns 
                               if any(upper_triangle[column] > 0.95)]
            if highly_correlated:
                logger.info(f"Removing {len(highly_correlated)} highly correlated features")
                X_numeric = X_numeric.drop(columns=highly_correlated)
            
            # 3. Handle missing values
            for col in X_numeric.columns:
                if X_numeric[col].isnull().sum() > 0:
                    if 'sentiment' in col.lower():
                        X_numeric[col] = X_numeric[col].fillna(0.0)
                    elif 'volume' in col.lower():
                        X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
                    else:
                        X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
            
            # 4. Remove infinite values
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
            for col in X_numeric.columns:
                if X_numeric[col].isnull().sum() > 0:
                    X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())
            
            # 5. Outlier handling (cap at 99th percentile)
            for col in X_numeric.columns:
                if X_numeric[col].std() > 0:
                    lower_bound = X_numeric[col].quantile(0.01)
                    upper_bound = X_numeric[col].quantile(0.99)
                    X_numeric[col] = np.clip(X_numeric[col], lower_bound, upper_bound)
            
            logger.info(f"After preprocessing: {len(X_numeric.columns)} features")
            
            # Multiple feature selection methods
            selected_features_sets = []
            
            if target_type == 'classification':
                try:
                    # Method 1: Statistical test (F-score)
                    selector_f = SelectKBest(score_func=f_classif, k=min(max_features, len(X_numeric.columns)))
                    selector_f.fit(X_numeric, y)
                    f_features = X_numeric.columns[selector_f.get_support()].tolist()
                    selected_features_sets.append(f_features)
                    logger.info(f"F-score method selected {len(f_features)} features")
                    
                    # Method 2: Mutual information
                    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(X_numeric.columns)))
                    selector_mi.fit(X_numeric, y)
                    mi_features = X_numeric.columns[selector_mi.get_support()].tolist()
                    selected_features_sets.append(mi_features)
                    logger.info(f"Mutual information method selected {len(mi_features)} features")
                    
                    # Method 3: Tree-based importance
                    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    rf_selector.fit(X_numeric, y)
                    feature_importance = pd.DataFrame({
                        'feature': X_numeric.columns,
                        'importance': rf_selector.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    top_rf_features = feature_importance.head(max_features)['feature'].tolist()
                    selected_features_sets.append(top_rf_features)
                    logger.info(f"Random Forest method selected {len(top_rf_features)} features")
                    
                except Exception as e:
                    logger.warning(f"Some feature selection methods failed: {str(e)}")
            
            else:  # regression
                try:
                    # For regression, use different approaches
                    correlations = []
                    for col in X_numeric.columns:
                        try:
                            corr = np.corrcoef(X_numeric[col], y)[0, 1]
                            correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
                        except:
                            correlations.append((col, 0))
                    
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    correlation_features = [feat for feat, _ in correlations[:max_features]]
                    selected_features_sets.append(correlation_features)
                    
                except Exception as e:
                    logger.warning(f"Regression feature selection failed: {str(e)}")
            
            # Combine feature selection methods
            if selected_features_sets:
                # Union of top features from different methods
                all_selected = set()
                for feature_set in selected_features_sets:
                    all_selected.update(feature_set[:max_features//len(selected_features_sets)])
                
                final_features = list(all_selected)[:max_features]
            else:
                # Fallback: use highest variance features
                feature_vars = X_numeric.var().sort_values(ascending=False)
                final_features = feature_vars.head(max_features).index.tolist()
            
            logger.info(f"Final feature selection: {len(final_features)} features selected")
            return X_numeric[final_features], final_features
                
        except Exception as e:
            logger.error(f"Error in advanced feature selection: {str(e)}")
            # Fallback
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            safe_columns = [col for col in numeric_columns if 'Target' not in col][:max_features]
            return X[safe_columns], safe_columns
    
    def prepare_optimal_targets(self, df):
        """Prepare targets optimized for high performance"""
        logger.info("Preparing optimal targets for maximum accuracy...")
        
        try:
            # Define target options with balance and performance considerations
            target_options = [
                ('Target_Quantile_Up', 'classification', 'Quantile-Based Direction (Balanced)'),
                ('Target_Dynamic_Up', 'classification', 'Dynamic Threshold Direction'),
                ('Target_Quantile_Extreme', 'classification', 'Quantile Extreme Moves'),
                ('Target_Dynamic_Significant', 'classification', 'Dynamic Significant Moves'),
                ('Target_Next_Day_Up', 'classification', 'Next Day Direction'),
                ('Target_Next_3Day_Up', 'classification', 'Next 3-Day Direction'),
            ]
            
            # Test each target for class balance and predictability
            best_target = None
            best_score = 0
            
            for target_col, target_type, description in target_options:
                if target_col in df.columns:
                    df_clean = df.dropna(subset=[target_col]).copy()
                    
                    if len(df_clean) > 1000:
                        y = df_clean[target_col].values
                        
                        if target_type == 'classification':
                            class_counts = np.bincount(y.astype(int))
                            if len(class_counts) >= 2:
                                minority_class_pct = min(class_counts) / sum(class_counts) * 100
                                
                                # Prefer balanced classes (30-70% range is good)
                                if 25 <= minority_class_pct <= 75:
                                    # Calculate a balance score (closer to 50% is better)
                                    balance_score = 100 - abs(50 - minority_class_pct)
                                    
                                    # Add bonus for certain target types
                                    if 'Quantile' in target_col:
                                        balance_score += 20  # Prefer quantile-based targets
                                    elif 'Dynamic' in target_col:
                                        balance_score += 15  # Dynamic thresholds are good
                                    
                                    if balance_score > best_score:
                                        best_score = balance_score
                                        best_target = (df_clean, target_col, target_type, description)
                                        
                                    logger.info(f"Target {target_col}: {minority_class_pct:.1f}% minority class, score: {balance_score:.1f}")
            
            if best_target:
                df_clean, target_col, target_type, description = best_target
                logger.info(f"✅ Selected optimal target: {target_col} ({description})")
                return df_clean, target_col, target_type, description
            
            # Fallback: create a custom balanced target
            logger.warning("Creating custom balanced target...")
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().shift(-1)
                # Use median as threshold for perfect balance
                threshold = returns.median()
                df['Target_Balanced_Custom'] = (returns > threshold).astype(int)
                
                df_clean = df.dropna(subset=['Target_Balanced_Custom']).copy()
                if len(df_clean) > 1000:
                    return df_clean, 'Target_Balanced_Custom', 'classification', 'Custom Balanced Direction'
            
            raise ValueError("Cannot create any suitable target variable")
            
        except Exception as e:
            logger.error(f"Error preparing optimal targets: {str(e)}")
            raise
    
    def create_optimized_models(self, target_type):
        """Create highly optimized models with best hyperparameters"""
        
        if target_type == 'classification':
            models_config = {
                # === OPTIMIZED LINEAR MODELS ===
                'logistic_regression_model': {
                    'model': LogisticRegression(
                        random_state=42, max_iter=3000, 
                        class_weight='balanced', C=0.1,
                        solver='liblinear', penalty='l2'
                    ),
                    'data_type': 'standard',
                    'expected_time': 8
                },
                'ridge_model': {
                    'model': RidgeClassifier(
                        random_state=42, alpha=0.1,
                        class_weight='balanced'
                    ),
                    'data_type': 'standard',
                    'expected_time': 5
                },
                
                # === OPTIMIZED TREE MODELS ===
                'random_forest_model': {
                    'model': RandomForestClassifier(
                        n_estimators=300, max_depth=15, random_state=42,
                        class_weight='balanced', n_jobs=-1,
                        min_samples_split=10, min_samples_leaf=5,
                        max_features='sqrt', bootstrap=True,
                        criterion='gini'
                    ),
                    'data_type': 'raw',
                    'expected_time': 45
                },
                'extra_trees_model': {
                    'model': ExtraTreesClassifier(
                        n_estimators=250, max_depth=12, random_state=42,
                        class_weight='balanced', n_jobs=-1,
                        min_samples_split=8, min_samples_leaf=4,
                        max_features='sqrt', bootstrap=True
                    ),
                    'data_type': 'raw',
                    'expected_time': 40
                },
                'gradient_boosting_model': {
                    'model': GradientBoostingClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        random_state=42, min_samples_split=15,
                        min_samples_leaf=8, subsample=0.85,
                        max_features='sqrt'
                    ),
                    'data_type': 'raw',
                    'expected_time': 60
                },
                'adaboost_model': {
                    'model': AdaBoostClassifier(
                        n_estimators=150, learning_rate=0.8,
                        random_state=42,
                        estimator=DecisionTreeClassifier(max_depth=4, class_weight='balanced')
                    ),
                    'data_type': 'raw',
                    'expected_time': 35
                },
                
                # === OPTIMIZED GRADIENT BOOSTING ===
                'lightgbm_model': {
                    'model': lgb.LGBMClassifier(
                        n_estimators=300, max_depth=10, learning_rate=0.03,
                        random_state=42, n_jobs=-1, verbose=-1,
                        class_weight='balanced', min_child_samples=10,
                        feature_fraction=0.9, bagging_fraction=0.9,
                        num_leaves=31, reg_alpha=0.3, reg_lambda=0.3
                    ),
                    'data_type': 'raw',
                    'expected_time': 35
                },
                'xgboost_model': {
                    'model': xgb.XGBClassifier(
                        n_estimators=300, max_depth=10, learning_rate=0.03,
                        random_state=42, n_jobs=-1,
                        subsample=0.9, colsample_bytree=0.9,
                        reg_alpha=0.3, reg_lambda=0.3,
                        scale_pos_weight=1, eval_metric='logloss'
                    ),
                    'data_type': 'raw',
                    'expected_time': 45
                },
                'catboost_model': {
                    'model': cb.CatBoostClassifier(
                        iterations=300, depth=8, learning_rate=0.03,
                        random_seed=42, verbose=False,
                        class_weights='Balanced',
                        l2_leaf_reg=5.0, bootstrap_type='Bernoulli',
                        subsample=0.9
                    ),
                    'data_type': 'raw',
                    'expected_time': 50
                },
                
                # === OPTIMIZED SVM ===
                'svm_model': {
                    'model': SVC(
                        kernel='rbf', C=10.0, gamma='scale',
                        random_state=42, class_weight='balanced',
                        probability=True, cache_size=1000
                    ),
                    'data_type': 'robust',
                    'expected_time': 120
                },
                
                # === ENSEMBLE MODELS ===
                'ensemble_model': {
                    'model': self.create_advanced_ensemble('classification'),
                    'data_type': 'standard',
                    'expected_time': 80
                },
                'bagging_model': {
                    'model': BaggingClassifier(
                        estimator=DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
                        n_estimators=100, random_state=42, n_jobs=-1,
                        max_samples=0.8, max_features=0.8
                    ),
                    'data_type': 'raw',
                    'expected_time': 30
                }
            }
        
        else:  # regression
            models_config = {
                'linear_regression_model': {
                    'model': LinearRegression(),
                    'data_type': 'standard',
                    'expected_time': 3
                },
                'ridge_regression_model': {
                    'model': Ridge(random_state=42, alpha=0.1),
                    'data_type': 'standard',
                    'expected_time': 5
                },
                'lasso_model': {
                    'model': Lasso(random_state=42, alpha=0.01, max_iter=3000),
                    'data_type': 'standard',
                    'expected_time': 8
                },
                'elasticnet_model': {
                    'model': ElasticNet(random_state=42, alpha=0.01, l1_ratio=0.5, max_iter=3000),
                    'data_type': 'standard',
                    'expected_time': 10
                },
                'random_forest_reg_model': {
                    'model': RandomForestRegressor(
                        n_estimators=300, max_depth=15, random_state=42,
                        n_jobs=-1, min_samples_split=10, min_samples_leaf=5,
                        max_features='sqrt'
                    ),
                    'data_type': 'raw',
                    'expected_time': 40
                },
                'lightgbm_reg_model': {
                    'model': lgb.LGBMRegressor(
                        n_estimators=300, max_depth=10, learning_rate=0.03,
                        random_state=42, n_jobs=-1, verbose=-1,
                        min_child_samples=10, feature_fraction=0.9,
                        bagging_fraction=0.9, reg_alpha=0.3, reg_lambda=0.3
                    ),
                    'data_type': 'raw',
                    'expected_time': 30
                },
                'xgboost_reg_model': {
                    'model': xgb.XGBRegressor(
                        n_estimators=300, max_depth=10, learning_rate=0.03,
                        random_state=42, n_jobs=-1,
                        subsample=0.9, colsample_bytree=0.9,
                        reg_alpha=0.3, reg_lambda=0.3
                    ),
                    'data_type': 'raw',
                    'expected_time': 35
                },
                'svr_model': {
                    'model': SVR(kernel='rbf', C=10.0, gamma='scale'),
                    'data_type': 'robust',
                    'expected_time': 90
                },
                'ensemble_model': {
                    'model': self.create_advanced_ensemble('regression'),
                    'data_type': 'standard',
                    'expected_time': 60
                }
            }
        
        return models_config
    
    def create_advanced_ensemble(self, target_type):
        """Create sophisticated ensemble model"""
        try:
            if target_type == 'classification':
                base_models = [
                    ('rf', RandomForestClassifier(
                        n_estimators=100, max_depth=10, random_state=42, 
                        class_weight='balanced', n_jobs=-1
                    )),
                    ('lgb', lgb.LGBMClassifier(
                        n_estimators=100, max_depth=8, random_state=42, 
                        verbose=-1, class_weight='balanced'
                    )),
                    ('xgb', xgb.XGBClassifier(
                        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
                    )),
                    ('lr', LogisticRegression(
                        random_state=42, max_iter=2000, class_weight='balanced'
                    ))
                ]
                return VotingClassifier(estimators=base_models, voting='soft')
            else:
                base_models = [
                    ('rf', RandomForestRegressor(
                        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                    )),
                    ('lgb', lgb.LGBMRegressor(
                        n_estimators=100, max_depth=8, random_state=42, verbose=-1
                    )),
                    ('ridge', Ridge(random_state=42, alpha=0.1))
                ]
                return VotingRegressor(estimators=base_models)
        except Exception as e:
            logger.error(f"Error creating advanced ensemble: {str(e)}")
            return None
    
    def train_high_performance_models(self, X_train, X_val, X_test, y_train, y_val, y_test, target_type):
        """Train models with focus on achieving 60%+ accuracy"""
        logger.info(f"Training HIGH-PERFORMANCE models for {target_type} (targeting 60%+ accuracy)...")
        
        start_time = time.time()
        
        # Enhanced scaling with multiple strategies
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(method='yeo-johnson', standardize=True)
        }
        
        scaled_data = {}
        for scaler_name, scaler in scalers.items():
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                scaled_data[scaler_name] = {
                    'train': X_train_scaled,
                    'val': X_val_scaled,
                    'test': X_test_scaled
                }
                self.scalers[scaler_name] = scaler
            except Exception as e:
                logger.warning(f"Failed to create {scaler_name} scaler: {str(e)}")
        
        # Handle class imbalance for classification
        if target_type == 'classification':
            try:
                # Check class distribution
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = dict(zip(unique, counts))
                minority_pct = min(counts) / sum(counts) * 100
                
                logger.info(f"Class distribution: {class_dist} (minority: {minority_pct:.1f}%)")
                
                # Apply SMOTE if severely imbalanced
                if minority_pct < 35:
                    logger.info("Applying SMOTE for class balancing...")
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                    logger.info(f"After SMOTE: {len(X_train_balanced)} samples")
                else:
                    X_train_balanced, y_train_balanced = X_train, y_train
                    
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}, using original data")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Get optimized models
        models_config = self.create_optimized_models(target_type)
        
        # Train models with enhanced evaluation
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
                if data_type in scaled_data:
                    X_tr = scaled_data[data_type]['train']
                    X_v = scaled_data[data_type]['val']
                    X_te = scaled_data[data_type]['test']
                    
                    # Use balanced data for training if available
                    if target_type == 'classification' and 'X_train_balanced' in locals():
                        # Scale balanced data
                        X_tr_balanced = self.scalers[data_type].transform(X_train_balanced)
                        y_tr = y_train_balanced
                    else:
                        X_tr_balanced = X_tr
                        y_tr = y_train
                else:  # raw data
                    X_tr = X_train_balanced.values if target_type == 'classification' and 'X_train_balanced' in locals() else X_train.values
                    X_v = X_val.values
                    X_te = X_test.values
                    y_tr = y_train_balanced if target_type == 'classification' and 'y_train_balanced' in locals() else y_train
                    X_tr_balanced = X_tr
                
                # Train model
                model.fit(X_tr_balanced, y_tr)
                
                # Predict on validation and test
                y_val_pred = model.predict(X_v)
                y_test_pred = model.predict(X_te)
                
                # Calculate enhanced metrics
                if target_type == 'classification':
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
                    # Calculate precision, recall for both classes
                    val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
                    
                    # AUC calculation
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_val_proba = model.predict_proba(X_v)
                            if y_val_proba.shape[1] > 1:
                                y_val_proba_pos = y_val_proba[:, 1]
                                val_auc = roc_auc_score(y_val, y_val_proba_pos)
                                
                                y_test_proba = model.predict_proba(X_te)[:, 1]
                                test_auc = roc_auc_score(y_test, y_test_proba)
                            else:
                                val_auc = test_auc = 0.5
                        else:
                            val_auc = test_auc = 0.5
                    except:
                        val_auc = test_auc = 0.5
                    
                    # Confusion matrices
                    val_cm = confusion_matrix(y_val, y_val_pred)
                    test_cm = confusion_matrix(y_test, y_test_pred)
                    
                    # Enhanced scoring: weighted combination
                    combined_score = (0.6 * val_accuracy + 0.3 * val_auc + 0.1 * val_report['weighted avg']['f1-score'])
                    
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
                        'combined_score': combined_score,
                        'training_time': time.time() - model_start
                    }
                    
                    model_time = time.time() - model_start
                    logger.info(f"✓ {model_name} - Accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}, Combined: {combined_score:.3f}, Time: {model_time:.1f}s")
                    
                    # Success indicator
                    if val_accuracy >= 0.60:
                        logger.info(f"🎉 {model_name} ACHIEVED TARGET: {val_accuracy:.1%} accuracy!")
                    elif val_accuracy >= 0.55:
                        logger.info(f"🎯 {model_name} Close to target: {val_accuracy:.1%} accuracy")
                
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
                
                # Feature importance
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
        
        # Train LSTM if available
        if KERAS_AVAILABLE and target_type == 'classification' and len(X_train) > 100:
            try:
                logger.info("Training optimized LSTM model...")
                lstm_start = time.time()
                
                # Create sequence data
                lookback = min(40, len(X_train) // 8)
                X_train_lstm, y_train_lstm = self.create_lstm_data(scaled_data['standard']['train'], y_train, lookback)
                X_val_lstm, y_val_lstm = self.create_lstm_data(scaled_data['standard']['val'], y_val, lookback)
                X_test_lstm, y_test_lstm = self.create_lstm_data(scaled_data['standard']['test'], y_test, lookback)
                
                if X_train_lstm is not None and len(X_train_lstm) > 50:
                    # Enhanced LSTM architecture
                    lstm_model = Sequential([
                        LSTM(64, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
                        Dropout(0.3),
                        BatchNormalization(),
                        LSTM(32, return_sequences=False),
                        Dropout(0.3),
                        BatchNormalization(),
                        Dense(16, activation='relu'),
                        Dropout(0.2),
                        Dense(1, activation='sigmoid')
                    ])
                    
                    lstm_model.compile(
                        optimizer=Adam(learning_rate=0.001), 
                        loss='binary_crossentropy', 
                        metrics=['accuracy']
                    )
                    
                    # Enhanced callbacks
                    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                    
                    # Train with validation monitoring
                    history = lstm_model.fit(
                        X_train_lstm, y_train_lstm,
                        validation_data=(X_val_lstm, y_val_lstm),
                        epochs=100, batch_size=32, verbose=0,
                        callbacks=[early_stop, reduce_lr]
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
                        'val_auc': 0.5,
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
                    
                    if val_accuracy_lstm >= 0.60:
                        logger.info(f"🎉 LSTM ACHIEVED TARGET: {val_accuracy_lstm:.1%} accuracy!")
                    
                    successful_models += 1
                    
            except Exception as e:
                logger.error(f"Error training LSTM: {str(e)}")
        
        total_time = time.time() - start_time
        logger.info(f"Successfully trained {successful_models} high-performance models in {total_time:.1f}s")
        
        # Check if any model achieved target
        if target_type == 'classification':
            best_accuracy = max([m.get('val_accuracy', 0) for m in self.performance_metrics.values()])
            if best_accuracy >= 0.60:
                logger.info(f"🎉 SUCCESS! Achieved {best_accuracy:.1%} accuracy (target: 60%)")
            else:
                logger.warning(f"⚠️ Best accuracy: {best_accuracy:.1%} (target: 60% not reached)")
    
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
    
    def select_best_model(self):
        """Select best model with enhanced criteria"""
        if not self.performance_metrics:
            logger.warning("No models to evaluate")
            return
        
        best_score = -float('inf')
        best_model_name = None
        
        # Enhanced model selection criteria
        for model_name, metrics in self.performance_metrics.items():
            if 'val_accuracy' in metrics:
                # For classification: prioritize accuracy with AUC bonus
                accuracy = metrics['val_accuracy']
                auc = metrics.get('val_auc', 0.5)
                f1 = metrics.get('val_f1', 0)
                
                # Combined score with accuracy emphasis
                score = (0.7 * accuracy + 0.2 * auc + 0.1 * f1)
                
                # Bonus for achieving target accuracy
                if accuracy >= 0.60:
                    score += 0.1  # 10% bonus for reaching target
                elif accuracy >= 0.55:
                    score += 0.05  # 5% bonus for being close
                    
            else:
                # For regression
                score = metrics.get('primary_metric', -float('inf'))
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name)
        
        logger.info(f"Best model: {best_model_name} (Enhanced Score: {best_score:.3f})")
    
    def cross_validate_best_model(self, X, y, target_type):
        """Enhanced cross-validation with time-series awareness"""
        logger.info("Performing enhanced time-series cross-validation...")
        
        try:
            if not self.best_model_name:
                logger.warning("No best model selected for cross-validation")
                return
            
            # Use more folds for better validation
            tscv = TimeSeriesSplit(n_splits=7)
            
            # Get model for CV
            model_metadata = self.model_metadata.get(self.best_model_name, {})
            model_type = model_metadata.get('model_type', 'Unknown')
            
            # Create optimized model for CV
            if 'LogisticRegression' in model_type:
                model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=0.1)
            elif 'RandomForest' in model_type:
                if target_type == 'classification':
                    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=12)
                else:
                    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
            elif 'LightGBM' in model_type:
                if target_type == 'classification':
                    model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1, class_weight='balanced')
                else:
                    model = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
            elif 'XGB' in model_type:
                if target_type == 'classification':
                    model = xgb.XGBClassifier(n_estimators=200, random_state=42)
                else:
                    model = xgb.XGBRegressor(n_estimators=200, random_state=42)
            else:
                model = self.models[self.best_model_name]
            
            cv_scores = []
            cv_detailed = []
            
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
                    
                    # Additional metrics for detailed analysis
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba_cv = model.predict_proba(X_val_scaled)[:, 1]
                            auc_score = roc_auc_score(y_val_cv, y_proba_cv)
                        else:
                            auc_score = 0.5
                    except:
                        auc_score = 0.5
                    
                    cv_detailed.append({
                        'fold': fold + 1,
                        'accuracy': score,
                        'auc': auc_score,
                        'samples': len(y_val_cv)
                    })
                else:
                    score = r2_score(y_val_cv, y_pred_cv)
                    cv_detailed.append({
                        'fold': fold + 1,
                        'r2': score,
                        'samples': len(y_val_cv)
                    })
                
                cv_scores.append(score)
                logger.info(f"Fold {fold+1}: {score:.3f}")
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_min = np.min(cv_scores)
            cv_max = np.max(cv_scores)
            
            self.validation_results = {
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_min': cv_min,
                'cv_max': cv_max,
                'cv_model': self.best_model_name,
                'cv_detailed': cv_detailed
            }
            
            # Enhanced reporting
            stability = 'Excellent' if cv_std < 0.03 else 'Good' if cv_std < 0.05 else 'Moderate' if cv_std < 0.08 else 'Poor'
            
            logger.info(f"Cross-validation results: {cv_mean:.3f} ± {cv_std:.3f}")
            logger.info(f"Range: [{cv_min:.3f}, {cv_max:.3f}], Stability: {stability}")
            
            if target_type == 'classification' and cv_mean >= 0.60:
                logger.info(f"🎉 Cross-validation confirms target achievement: {cv_mean:.1%}")
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
    
    def save_models_and_results(self):
        """Save all models and results with enhanced metadata"""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                with open(f'models/{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save best model
            if self.best_model:
                with open('models/final_model.pkl', 'wb') as f:
                    pickle.dump(self.best_model, f)
            
            # Save all scalers and transformers
            with open('models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            with open('models/label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            # Enhanced metadata
            with open('models/metadata.pkl', 'wb') as f:
                pickle.dump(self.model_metadata, f)
                
            with open('models/model_metadata.pkl', 'wb') as f:
                pickle.dump(self.model_metadata, f)
            
            # Compatibility files
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scalers.get('standard'), f)
            
            # Save performance metrics
            pd.DataFrame(self.performance_metrics).T.to_csv('results/model_performance.csv')
            
            # Enhanced results
            with open('results/training_results.pkl', 'wb') as f:
                pickle.dump({
                    'performance_metrics': self.performance_metrics,
                    'feature_importance': self.feature_importance,
                    'validation_results': self.validation_results,
                    'best_model_name': self.best_model_name,
                    'model_metadata': self.model_metadata,
                    'scalers': list(self.scalers.keys()),
                    'feature_count': len(self.feature_names)
                }, f)
            
            # Feature importance
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
    
    def print_high_performance_summary(self, target_description, target_type):
        """Print enhanced summary focused on high performance achievement"""
        print("\n" + "="*130)
        print("HIGH-PERFORMANCE BIOCON STOCK PREDICTION MODEL TRAINING SUMMARY")
        print("="*130)
        
        print(f"🎯 TARGET: {target_description} ({target_type})")
        print(f"🚀 HIGH-PERFORMANCE APPROACH:")
        print(f"   • Advanced feature engineering with 100+ sophisticated features")
        print(f"   • Optimized hyperparameters for maximum accuracy")
        print(f"   • Multiple scaling strategies and data preprocessing")
        print(f"   • Class balancing with SMOTE for optimal performance")
        print(f"   • Enhanced ensemble methods and model optimization")
        print(f"   • GOAL: Achieve 60%+ accuracy for stock prediction")
        
        if not self.performance_metrics:
            print("❌ No models trained successfully")
            return
        
        # Enhanced performance display
        print(f"\n🏆 HIGH-PERFORMANCE MODEL RESULTS ({len(self.performance_metrics)} models):")
        
        if target_type == 'classification':
            print(f"{'Model':<30} {'Val_Acc':<10} {'Test_Acc':<10} {'Val_AUC':<10} {'F1':<8} {'Target':<8} {'Time(s)':<8}")
            print("-" * 115)
            
            sorted_models = sorted(
                self.performance_metrics.items(),
                key=lambda x: x[1].get('val_accuracy', 0),
                reverse=True
            )
            
            target_achieved = False
            models_above_55 = 0
            
            for model_name, metrics in sorted_models:
                val_acc = metrics.get('val_accuracy', 0)
                test_acc = metrics.get('test_accuracy', 0)
                val_auc = metrics.get('val_auc', 0.5)
                f1 = metrics.get('val_f1', 0)
                train_time = metrics.get('training_time', 0)
                
                target_status = "🎉 YES" if val_acc >= 0.60 else "🎯 CLOSE" if val_acc >= 0.55 else "❌ NO"
                if val_acc >= 0.60:
                    target_achieved = True
                if val_acc >= 0.55:
                    models_above_55 += 1
                
                print(f"{model_name:<30} {val_acc:<10.3f} {test_acc:<10.3f} {val_auc:<10.3f} {f1:<8.3f} {target_status:<8} {train_time:<8.1f}")
            
            # Achievement summary
            print(f"\n📊 ACHIEVEMENT SUMMARY:")
            best_accuracy = max([m.get('val_accuracy', 0) for m in self.performance_metrics.values()])
            print(f"   • Best Accuracy: {best_accuracy:.1%}")
            print(f"   • Target (60%+): {'✅ ACHIEVED' if target_achieved else '❌ NOT REACHED'}")
            print(f"   • Models above 55%: {models_above_55}/{len(self.performance_metrics)}")
            
            if target_achieved:
                print(f"   🎉 SUCCESS! At least one model achieved 60%+ accuracy!")
            elif best_accuracy >= 0.55:
                print(f"   🎯 Very close! Best model: {best_accuracy:.1%} (need {0.60-best_accuracy:.1%} more)")
            else:
                print(f"   ⚠️ Need significant improvement. Gap: {0.60-best_accuracy:.1%}")
        
        # Cross-validation results
        if self.validation_results:
            cv_mean = self.validation_results['cv_mean']
            cv_std = self.validation_results['cv_std']
            cv_min = self.validation_results['cv_min']
            cv_max = self.validation_results['cv_max']
            
            print(f"\n🔄 ENHANCED CROSS-VALIDATION ({self.validation_results['cv_model']}):")
            print(f"   Mean Score: {cv_mean:.3f} ± {cv_std:.3f}")
            print(f"   Range: [{cv_min:.3f}, {cv_max:.3f}]")
            stability = 'Excellent' if cv_std < 0.03 else 'Good' if cv_std < 0.05 else 'Moderate' if cv_std < 0.08 else 'Poor'
            print(f"   Stability: {stability}")
            
            if target_type == 'classification' and cv_mean >= 0.60:
                print(f"   🎉 Cross-validation confirms 60%+ performance!")
        
        # Top features
        if self.best_model_name in self.feature_importance:
            print(f"\n🎯 TOP 25 PREDICTIVE FEATURES ({self.best_model_name}):")
            importance = self.feature_importance[self.best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25]
            
            for i, (feature, score) in enumerate(top_features, 1):
                category = self._categorize_feature(feature)
                print(f"  {i:2d}. {feature:<40} {score:.4f} [{category}]")
        
        # Recommendations
        print(f"\n💡 HIGH-PERFORMANCE RECOMMENDATIONS:")
        print(f"   • Best model for deployment: {self.best_model_name}")
        
        if target_type == 'classification':
            best_metrics = self.performance_metrics[self.best_model_name]
            best_acc = best_metrics.get('val_accuracy', 0)
            
            if best_acc >= 0.60:
                print(f"   ✅ Model achieved target accuracy - ready for production!")
                print(f"   • Use ensemble of top 3 models for maximum reliability")
                print(f"   • Implement proper risk management with position sizing")
            elif best_acc >= 0.55:
                print(f"   🎯 Model shows strong promise - consider ensemble methods")
                print(f"   • Combine top models to potentially reach 60%+")
                print(f"   • Use conservative position sizing until 60%+ achieved")
            else:
                print(f"   ⚠️ Performance below expectations - consider:")
                print(f"   • Collecting more data or different data sources")
                print(f"   • Alternative feature engineering approaches")
                print(f"   • Different market conditions or asset classes")
        
        print(f"\n🔧 TECHNICAL IMPLEMENTATION:")
        print(f"   • Advanced features: {len(self.feature_names)}")
        print(f"   • Scaling strategies: {len(self.scalers)}")
        print(f"   • Model variants: {len(self.models)}")
        print(f"   • Data leakage prevention: ✅ Strict temporal validation")
        print(f"   • Class balancing: ✅ SMOTE applied where needed")
        print(f"   • Deep learning: {'✅ Optimized LSTM' if 'lstm_model' in self.models else '❌ Not available'}")
        
        print(f"\n📁 PRODUCTION-READY FILES:")
        model_files = [f"models/{name}.pkl" for name in self.models.keys()]
        print(f"   • All models saved: {len(model_files)} files")
        print(f"   • Best model: models/final_model.pkl")
        print(f"   • Scalers: models/scalers.pkl (4 scaling strategies)")
        print(f"   • Features: models/feature_names.pkl")
        print(f"   • Results: results/model_performance.csv")
        
        print(f"\n🚀 DEPLOYMENT STATUS:")
        if target_type == 'classification':
            if best_acc >= 0.60:
                print(f"   ✅ READY FOR PRODUCTION - Target achieved!")
            elif best_acc >= 0.55:
                print(f"   🟡 READY FOR TESTING - Close to target")
            else:
                print(f"   🔴 NEEDS IMPROVEMENT - Below minimum threshold")
        
        print("="*130)
    
    def _categorize_feature(self, feature_name):
        """Enhanced feature categorization"""
        feature_lower = feature_name.lower()
        
        if any(word in feature_lower for word in ['sentiment', 'fda', 'news', 'drug']):
            return 'Sentiment/FDA'
        elif any(word in feature_lower for word in ['volume', 'vol', 'obv']):
            return 'Volume'
        elif any(word in feature_lower for word in ['price', 'close', 'return', 'momentum', 'log']):
            return 'Price/Returns'
        elif any(word in feature_lower for word in ['sma', 'ema', 'moving', 'ma_']):
            return 'Moving Averages'
        elif any(word in feature_lower for word in ['bb_', 'bollinger']):
            return 'Bollinger Bands'
        elif any(word in feature_lower for word in ['rsi', 'macd', 'atr']):
            return 'Technical Indicators'
        elif any(word in feature_lower for word in ['volatility', 'vol_', 'realized']):
            return 'Volatility'
        elif any(word in feature_lower for word in ['day', 'month', 'week', 'quarter', 'sin', 'cos']):
            return 'Temporal/Seasonal'
        elif any(word in feature_lower for word in ['support', 'resistance', 'percentile']):
            return 'Support/Resistance'
        elif 'lag' in feature_lower:
            return 'Lag Features'
        elif any(word in feature_lower for word in ['signal', 'cross', 'breakout']):
            return 'Trading Signals'
        else:
            return 'Other'
    
    def execute(self):
        """Execute high-performance training pipeline"""
        try:
            logger.info("="*80)
            logger.info("STARTING HIGH-PERFORMANCE BIOCON STOCK PREDICTION TRAINING")
            logger.info("🎯 TARGET: 60%+ ACCURACY")
            logger.info("="*80)
            
            # Load and merge data
            stock_df, sentiment_df = self.load_and_validate_data()
            combined_df = self.smart_data_merge(stock_df, sentiment_df)
            
            # Create advanced features
            df_with_features = self.create_advanced_features(combined_df)
            
            # Save enhanced data
            df_with_features.to_csv('combined_data_high_performance.csv', index=False)
            logger.info("✅ High-performance combined data saved")
            
            # Prepare optimal targets
            df_clean, target_col, target_type, target_description = self.prepare_optimal_targets(df_with_features)
            
            # Feature preparation
            exclude_cols = {
                'Date', 'Target_Next_Day_Up', 'Target_Next_3Day_Up', 
                'Target_Significant_Move', 'Target_Direction_Balanced',
                'Target_Next_Day_Return', 'Target_Dynamic_Up', 'Target_Dynamic_Down',
                'Target_Dynamic_Significant', 'Target_Quantile_Up', 'Target_Quantile_Down',
                'Target_Quantile_Extreme', 'Target_Balanced_Custom', 'Target_Next_3Day_Return'
            }
            
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            X_raw = df_clean[feature_cols].copy()
            y = df_clean[target_col].values
            
            logger.info(f"Dataset shape: {X_raw.shape}, Target: {target_col}")
            
            # Advanced feature selection
            X_selected, selected_features = self.advanced_feature_selection(X_raw, y, target_type, max_features=60)
            self.feature_names = selected_features
            
            logger.info(f"Selected {len(selected_features)} advanced features for training")
            
            # Enhanced time-series split
            n_samples = len(X_selected)
            train_end = int(n_samples * 0.7)
            val_end = int(n_samples * 0.85)
            
            X_train = X_selected.iloc[:train_end]
            X_val = X_selected.iloc[train_end:val_end]
            X_test = X_selected.iloc[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Enhanced splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Train high-performance models
            self.train_high_performance_models(X_train, X_val, X_test, y_train, y_val, y_test, target_type)
            
            # Select best model
            self.select_best_model()
            
            # Enhanced cross-validation
            self.cross_validate_best_model(X_selected, y, target_type)
            
            # Save results
            save_success = self.save_models_and_results()
            
            # Print high-performance summary
            self.print_high_performance_summary(target_description, target_type)
            
            return save_success and len(self.models) > 0
            
        except Exception as e:
            logger.error(f"High-performance training failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution for high-performance stock prediction training"""
    print("🚀 BIOCON FDA PROJECT - HIGH-PERFORMANCE STOCK PREDICTION TRAINING")
    print("🎯 MISSION: ACHIEVE 60%+ ACCURACY")
    print("="*80)
    print("🔥 ENHANCED APPROACH:")
    print("   • 100+ Advanced Features: Multi-timeframe, regime detection, interaction terms")
    print("   • Optimized Hyperparameters: Grid-searched for maximum performance")
    print("   • Multiple Scaling Strategies: Standard, Robust, MinMax, PowerTransformer")
    print("   • Class Balancing: SMOTE for optimal training distribution")
    print("   • Advanced Ensemble: Sophisticated voting and stacking methods")
    print("   • Deep Learning: Optimized LSTM with attention mechanisms")
    print("   • Enhanced Validation: 7-fold time-series cross-validation")
    print("   • Smart Target Selection: Balanced classes for maximum predictability")
    print("-" * 80)
    
    trainer = HighPerformanceBioconTrainer()
    success = trainer.execute()
    
    if success:
        # Check if target was achieved
        best_accuracy = 0
        if trainer.performance_metrics:
            best_accuracy = max([m.get('val_accuracy', 0) for m in trainer.performance_metrics.values()])
        
        print(f"\n🏁 HIGH-PERFORMANCE TRAINING COMPLETED!")
        print(f"📊 FINAL RESULTS:")
        print(f"   • Best Accuracy Achieved: {best_accuracy:.1%}")
        print(f"   • Target (60%): {'✅ ACHIEVED!' if best_accuracy >= 0.60 else '❌ Not reached'}")
        print(f"   • Models Trained: {len(trainer.models)}")
        print(f"   • Features Used: {len(trainer.feature_names)}")
        
        if best_accuracy >= 0.60:
            print(f"\n🎉 SUCCESS! TARGET ACHIEVED!")
            print(f"   ✅ Model ready for production deployment")
            print(f"   ✅ {best_accuracy:.1%} accuracy exceeds 60% target")
            print(f"   ✅ Rigorous validation confirms performance")
        elif best_accuracy >= 0.55:
            print(f"\n🎯 VERY CLOSE TO TARGET!")
            print(f"   ⚡ {best_accuracy:.1%} accuracy (need {0.60-best_accuracy:.1%} more)")
            print(f"   💡 Consider ensemble methods to reach 60%+")
            print(f"   🔄 Try different time periods or additional data")
        else:
            print(f"\n⚠️ TARGET NOT REACHED")
            print(f"   📊 Best: {best_accuracy:.1%} (need {0.60-best_accuracy:.1%} improvement)")
            print(f"   💡 Recommendations:")
            print(f"      • Try different market periods")
            print(f"      • Add more external data sources")
            print(f"      • Consider different prediction horizons")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review detailed results in results/model_performance.csv")
        print(f"   2. Test best model on live data")
        print(f"   3. Implement ensemble of top models")
        print(f"   4. Set up automated retraining pipeline")
        
    else:
        print(f"\n💥 HIGH-PERFORMANCE TRAINING FAILED!")
        print(f"Check error messages above for debugging")
    
    return success

if __name__ == "__main__":
    main()