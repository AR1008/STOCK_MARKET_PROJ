import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticBioconModelTrainer:
    """
    FINAL FIX: Realistic Biocon Stock Prediction Training
    Focus on achievable targets: direction prediction and significant moves
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = []
        self.combined_df_final = None
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'results', 'results/charts', 'data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_and_validate_data(self):
        """Load and validate data"""
        logger.info("Loading and validating data...")
        
        try:
            # Load stock data
            stock_df = pd.read_csv('data/stock_data.csv')
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)
            logger.info(f"✓ Loaded stock data: {len(stock_df)} records")
            
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
        """Smart data merge preserving all stock trading days"""
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
            
            logger.info(f"Smart merge completed: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in smart merge: {str(e)}")
            raise
    
    def create_realistic_features(self, df):
        """Create realistic features focused on predictable patterns"""
        logger.info("Creating realistic features focused on predictable patterns...")
        
        try:
            df = df.sort_values('Date').reset_index(drop=True)
            
            # === CORE PRICE FEATURES ===
            if 'Close' in df.columns:
                # Returns with different horizons
                df['Return_1D'] = df['Close'].pct_change()
                df['Return_3D'] = df['Close'].pct_change(3)
                df['Return_5D'] = df['Close'].pct_change(5)
                
                # Moving averages
                for window in [5, 10, 20]:
                    df[f'MA_{window}'] = df['Close'].rolling(window).mean()
                    df[f'Price_Above_MA_{window}'] = (df['Close'] > df[f'MA_{window}']).astype(int)
                
                # Volatility (key for stock prediction)
                df['Volatility_5D'] = df['Return_1D'].rolling(5).std() * np.sqrt(252)
                df['Volatility_20D'] = df['Return_1D'].rolling(20).std() * np.sqrt(252)
                
                # Price momentum
                df['Momentum_5D'] = df['Return_5D']
                df['Momentum_20D'] = df['Close'].pct_change(20)
            
            # === VOLUME FEATURES ===
            if 'Volume' in df.columns:
                df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
                df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA_10'] + 1)
                df['High_Volume'] = (df['Volume_Ratio'] > 2.0).astype(int)
            
            # === SENTIMENT FEATURES (KEY FOR FDA EVENTS) ===
            if 'avg_sentiment' in df.columns:
                # Clean sentiment
                df['avg_sentiment'] = np.clip(df['avg_sentiment'], -1, 1)
                
                # Sentiment signals
                df['Positive_Sentiment'] = (df['avg_sentiment'] > 0.1).astype(int)
                df['Negative_Sentiment'] = (df['avg_sentiment'] < -0.1).astype(int)
                df['Strong_Sentiment'] = (np.abs(df['avg_sentiment']) > 0.5).astype(int)
                
                # Sentiment momentum
                df['Sentiment_Change'] = df['avg_sentiment'].diff()
                df['Sentiment_MA_3'] = df['avg_sentiment'].rolling(3).mean()
            
            # === FDA EVENT FEATURES (CORE VALUE) ===
            if 'day_importance_score' in df.columns:
                # FDA event flags
                df['Major_FDA_Event'] = (df['day_importance_score'] > 25).astype(int)
                df['Minor_FDA_Event'] = ((df['day_importance_score'] > 10) & (df['day_importance_score'] <= 25)).astype(int)
                
                # Days since FDA event
                fda_events = df['day_importance_score'] > 15
                df['Days_Since_FDA'] = 0
                
                days_counter = 0
                for i in range(len(df)):
                    if fda_events.iloc[i]:
                        days_counter = 0
                    else:
                        days_counter += 1
                    df.loc[i, 'Days_Since_FDA'] = min(days_counter, 30)  # Cap at 30
            
            # === MARKET TIMING FEATURES ===
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Is_Monday'] = (df['Day_of_Week'] == 0).astype(int)
            df['Is_Friday'] = (df['Day_of_Week'] == 4).astype(int)
            df['Month'] = df['Date'].dt.month
            df['Is_Earnings_Month'] = df['Month'].isin([1, 4, 7, 10]).astype(int)
            
            # === TECHNICAL INDICATORS (IF AVAILABLE) ===
            technical_cols = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower']
            for col in technical_cols:
                if col in df.columns:
                    if col == 'RSI':
                        df['RSI_Oversold'] = (df[col] < 30).astype(int)
                        df['RSI_Overbought'] = (df[col] > 70).astype(int)
                    elif col == 'MACD':
                        df['MACD_Positive'] = (df[col] > 0).astype(int)
            
            # === INTERACTION FEATURES ===
            # Sentiment + Volume (strong signal)
            if 'Strong_Sentiment' in df.columns and 'High_Volume' in df.columns:
                df['Sentiment_Volume_Signal'] = df['Strong_Sentiment'] * df['High_Volume']
            
            # FDA + Sentiment
            if 'Major_FDA_Event' in df.columns and 'avg_sentiment' in df.columns:
                df['FDA_Sentiment_Signal'] = df['Major_FDA_Event'] * df['avg_sentiment']
            
            # === REALISTIC TARGETS ===
            if 'Close' in df.columns:
                # Direction prediction (classification) - much more achievable
                df['Next_Day_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                df['Next_3Day_Up'] = (df['Close'].shift(-3) > df['Close']).astype(int)
                
                # Significant move prediction (classification)
                returns = df['Close'].pct_change().shift(-1)
                threshold = returns.std() * 0.5  # Half standard deviation
                df['Next_Day_Significant_Up'] = (returns > threshold).astype(int)
                df['Next_Day_Significant_Down'] = (returns < -threshold).astype(int)
                df['Next_Day_Significant_Move'] = ((np.abs(returns) > threshold)).astype(int)
                
                # Return magnitude for significant moves only
                df['Next_Day_Return_Clean'] = np.where(
                    np.abs(returns) > threshold, returns, 0
                )
            
            logger.info(f"Realistic feature engineering completed. Total features: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error in realistic feature engineering: {str(e)}")
            raise
    
    def select_predictive_features(self, X, y, target_type, max_features=25):
        """Select most predictive features for the specific target"""
        logger.info(f"Selecting predictive features for {target_type}...")
        
        try:
            # Remove low-quality features
            good_features = []
            for col in X.columns:
                missing_pct = X[col].isnull().sum() / len(X)
                unique_count = X[col].nunique()
                
                if missing_pct < 0.5 and unique_count > 1:
                    good_features.append(col)
            
            X_filtered = X[good_features].copy()
            
            # Fill missing values
            for col in X_filtered.columns:
                if X_filtered[col].dtype in ['int64', 'float64']:
                    X_filtered[col] = X_filtered[col].fillna(X_filtered[col].median())
                else:
                    X_filtered[col] = X_filtered[col].fillna(0)
            
            # Feature selection based on target type
            if target_type == 'classification':
                if len(np.unique(y)) > 1:  # Check if we have both classes
                    selector = SelectKBest(score_func=f_classif, k=min(max_features, len(good_features)))
                    X_selected = selector.fit_transform(X_filtered, y)
                    selected_mask = selector.get_support()
                    selected_features = [feat for feat, selected in zip(good_features, selected_mask) if selected]
                    
                    # Log top features
                    feature_scores = dict(zip(good_features, selector.scores_))
                    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info(f"Top features: {[f[0] for f in top_features]}")
                    
                    return X_filtered[selected_features], selected_features
                else:
                    logger.warning("Only one class in target, using all features")
                    return X_filtered, good_features
            else:
                # For regression, use correlation-based selection
                correlations = []
                for col in good_features:
                    try:
                        corr = np.corrcoef(X_filtered[col], y)[0, 1]
                        correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
                    except:
                        correlations.append((col, 0))
                
                # Sort by correlation and take top features
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected_features = [feat for feat, _ in correlations[:max_features]]
                
                logger.info(f"Top correlated features: {[f for f, _ in correlations[:10]]}")
                return X_filtered[selected_features], selected_features
                
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return X, list(X.columns)
    
    def prepare_realistic_targets(self, df):
        """Prepare realistic, achievable targets"""
        logger.info("Preparing realistic, achievable targets...")
        
        try:
            # Available target options (in order of preference)
            target_options = [
                ('Next_Day_Up', 'classification', 'Direction (1-day)'),
                ('Next_Day_Significant_Move', 'classification', 'Significant Move'),
                ('Next_3Day_Up', 'classification', 'Direction (3-day)'),
                ('Next_Day_Significant_Up', 'classification', 'Significant Up Move'),
            ]
            
            # Find the best available target
            for target_col, target_type, description in target_options:
                if target_col in df.columns:
                    df_clean = df.dropna(subset=[target_col]).copy()
                    
                    if len(df_clean) > 500:  # Need sufficient data
                        y = df_clean[target_col].values
                        
                        # Check class balance for classification
                        if target_type == 'classification':
                            class_counts = np.bincount(y.astype(int))
                            minority_class_pct = min(class_counts) / sum(class_counts) * 100
                            
                            if minority_class_pct >= 20:  # At least 20% minority class
                                logger.info(f"Selected target: {target_col} ({description})")
                                logger.info(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
                                logger.info(f"Minority class: {minority_class_pct:.1f}%")
                                
                                return df_clean, target_col, target_type, description
            
            # Fallback: create a balanced direction target
            logger.warning("Creating fallback balanced direction target...")
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().shift(-1)
                # Use median split for balanced classes
                median_return = returns.median()
                df['Next_Day_Above_Median'] = (returns > median_return).astype(int)
                
                df_clean = df.dropna(subset=['Next_Day_Above_Median']).copy()
                return df_clean, 'Next_Day_Above_Median', 'classification', 'Above Median Return'
            
            raise ValueError("Cannot create any suitable target variable")
            
        except Exception as e:
            logger.error(f"Error preparing targets: {str(e)}")
            raise
    
    def train_focused_models(self, X_train, X_val, X_test, y_train, y_val, y_test, target_type):
        """Train focused models for the specific target type"""
        logger.info(f"Training focused models for {target_type}...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        if target_type == 'classification':
            models_config = {
                'Logistic_Regression': {
                    'model': LogisticRegression(
                        random_state=42, max_iter=1000, 
                        class_weight='balanced', C=1.0
                    ),
                    'use_scaled': True
                },
                'Random_Forest': {
                    'model': RandomForestClassifier(
                        n_estimators=100, max_depth=8, random_state=42,
                        class_weight='balanced', n_jobs=-1,
                        min_samples_split=20, min_samples_leaf=10
                    ),
                    'use_scaled': False
                },
                'LightGBM': {
                    'model': lgb.LGBMClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=42, n_jobs=-1, verbose=-1,
                        class_weight='balanced', min_child_samples=20
                    ),
                    'use_scaled': False
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=42, n_jobs=-1,
                        scale_pos_weight=1  # Will adjust if needed
                    ),
                    'use_scaled': False
                }
            }
        else:
            # Regression models (if needed)
            models_config = {
                'ElasticNet': {
                    'model': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
                    'use_scaled': True
                },
                'Random_Forest': {
                    'model': RandomForestRegressor(
                        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
                    ),
                    'use_scaled': False
                }
            }
        
        # Train models
        for model_name, config in models_config.items():
            try:
                logger.info(f"Training {model_name}...")
                
                model = config['model']
                use_scaled = config['use_scaled']
                
                # Select data
                if use_scaled:
                    X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
                else:
                    X_tr, X_v, X_te = X_train.values, X_val.values, X_test.values
                
                # Train
                model.fit(X_tr, y_train)
                
                # Predict
                y_val_pred = model.predict(X_v)
                y_test_pred = model.predict(X_te)
                
                # Calculate metrics
                if target_type == 'classification':
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    
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
                    
                    metrics = {
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'val_auc': val_auc,
                        'test_auc': test_auc,
                        'primary_metric': val_accuracy
                    }
                    
                    logger.info(f"✓ {model_name} - Accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}")
                else:
                    val_r2 = r2_score(y_val, y_val_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    val_mse = mean_squared_error(y_val, y_val_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    
                    metrics = {
                        'val_r2': val_r2,
                        'test_r2': test_r2,
                        'val_rmse': np.sqrt(val_mse),
                        'test_rmse': np.sqrt(test_mse),
                        'primary_metric': val_r2
                    }
                    
                    logger.info(f"✓ {model_name} - R²: {val_r2:.3f}")
                
                # Store results
                self.models[model_name] = model
                self.performance_metrics[model_name] = metrics
                
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
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully trained {len(self.models)} models")
    
    def select_best_model(self):
        """Select best model"""
        if not self.performance_metrics:
            return
        
        best_score = -float('inf')
        best_model_name = None
        
        for model_name, metrics in self.performance_metrics.items():
            score = metrics.get('primary_metric', -float('inf'))
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name)
        
        logger.info(f"Best model: {best_model_name} (Score: {best_score:.3f})")
    
    def save_models_and_results(self):
        """Save models and results"""
        try:
            # Save models
            for model_name, model in self.models.items():
                with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save best model
            if self.best_model:
                with open('models/final_model.pkl', 'wb') as f:
                    pickle.dump(self.best_model, f)
            
            # Save scalers and metadata
            with open('models/scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            with open('models/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            # Save performance
            pd.DataFrame(self.performance_metrics).T.to_csv('results/model_performance.csv')
            
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
    
    def print_realistic_summary(self, target_description, target_type):
        """Print realistic training summary"""
        print("\n" + "="*90)
        print("REALISTIC BIOCON MODEL TRAINING SUMMARY (FINAL FIX)")
        print("="*90)
        
        print(f"🎯 TARGET: {target_description} ({target_type})")
        print(f"✅ REALISTIC APPROACH:")
        print(f"   • Focus on achievable prediction tasks")
        print(f"   • Balanced classes for classification")
        print(f"   • Domain-specific feature engineering")
        print(f"   • Time-series aware validation")
        
        if not self.performance_metrics:
            print("❌ No models trained successfully")
            return
        
        print(f"\n📊 MODEL PERFORMANCE:")
        if target_type == 'classification':
            print(f"{'Model':<20} {'Accuracy':<12} {'AUC':<12} {'Primary':<12}")
            print("-" * 60)
            
            sorted_models = sorted(
                self.performance_metrics.items(),
                key=lambda x: x[1].get('primary_metric', 0),
                reverse=True
            )
            
            for model_name, metrics in sorted_models:
                acc = metrics.get('val_accuracy', 0)
                auc = metrics.get('val_auc', 0.5)
                primary = metrics.get('primary_metric', 0)
                print(f"{model_name:<20} {acc:<12.3f} {auc:<12.3f} {primary:<12.3f}")
        else:
            print(f"{'Model':<20} {'R²':<12} {'RMSE':<12} {'Primary':<12}")
            print("-" * 60)
            
            for model_name, metrics in self.performance_metrics.items():
                r2 = metrics.get('val_r2', 0)
                rmse = metrics.get('val_rmse', 0)
                primary = metrics.get('primary_metric', 0)
                print(f"{model_name:<20} {r2:<12.3f} {rmse:<12.6f} {primary:<12.3f}")
        
        # Top features
        if self.best_model_name in self.feature_importance:
            print(f"\n🎯 TOP 10 PREDICTIVE FEATURES ({self.best_model_name}):")
            importance = self.feature_importance[self.best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"  {i:2d}. {feature:<30} {score:.4f}")
        
        print(f"\n🚀 READY FOR DAY 3 TESTING!")
        print("="*90)
    
    def execute(self):
        """Execute realistic training pipeline"""
        try:
            logger.info("="*60)
            logger.info("STARTING REALISTIC BIOCON TRAINING (FINAL FIX)")
            logger.info("="*60)
            
            # Load data
            stock_df, sentiment_df = self.load_and_validate_data()
            
            # Merge data
            combined_df = self.smart_data_merge(stock_df, sentiment_df)
            
            # Create realistic features
            df_with_features = self.create_realistic_features(combined_df)
            
            # Save combined data
            df_with_features.to_csv('data/combined_data.csv', index=False)
            logger.info("✅ Combined data saved")
            
            # Prepare realistic targets
            df_clean, target_col, target_type, target_description = self.prepare_realistic_targets(df_with_features)
            
            # Prepare features and target
            exclude_cols = {
                'Date', 'Next_Day_Up', 'Next_3Day_Up', 'Next_Day_Significant_Up',
                'Next_Day_Significant_Down', 'Next_Day_Significant_Move',
                'Next_Day_Above_Median', 'Next_Day_Return_Clean'
            }
            
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            X_raw = df_clean[feature_cols].copy()
            y = df_clean[target_col].values
            
            # Feature selection
            X_selected, selected_features = self.select_predictive_features(X_raw, y, target_type)
            self.feature_names = selected_features
            
            # Time-series split
            n_samples = len(X_selected)
            train_end = int(n_samples * 0.7)
            val_end = int(n_samples * 0.85)
            
            X_train = X_selected.iloc[:train_end]
            X_val = X_selected.iloc[train_end:val_end]
            X_test = X_selected.iloc[val_end:]
            
            y_train = y[:train_end]
            y_val = y[train_end:val_end]
            y_test = y[val_end:]
            
            logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Train focused models
            self.train_focused_models(X_train, X_val, X_test, y_train, y_val, y_test, target_type)
            
            # Select best model
            self.select_best_model()
            
            # Save results
            save_success = self.save_models_and_results()
            
            # Print summary
            self.print_realistic_summary(target_description, target_type)
            
            return save_success and len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Realistic training failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution for realistic training"""
    print("🚀 BIOCON FDA PROJECT - FINAL FIX: REALISTIC MODEL TRAINING")
    print("🎯 REALISTIC APPROACH:")
    print("   • Focus on direction prediction (classification)")
    print("   • Balanced classes for better learning")
    print("   • FDA event-focused features")
    print("   • Achievable accuracy targets (>55%)")
    print("   • Proper time-series validation")
    print("-" * 80)
    
    trainer = RealisticBioconModelTrainer()
    success = trainer.execute()
    
    if success:
        print("\n🎉 REALISTIC TRAINING SUCCESS!")
        print("✅ Models trained on achievable prediction task")
        print("✅ Should see accuracy > 55% (better than random)")
        print("✅ FDA event features properly incorporated")
        print("✅ Ready for realistic Day 3 testing")
        print("\n🚀 NEXT: python code/4_test_model.py")
    else:
        print("\n💥 TRAINING FAILED!")
        print("Check error messages above")
    
    return success

if __name__ == "__main__":
    main()