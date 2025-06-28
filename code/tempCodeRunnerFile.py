"""
Advanced Model Training for Biocon FDA Project
Day 2 - Step 3: Train multiple machine learning models for stock price prediction

Features:
- Combines stock data with news sentiment and FDA events
- Multiple model architectures (Linear Regression, Random Forest, Gradient Boosting, LSTM, BERT)
- Comprehensive feature selection and preprocessing
- Cross-validation and hyperparameter tuning
- Model performance evaluation and visualization
- Ensemble model creation
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
    MODEL_CONFIG = {}  # Fallback to empty dict if MODEL_CONFIG is not defined
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
    """
    Advanced model trainer for stock price prediction with multiple algorithms
    """
    
    def __init__(self):
        self.start_date = DATA_START_DATE
        self.end_date = DATA_END_DATE
        self.data =None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}
        self.models = {}
        self.feature_names = []
        
        create_directories()
        validate_config()
        
        logger.info("🚀 Advanced Model Trainer Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"🏢 Target Company: {COMPANY_INFO['name']}")

    def load_and_prepare_data(self):
        """
        Load and combine stock, technical indicators, sentiment, and benchmark data
        """
        logger.info("📊 Loading and preparing data...")
        
        try:
            # Load stock data
            stock_file = PATHS['data'] / DATA_FILES['stock_data']
            stock_df = pd.read_csv(stock_file)
            stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)  # Remove timezone
            
            # Load daily sentiment data
            sentiment_file = PATHS['data'] / DATA_FILES['daily_sentiment']
            sentiment_df = pd.read_csv(sentiment_file)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)  # Remove timezone
            
            # Load FDA events
            fda_file = PATHS['data'] / DATA_FILES['fda_events']
            if fda_file.exists():
                fda_df = pd.read_csv(fda_file)
                fda_df['date'] = pd.to_datetime(fda_df['date']).dt.tz_localize(None)  # Remove timezone
            else:
                fda_df = pd.DataFrame()
            
            # Load benchmark indices
            nifty_50_file = PATHS['data'] / DATA_FILES['nifty_50']
            nifty_pharma_file = PATHS['data'] / DATA_FILES['nifty_pharma']
            if nifty_50_file.exists():
                nifty_50_df = pd.read_csv(nifty_50_file)
                nifty_50_df['Date'] = pd.to_datetime(nifty_50_df['Date']).dt.tz_localize(None)
            else:
                nifty_50_df = pd.DataFrame()
            if nifty_pharma_file.exists():
                nifty_pharma_df = pd.read_csv(nifty_pharma_file)
                nifty_pharma_df['Date'] = pd.to_datetime(nifty_pharma_df['Date']).dt.tz_localize(None)
            else:
                nifty_pharma_df = pd.DataFrame()
            
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
            if not nifty_50_df.empty:
                df = pd.merge(df, nifty_50_df.add_prefix('Nifty50_'), left_on='Date', right_on='Nifty50_Date', how='left')
                df = df.drop(columns=['Nifty50_Date'], errors='ignore')
            if not nifty_pharma_df.empty:
                df = pd.merge(df, nifty_pharma_df.add_prefix('NiftyPharma_'), left_on='Date', right_on='NiftyPharma_Date', how='left')
                df = df.drop(columns=['NiftyPharma_Date'], errors='ignore')
            
            # Clip outliers in target variable
            target_col = 'Forward_Return_5D'
            if target_col in df.columns:
                q1, q3 = df[target_col].quantile([0.01, 0.99])
                df[target_col] = df[target_col].clip(q1, q3)
            
            # Filter date range to match config
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
            target_col = 'Forward_Return_5D'  # Predicting 5-day forward return
            
            self.data = df
            self.feature_names = feature_cols
            
            # Prepare X and y
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
            
            # Feature selection using Random Forest
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = importances.nlargest(50).index  # Select top 50 features
            X = X[top_features]
            self.feature_names = top_features.tolist()
            logger.info(f"Selected Features: {self.feature_names}")
            
            # Split data as per POA: 70% train (2015-2021), 15% validation (2022-2023), 15% test (2024-2025)
            train_end = pd.to_datetime('2021-12-31')
            val_end = pd.to_datetime('2023-12-31')
            train_idx = df['Date'] <= train_end
            val_idx = (df['Date'] > train_end) & (df['Date'] <= val_end)
            test_idx = df['Date'] > val_end
            
            self.X_train = X[train_idx]
            self.y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            self.X_test = X[test_idx]
            self.y_test = y[test_idx]
            
            # Combine train and validation for final training
            self.X_train = pd.concat([self.X_train, X_val])
            self.y_train = pd.concat([self.y_train, y_val])
            
            # Scale features
            self.scalers['X'] = StandardScaler()
            self.X_train = self.scalers['X'].fit_transform(self.X_train)
            self.X_test = self.scalers['X'].transform(self.X_test)
            
            # Scale target variable
            self.scalers['y'] = StandardScaler()
            self.y_train = self.scalers['y'].fit_transform(self.y_train.values.reshape(-1, 1)).flatten()
            self.y_test = self.scalers['y'].transform(self.y_test.values.reshape(-1, 1)).flatten()
            
            logger.info(f"✅ Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
            logger.info(f"📈 Features selected: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {str(e)}")
            raise

    def train_linear_models(self):
        """
        Train linear regression models with regularization
        """
        logger.info("🧠 Training linear models...")
        
        try:
            # Linear Regression
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            self.models['linear_regression'] = lr
            logger.info("✅ Linear Regression trained")
            
            # Ridge Regression
            ridge_params = MODEL_CONFIG.get('ridge_regression', {'alpha': [0.1, 1.0, 10.0, 100.0]})
            ridge = GridSearchCV(Ridge(), ridge_params, cv=TimeSeriesSplit(n_splits=5))
            ridge.fit(self.X_train, self.y_train)
            self.models['ridge_regression'] = ridge.best_estimator_
            logger.info(f"✅ Ridge Regression trained (best alpha: {ridge.best_params_['alpha']})")
            
            # Lasso Regression
            lasso_params = MODEL_CONFIG.get('lasso_regression', {'alpha': [0.01, 0.1, 1.0, 10.0]})
            lasso = GridSearchCV(Lasso(), lasso_params, cv=TimeSeriesSplit(n_splits=5))
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
            # Random Forest
            rf_params = MODEL_CONFIG.get('random_forest', {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            })
            rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=TimeSeriesSplit(n_splits=5))
            rf.fit(self.X_train, self.y_train)
            self.models['random_forest'] = rf.best_estimator_
            logger.info(f"✅ Random Forest trained (best params: {rf.best_params_})")
            
            # Gradient Boosting
            gb_params = MODEL_CONFIG.get('gradient_boosting', {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            })
            gb = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=TimeSeriesSplit(n_splits=5))
            gb.fit(self.X_train, self.y_train)
            self.models['gradient_boosting'] = gb.best_estimator_
            logger.info(f"✅ Gradient Boosting trained (best params: {gb.best_params_})")
            
            # LightGBM
            lgb_params = MODEL_CONFIG.get('lightgbm', {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50],
                'min_child_samples': [10],
                'min_split_gain': [0.0]
            })
            lgb_model = GridSearchCV(lgb.LGBMRegressor(random_state=42, force_col_wise=True), lgb_params, cv=TimeSeriesSplit(n_splits=5))
            lgb_model.fit(self.X_train, self.y_train)
            self.models['lightgbm'] = lgb_model.best_estimator_
            logger.info(f"✅ LightGBM trained (best params: {lgb_model.best_params_})")
            
            # XGBoost
            xgb_params = MODEL_CONFIG.get('xgboost', {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            })
            xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=TimeSeriesSplit(n_splits=5))
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
            svr_params = MODEL_CONFIG.get('svr', {
                'kernel': ['rbf'],
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1]
            })
            svr = GridSearchCV(SVR(), svr_params, cv=TimeSeriesSplit(n_splits=5))
            svr.fit(self.X_train, self.y_train)
            self.models['svr'] = svr.best_estimator_
            logger.info(f"✅ SVR trained (best params: {svr.best_params_})")
            
        except Exception as e:
            logger.error(f"❌ Error training SVR model: {str(e)}")
            raise

    def train_lstm(self):
        """
        Train LSTM model for time series prediction
        """
        logger.info("🧠 Training LSTM model...")
        
        try:
            lstm_params = MODEL_CONFIG.get('lstm', {
                'epochs': 50,
                'batch_size': 32,
                'units': [50, 25]
            })
            X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
            X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
            
            model = Sequential([
                LSTM(lstm_params['units'][0], activation='relu', input_shape=(1, self.X_train.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(lstm_params['units'][1], activation='relu'),
                Dropout(0.2),
                Dense(lstm_params['units'][1] // 2, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_reshaped, self.y_train, epochs=lstm_params['epochs'], batch_size=lstm_params['batch_size'], verbose=0)
            self.models['lstm'] = model
            logger.info("✅ LSTM model trained")
            
        except Exception as e:
            logger.error(f"❌ Error training LSTM model: {str(e)}")
            raise

    def train_bert_sentiment(self):
        """
        Placeholder for BERT sentiment model (assumes sentiment scores in daily_sentiment.csv)
        """
        logger.info("🧠 Note: BERT sentiment model assumes precomputed scores in daily_sentiment.csv")
        # Implementation of BERT model requires Hugging Face Transformers and is handled in 2_collect_news_data.py
        # This method is a placeholder to align with POA requirements

    def create_ensemble(self):
        """
        Create ensemble predictions using weighted average of model predictions
        """
        logger.info("🤝 Creating ensemble model...")
        
        try:
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                if name == 'lstm':
                    X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
                    predictions[name] = model.predict(X_test_reshaped).flatten()
                else:
                    predictions[name] = model.predict(self.X_test)
            
            # Weighted ensemble (downweight linear_regression due to potential poor performance)
            weights = {
                'linear_regression': 0.05,  # Reduced weight
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
            
            # Inverse transform ensemble predictions to original scale
            ensemble_predictions = self.scalers['y'].inverse_transform(ensemble_predictions.reshape(-1, 1)).flatten()
            
            self.models['ensemble'] = {'predictions': ensemble_predictions, 'weights': weights}
            logger.info("✅ Ensemble model created")
            
        except Exception as e:
            logger.error(f"❌ Error creating ensemble model: {str(e)}")
            raise

    def evaluate_models(self):
        """
        Evaluate all models and generate performance metrics
        """
        logger.info("📈 Evaluating model performance...")
        
        try:
            performance_metrics = {}
            predictions_df = pd.DataFrame()
            
            # Inverse transform y_test for evaluation
            y_test_orig = self.scalers['y'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
            
            for name, model in self.models.items():
                if name == 'ensemble':
                    y_pred = model['predictions']  # Already inverse transformed
                elif name == 'lstm':
                    X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
                    y_pred = model.predict(X_test_reshaped).flatten()
                    y_pred = self.scalers['y'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
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
                
                predictions_df[name] = y_pred
            
            # Save performance metrics
            metrics_df = pd.DataFrame(performance_metrics).T
            metrics_df.to_csv(PATHS['results'] / 'model_performance.csv')
            
            # Save predictions
            predictions_df['Actual'] = y_test_orig
            predictions_df['Date'] = self.data.iloc[-len(self.y_test):]['Date']
            predictions_df.to_csv(PATHS['results'] / 'predictions.csv')
            
            # Generate visualization
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
                if name != 'ensemble':
                    with open(PATHS['models'] / f"{name}_model.pkl", 'wb') as f:
                        pickle.dump(model, f)
            
            # Save ensemble weights separately
            if 'ensemble' in self.models:
                with open(PATHS['models'] / 'ensemble_model.pkl', 'wb') as f:
                    pickle.dump(self.models['ensemble']['weights'], f)
            
            # Save feature names
            with open(PATHS['models'] / 'feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            # Save scalers
            with open(PATHS['models'] / 'scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save model metadata
            metadata = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_count': len(self.feature_names),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
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
        
        print(f"\n📁 Output Files:")
        print(f"   ✅ models/*_model.pkl - Trained model files")
        print(f"   ✅ models/feature_names.pkl - Feature names")
        print(f"   ✅ models/scalers.pkl - Data scalers")
        print(f"   ✅ models/model_metadata.pkl - Model metadata")
        print(f"   ✅ results/model_performance.csv - Performance metrics")
        print(f"   ✅ results/predictions.csv - Model predictions")
        print(f"   ✅ results/charts/prediction_vs_actual.png - Visualization")
        
        print(f"\n🎉 Ready for Day 2 Step 4: Model Testing")
        print("="*80)

    def execute(self):
        """
        Execute the complete model training pipeline
        """
        try:
            logger.info("🚀 Starting Advanced Model Training Pipeline...")
            
            # Step 1: Prepare data
            self.load_and_prepare_data()
            
            # Step 2: Train models
            self.train_linear_models()
            self.train_tree_models()
            self.train_svr()
            self.train_lstm()
            self.train_bert_sentiment()  # Placeholder for BERT
            self.create_ensemble()
            
            # Step 3: Evaluate models
            performance_metrics = self.evaluate_models()
            
            # Step 4: Save models
            self.save_models()
            
            # Step 5: Print summary
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
    
    # Initialize and run trainer
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