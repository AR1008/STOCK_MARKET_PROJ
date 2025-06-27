"""
Advanced Configuration for Biocon FDA Project
Comprehensive settings for ML pipeline with FDA milestone analysis
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_NAME = "Biocon FDA Drug Journey Analysis"
PROJECT_VERSION = "2.0"
PROJECT_DESCRIPTION = "Advanced ML pipeline for stock prediction using FDA milestones and financial NLP"

# =============================================================================
# DATA COLLECTION CONFIGURATION
# =============================================================================

# Date ranges
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2025-06-28"  # Current date
FDA_FOCUS_START = "2017-01-01"  # When FDA application started

# Company and Drug Information
COMPANY_INFO = {
    'name': 'Biocon',
    'ticker': 'BIOCON.NS',
    'sector': 'Pharmaceutical',
    'subsidiary_names': ['Biocon Biologics', 'Biocon Pharma'],
    'market_cap_category': 'Large Cap',
    'exchange': 'NSE',
    'bloomberg_ticker': 'BIOS:IN',
    'reuters_ric': 'BIOS.NS'
}

DRUG_INFO = {
    'name': 'Semglee',
    'scientific_name': 'insulin glargine-yfgn',
    'full_name': 'Semglee (insulin glargine-yfgn)',
    'drug_type': 'insulin biosimilar',
    'indication': 'diabetes',
    'application_year': 2017,
    'approval_year': 2021,
    'launch_year': 2021,
    'therapeutic_area': 'Endocrinology',
    'competitor_drug': 'Lantus',
    'nda_number': 'BLA125159',
    'fda_orange_book_id': 'N021536'
}

# =============================================================================
# ADVANCED ML MODEL CONFIGURATION
# =============================================================================

# 1. News Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'models': {
        'finbert': {
            'model_name': 'ProsusAI/finbert',
            'task': 'financial-sentiment-analysis',
            'max_length': 512,
            'batch_size': 16
        },
        'distilbert': {
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'task': 'sentiment-analysis',
            'max_length': 512,
            'batch_size': 32
        },
        'vader': {
            'compound_threshold': {'positive': 0.05, 'negative': -0.05}
        },
        'textblob': {
            'polarity_threshold': {'positive': 0.1, 'negative': -0.1}
        }
    },
    'ensemble_weights': {
        'finbert': 0.4,
        'distilbert': 0.3,
        'vader': 0.2,
        'textblob': 0.1
    }
}

# 2. Time-Series Price Prediction Configuration
LSTM_CONFIG = {
    'sequence_length': 60,  # Look back window
    'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_position'],
    'architecture': {
        'layers': [
            {'type': 'LSTM', 'units': 128, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'LSTM', 'units': 64, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'LSTM', 'units': 32, 'dropout': 0.2},
            {'type': 'Dense', 'units': 16, 'activation': 'relu'},
            {'type': 'Dense', 'units': 1, 'activation': 'linear'}
        ]
    },
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss': 'mse',
    'epochs': 200,
    'batch_size': 32,
    'validation_split': 0.2
}

# 3. Classification Model Configuration
CLASSIFICATION_CONFIG = {
    'target': 'price_direction',  # Up/Down prediction
    'models': {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        }
    }
}

# 4. Multimodal Fusion Configuration
FUSION_CONFIG = {
    'architecture': 'late_fusion',  # early_fusion, late_fusion, attention_fusion
    'lstm_features_dim': 32,
    'sentiment_features_dim': 16,
    'fusion_layer': {
        'type': 'Dense',
        'units': 64,
        'activation': 'relu',
        'dropout': 0.3
    },
    'output_layer': {
        'units': 1,
        'activation': 'sigmoid'  # for binary classification
    }
}

# 5. Event Impact Modeling Configuration
CAUSAL_CONFIG = {
    'methods': {
        'difference_in_differences': {
            'treatment_window': 5,  # days before/after event
            'control_group': 'NIFTY_PHARMA'
        },
        'propensity_score_matching': {
            'caliper': 0.1,
            'matching_method': 'nearest'
        },
        'synthetic_control': {
            'donor_pool': ['DRREDDY.NS', 'CIPLA.NS', 'SUNPHARMA.NS']
        }
    }
}

# 6. Anomaly Detection Configuration
ANOMALY_CONFIG = {
    'methods': {
        'isolation_forest': {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42
        },
        'autoencoder': {
            'encoding_dim': 32,
            'epochs': 100,
            'batch_size': 32,
            'threshold_percentile': 95
        },
        'statistical': {
            'zscore_threshold': 3,
            'bollinger_std': 2
        }
    }
}

# 7. Baseline Models Configuration
BASELINE_CONFIG = {
    'arima': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 5),  # Weekly seasonality
        'auto_arima': True
    },
    'sarima': {
        'seasonal_periods': [5, 21, 252]  # Weekly, Monthly, Yearly
    }
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    'technical_indicators': {
        'trend': ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26'],
        'momentum': ['RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_10'],
        'volatility': ['BB_Upper', 'BB_Lower', 'BB_Width', 'ATR_14', 'Volatility_20'],
        'volume': ['Volume_SMA_20', 'Volume_Ratio', 'OBV', 'VWAP']
    },
    'sentiment_features': {
        'aggregation_windows': [1, 3, 7, 14, 30],
        'sentiment_metrics': ['mean', 'std', 'min', 'max', 'skew'],
        'event_flags': ['FDA_approval', 'Clinical_trial', 'Earnings', 'Partnership']
    },
    'lag_features': {
        'price_lags': [1, 2, 3, 5, 10],
        'return_lags': [1, 2, 3, 5, 10],
        'sentiment_lags': [1, 2, 3, 5]
    },
    'interaction_features': {
        'sentiment_volume': True,
        'sentiment_volatility': True,
        'event_volume': True
    }
}

# =============================================================================
# FDA MILESTONE CONFIGURATION
# =============================================================================

FDA_MILESTONES = {
    'application_phase': {
        'keywords': [
            'IND application', 'investigational new drug', 'pre-clinical',
            'FDA submission', 'regulatory filing', 'drug application', 'BLA submission',
            'pre-IND meeting', 'FDA guidance', 'regulatory pathway'
        ],
        'weight': 1.6,
        'importance': 8,
        'category': 'regulatory'
    },
    'clinical_trials': {
        'keywords': [
            'phase I trial', 'phase II trial', 'phase III trial', 'clinical trial',
            'study results', 'trial data', 'clinical endpoint', 'patient enrollment',
            'bioequivalence study', 'clinical data', 'primary endpoint', 'interim analysis'
        ],
        'weight': 1.5,
        'importance': 7,
        'category': 'clinical'
    },
    'regulatory_review': {
        'keywords': [
            'FDA review', 'regulatory review', 'FDA meeting', 'advisory committee',
            'FDA inspection', 'manufacturing inspection', 'facility inspection',
            'PDUFA date', 'FDA letter', 'complete response letter', 'CRL'
        ],
        'weight': 1.8,
        'importance': 9,
        'category': 'regulatory'
    },
    'approval_process': {
        'keywords': [
            'FDA approval', 'drug approval', 'marketing authorization', 'BLA approval',
            'NDA approval', 'biosimilar approval', 'interchangeable designation',
            'regulatory approval', 'FDA clearance', 'approval letter'
        ],
        'weight': 2.0,
        'importance': 10,
        'category': 'approval'
    },
    'post_approval': {
        'keywords': [
            'product launch', 'commercial launch', 'market launch', 'hospital adoption',
            'prescription volume', 'market penetration', 'real-world evidence',
            'post-market surveillance', 'market access', 'formulary inclusion'
        ],
        'weight': 1.3,
        'importance': 6,
        'category': 'commercial'
    },
    'regulatory_issues': {
        'keywords': [
            'FDA warning letter', 'recall', 'safety concern', 'adverse event',
            'manufacturing issue', 'quality issue', 'FDA inspection deficiency',
            'regulatory action', 'compliance issue', 'import alert'
        ],
        'weight': 1.7,
        'importance': 8,
        'category': 'risk'
    }
}

# =============================================================================
# DATA SOURCES CONFIGURATION
# =============================================================================

DATA_SOURCES = {
    'stock_data': {
        'primary': 'yfinance',
        'backup': ['alpha_vantage', 'quandl'],
        'symbols': {
            'BIOCON': ['BIOCON.NS', 'BIOCON.BO', '532523.BO'],
            'NIFTY50': ['^NSEI', 'NIFTY50.NS'],
            'NIFTY_PHARMA': ['^CNXPHARMA', 'CNXPHARMA.NS']
        }
    },
    'news_data': {
        'sources': [
            'google_news',
            'yahoo_finance_news',
            'reuters_api',
            'bloomberg_api',
            'financial_rss_feeds'
        ],
        'rss_feeds': [
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'https://www.business-standard.com/rss/markets-106.rss',
            'https://www.livemint.com/rss/companies'
        ]
    },
    'fda_data': {
        'sources': [
            'fda.gov',
            'drugs@fda',
            'orange_book',
            'purple_book'
        ]
    }
}

# =============================================================================
# MODEL EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    'metrics': {
        'regression': ['mse', 'rmse', 'mae', 'r2', 'mape'],
        'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
        'financial': ['sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio']
    },
    'cross_validation': {
        'method': 'TimeSeriesSplit',
        'n_splits': 5,
        'test_size': 0.2
    },
    'backtesting': {
        'method': 'walk_forward',
        'window_size': 252,  # 1 year
        'step_size': 21,    # 1 month
        'min_train_size': 500
    }
}

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

HPO_CONFIG = {
    'framework': 'optuna',  # optuna, hyperopt, scikit-optimize
    'n_trials': 100,
    'optimization_metric': 'val_r2',
    'direction': 'maximize',
    'pruning': True,
    'parallel_jobs': 4
}

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

DEPLOYMENT_CONFIG = {
    'model_registry': 'mlflow',
    'monitoring': 'wandb',
    'api_framework': 'fastapi',
    'containerization': 'docker',
    'cloud_platform': 'aws'
}

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

PATHS = {
    'data': Path('data'),
    'models': Path('models'),
    'results': Path('results'),
    'charts': Path('results/charts'),
    'notebooks': Path('notebooks'),
    'logs': Path('logs'),
    'temp': Path('temp'),
    'config': Path('config')
}

# Data file names
DATA_FILES = {
    'stock_data': 'stock_data.csv',
    'news_data': 'news_data.csv',
    'daily_sentiment': 'daily_sentiment.csv',
    'combined_data': 'combined_data.csv',
    'processed_features': 'processed_features.csv',
    'fda_events': 'fda_events.csv',
    'technical_indicators': 'technical_indicators.csv'
}

# Model file names
MODEL_FILES = {
    'finbert_model': 'finbert_sentiment.pkl',
    'lstm_model': 'lstm_price_prediction.h5',
    'classification_model': 'direction_classifier.pkl',
    'fusion_model': 'multimodal_fusion.h5',
    'causal_model': 'causal_inference.pkl',
    'anomaly_model': 'anomaly_detector.pkl',
    'scalers': 'feature_scalers.pkl',
    'encoders': 'categorical_encoders.pkl'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create all necessary directories"""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def get_data_path(filename):
    """Get full path for data file"""
    return PATHS['data'] / filename

def get_model_path(filename):
    """Get full path for model file"""
    return PATHS['models'] / filename

def get_results_path(filename):
    """Get full path for results file"""
    return PATHS['results'] / filename

def validate_config():
    """Validate configuration settings"""
    required_paths = ['data', 'models', 'results']
    for path_name in required_paths:
        if not PATHS[path_name].exists():
            PATHS[path_name].mkdir(parents=True, exist_ok=True)
    
    return True

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/biocon_analysis.log',
            'level': 'DEBUG',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'biocon_analysis': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    }
}

if __name__ == "__main__":
    """Test configuration"""
    print(f"🚀 {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"📊 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"💊 Drug: {DRUG_INFO['full_name']}")
    print(f"📅 Analysis Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"🔬 FDA Journey: {DRUG_INFO['application_year']} - {DRUG_INFO['launch_year']}")
    
    # Create directories
    create_directories()
    print("✅ Directories created")
    
    # Validate configuration
    validate_config()
    print("✅ Configuration validated")
    
    print("\n🤖 Advanced ML Pipeline Configured:")
    print(f"  📰 Sentiment: FinBERT + DistilBERT + VADER + TextBlob")
    print(f"  📈 Time Series: LSTM with {LSTM_CONFIG['sequence_length']} day lookback")
    print(f"  🎯 Classification: Random Forest + XGBoost + SVM")
    print(f"  🔗 Fusion: Multimodal architecture")
    print(f"  📊 Causal: Difference-in-Differences + PSM")
    print(f"  🚨 Anomaly: Isolation Forest + Autoencoder")
    print(f"  📐 Baseline: ARIMA + SARIMA")
    
    print("\n✅ Configuration loaded successfully!")