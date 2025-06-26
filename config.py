"""
Configuration file for Biocon FDA Project
Contains all settings, API keys, and parameters
"""

import os
from datetime import datetime, timedelta

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_NAME = "Biocon FDA Drug Journey Analysis"
PROJECT_VERSION = "1.0"
PROJECT_DESCRIPTION = "Stock price prediction using FDA milestones and news sentiment"

# =============================================================================
# DATA COLLECTION CONFIGURATION
# =============================================================================

# Date ranges
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2025-06-26"  # Current date
FDA_FOCUS_START = "2017-01-01"  # When FDA application started

# Company and Drug Information
COMPANY_INFO = {
    'name': 'Biocon',
    'ticker': 'BIOCON.NS',
    'sector': 'Pharmaceutical',
    'subsidiary_names': ['Biocon Biologics', 'Biocon Pharma'],
    'market_cap_category': 'Large Cap',
    'exchange': 'NSE'
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
    'therapeutic_area': 'Endocrinology'
}

# Stock data configuration
STOCK_CONFIG = {
    'symbols': {
        'BIOCON': {
            'primary': 'BIOCON.NS',
            'alternatives': ['BIOCON.BO', '532523.BO'],
            'name': 'Biocon Limited',
            'type': 'stock'
        },
        'NIFTY50': {
            'primary': '^NSEI',
            'alternatives': ['NIFTY50.NS'],
            'name': 'Nifty 50',
            'type': 'index'
        },
        'NIFTY_PHARMA': {
            'primary': '^CNXPHARMA',
            'alternatives': ['CNXPHARMA.NS', 'NIFTYPHARMA.NS'],
            'name': 'Nifty Pharma',
            'type': 'index'
        }
    },
    'technical_indicators': {
        'moving_averages': [20, 50, 200],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2
    }
}

# News data configuration
NEWS_CONFIG = {
    'min_articles_threshold': 10,
    'sentiment_sources': ['TextBlob', 'NLTK_VADER'],
    'news_sources': [
        'Google News',
        'Yahoo Finance',
        'Financial RSS Feeds'
    ],
    'search_delay': 1.5,  # seconds between searches
    'max_articles_per_query': 100
}

# =============================================================================
# API KEYS AND CREDENTIALS (Set as environment variables)
# =============================================================================

# Yahoo Finance (usually no API key needed)
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY', '')

# News API (if using News API service)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Alpha Vantage (alternative stock data source)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

# Quandl (alternative financial data source)
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY', '')

# =============================================================================
# MODEL TRAINING CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'target_variable': 'Next_Day_Return',
    'feature_selection': {
        'method': 'SelectKBest',
        'k': 50,  # Number of top features to select
        'score_func': 'f_regression'
    },
    'train_validation_test_split': [0.6, 0.2, 0.2],  # 60% train, 20% val, 20% test
    'cross_validation': {
        'method': 'TimeSeriesSplit',
        'n_splits': 5
    },
    'random_state': 42
}

# Traditional ML Models Configuration
TRADITIONAL_MODELS = {
    'Linear_Regression': {
        'params': {}
    },
    'Ridge_Regression': {
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky']
        }
    },
    'Lasso_Regression': {
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'max_iter': [1000, 2000]
        }
    },
    'Random_Forest': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient_Boosting': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'XGBoost': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    },
    'LightGBM': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [20, 31, 50],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    'SVR': {
        'params': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }
    }
}

# LSTM Model Configuration
LSTM_CONFIG = {
    'sequence_length': 30,  # Number of days to look back
    'layers': [
        {'type': 'LSTM', 'units': 50, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'LSTM', 'units': 50, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'LSTM', 'units': 50, 'dropout': 0.2},
        {'type': 'Dense', 'units': 25},
        {'type': 'Dense', 'units': 1}
    ],
    'optimizer': {
        'type': 'Adam',
        'learning_rate': 0.001
    },
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'callbacks': {
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 10,
            'restore_best_weights': True
        },
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.2,
            'patience': 5,
            'min_lr': 0.0001
        }
    }
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

FEATURE_CONFIG = {
    'price_features': {
        'moving_averages': [5, 10, 20, 50, 200],
        'momentum_periods': [5, 10, 20],
        'volatility_periods': [5, 10, 20],
        'lag_features': [1, 2, 3, 5]
    },
    'sentiment_features': {
        'moving_averages': [3, 7, 14],
        'momentum_periods': [3, 7],
        'lag_features': [1, 2, 3]
    },
    'volume_features': {
        'moving_averages': [5, 10, 20],
        'ratios': True
    },
    'technical_indicators': {
        'rsi': True,
        'macd': True,
        'bollinger_bands': True,
        'stochastic': False,
        'williams_r': False
    },
    'date_features': {
        'day_of_week': True,
        'month': True,
        'quarter': True,
        'is_month_end': True,
        'is_quarter_end': True,
        'is_year_end': False
    }
}

# =============================================================================
# FDA MILESTONE CONFIGURATION
# =============================================================================

FDA_MILESTONES = {
    'application_phase': {
        'keywords': [
            'IND application', 'investigational new drug', 'pre-clinical',
            'FDA submission', 'regulatory filing', 'drug application', 'BLA submission'
        ],
        'weight': 1.6,
        'importance': 8
    },
    'clinical_trials': {
        'keywords': [
            'phase I trial', 'phase II trial', 'phase III trial', 'clinical trial',
            'study results', 'trial data', 'clinical endpoint', 'patient enrollment',
            'bioequivalence study', 'clinical data'
        ],
        'weight': 1.5,
        'importance': 7
    },
    'regulatory_review': {
        'keywords': [
            'FDA review', 'regulatory review', 'FDA meeting', 'advisory committee',
            'FDA inspection', 'manufacturing inspection', 'facility inspection',
            'PDUFA date', 'FDA letter'
        ],
        'weight': 1.8,
        'importance': 9
    },
    'approval_process': {
        'keywords': [
            'FDA approval', 'drug approval', 'marketing authorization', 'BLA approval',
            'NDA approval', 'biosimilar approval', 'interchangeable designation',
            'regulatory approval', 'FDA clearance'
        ],
        'weight': 2.0,
        'importance': 10
    },
    'post_approval': {
        'keywords': [
            'product launch', 'commercial launch', 'market launch', 'hospital adoption',
            'prescription volume', 'market penetration', 'real-world evidence',
            'post-market surveillance'
        ],
        'weight': 1.3,
        'importance': 6
    },
    'regulatory_issues': {
        'keywords': [
            'FDA warning letter', 'recall', 'safety concern', 'adverse event',
            'manufacturing issue', 'quality issue', 'FDA inspection deficiency',
            'regulatory action', 'compliance issue'
        ],
        'weight': 1.7,
        'importance': 8
    }
}

# =============================================================================
# MODEL EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG = {
    'metrics': [
        'mse', 'rmse', 'mae', 'r2_score', 'mean_absolute_percentage_error'
    ],
    'prediction_horizons': [1, 5, 10, 20],  # days ahead to predict
    'backtesting': {
        'method': 'walk_forward',
        'window_size': 252,  # trading days (1 year)
        'step_size': 21  # trading days (1 month)
    },
    'significance_tests': {
        'alpha': 0.05,
        'tests': ['shapiro', 'jarque_bera', 'durbin_watson']
    }
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'save_format': 'png',
    'charts_to_generate': [
        'stock_price_timeline',
        'sentiment_timeline',
        'correlation_heatmap',
        'feature_importance',
        'model_performance_comparison',
        'prediction_vs_actual',
        'residual_analysis'
    ]
}

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

PATHS = {
    'data': 'data/',
    'models': 'models/',
    'results': 'results/',
    'charts': 'results/charts/',
    'notebooks': 'notebooks/',
    'logs': 'logs/',
    'temp': 'temp/'
}

# Data file names
DATA_FILES = {
    'stock_data': 'stock_data.csv',
    'news_data': 'news_data.csv',
    'daily_sentiment': 'daily_sentiment.csv',
    'combined_data': 'combined_data.csv',
    'processed_features': 'processed_features.csv'
}

# Model file names
MODEL_FILES = {
    'final_model': 'final_model.pkl',
    'lstm_model': 'lstm_model.h5',
    'sentiment_model': 'sentiment_model.pkl',
    'scalers': 'scalers.pkl',
    'feature_names': 'feature_names.pkl'
}

# Results file names
RESULTS_FILES = {
    'model_performance': 'model_performance.csv',
    'predictions': 'predictions.csv',
    'correlation_analysis': 'correlation_analysis.csv',
    'feature_importance': 'feature_importance.csv',
    'backtesting_results': 'backtesting_results.csv'
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': {
            'filename': 'logs/biocon_analysis.log',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        },
        'console': {
            'stream': 'ext://sys.stdout'
        }
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_path(filename):
    """Get full path for data file"""
    return os.path.join(PATHS['data'], filename)

def get_model_path(filename):
    """Get full path for model file"""
    return os.path.join(PATHS['models'], filename)

def get_results_path(filename):
    """Get full path for results file"""
    return os.path.join(PATHS['results'], filename)

def get_charts_path(filename):
    """Get full path for charts file"""
    return os.path.join(PATHS['charts'], filename)

def create_directories():
    """Create all necessary directories"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = []  # Add any required environment variables
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
    
    return True

# =============================================================================
# MARKET CALENDAR CONFIGURATION
# =============================================================================

MARKET_CONFIG = {
    'trading_hours': {
        'nse': {
            'open': '09:15',
            'close': '15:30',
            'timezone': 'Asia/Kolkata'
        }
    },
    'holidays': {
        # Indian stock market holidays (add as needed)
        'nse': [
            '2024-01-26',  # Republic Day
            '2024-03-08',  # Holi
            '2024-03-29',  # Good Friday
            '2024-04-11',  # Ram Navami
            '2024-05-01',  # Labour Day
            '2024-08-15',  # Independence Day
            '2024-10-02',  # Gandhi Jayanti
            '2024-11-01',  # Diwali
            '2024-11-15',  # Guru Nanak Jayanti
        ]
    }
}

# =============================================================================
# ALERT AND NOTIFICATION CONFIGURATION
# =============================================================================

ALERTS_CONFIG = {
    'thresholds': {
        'high_sentiment_change': 0.5,
        'high_price_change': 0.05,  # 5%
        'high_volume_change': 2.0,  # 2x average
        'fda_milestone_detected': True
    },
    'notification_methods': {
        'email': False,
        'slack': False,
        'console': True,
        'log_file': True
    }
}

if __name__ == "__main__":
    """Test configuration"""
    print(f"Project: {PROJECT_NAME}")
    print(f"Version: {PROJECT_VERSION}")
    print(f"Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"Drug: {DRUG_INFO['full_name']}")
    print(f"Data Period: {DATA_START_DATE} to {DATA_END_DATE}")
    
    # Create directories
    create_directories()
    print("✓ Directories created")
    
    # Validate environment (if needed)
    try:
        validate_environment()
        print("✓ Environment validated")
    except EnvironmentError as e:
        print(f"⚠️  Environment warning: {e}")
    
    print("Configuration loaded successfully!")