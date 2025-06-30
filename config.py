"""
Configuration file for Biocon FDA Project
Defines paths, data sources, company information, and model parameters
"""

from pathlib import Path
from datetime import datetime
import os
import logging

# Setup logging for config validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Company Information
COMPANY_INFO = {
    'name': 'Biocon Limited',
    'ticker': 'BIOCON.NS',
    'bse_code': '532523'
}

# Date Range
DATA_START_DATE = datetime(2015, 1, 1)
DATA_END_DATE = datetime(2025, 7, 1, 1, 44)  # Fixed for reproducibility

# Data Sources
DATA_SOURCES = {
    'stock_data': {
        'symbols': {
            'BIOCON': ['BIOCON.NS'],
            'NIFTY50': ['^NSEI', 'NIFTY 50'],
            'SENSEX': ['^BSESN', 'SENSEX'],
            'NIFTY_PHARMA': ['^CNXPHARMA', 'NIFTY PHARMA']
        },
        'primary_source': 'Yahoo Finance',
        'fallback_sources': ['NSE India', 'NSE India (nsepy)', 'BSE India']
    },
    'sentiment_data': {
        'source': 'news_api',
        'backup': 'web_scraping'
    },
    'fda_data': {
        'source': 'fda_gov_api',
        'backup': 'clinical_trials_gov'
    }
}

# Proxy Configuration
PROXY_CONFIG = {
    'http': os.getenv('HTTP_PROXY', None),
    'https': os.getenv('HTTPS_PROXY', None)
}

# File Paths
BASE_PATH = Path(__file__).parent
PATHS = {
    'data': BASE_PATH / 'data',
    'models': BASE_PATH / 'models',
    'results': BASE_PATH / 'results',
    'logs': BASE_PATH / 'logs'
}

# Data Files
DATA_FILES = {
    'stock_data': 'stock_data.csv',
    'technical_indicators': 'technical_indicators.csv',
    'daily_sentiment': 'daily_sentiment.csv',
    'fda_events': 'fda_events.csv'
}

# Feature Configuration
FEATURE_CONFIG = {
    'technical_indicators': {
        'sma_periods': [5, 10, 20, 50, 100, 200],
        'ema_periods': [12, 26, 50],
        'rsi_periods': [14, 21],
        'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger_period': 20,
        'atr_periods': [14, 21],
        'volume_sma_periods': [10, 20, 50]
    },
    'market_metrics': {
        'beta_windows': [60, 252],
        'correlation_window': 60,
        'volatility_windows': [10, 20, 30]
    },
    'advanced_features': {
        'lag_periods': [1, 2, 3, 5],
        'forward_horizons': [1, 3, 5, 10],
        'sentiment_lags': [1, 2, 3, 5]
    }
}

# Model Configuration
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
    'catboost': {
        'iterations': [100, 200, 300],
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    'adaboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'extra_trees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
}

def create_directories():
    """
    Create necessary directories for data, models, results, and logs
    """
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def validate_config():
    """
    Validate configuration settings
    """
    # Validate date range
    if DATA_START_DATE >= DATA_END_DATE:
        logger.error("Invalid date range: start_date must be before end_date")
        raise ValueError("Invalid date range")
    
    # Validate paths
    for path_name, path in PATHS.items():
        if not path.exists():
            logger.info(f"Creating {path_name} directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Validate company information
    if not all(key in COMPANY_INFO for key in ['name', 'ticker', 'bse_code']):
        logger.error("Incomplete company information")
        raise ValueError("Incomplete company information")
    
    # Validate data files
    for file_name, file_path in DATA_FILES.items():
        full_path = PATHS['data'] / file_path
        if not full_path.exists():
            logger.warning(f"Data file not found: {full_path}")
    
    # Skip proxy validation if PROXY_CONFIG is None
    if PROXY_CONFIG and (PROXY_CONFIG['http'] is None or PROXY_CONFIG['https'] is None):
        logger.info("ℹ️ No proxy configured, proceeding without proxy")
    
    logger.info("Configuration validated successfully")