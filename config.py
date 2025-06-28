"""
Configuration file for Biocon FDA Project
Defines constants and utility functions for data collection and model training
"""

from pathlib import Path
from datetime import datetime

# Company Information
COMPANY_INFO = {
    'name': 'Biocon',
    'ticker': 'BIOCON.NS'  # Yahoo Finance ticker for Biocon Ltd.
}

# Date Range for Data Collection
DATA_START_DATE = '2015-01-01'  # Updated to match POA requirement
DATA_END_DATE = '2025-06-28'    # Updated to match user clarification

# Directory Paths
BASE_DIR = Path(__file__).parent
PATHS = {
    'data': BASE_DIR / 'data',
    'models': BASE_DIR / 'models',
    'results': BASE_DIR / 'results' / 'charts',
    'logs': BASE_DIR / 'logs'
}

# Data Files
DATA_FILES = {
    'stock_data': 'stock_data.csv',
    'daily_sentiment': 'daily_sentiment.csv',
    'fda_events': 'fda_events.csv',
    'nifty_50': 'nifty_50.csv',
    'nifty_pharma': 'nifty_pharma.csv'
}

# Data Sources
STOCK_APIS = ['Yahoo Finance', 'Alpha Vantage']  # NSE API may require authentication
NEWS_SOURCES = ['Reuters', 'BioPharma Dive', 'FiercePharma', 'FDA.gov', 'Biocon Press Releases']
BENCHMARK_INDICES = {
    'Nifty 50': '^NSEI',
    'Nifty Pharma': '^NIPHARM'
}

# Model Configuration
MODEL_CONFIG = {
    'linear_regression': {},
    'ridge_regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    'lasso_regression': {'alpha': [0.01, 0.1, 1.0, 10.0]},
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'lightgbm': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'min_child_samples': [10],  # Reduced to allow more splits
        'min_split_gain': [0.0]    # Allow splits with smaller gains
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'svr': {
        'kernel': ['rbf'],
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1]
    },
    'lstm': {
        'epochs': 50,
        'batch_size': 32,
        'units': [50, 25]
    },
    'bert_sentiment': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 512,
        'batch_size': 16
    }
}

def create_directories():
    """
    Create necessary directories if they don't exist.
    """
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

def validate_config():
    """
    Validate configuration settings.
    Warn if data files are missing, but don't raise errors to allow partial execution.
    """
    for path in PATHS.values():
        if not path.exists():
            print(f"Warning: Directory not found, creating: {path}")
            path.mkdir(parents=True, exist_ok=True)
    for file in DATA_FILES.values():
        if not (PATHS['data'] / file).exists():
            print(f"Warning: Data file {file} not found in {PATHS['data']}")