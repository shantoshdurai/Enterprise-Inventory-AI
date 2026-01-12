"""
Configuration file for Inventory Forecasting System
O9 Solutions-grade enterprise configuration
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_FILE = BASE_DIR / "retail_store_inventory.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.pkl"

# Feature engineering configuration
FEATURE_CONFIG = {
    'lag_features': [1, 7, 14, 30],  # Days to lag
    'rolling_windows': [7, 14, 30],  # Rolling window sizes
    'categorical_features': ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'],
    'numerical_features': ['Inventory Level', 'Price', 'Discount', 'Competitor Pricing'],
    'target': 'Units Sold',
    'date_column': 'Date'
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'forecast_horizon': 30,  # Days to forecast
    'confidence_level': 0.95
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# CatBoost hyperparameters
CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_state': 42,
    'verbose': False
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# LSTM configuration
LSTM_CONFIG = {
    'sequence_length': 30,
    'units': [128, 64],
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 10
}

# Prophet configuration
PROPHET_CONFIG = {
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'changepoint_prior_scale': 0.05
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    'method': 'weighted',  # 'weighted', 'stacking', or 'voting'
    'optimize_weights': True,
    'meta_model': 'ridge'  # For stacking
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'host': '127.0.0.1',
    'port': 8050,
    'debug': True,
    'theme': 'BOOTSTRAP',
    'update_interval': 1000,  # milliseconds
    'cache_timeout': 300  # seconds
}

# Evaluation metrics
METRICS = ['MAPE', 'RMSE', 'MAE', 'R2', 'Bias']

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'inventory_forecast.log',
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
