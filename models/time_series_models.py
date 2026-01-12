"""
Statistical Time Series Models for Inventory Forecasting
ARIMA and Prophet implementation
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from pmdarima import auto_arima
import logging
import joblib
from pathlib import Path
from config import *

logger = logging.getLogger(__name__)

class TimeSeriesModels:
    """
    Implementation of Prophet and ARIMA for baseline structural forecasting
    """
    
    def __init__(self):
        self.prophet_models = {}  # One per SKU/Location usually
        self.arima_models = {}
        
    def train_prophet(self, df, sku_id):
        """Train Prophet model for a specific SKU"""
        logger.info(f"Training Prophet for SKU: {sku_id}")
        
        # Prophet expects 'ds' and 'y' columns
        prophet_df = df[df['Product ID'] == sku_id][[FEATURE_CONFIG['date_column'], FEATURE_CONFIG['target']]].copy()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(**PROPHET_CONFIG)
        
        # Add regressors for external drivers if they exist
        if 'Price' in df.columns:
            model.add_regressor('Price')
        if 'Discount' in df.columns:
            model.add_regressor('Discount')
            
        # Add regressors back to the training df
        extra_cols = ['Price', 'Discount']
        for col in extra_cols:
            prophet_df[col] = df[df['Product ID'] == sku_id][col].values
            
        model.fit(prophet_df)
        self.prophet_models[sku_id] = model
        return model

    def predict_prophet(self, sku_id, periods=30, future_df=None):
        """Predict using Prophet"""
        if sku_id not in self.prophet_models:
            raise ValueError(f"No Prophet model trained for SKU {sku_id}")
            
        model = self.prophet_models[sku_id]
        
        if future_df is None:
            future = model.make_future_dataframe(periods=periods)
            # In a real scenario, we'd need future values for Price/Discount
            # For this demo, we'll carry forward last values if not provided
            for col in ['Price', 'Discount']:
                future[col] = 0 # Placeholder
        else:
            future = future_df.copy()
            future.columns = ['ds' if col == FEATURE_CONFIG['date_column'] else col for col in future.columns]
            
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def train_arima(self, series):
        """Train Auto-ARIMA model"""
        logger.info("Training Auto-ARIMA")
        model = auto_arima(series, seasonal=True, m=7, suppress_warnings=True)
        return model

    def save_models(self, directory=None):
        """Save statistical models"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.prophet_models, directory / "prophet_models.pkl")
        logger.info(f"Saved Prophet models to {directory}")

    def load_models(self, directory=None):
        """Load statistical models"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        if (directory / "prophet_models.pkl").exists():
            self.prophet_models = joblib.load(directory / "prophet_models.pkl")
            logger.info("Loaded Prophet models")
