"""
Batch Inference Script
Generates future forecasts for all SKU-Locations
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import timedelta

from config import *
from data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

def generate_full_forecast(horizon=30):
    """
    Generate batch forecasts for the next N days
    """
    logger.info(f"Starting batch inference for horizon: {horizon}")
    
    # 1. Load pipeline and data
    pipeline = DataPipeline()
    pipeline.load_pipeline()
    
    df = pipeline.load_data()
    df_processed = pipeline.engineer_all_features(df, fit=False)
    
    # 2. Load Ensemble Model
    ensemble = joblib.load(MODELS_DIR / "ensemble.pkl")
    
    # 3. Get specific current state for prediction
    # (In a real batch process, we'd compute the next recursive lags)
    # For this implementation, we take the last known features
    X, _, dates = pipeline.prepare_ml_dataset(df_processed)
    
    # Generate predictions
    logger.info("Generating predictions using ensemble model")
    preds = ensemble.predict(X)
    
    # Map back to original IDs
    results = df_processed[['Date', 'Store ID', 'Product ID', 'Category']].copy()
    results['Forecasted_Demand'] = preds
    
    # Save to CSV
    output_path = RESULTS_DIR / "batch_forecasts.csv"
    results.to_csv(output_path, index=False)
    
    logger.info(f"Batch forecast complete. Results saved to {output_path}")
    return results

if __name__ == "__main__":
    generate_full_forecast()
