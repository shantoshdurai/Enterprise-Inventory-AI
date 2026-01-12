"""
Main Forecasting Engine
Orchestrates training and inference for all model types
O9 Solutions-grade integration
"""

import logging.config
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import *
from data_pipeline import DataPipeline
from models.ml_models import MLModels, EnsembleModel
from models.deep_learning_models import LSTMModel
from models.time_series_models import TimeSeriesModels

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """
    Unified engine to manage the end-to-end forecasting lifecycle
    """
    
    def __init__(self):
        self.pipeline = DataPipeline()
        self.ml_models = MLModels()
        self.ts_models = TimeSeriesModels()
        self.lstm_model = LSTMModel(sequence_length=LSTM_CONFIG['sequence_length'])
        self.ensemble = None
        
    def run_full_training(self):
        """Execute the full training pipeline"""
        logger.info("Initializing full enterprise training cycle")
        
        # 1. Load and Process Data
        df = self.pipeline.load_data()
        df_processed = self.pipeline.engineer_all_features(df, fit=True)
        self.pipeline.save_pipeline()
        
        # 2. Split Data
        train_df, test_df = self.pipeline.create_train_test_split(df_processed)
        X_train, y_train, dates_train = self.pipeline.prepare_ml_dataset(train_df)
        X_test, y_test, dates_test = self.pipeline.prepare_ml_dataset(test_df)
        
        # Split train for validation
        val_size = int(len(X_train) * MODEL_CONFIG['validation_size'])
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_sub = X_train[:-val_size]
        y_train_sub = y_train[:-val_size]
        
        # 3. Train ML Models
        logger.info("Step 3: Training ML Models suite")
        self.ml_models.train_all_models(X_train_sub, y_train_sub, X_val, y_val)
        ml_metrics = self.ml_models.evaluate_all_models(X_test, y_test)
        self.ml_models.save_models()
        
        # 4. Train Ensemble
        logger.info("Step 4: Optimizing Ensemble weights")
        self.ensemble = EnsembleModel(self.ml_models.models)
        self.ensemble.optimize_weights(X_val, y_val)
        ensemble_metrics = self.ensemble.evaluate(X_test, y_test)
        joblib.dump(self.ensemble, MODELS_DIR / "ensemble.pkl")
        
        # 5. Train Deep Learning (LSTM)
        # Note: Large datasets might need optimization for LSTM
        logger.info("Step 5: Training Deep Learning (LSTM) model")
        try:
            self.lstm_model.train(X_train_sub, y_train_sub, X_val, y_val)
            self.lstm_model.save_model()
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            
        # 6. Train Statistical Models (Prophet) for top SKUs
        # For hackathon, we'll pick top 5 SKUs to demonstrate
        logger.info("Step 6: Training statistical foundations (Prophet)")
        top_skus = df['Product ID'].value_counts().head(5).index
        for sku in top_skus:
            try:
                self.ts_models.train_prophet(df, sku)
            except Exception as e:
                logger.error(f"Prophet training failed for {sku}: {str(e)}")
        self.ts_models.save_models()
        
        # 7. Final Report
        logger.info("Enterprise training cycle complete")
        return {
            'ml_metrics': ml_metrics,
            'ensemble_metrics': ensemble_metrics
        }

    def generate_recommendations(self, sku_id, store_id):
        """High-level business logic for inventory replenishment"""
        # This would normally be much more complex in o9
        # Here we provide a simplified version
        logger.info(f"Generating recommendations for {sku_id} at {store_id}")
        
        # 1. Get latest data
        # 2. Predict next 30 days
        # 3. Calculate Reorder Point = (Avg Daily Demand * Lead Time) + Safety Stock
        # 4. Return action
        
        return {
            'action': 'Maintain',
            'reorder_point': 150,
            'safety_stock': 45,
            'stockout_risk': 'Low'
        }

if __name__ == "__main__":
    engine = ForecastingEngine()
    results = engine.run_full_training()
    print("\nTraining Results Summary:")
    print(results['ensemble_metrics'])
