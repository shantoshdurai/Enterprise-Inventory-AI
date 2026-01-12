"""
Machine Learning Models for Inventory Forecasting
O9 Solutions-grade ML implementation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

logger = logging.getLogger(__name__)


class MLModels:
    """
    Enterprise-grade ML models for demand forecasting
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model"""
        logger.info("Training Random Forest model")
        
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        # Validation score
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"Random Forest validation R²: {val_score:.4f}")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        logger.info("Training XGBoost model")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"XGBoost validation R²: {val_score:.4f}")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        logger.info("Training LightGBM model")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"LightGBM validation R²: {val_score:.4f}")
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model"""
        logger.info("Training CatBoost model")
        
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            model = CatBoostRegressor(**CATBOOST_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model = CatBoostRegressor(**CATBOOST_PARAMS)
            model.fit(X_train, y_train, verbose=False)
        
        self.models['catboost'] = model
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"CatBoost validation R²: {val_score:.4f}")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_val=None, y_val=None):
        """Train Gradient Boosting model"""
        logger.info("Training Gradient Boosting model")
        
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"Gradient Boosting validation R²: {val_score:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all ML models"""
        logger.info("Training all ML models")
        
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_catboost(X_train, y_train, X_val, y_val)
        self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        
        logger.info(f"Trained {len(self.models)} models successfully")
        
        return self.models
    
    def predict(self, X, model_name=None):
        """Make predictions with specified model or all models"""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name].predict(X)
        else:
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            return predictions
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        y_pred = self.models[model_name].predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        # Bias
        bias = np.mean(y_pred - y_test)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Bias': bias
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"{model_name} - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        logger.info("Evaluating all models")
        
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(model_name, X_test, y_test)
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df.sort_values('MAPE')
        
        logger.info("\nModel Performance Comparison:")
        logger.info(f"\n{metrics_df.to_string()}")
        
        return metrics_df
    
    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return None
    
    def save_models(self, directory=None):
        """Save all trained models"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = directory / f"{model_name}.pkl"
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")
        
        # Save metrics
        if self.metrics:
            metrics_file = directory / "metrics.pkl"
            joblib.dump(self.metrics, metrics_file)
            logger.info(f"Saved metrics to {metrics_file}")
    
    def load_models(self, directory=None):
        """Load saved models"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        
        for model_file in directory.glob("*.pkl"):
            if model_file.stem != "metrics":
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load metrics if available
        metrics_file = directory / "metrics.pkl"
        if metrics_file.exists():
            self.metrics = joblib.load(metrics_file)
            logger.info(f"Loaded metrics from {metrics_file}")


class EnsembleModel:
    """
    Ensemble model combining multiple forecasts
    """
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.weights = None
        
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights based on validation performance"""
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_val)
        
        # Calculate MAPE for each model
        mapes = {}
        for name, preds in predictions.items():
            mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-10))) * 100
            mapes[name] = mape
        
        # Inverse MAPE for weights (lower MAPE = higher weight)
        inv_mapes = {name: 1 / (mape + 1e-10) for name, mape in mapes.items()}
        total = sum(inv_mapes.values())
        self.weights = {name: inv / total for name, inv in inv_mapes.items()}
        
        logger.info("Optimized weights:")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return self.weights
    
    def predict(self, X):
        """Make ensemble prediction"""
        if self.weights is None:
            # Equal weights if not optimized
            self.weights = {name: 1/len(self.models) for name in self.models.keys()}
        
        # Weighted average of predictions
        ensemble_pred = np.zeros(len(X))
        for name, model in self.models.items():
            ensemble_pred += self.weights[name] * model.predict(X)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble model"""
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        bias = np.mean(y_pred - y_test)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Bias': bias
        }
        
        logger.info(f"Ensemble - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return metrics


def main():
    """Test ML models"""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Load processed data
    logger.info("Loading processed data")
    data = joblib.load(PROCESSED_DATA_FILE)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split train into train/val
    val_size = int(len(X_train) * MODEL_CONFIG['validation_size'])
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]
    
    # Train models
    ml_models = MLModels()
    ml_models.train_all_models(X_train_sub, y_train_sub, X_val, y_val)
    
    # Evaluate models
    metrics_df = ml_models.evaluate_all_models(X_test, y_test)
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(metrics_df)
    
    # Create ensemble
    ensemble = EnsembleModel(ml_models.models)
    ensemble.optimize_weights(X_val, y_val)
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("="*80)
    for metric, value in ensemble_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save models
    ml_models.save_models()
    
    # Save ensemble
    joblib.dump(ensemble, MODELS_DIR / "ensemble.pkl")
    logger.info("Ensemble model saved")


if __name__ == "__main__":
    main()
