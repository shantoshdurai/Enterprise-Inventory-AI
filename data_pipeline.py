"""
Data Pipeline and Feature Engineering Module
O9 Solutions-grade data processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Enterprise-grade data pipeline for inventory forecasting
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath=None):
        """Load and perform initial data validation"""
        if filepath is None:
            filepath = DATA_FILE
            
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Convert date column
        df[FEATURE_CONFIG['date_column']] = pd.to_datetime(df[FEATURE_CONFIG['date_column']])
        
        # Sort by date
        df = df.sort_values(FEATURE_CONFIG['date_column']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records from {df[FEATURE_CONFIG['date_column']].min()} to {df[FEATURE_CONFIG['date_column']].max()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        logger.info("Handling missing values")
        
        # For numerical columns, use forward fill then backward fill
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(method='ffill').fillna(method='bfill')
        
        # For categorical, use mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != FEATURE_CONFIG['date_column']:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def create_time_features(self, df):
        """Create comprehensive time-based features"""
        logger.info("Creating time features")
        
        date_col = FEATURE_CONFIG['date_column']
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_lag_features(self, df, group_cols=['Store ID', 'Product ID']):
        """Create lag features for time series"""
        logger.info("Creating lag features")
        
        target = FEATURE_CONFIG['target']
        
        for lag in FEATURE_CONFIG['lag_features']:
            df[f'{target}_lag_{lag}'] = df.groupby(group_cols)[target].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, group_cols=['Store ID', 'Product ID']):
        """Create rolling window statistics"""
        logger.info("Creating rolling window features")
        
        target = FEATURE_CONFIG['target']
        
        for window in FEATURE_CONFIG['rolling_windows']:
            # Rolling mean
            df[f'{target}_rolling_mean_{window}'] = df.groupby(group_cols)[target].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{target}_rolling_std_{window}'] = df.groupby(group_cols)[target].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{target}_rolling_min_{window}'] = df.groupby(group_cols)[target].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'{target}_rolling_max_{window}'] = df.groupby(group_cols)[target].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        return df
    
    def create_price_features(self, df):
        """Create price-related features"""
        logger.info("Creating price features")
        
        # Price relative to competitor
        df['price_vs_competitor'] = df['Price'] - df['Competitor Pricing']
        df['price_ratio_competitor'] = df['Price'] / (df['Competitor Pricing'] + 1e-6)
        
        # Discount bins
        df['discount_bin'] = pd.cut(df['Discount'], bins=[0, 5, 10, 15, 20, 100], labels=['0-5', '5-10', '10-15', '15-20', '20+'])
        
        # Price change (compared to previous period)
        df['price_change'] = df.groupby(['Store ID', 'Product ID'])['Price'].diff()
        
        return df
    
    def create_inventory_features(self, df):
        """Create inventory-related features"""
        logger.info("Creating inventory features")
        
        # Stock-to-demand ratio
        df['stock_to_demand_ratio'] = df['Inventory Level'] / (df['Units Sold'] + 1)
        
        # Stockout indicator (inventory below threshold)
        df['low_stock'] = (df['Inventory Level'] < 100).astype(int)
        
        # Overstock indicator
        df['high_stock'] = (df['Inventory Level'] > 400).astype(int)
        
        return df
    
    def create_aggregation_features(self, df):
        """Create aggregated features by category, region, etc."""
        logger.info("Creating aggregation features")
        
        target = FEATURE_CONFIG['target']
        
        # Category-level average demand
        category_avg = df.groupby(['Category', FEATURE_CONFIG['date_column']])[target].transform('mean')
        df['category_avg_demand'] = category_avg
        
        # Region-level average demand
        region_avg = df.groupby(['Region', FEATURE_CONFIG['date_column']])[target].transform('mean')
        df['region_avg_demand'] = region_avg
        
        # Store-level average demand
        store_avg = df.groupby(['Store ID', FEATURE_CONFIG['date_column']])[target].transform('mean')
        df['store_avg_demand'] = store_avg
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        logger.info("Encoding categorical features")
        
        categorical_features = FEATURE_CONFIG['categorical_features']
        
        for col in categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def engineer_all_features(self, df, fit=True):
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create rolling features
        df = self.create_rolling_features(df)
        
        # Create price features
        df = self.create_price_features(df)
        
        # Create inventory features
        df = self.create_inventory_features(df)
        
        # Create aggregation features
        df = self.create_aggregation_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Drop rows with NaN created by lag features
        df = df.dropna()
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def prepare_ml_dataset(self, df):
        """Prepare dataset for ML models"""
        logger.info("Preparing ML dataset")
        
        # Select feature columns
        feature_cols = []
        
        # Add encoded categorical features
        for col in FEATURE_CONFIG['categorical_features']:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
        
        # Add numerical features
        for col in FEATURE_CONFIG['numerical_features']:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add engineered features
        engineered_patterns = [
            'lag_', 'rolling_', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_month_', 'is_quarter_', 'price_vs_', 'price_ratio_',
            'stock_to_demand', 'low_stock', 'high_stock', 'category_avg', 'region_avg', 'store_avg',
            'year', 'month', 'day', 'day_of_week', 'quarter', 'week_of_year'
        ]
        
        for pattern in engineered_patterns:
            feature_cols.extend([col for col in df.columns if pattern in col and col not in feature_cols])
        
        # Remove duplicates and sort
        feature_cols = sorted(list(set(feature_cols)))
        
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df[FEATURE_CONFIG['target']]
        
        logger.info(f"Prepared dataset with {len(feature_cols)} features")
        
        return X, y, df[FEATURE_CONFIG['date_column']]
    
    def create_train_test_split(self, df):
        """Time-aware train/test split"""
        logger.info("Creating train/test split")
        
        # Sort by date
        df = df.sort_values(FEATURE_CONFIG['date_column'])
        
        # Split based on time
        split_idx = int(len(df) * (1 - MODEL_CONFIG['test_size']))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train_df)} records, Test: {len(test_df)} records")
        logger.info(f"Train period: {train_df[FEATURE_CONFIG['date_column']].min()} to {train_df[FEATURE_CONFIG['date_column']].max()}")
        logger.info(f"Test period: {test_df[FEATURE_CONFIG['date_column']].min()} to {test_df[FEATURE_CONFIG['date_column']].max()}")
        
        return train_df, test_df
    
    def save_pipeline(self, filepath=None):
        """Save the pipeline for later use"""
        if filepath is None:
            filepath = DATA_DIR / "pipeline.pkl"
        
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath=None):
        """Load a saved pipeline"""
        if filepath is None:
            filepath = DATA_DIR / "pipeline.pkl"
        
        pipeline_data = joblib.load(filepath)
        self.label_encoders = pipeline_data['label_encoders']
        self.scaler = pipeline_data['scaler']
        self.feature_names = pipeline_data['feature_names']
        
        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Test the data pipeline"""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Load data
    df = pipeline.load_data()
    
    # Engineer features
    df_processed = pipeline.engineer_all_features(df, fit=True)
    
    # Create train/test split
    train_df, test_df = pipeline.create_train_test_split(df_processed)
    
    # Prepare ML datasets
    X_train, y_train, dates_train = pipeline.prepare_ml_dataset(train_df)
    X_test, y_test, dates_test = pipeline.prepare_ml_dataset(test_df)
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {len(pipeline.feature_names)}")
    print(f"\nFeature names (first 20):")
    print(pipeline.feature_names[:20])
    
    # Save pipeline
    pipeline.save_pipeline()
    
    # Save processed data
    joblib.dump({
        'X_train': X_train,
        'y_train': y_train,
        'dates_train': dates_train,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
        'train_df': train_df,
        'test_df': test_df
    }, PROCESSED_DATA_FILE)
    
    logger.info(f"Processed data saved to {PROCESSED_DATA_FILE}")


if __name__ == "__main__":
    main()
