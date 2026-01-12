"""
Deep Learning Models for Inventory Forecasting
O9 Solutions-grade LSTM/GRU implementation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    Enterprise-grade LSTM model for sequential demand patterns
    """
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_sequences(self, X, y):
        """Transform tabular data into windowed sequences for LSTM"""
        X_seq, y_seq = [], []
        
        # We assume X and y are already matching and sorted by time per SKU if necessary
        # For simplicity in this implementation, we treat the entire dataset as a sequence
        # In a real O9 project, we'd do this per SKU-Location combination
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """Build the LSTM architecture"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(LSTM_CONFIG['units'][0], return_sequences=True),
            Dropout(LSTM_CONFIG['dropout']),
            LSTM(LSTM_CONFIG['units'][1]),
            Dropout(LSTM_CONFIG['dropout']),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_CONFIG['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        logger.info("Training LSTM model")
        
        # Scaling
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1))
        
        # Sequence preparation
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
        
        # Build model
        self.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=LSTM_CONFIG['early_stopping_patience'],
            restore_best_weights=True
        )
        
        # Fit
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        logger.info("LSTM training complete")
        return history
    
    def predict(self, X):
        """Make predictions with the LSTM model"""
        X_scaled = self.scaler_X.transform(X)
        
        # We need the last 'sequence_length' values to predict the next value
        # This implementation expects X to be a continuous sequence
        predictions = []
        
        # Simple sliding window prediction for the entire X
        # In production, we'd handle the 'cold start' and overlap properly
        for i in range(len(X_scaled) - self.sequence_length + 1):
            seq = X_scaled[i:i+self.sequence_length].reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(seq, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled)
            predictions.append(pred[0][0])
            
        # Pad beginning with NaNs or first prediction to match output shape
        padding = [predictions[0]] * (self.sequence_length - 1)
        return np.array(padding + predictions)

    def save_model(self, directory=None):
        """Save the LSTM model and scalers"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        self.model.save(directory / "lstm_model.h5")
        joblib.dump(self.scaler_X, directory / "lstm_scaler_X.pkl")
        joblib.dump(self.scaler_y, directory / "lstm_scaler_y.pkl")
        logger.info(f"Saved LSTM model and scalers to {directory}")

    def load_model(self, directory=None):
        """Load the LSTM model and scalers"""
        if directory is None:
            directory = MODELS_DIR
        
        directory = Path(directory)
        self.model = tf.keras.models.load_model(directory / "lstm_model.h5")
        self.scaler_X = joblib.load(directory / "lstm_scaler_X.pkl")
        self.scaler_y = joblib.load(directory / "lstm_scaler_y.pkl")
        logger.info(f"Loaded LSTM model and scalers from {directory}")
