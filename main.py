#!/usr/bin/env python3
"""
Ice Cream Sales Predictor
Main module for training and deploying the ML model

Author: Hernane Bini
Date: February 2026
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IceCreamSalesPredictor:
    """Machine Learning model for predicting ice cream sales based on temperature."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the predictor."""
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        X = df[['Temperature']].values
        y = df['Sales'].values
        logger.info(f"Data loaded: {len(df)} records")
        return X, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Split data into train and test sets."""
        logger.info("Preparing data for training")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
    
    def train(self) -> dict:
        """Train the linear regression model."""
        logger.info("Training Linear Regression model")
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        return self._evaluate()
    
    def _evaluate(self) -> dict:
        """Evaluate model performance."""
        logger.info("Evaluating model")
        y_pred = self.model.predict(self.X_test)
        
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        metrics = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'coefficient': self.model.coef_[0],
            'intercept': self.model.intercept_
        }
        
        logger.info(f"Model Metrics - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return metrics
    
    def predict(self, temperature: float) -> float:
        """Make prediction for a given temperature."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict([[temperature]])[0]
    
    def log_to_mlflow(self, metrics: dict, params: dict = None) -> None:
        """Log model and metrics to MLflow."""
        with mlflow.start_run():
            params = params or {'test_size': self.test_size, 'random_state': self.random_state}
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "model")
            logger.info("Model logged to MLflow")


def main():
    """Main execution function."""
    logger.info("Starting Ice Cream Sales Predictor")
    
    # Initialize predictor
    predictor = IceCreamSalesPredictor()
    
    # Load and prepare data
    # X, y = predictor.load_data('data/ice_cream_sales.csv')
    # predictor.prepare_data(X, y)
    
    # Train model
    # metrics = predictor.train()
    
    # Log to MLflow
    # predictor.log_to_mlflow(metrics)
    
    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()
