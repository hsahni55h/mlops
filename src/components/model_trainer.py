"""
This file handles the training of various regression models, evaluates their performance, and saves the best performing model to disk.
"""

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model trainer paths.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Class for handling the training and evaluation of regression models.
    """
    def __init__(self):
        """
        Initializes the ModelTrainer instance with the configuration paths.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiates the model training and evaluation process.

        Args:
        train_array: Array containing training data (features and target).
        test_array: Array containing test data (features and target).

        Returns:
        float: R^2 score of the best model on the test data.

        Raises:
        Custom_Exception: If any exception occurs during the model training and evaluation process.
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last one for features
                train_array[:, -1],   # The last column for target
                test_array[:, :-1],   # All columns except the last one for features
                test_array[:, -1]     # The last column for target
            )
            
            # Define a dictionary of regression models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate the models and get their performance report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the report
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # If the best model's score is below 0.6, raise an exception
            if best_model_score < 0.6:
                raise Custom_Exception("No best model found with score above 0.6")
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on the test data using the best model
            predicted = best_model.predict(X_test)

            # Calculate and return the R^2 score of the best model on the test data
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise Custom_Exception(e, sys)
