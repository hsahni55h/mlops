"""
This file contains utility functions used across the project. 
It includes functionality to save objects to disk, ensuring that important objects such as models and preprocessors can be persisted and later retrieved.
"""

import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score

from src.exception import Custom_Exception

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Args:
    file_path (str): The path where the object should be saved.
    obj: The object to be saved.

    Raises:
    Custom_Exception: If any exception occurs during the save operation, it raises a custom exception with the error details.
    """
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise Custom_Exception(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return a report with their performance on the test set.

    Args:
    X_train: Training data features.
    y_train: Training data target.
    X_test: Test data features.
    y_test: Test data target.
    models (dict): A dictionary containing model names as keys and instantiated model objects as values.

    Returns:
    dict: A report containing the test R^2 scores for each model.

    Raises:
    Custom_Exception: If any exception occurs during model evaluation, it raises a custom exception with the error details.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Train the model on the training data
            model.fit(X_train, y_train)

            # Predict on the training data
            y_train_pred = model.predict(X_train)

            # Predict on the test data
            y_test_pred = model.predict(X_test)

            # Calculate R^2 score for the training data
            train_model_score = r2_score(y_train, y_train_pred)

            # Calculate R^2 score for the test data
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R^2 score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise Custom_Exception(e, sys)
    
def load_object(file_path):
    """
    Load an object from a file using pickle.

    Args:
    file_path (str): The path to the file from which the object should be loaded.

    Returns:
    The loaded object.

    Raises:
    Custom_Exception: If any exception occurs during the load operation, it raises a custom exception with the error details.
    """
    try:
        # Open the file in read-binary mode and load the object using pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise Custom_Exception(e, sys)
