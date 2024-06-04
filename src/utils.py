"""

This file contains utility functions used across the project. It includes functionality to save objects to disk, ensuring that important objects such as models and preprocessors can be persisted and later retrieved.

"""

import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

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
