"""
This file handles the data transformation process for the project. 
It includes functionality to create preprocessing pipelines for numerical and categorical features, and to apply these transformations to the training and test datasets.
"""

import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custom_Exception
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation paths.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    """
    Class for handling the data transformation process.
    """
    def __init__(self):
        """
        Initializes the DataTransformation instance with the configuration paths.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor object for data transformation.
        
        This function creates separate pipelines for numerical and categorical columns,
        and combines them into a single ColumnTransformer.

        Returns:
        ColumnTransformer: A combined preprocessor object for numerical and categorical data.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise Custom_Exception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process by applying the preprocessing pipelines to the training and test datasets.
        
        Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.

        Returns:
        tuple: Transformed training and test arrays, and the path to the saved preprocessor object.
        """
        try:
            # Read training and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature for test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Transform input features using the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed input features with target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object to disk
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise Custom_Exception(e, sys)

if __name__ == "__main__":
    # Example usage to initiate data transformation
    data_transformation = DataTransformation()
    train_path = "path/to/train.csv"  # Replace with actual path
    test_path = "path/to/test.csv"  # Replace with actual path
    data_transformation.initiate_data_transformation(train_path, test_path)
