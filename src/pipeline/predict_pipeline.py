"""
This file handles the prediction process using a trained model and preprocessor. 
It includes a class to manage prediction pipeline and a class to handle custom input data.
"""

import sys
import os
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object

class PredictPipeline:
    """
    Class for handling the prediction pipeline process.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predict the target using the loaded model and preprocessor.

        Args:
        features (pd.DataFrame): DataFrame containing the input features for prediction.

        Returns:
        np.ndarray: Predicted values.

        Raises:
        Custom_Exception: If any exception occurs during prediction.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")
            model = load_object(file_path=model_path)  # Load the trained model
            preprocessor = load_object(file_path=preprocessor_path)  # Load the preprocessor object
            print("After Loading")

            data_scaled = preprocessor.transform(features)  # Preprocess the input features
            preds = model.predict(data_scaled)  # Predict using the preprocessed features
            return preds

        except Exception as e:
            raise Custom_Exception(e, sys)

class CustomData:
    """
    Class for handling custom input data.
    """
    def __init__(self, 
                 gender: str, 
                 race_ethnicity: str, 
                 parental_level_of_education, 
                 lunch: str,
                 test_preparation_course: str, 
                 reading_score: int, 
                 writing_score: int):
        """
        Initializes the CustomData instance with input features.

        Args:
        gender (str): Gender of the student.
        race_ethnicity (str): Race/ethnicity of the student.
        parental_level_of_education (str): Parental level of education.
        lunch (str): Type of lunch.
        test_preparation_course (str): Test preparation course.
        reading_score (int): Reading score.
        writing_score (int): Writing score.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the custom input data to a DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the input features.

        Raises:
        Custom_Exception: If any exception occurs during the DataFrame creation.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Custom_Exception(e, sys)
