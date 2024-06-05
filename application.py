"""
This file creates a Flask web application that handles user input, processes it using the trained model, and returns the prediction result.
"""

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a Flask application instance
application = Flask(__name__)

# Alias for the application instance
app = application

# Route for the home page
@app.route('/')
def index():
    """
    Render the index.html template for the home page.
    """
    return render_template('index.html')

# Route to handle data prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle the data prediction process.
    """
    if request.method == 'GET':
        # Render the home.html template for GET requests
        return render_template('home.html')
    else:
        # Create an instance of CustomData with input data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        # Convert the custom data to a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Create an instance of PredictPipeline and predict the results
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Render the home.html template with the prediction results
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    # Run the Flask application
    app.run(host="0.0.0.0")
