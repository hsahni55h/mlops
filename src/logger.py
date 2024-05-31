"""
This file sets up the logging configuration for the project. 
It ensures that all log messages are saved to a log file with a timestamped filename in a dedicated logs directory.
"""

import logging
import os
from datetime import datetime

# Generate a log file name with the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Ensure the logs directory exists, create if it does not
os.makedirs(logs_path, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Set the log message format
    level=logging.INFO,  # Set the logging level to INFO
)


# if __name__ == "__main__":
#     logging.info("Logging has started")
