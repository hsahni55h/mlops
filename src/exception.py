"""
This file contains custom exception handling logic for the project. 
It includes a function to format error messages with detailed information and a custom exception class to raise more informative errors.
"""

import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Formats detailed error messages.

    Args:
    error: The exception object.
    error_detail (sys): The sys module to get the current exception details.

    Returns:
    str: A formatted string containing the file name, line number, and error message.
    """
    _, _, exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exec_tb.tb_lineno, str(error))
    return error_message

class Custom_Exception(Exception):
    """
    Custom exception class to handle exceptions with detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the custom exception with an error message and details.

        Args:
        error_message (str): The error message to be displayed.
        error_detail (sys): The sys module to get the current exception details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        """
        Returns the error message when the exception is converted to a string.
        
        Returns:
        str: The detailed error message.
        """
        return self.error_message
     

if __name__ == "__main__":
    # Example usage of the custom exception
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero")
        raise Custom_Exception(e, sys)
