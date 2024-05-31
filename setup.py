from setuptools import find_packages, setup
from typing import List

# Constant to represent the editable install flag in requirements file
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements file and returns a list of dependencies,
    excluding any editable install flags.

    Args:
    file_path (str): The path to the requirements file.

    Returns:
    List[str]: A list of package requirements.
    '''
    requirements = []
    # Open the file in read mode
    with open(file_path) as file_obj:
        # Read all lines from the file
        requirements = file_obj.readlines()
        # Remove newline characters from each requirement
        requirements = [req.replace("\n", "") for req in requirements]

        # If the editable install flag is present, remove it from the list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# Setup function to configure the package
setup(
    name='mlproject',  # The name of the package
    version='0.0.1',  # Initial version of the package
    author='Himanshu',  # Author of the package
    author_email='h.sahni1998@gmail.com',  # Author's email
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=get_requirements('requirements.txt')  # List of dependencies
)
