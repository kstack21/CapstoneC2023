"""
Module Name: functions.py
Description: This module contains a collection of functions used throughout the program. 
"""
# Import libraries
import pandas
import os

def path_back_to(new_folder_name):
    """
    Navigate to a new folder path relative to the current script's location.

    This function takes a `new_folder_name` as input and calculates a new path by
    moving up from the current script's location and appending the specified folder name.

    Args:
        new_folder_name (str or List[str]): The name of the folder(s) to navigate to,
            relative to the current script's location. Can be a single string or a list
            of strings for nested folders.

    Returns:
        str: The newly constructed path based on the provided folder name(s).

    Example:
        # Navigate to a folder named "data" one level above the script's location
            new_path = path_back_to("data")
    """
    # Get the directory name of the provided path
    directory_name = os.path.dirname(__file__)

    # Split the directory path into components
    directory_components = directory_name.split(os.path.sep)

    # # Remove the last folder 
    # if directory_components[-1]:
    #     directory_components.pop()

    # Add the new folder to the path
    for file in new_folder_name:
        directory_components.append(file)

    # Join the modified components to create the new path
    new_path = os.path.sep.join(directory_components)

    return new_path

def data_demographics(data):
    pass