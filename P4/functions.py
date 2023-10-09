import fnmatch
from typing import List, Dict
import pandas as pd
import os


def find_csv_files(dir_path: str) -> List[str]:
    """
    Finds all CSV files in a given directory and its sub-directories 
    and returns a list of paths relative to the current working directory.
    
    Parameters:
        dir_path (str): The absolute path of the directory to search.
        
    Returns:
        List[str]: A list of relative paths to the found CSV files.
        
    Raises:
        ValueError: If the provided path is not a valid directory.
    """
    # Validate that the path is a directory
    if not os.path.isdir(dir_path):
        raise ValueError("Provided path is not a valid directory")
    
    # Get the current working directory
    current_working_dir = os.getcwd()
    
    # Initialize an empty list for storing relative paths of CSV files
    csv_files = []
    
    # Walk through the directory
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            # Identify if the file is a CSV file
            if fnmatch.fnmatch(filename, '*.csv'):
                # Calculate and store the relative path
                absolute_path = os.path.join(root, filename)
                relative_path = os.path.relpath(absolute_path, current_working_dir)
                csv_files.append(relative_path)
    
    return csv_files

def load_csvs_to_dict(file_list: List[str]) -> Dict[str, pd.DataFrame]:
    """
    This function takes a list of CSV filenames and loads each into a DataFrame.
    It then stores each DataFrame in a dictionary with the base filename as the key.
    
    Parameters:
    - file_list (List[str]): List of CSV filenames to load.
    
    Returns:
    - df_dict (Dict[str, pd.DataFrame]): Dictionary where keys are base filenames and values are DataFrames.
    """
    df_dict: Dict[str, pd.DataFrame] = {}  # Initialize an empty dictionary to hold DataFrames
    
    for file_name in file_list:
        # Check if the file exists
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} does not exist. Skipping.")
            continue
        
        # Read the CSV file into a DataFrame
        df: pd.DataFrame = pd.read_csv(file_name)
        
        # Get the base name of the file and remove the extension to use as the key
        base_name: str = os.path.basename(file_name)
        key_name: str = os.path.splitext(base_name)[0]
        
        # Store the DataFrame in the dictionary
        df_dict[key_name] = df
    
    return df_dict
