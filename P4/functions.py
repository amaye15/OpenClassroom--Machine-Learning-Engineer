import fnmatch
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import requests
import json
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

def sort_and_classify_column(df: pd.DataFrame, column_name: str, datetime: bool = False, new_column_name: Optional[str] = None) -> pd.DataFrame:
    """
    This function sorts a DataFrame by a specified column in ascending order and then
    classifies the sorted values into three categories: 'Low', 'Medium', and 'High'.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to sort and classify.
    - column_name (str): The name of the column to sort and classify.
    - new_column_name (Optional[str]): The name of the new column to store the classes. 
                                       Defaults to `column_name + '_class'` if not provided.
    
    Returns:
    - df (pd.DataFrame): The DataFrame with the sorted column and new classification column.
    """

    if datetime:
         df[column_name] = pd.to_datetime(df[column_name])
    
    # Sort the DataFrame by the specified column in ascending order
    df = df.sort_values(by=column_name)
    
    # Create labels for the three categories
    labels = ['Low', 'Medium', 'High']
    
    # If a new column name is not provided, create one based on the original column name
    if new_column_name is None:
        new_column_name = f"{column_name}_class"
    
    # Classify the sorted column into three categories and store it in a new column
    df[new_column_name] = pd.cut(df[column_name], bins=3, labels=labels)
    
    return df


def get_country_bounding_box(country_name: str) -> Union[Dict[str, float], Dict[str, str]]:
    """
    Fetch the bounding box coordinates of a given country from the OpenStreetMap Nominatim API.
    
    Parameters:
        country_name (str): The name of the country for which the bounding box is required.
    
    Returns:
        Union[Dict[str, float], Dict[str, str]]: A dictionary containing the south, north, west, and east coordinates
                                                  of the country's bounding box. If an error occurs, a dictionary 
                                                  containing an 'error' key is returned.
    
    Example:
        >>> get_country_bounding_box('France')
        {'south': 41.3032, 'north': 51.126, 'west': -4.7904, 'east': 9.5615}
    """
    # Define API Endpoint and Parameters
    api_endpoint = "https://nominatim.openstreetmap.org/search"
    params = {
        'country': country_name,
        'format': 'json',
        'limit': 1
    }
    
    # Send HTTP GET Request
    response = requests.get(api_endpoint, params=params)
    
    # Check if the response is valid
    if response.status_code != 200:
        return {'error': 'Invalid API Response'}
    
    # Parse JSON Response
    data = json.loads(response.text)
    if not data:
        return {'error': 'Country not found'}
    
    bounding_box = data[0]['boundingbox']
    
    # Extract Coordinates
    south = float(bounding_box[0])
    north = float(bounding_box[1])
    west = float(bounding_box[2])
    east = float(bounding_box[3])
    
    # Create and Return Dictionary
    return {
        'south': south,
        'north': north,
        'west': west,
        'east': east
    }

from sklearn.impute import SimpleImputer

def set_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Set the 'Outliers' column in the DataFrame based on latitude boundaries.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'geolocation_lat' column.
    - bounds (dict): A dictionary containing the latitude boundaries ('west', 'east', 'north', 'south').

    Returns:
    - pd.DataFrame: The updated DataFrame with the 'Outliers' column set.
    """
    c1 = (df["geolocation_lat"] >= bounds["west"]) & (df["geolocation_lat"] <= bounds["east"])
    c2 = (df["geolocation_lat"] <= bounds["north"]) & (df["geolocation_lat"] >= bounds["south"])
    conditions = [c1, c2]
    choices = ["Normal", "Normal"]
    df["Outliers"] = np.select(conditions, choices, default='Outlier')
    return df

def impute_values(df: pd.DataFrame, condition: np.ndarray, imputer: SimpleImputer) -> pd.DataFrame:
    """
    Impute missing latitude and longitude values in the DataFrame based on a condition.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'geolocation_lat' and 'geolocation_lng' columns.
    - condition (np.ndarray): A boolean NumPy array representing the condition to filter the DataFrame.
    - imputer (SimpleImputer): A SimpleImputer object for imputing missing values.

    Returns:
    - pd.DataFrame: The updated DataFrame with imputed values for 'geolocation_lat' and 'geolocation_lng'.
    """
    lat_values = imputer.fit_transform(df.loc[condition, "geolocation_lat"].values.reshape(-1, 1))
    lng_values = imputer.fit_transform(df.loc[condition, "geolocation_lng"].values.reshape(-1, 1))
    if lat_values.shape[1] > 0:
        df.loc[condition, "geolocation_lat"] = lat_values
        df.loc[condition, "geolocation_lng"] = lng_values
    return df
