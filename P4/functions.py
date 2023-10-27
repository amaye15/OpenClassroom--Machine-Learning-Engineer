import fnmatch
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pandas import DataFrame, Timestamp, to_datetime
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from adjustText import adjust_text


def plot_variance_explained(pca: PCA) -> None:
    """
    Plot the variance explained by each component.
    
    Parameters:
    - pca: PCA object containing the principal components
    
    Returns:
    - None
    """
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.5, align='center',
            label='Individual explained variance', color='g')
    plt.step(range(1, len(explained_var) + 1), cum_explained_var, where='mid',
             label='Cumulative explained variance', color='r')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title('Scree Plot: Variance Explained by Principal Components')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loading_heatmap(pca: PCA, df: DataFrame) -> None:
    """
    Plot a heatmap of the PCA components' coefficients.
    
    Parameters:
    - pca: PCA object containing the principal components
    - df: DataFrame containing the original data
    
    Returns:
    - None
    """
    components = pca.components_
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()

    plt.figure(figsize=(10, len(components)*0.5))
    sns.heatmap(components, cmap='viridis', yticklabels=numeric_cols,
                xticklabels=[f'PC{i+1}' for i in range(len(components))],
                cbar_kws={"orientation": "vertical"})
    plt.yticks(rotation=0)
    plt.title('Heatmap of Component Coefficients')
    plt.tight_layout()
    plt.show()

def plot_correlation_circle(df: DataFrame, dimension: str = '2d') -> None:
    """
    Plot a correlation circle or sphere for PCA results.
    
    Parameters:
    - df: DataFrame containing the original data
    - dimension: String specifying whether to plot in '2d' or '3d'
    
    Returns:
    - None
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
    #df_std = StandardScaler().fit_transform(df[numeric_cols])
    df_std = df.values
    pca = PCA(random_state=42).fit(df_std)

    plot_variance_explained(pca)
    plot_loading_heatmap(pca, df)

    if dimension == '2d':
        fig, ax = plt.subplots(figsize=(8, 8))
        pcs = pca.components_
        texts = []
        for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
            plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.1)
            texts.append(plt.text(x, y, df.columns[i]))

        circle = plt.Circle((0, 0), 1, color='gray', fill=False)
        ax.add_artist(circle)
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.axvline(0, color='grey', linestyle='--')
        plt.axhline(0, color='grey', linestyle='--')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.title("Correlation Circle")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

    elif dimension == '3d':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        pcs = pca.components_

        for i, (x, y, z) in enumerate(zip(pcs[0, :], pcs[1, :], pcs[2, :])):
            ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.1, color="black")
            ax.text(x, y, z, df.columns[i], fontsize=10)

        # Create a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%})")
        ax.set_title("Correlation Sphere")
        plt.show()

    else:
        raise ValueError("Only '2d' and '3d' are valid dimensions.")

def split_dataframe_by_time(df: DataFrame, datetime_col: str, n_splits: int) -> Union[Dict[str, DataFrame], None]:
    """
    Split a DataFrame into n_splits based on a datetime column.
    
    Parameters:
    - df: DataFrame to be split
    - datetime_col: The name of the datetime column
    - n_splits: The number of splits
    
    Returns:
    - Dictionary of DataFrames, each corresponding to a time interval or None if conversion fails
    """
    # Check if the datetime_col is already of datetime dtype, if not try converting it
    if df[datetime_col].dtype != 'datetime64[ns]':
        try:
            df[datetime_col] = to_datetime(df[datetime_col])
        except Exception as e:
            print(f"Failed to convert column {datetime_col} to datetime. Error: {e}")
            return None
    
    # Sort DataFrame by datetime column
    df_sorted: DataFrame = df.sort_values(by=datetime_col)
    
    # Calculate time intervals
    min_time: Timestamp = df_sorted[datetime_col].min()
    max_time: Timestamp = df_sorted[datetime_col].max()
    delta: pd.Timedelta = (max_time - min_time) / n_splits
    
    # Generate splits
    splits: Dict[str, DataFrame] = {}
    for i in range(n_splits):
        start_time: Timestamp = min_time + i * delta
        end_time: Timestamp = min_time + (i + 1) * delta
        key: str = f"{start_time} to {end_time}"
        split_df: DataFrame = df_sorted[(df_sorted[datetime_col] >= start_time) & (df_sorted[datetime_col] < end_time)]
        splits[key] = split_df
        
    return splits

def transform_columns(df: DataFrame) -> DataFrame:
    """
    Automatically transform columns of a DataFrame based on their dtype.

    Parameters:
    - df: DataFrame to be transformed

    Returns:
    - Transformed DataFrame
    """
    transformed_df: DataFrame = df.copy()
    
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype in ['int64', 'float64']:
            ss: QuantileTransformer = QuantileTransformer(output_distribution='normal')
            transformed_values: ndarray = ss.fit_transform(df[col].values.reshape(-1, 1))
            transformed_df[col] = transformed_values
        elif col_dtype == 'object':
            le: LabelEncoder = LabelEncoder()
            ss: QuantileTransformer = QuantileTransformer(output_distribution='normal')
            transformed_values: ndarray = ss.fit_transform(le.fit_transform(df[col]).reshape(-1, 1))
            transformed_df[col] = transformed_values
        else:
            print(f"Column {col} has an unsupported dtype {col_dtype}. Skipping.")
            
    return transformed_df

def optimize_dbscan(df: DataFrame, 
                    eps_range: List[float] = [0.1, 0.5, 1.0], 
                    min_samples_range: Union[List[int], Tuple[int, ...]] = [2, 5, 10]) -> Dict[str, Union[float, int, List[Union[float, int]]]]:
    """
    Apply and optimize DBSCAN clustering on a given DataFrame.
    
    Parameters:
    - df: DataFrame, data for clustering
    - eps_range: list or tuple, range of eps values to try
    - min_samples_range: list or tuple, range of min_samples values to try
    
    Returns:
    - dict, containing optimal eps, min_samples and metrics
    """
    # Initialize variables to store metrics
    eps_values: List[float] = []
    min_samples_values: List[int] = []
    silhouette_scores: List[float] = []
    davies_bouldin_scores: List[float] = []
    number_of_labels: List[int] = []
    
    # Loop through different values of eps and min_samples to find the optimal ones
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Fit DBSCAN model
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels: np.ndarray = dbscan.fit_predict(df)
            
            # Only calculate metrics if more than one cluster is found
            if len(set(labels)) > 1:
                silhouette: float = silhouette_score(df, labels)
                davies_bouldin: float = davies_bouldin_score(df, labels)
                
                # Store metrics
                eps_values.append(eps)
                min_samples_values.append(min_samples)
                silhouette_scores.append(silhouette)
                davies_bouldin_scores.append(davies_bouldin)
                number_of_labels.append(len(set(labels)))
    
    # Finding the optimal eps and min_samples based on metrics
    # Lower Davies-Bouldin score is better. Higher silhouette score is better.
    optimal_index: int = np.argmin(davies_bouldin_scores)  # Change this based on the metric you prioritize
    optimal_eps: float = eps_values[optimal_index]
    optimal_min_samples: int = min_samples_values[optimal_index]
    optimal_labels: int = number_of_labels[optimal_index]
    
    # Compile metrics
    metrics: Dict[str, Union[float, int, List[Union[float, int]]]] = {
        'eps_values': eps_values,
        'min_samples_values': min_samples_values,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_eps': optimal_eps,
        'optimal_min_samples': optimal_min_samples,
        'optimal_labels': optimal_labels
    }
    
    return metrics

def optimize_kmeans(df: DataFrame, k_range: Tuple[int, int] = (2, 10)) -> Dict[str, List[object]]:
    """
    Apply and optimize K-means clustering on a given DataFrame.
    
    Parameters:
    - df: DataFrame, data for clustering
    - k_range: tuple, range of k values to try (inclusive)
    
    Returns:
    - dict, containing optimal k and metrics
    """
    # Initialize variables to store metrics
    k_values: List[int] = []
    inertias: List[float] = []
    silhouette_scores: List[float] = []
    davies_bouldin_scores: List[float] = []
    
    # Loop through different values of k to find the optimal one
    for k in range(k_range[0], k_range[1] + 1):  # Adding 1 to make the range inclusive
        # Fit K-means model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(df) 
        
        # Get cluster labels
        labels: np.ndarray = kmeans.labels_
        
        # Calculate metrics
        inertia: float = kmeans.inertia_
        silhouette: float = silhouette_score(df, labels)
        davies_bouldin: float = davies_bouldin_score(df, labels)
        
        # Store metrics
        k_values.append(k)
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        
    # Finding the optimal k based on metrics
    # Lower inertia and Davies-Bouldin score is better. Higher silhouette score is better.
    optimal_k: int = k_values[np.argmin(davies_bouldin_scores)]  # Change this based on the metric you prioritize
    
    # Compile metrics
    metrics: Dict[str, List[object]] = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k
    }
    
    return metrics

def csv_to_gzip_pandas_and_delete(csv_files: List[str]) -> None:
    """
    Convert a list of CSV files to gzipped files and delete the original CSV files.
    
    Parameters:
        csv_files (List[str]): List of paths to the CSV files to be converted.
        
    Returns:
        None: The function performs file operations and does not return any value.
    """
    for csv_file in csv_files:
        # Define the gzipped filename based on the original csv filename
        gzip_file = f"{csv_file}.gz"
        
        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Write the DataFrame to a GZIP file
        df.to_csv(gzip_file, compression='gzip', index=False)
        
        # Delete the original CSV file
        os.remove(csv_file)
        
        print(f"Converted {csv_file} to {gzip_file} and deleted the original file.")
    return

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
            if fnmatch.fnmatch(filename, '*.csv*'):
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
        key_name: str = os.path.splitext(os.path.splitext(base_name)[0])[0]
        
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

def transform_to_days(column: pd.Series) -> pd.Series:
    """
    Transforms a Pandas Series containing date-time strings into a Series 
    of integers representing the number of days from each date-time to the current date-time.
    
    Parameters:
        column (pd.Series): A Pandas Series containing date-time values.
        
    Returns:
        pd.Series: A Pandas Series containing integers representing the number of days from
                   each date-time to the current date-time.
                   
    Example:
        >>> transform_to_days(pd.Series(['2021-01-01', '2022-05-15', '2022-09-30']))
        0    650
        1    151
        2     15
        dtype: int64
    """
    # Convert the column to datetime format
    column_datetime = pd.to_datetime(column)
    
    # Get the current datetime
    current_datetime = datetime.now()
    
    # Subtract the datetime column from the current datetime
    time_delta = current_datetime - column_datetime
    
    # Extract the number of days from the time delta
    days = time_delta.dt.days
    
    return days





### Legacy Code ###

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

