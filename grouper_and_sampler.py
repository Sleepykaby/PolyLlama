#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd


def grouper(*path):
    dataframes = []
    for p in path:
        try:
            df = pd.read_excel(p)
            dataframes.append(df)
            print(f"Successfully read the dataset: '{p}'")
        except Exception as e:
            print(f"Misson fail: {path}: {e}")
    
    return pd.concat(dataframes, axis=0, ignore_index=True)

def sampler(path: str, dataframe: pd.DataFrame = None, n_samples: int = 1, prune: bool = True, seed: int = None):
    """
    Sample data from an Excel file or a provided DataFrame based on the 'total count' column.

    Parameters:
    - path (str): The file path to the Excel file.
    - dataframe (pd.DataFrame): An optional DataFrame to use instead of reading from a file.
    - n_samples (int): The number of samples to draw from each group. Default is 1.
    - prune (bool): If True, return the remaining data after sampling. Default is True.
    - seed (int): Random seed for reproducibility. Default is None.

    Returns:
    - pd.DataFrame: Sampled DataFrame, and if prune is True, also returns remaining DataFrame.
    """
    # Read data from Excel file if path is provided and dataframe is not
    if (path is not None) and (dataframe is None):
        df = pd.read_excel(path)
    # Use the provided DataFrame if path is not provided
    elif (path is None) and (dataframe is not None):
        df = dataframe
    # Raise an error if both path and dataframe are provided
    elif (path is not None) and (dataframe is not None):
        raise ValueError("Param 'path' and 'dataframe' cannot be assigned together")
    
    # Create a dictionary to hold the number of samples for each unique group in 'total count'
    n_samples_dict = {group: n_samples for group in df['total count'].unique()}

    # List to store sampled DataFrames
    sampled_results = []

    # Iterate over each group and its corresponding number of samples
    for group, n in n_samples_dict.items():
        # Filter the DataFrame for the current group
        group_df = df[df['total count'] == group]
        
        # Skip if there are no entries for the current group
        if len(group_df) == 0:
            continue

        # Sample from the group DataFrame without replacement
        sampled_df = group_df.sample(n=min(len(group_df), n), replace=False, random_state=seed)
        sampled_results.append(sampled_df)

    # Concatenate all sampled DataFrames into a single DataFrame
    df_sampled = pd.concat(sampled_results, ignore_index=False)
    
    # If prune is True, create a DataFrame of the remaining data after sampling
    if prune:
        df_remain = df.drop(df_sampled.index)
        return df_sampled, df_remain

    return df_sampled
