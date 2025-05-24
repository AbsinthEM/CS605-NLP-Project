import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def select_columns_for_silver(df, columns=None):
    """
    Select columns to include in the Silver layer
    
    Args:
        df (pd.DataFrame): Input DataFrame (Bronze layer)
        columns (list, optional): List of columns to select if present
    
    Returns:
        pd.DataFrame: Silver layer DataFrame with selected columns
    """
    if columns is None:
        columns = ['integrated_review', 'stars', 'name', 'review', 'publishedAtDate']
    
    # Select available columns from the desired list
    available_columns = [col for col in columns if col in df.columns]
    
    # Check if we have at least some columns
    if not available_columns:
        raise ValueError(f"None of the requested columns {columns} exist in the dataframe")
    
    df_silver = df[available_columns].copy()
    print(f"Selected {len(available_columns)} columns for Silver layer: {available_columns}")
    
    return df_silver


def anonymize_user_data(df, columns_to_anonymize=None):
    """
    Anonymize sensitive user data
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_anonymize (list, optional): List of columns to anonymize.
            Defaults to ['name'].
    
    Returns:
        pd.DataFrame: DataFrame with anonymized data
    """
    if columns_to_anonymize is None:
        columns_to_anonymize = ['name']
    
    df_silver = df.copy()
    
    for col in columns_to_anonymize:
        if col in df_silver.columns and df_silver[col].notna().any():
            # Create a mapping of original values to hashed/anonymized values
            unique_values = df_silver[col].dropna().unique()
            value_map = {val: f"user_{i}" for i, val in enumerate(unique_values)}
            
            # Apply the mapping
            df_silver[col] = df_silver[col].map(lambda x: value_map.get(x, x))
            print(f"Anonymized {len(unique_values)} unique values in column: {col}")
    
    return df_silver

def extract_standardized_dates(df, date_col='publishedAtDate'):
    """
    Extract standardized dates from the publishedAtDate column.
    Keep the original column name but standardize the date format to YYYY-MM-DD.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Column containing date strings
    
    Returns:
        pd.DataFrame: DataFrame with standardized dates in the original column
    """
    import pandas as pd
    
    # Create a copy of the input DataFrame
    df_result = df.copy()
    
    # Skip if the date column doesn't exist
    if date_col not in df_result.columns:
        print(f"Column {date_col} not found. Skipping date extraction.")
        return df_result
    
    # Convert dates to standard format
    try:
        # Convert to datetime, then extract only date part (no time)
        df_result[date_col] = pd.to_datetime(df_result[date_col]).dt.date
        
        # Count successful conversions
        conversion_count = df_result[date_col].notna().sum()
        print(f"Converted {conversion_count} dates to standard format (YYYY-MM-DD)")
    except Exception as e:
        print(f"Error converting dates: {str(e)}")
    
    return df_result

def create_data_splits(df, split_column='data_split', test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits
    
    Args:
        df (pd.DataFrame): Input DataFrame
        split_column (str, optional): Name of column to store split information
        test_size (float, optional): Proportion for test set
        val_size (float, optional): Proportion for validation set
        random_state (int, optional): Random seed
    
    Returns:
        pd.DataFrame: DataFrame with added split column
    """
    df_silver = df.copy()
    
    # First split off the test set
    train_val, test = train_test_split(
        df_silver, test_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into train and validation
    # Calculate the validation size as a proportion of the train_val set
    effective_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=effective_val_size, random_state=random_state
    )
    
    # Add split information to the original DataFrame
    df_silver[split_column] = None
    df_silver.loc[train.index, split_column] = 'train'
    df_silver.loc[val.index, split_column] = 'val'
    df_silver.loc[test.index, split_column] = 'test'
    
    # Log split counts
    print(f"Data split: train={len(train)} ({len(train)/len(df_silver):.2%}), "
          f"val={len(val)} ({len(val)/len(df_silver):.2%}), "
          f"test={len(test)} ({len(test)/len(df_silver):.2%})")
    
    return df_silver

def transform_to_silver(df_bronze, output_path=None):
    """
    Transform Bronze layer data to Silver layer
    
    Args:
        df_bronze (pd.DataFrame): Bronze layer data
        output_path (str, optional): Path to save the Silver layer data
    
    Returns:
        pd.DataFrame: Silver layer DataFrame
    """
    from .cleaner import (
        remove_empty_records, clean_text_field, remove_special_chars, 
        process_translated_text, integrate_review_context
    )
    
    # Apply data cleaning steps
    print("Starting transformation from Bronze to Silver layer")
    
    df_clean = process_translated_text(df_bronze)
    df_clean = integrate_review_context(df_clean)
    df_clean = remove_empty_records(df_clean, required_columns=['review', 'stars', 'integrated_review', 'publishedAtDate'])
    df_clean = clean_text_field(df_clean, text_cols=['review', 'integrated_review'])
    df_clean = remove_special_chars(df_clean, text_cols=['review', 'integrated_review'])
    
    # Transform to Silver layer structure
    df_silver = extract_standardized_dates(df_clean)
    df_silver = select_columns_for_silver(df_silver)
    df_silver = anonymize_user_data(df_silver)
    df_silver = create_data_splits(df_silver)
    
    # Add a unique index
    df_silver.reset_index(drop=True, inplace=True)
    df_silver['review_index'] = df_silver.index
    
    print(f"Successfully created Silver layer with {len(df_silver)} records")
    
    # Save to file if output_path is provided
    if output_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        silver_csv_filename = 'USS_Reviews_Silver.csv'
        silver_parquet_fileanme = 'USS_Reviews_Silver.parquet'

        silver_csv_path = os.path.join(output_path, silver_csv_filename)
        silver_parquet_path = os.path.join(output_path, silver_parquet_fileanme)

        # Save to CSV
        df_silver.to_csv(silver_csv_path, index=False)
        print(f"Saved Silver layer csv data to {silver_csv_path}")

        # Save to Parquet
        df_silver.to_parquet(silver_parquet_path, index=False)
        print(f"Saved Silver layer parquet data to {silver_parquet_path}")
    
    return df_silver
