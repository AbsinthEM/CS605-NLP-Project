import pandas as pd
import os
from pathlib import Path

def load_raw_data(file_path, rename_columns=True):
    """
    Load raw USS review data from CSV file and optionally rename columns
    
    Args:
        file_path (str): Path to the raw CSV file
        rename_columns (bool): Whether to rename problematic columns
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid CSV
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Try to infer encoding, but default to utf-8
        df = pd.read_csv(file_path)
        
        # Rename columns if requested
        if rename_columns:
            # Create a mapping of old column names to new ones
            column_mapping = {
                'reviewContext/Visited on': 'visit_date',
                'reviewContext/Wait time': 'wait_time'
            }
            
            # Only rename columns that exist in the DataFrame
            columns_to_rename = {old: new for old, new in column_mapping.items() if old in df.columns}
            if columns_to_rename:
                df = df.rename(columns=columns_to_rename)
                print(f"Renamed columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
        
        print(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {str(e)}")

def validate_raw_data(df):
    """
    Perform basic validation on the raw data
    
    Args:
        df (pd.DataFrame): Raw data DataFrame
        
    Returns:
        bool: True if data passes basic validation
        
    Raises:
        ValueError: If critical columns are missing
    """
    # Define the critical columns that must be present
    critical_columns = ['text', 'stars']
    
    # Check if all critical columns are present
    missing_columns = [col for col in critical_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Critical columns missing: {missing_columns}")
    
    # Check for records with both text and stars
    valid_records = df['text'].notna() & df['stars'].notna()
    print(f"Data validation: {valid_records.sum()} of {len(df)} records have both text and stars")
    
    return True

def get_column_summary(df):
    """
    Generate a summary of columns in the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to summarize
        
    Returns:
        pd.DataFrame: Summary of column information
    """
    summary = []
    
    for col in df.columns:
        non_null = df[col].count()
        null_percent = (len(df) - non_null) / len(df) * 100
        unique_values = df[col].nunique()
        data_type = df[col].dtype
        
        summary.append({
            'Column': col,
            'Type': data_type,
            'Non-Null Count': non_null,
            'Null %': null_percent,
            'Unique Values': unique_values
        })
    
    return pd.DataFrame(summary)