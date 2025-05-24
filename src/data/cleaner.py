import pandas as pd
import numpy as np
import re
import unicodedata
from langdetect import detect, LangDetectException, DetectorFactory

def process_translated_text(df, filter_empty=True):
    """
    Process text and textTranslated fields to create a consistent review field.
    Uses language detection to determine if text is in English.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        filter_empty (bool): If True, filter out rows where review cannot be created
    
    Returns:
        pd.DataFrame: DataFrame with review field and reset index
    """
    required_columns = ['text', 'textTranslated', 'translatedLanguage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Required columns for translation processing not found: {missing_columns}")
        return df
    
    df_clean = df.copy()
    
    # Initialize review column
    df_clean['review'] = None

    # Set seed for language detection to ensure reproducibility
    DetectorFactory.seed = 42
    
    # Helper function to detect if text is English
    def is_english(text):
        if pd.isna(text) or text == '':
            return False
        try:
            return detect(str(text)) == 'en'
        except LangDetectException:
            return False
    
    # Apply language detection to the text column
    print("Detecting language for text column...")
    # Apply the function to text column for non-null values
    text_lang_mask = df_clean['text'].notna() & (df_clean['text'] != '')
    
    # First case: text is in English, use it directly
    for idx in df_clean[text_lang_mask].index:
        if is_english(df_clean.loc[idx, 'text']):
            df_clean.loc[idx, 'review'] = df_clean.loc[idx, 'text']
    
    # Second case: text is not English, check translatedLanguage and textTranslated
    remaining_mask = df_clean['review'].isna()
    translation_mask = remaining_mask & df_clean['textTranslated'].notna() & (df_clean['translatedLanguage'] == 'en')
    df_clean.loc[translation_mask, 'review'] = df_clean.loc[translation_mask, 'textTranslated']
    
    # Count how many records used each case
    eng_count = (~remaining_mask).sum()
    trans_count = translation_mask.sum()
    
    print(f"Created review field: {eng_count} records used original English text")
    print(f"{trans_count} records used translated text")
    
    # Filter out rows where review is still None or NaN
    if filter_empty:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['review'].notna()]
        filtered_count = initial_count - len(df_clean)
        print(f"Filtered out {filtered_count} rows ({filtered_count/initial_count:.2%}) without valid review text")
        
        # Reset index to ensure continuous indices and avoid index sparsity issues
        df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def integrate_review_context(df):
    """
    Integrate review context information into integrated_review field
    using only visit_date and wait_time columns
    Keeps the original review column
    
    Args:
        df (pd.DataFrame): Input DataFrame with review column
    
    Returns:
        pd.DataFrame: DataFrame with integrated context and original review column
    """
    df_clean = df.copy()
    
    # Create integrated review field starting with review text
    if 'review' in df_clean.columns:
        df_clean['integrated_review'] = df_clean['review'].fillna('')
    else:
        print("Warning: review column not found, context integration may fail")
        df_clean['integrated_review'] = ''
    
    # Add visited time information if available
    if 'visit_date' in df_clean.columns:
        mask = df_clean['visit_date'].notna()
        df_clean.loc[mask, 'integrated_review'] += ' [VISIT_TIME: ' + df_clean.loc[mask, 'visit_date'].astype(str) + ']'
    
    # Add wait time information if available
    if 'wait_time' in df_clean.columns:
        mask = df_clean['wait_time'].notna()
        df_clean.loc[mask, 'integrated_review'] += ' [WAIT_TIME: ' + df_clean.loc[mask, 'wait_time'].astype(str) + ']'
    
    print(f"Created integrated_review field with context information")
    
    return df_clean

def remove_empty_records(df, required_columns=None):
    """
    Remove records with empty values in required columns and reset index
    to ensure continuous indices for subsequent processing.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_columns (list, optional): List of columns that must contain values.
            Defaults to ['review', 'stars', 'integrated_review', 'publishAt']
    
    Returns:
        pd.DataFrame: DataFrame with empty records removed and index reset
    """
    if required_columns is None:
        required_columns = ['review', 'stars', 'integrated_review', 'publishAt']
    
    initial_count = len(df)
    df_clean = df.copy()
    
    # Remove records where required columns are empty
    for col in required_columns:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col].notna()]
    
    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} records ({removed_count/initial_count:.2%}) with missing values in {required_columns}")
    
    # Reset index to ensure continuous indices and avoid index sparsity issues
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

import unicodedata
import re
import numpy as np
import pandas as pd

def clean_text_field(df, text_cols=None):
    """
    Clean the text fields by removing extra whitespace, special characters, etc.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_cols (list, optional): List of column names to clean. 
                                  Defaults to ['review', 'integrated_review'].
    
    Returns:
        pd.DataFrame: DataFrame with cleaned text and empty rows removed
    """
    if text_cols is None:
        text_cols = ['review', 'integrated_review']
    
    df_clean = df.copy()
    
    # Process each column in the list
    for text_col in text_cols:
        if text_col not in df_clean.columns:
            print(f"Column {text_col} not found in DataFrame. Skipping text cleaning for this column.")
            continue
        
        # Count non-null values before cleaning
        non_null_before = df_clean[text_col].notna().sum()
        
        # Replace NaN with empty string to avoid errors
        df_clean[text_col] = df_clean[text_col].fillna('')
        
        # Clean text: normalize whitespace
        df_clean[text_col] = df_clean[text_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Remove HTML tags if present
        df_clean[text_col] = df_clean[text_col].str.replace(r'<[^>]+>', '', regex=True)
        
        # Convert empty strings back to NaN
        df_clean.loc[df_clean[text_col] == '', text_col] = np.nan
        
        # Count non-null values after cleaning
        non_null_after = df_clean[text_col].notna().sum()
        print(f"Text cleaning for {text_col}: {non_null_before - non_null_after} text fields became empty after cleaning")
    
    # Check for any rows with null values in specified text columns and remove them
    rows_before = len(df_clean)
    null_mask = df_clean[text_cols].isnull().any(axis=1)
    null_count = null_mask.sum()
    
    if null_count > 0:
        print(f"Found {null_count} rows with null values in text columns. Removing these rows.")
        df_clean = df_clean[~null_mask].reset_index(drop=True)
        rows_after = len(df_clean)
        print(f"Rows removed: {rows_before - rows_after}. Remaining rows: {rows_after}")
    
    return df_clean

def remove_special_chars(df, text_cols=None):
    """
    Remove garbled text and non-standard characters using a universal approach
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_cols (list, optional): List of column names to clean.
                                  Defaults to ['review', 'integrated_review'].
    
    Returns:
        pd.DataFrame: DataFrame with cleaned text and empty rows removed
    """
    if text_cols is None:
        text_cols = ['review', 'integrated_review']
    
    df_clean = df.copy()
    
    def clean_text(text):
        """
        Clean text by removing special characters and normalizing unicode
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Step 1: Normalize unicode characters (decompose then compose)
        text = unicodedata.normalize('NFKC', text)
        
        # Step 2: Keep only characters from allowed Unicode categories
        # This keeps letters, numbers, punctuation, and some symbols
        allowed_categories = {'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Ps', 'Pe', 
                             'Pi', 'Pf', 'Po', 'Zs'}
        filtered_text = ''.join(c for c in text if unicodedata.category(c) in allowed_categories)
        
        # Step 3: Remove any remaining non-ASCII characters that might be garbled
        ascii_text = re.sub(r'[^\x00-\x7F]+', '', filtered_text)
        
        # Step 4: Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', ascii_text).strip()
        
        return cleaned_text
    
    # Process each column in the list
    for text_col in text_cols:
        if text_col not in df_clean.columns:
            print(f"Column {text_col} not found in DataFrame. Skipping special character removal for this column.")
            continue
        
        # Count non-null values before cleaning
        non_null_before = df_clean[text_col].notna().sum()
        
        # Replace NaN with empty string to avoid errors
        df_clean[text_col] = df_clean[text_col].fillna('')
        
        # Apply cleaning function
        df_clean[text_col] = df_clean[text_col].apply(clean_text)
        
        # Convert empty strings back to NaN
        df_clean.loc[df_clean[text_col] == '', text_col] = np.nan
        
        # Count non-null values after cleaning
        non_null_after = df_clean[text_col].notna().sum()
        print(f"Special character removal for {text_col}: {non_null_before - non_null_after} text fields became empty after cleaning")
    
    # Check for any rows with null values in specified text columns and remove them
    rows_before = len(df_clean)
    null_mask = df_clean[text_cols].isnull().any(axis=1)
    null_count = null_mask.sum()
    
    if null_count > 0:
        print(f"Found {null_count} rows with null values in text columns. Removing these rows.")
        df_clean = df_clean[~null_mask].reset_index(drop=True)
        rows_after = len(df_clean)
        print(f"Rows removed: {rows_before - rows_after}. Remaining rows: {rows_after}")
    
    return df_clean