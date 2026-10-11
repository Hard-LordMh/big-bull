"""
data_validation.py
------------------
Module for validating and cleaning input stock market historical datasets (Daily & Intraday).
Ensures column presence, data types, chronological ordering, missing values,
and price logic sanity before feature engineering and modeling.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


REQUIRED_DAILY_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
REQUIRED_INTRADAY_COLUMNS = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']


def normalize_column_names(df: pd.DataFrame, is_intraday: bool = False) -> pd.DataFrame:
    """
    Standardize DataFrame column names to match ['Date'/'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'].
    Supports common variations like 'date', 'datetime', 'timestamp', 'close price', 'vol', etc.
    """
    df = df.copy()
    col_mapping = {}
    time_target = 'Timestamp' if is_intraday else 'Date'

    for col in df.columns:
        c_clean = str(col).strip().lower().replace(' ', '_').replace('_price', '')
        if c_clean in ['date', 'datetime', 'timestamp', 'time']:
            col_mapping[col] = time_target
        elif c_clean in ['open']:
            col_mapping[col] = 'Open'
        elif c_clean in ['high']:
            col_mapping[col] = 'High'
        elif c_clean in ['low']:
            col_mapping[col] = 'Low'
        elif c_clean in ['close', 'adj_close', 'adjusted_close', 'last']:
            col_mapping[col] = 'Close'
        elif c_clean in ['volume', 'vol', 'total_traded_quantity']:
            col_mapping[col] = 'Volume'
    
    return df.rename(columns=col_mapping)


def validate_stock_dataframe(df: pd.DataFrame, min_rows: int = 50) -> Tuple[bool, List[str], List[str], pd.DataFrame]:
    """
    Validates daily stock market DataFrame.

    Parameters:
        df (pd.DataFrame): Raw uploaded dataframe.
        min_rows (int): Minimum required trading days (default 50).

    Returns:
        is_valid (bool): True if dataset passes critical validation, False otherwise.
        errors (List[str]): Critical error messages preventing model training.
        warnings (List[str]): Non-fatal warnings for data quality awareness.
        cleaned_df (pd.DataFrame): Cleaned and chronologically sorted DataFrame.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return False, ["Uploaded file is empty or could not be parsed."], [], pd.DataFrame()

    # Step 1: Normalize column names
    df_clean = normalize_column_names(df, is_intraday=False)
    missing_cols = [col for col in REQUIRED_DAILY_COLUMNS if col not in df_clean.columns]
    
    if missing_cols:
        errors.append(f"Missing required column(s): {', '.join(missing_cols)}. Required columns: {REQUIRED_DAILY_COLUMNS}")
        return False, errors, warnings, pd.DataFrame()

    # Retain required columns plus optional Ticker column
    extra_cols = [c for c in df_clean.columns if c not in REQUIRED_DAILY_COLUMNS]
    cols_to_keep = REQUIRED_DAILY_COLUMNS + (['Ticker'] if 'Ticker' in extra_cols else [])
    df_clean = df_clean[cols_to_keep].copy()

    # Step 2: Validate & parse Date
    try:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    except Exception as e:
        errors.append(f"Failed to parse 'Date' column: {str(e)}")
        return False, errors, warnings, pd.DataFrame()

    null_dates = df_clean['Date'].isnull().sum()
    if null_dates > 0:
        errors.append(f"Found {null_dates} invalid or unparseable Date entry/entries.")
        return False, errors, warnings, pd.DataFrame()

    # Check duplicate dates
    duplicate_dates = df_clean['Date'].duplicated().sum()
    if duplicate_dates > 0:
        warnings.append(f"Found {duplicate_dates} duplicate date entries. Keeping the last observation for each date.")
        df_clean = df_clean.drop_duplicates(subset=['Date'], keep='last')

    # Sort chronologically ascending
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)

    # Step 3: Parse and validate numeric OHLCV values
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    null_counts = df_clean[numeric_cols].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        warnings.append(f"Found missing numerical values. Imputing with forward/backward fill.")
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()

    if df_clean[numeric_cols].isnull().sum().sum() > 0:
        errors.append("Dataset contains non-numeric data that could not be imputed.")
        return False, errors, warnings, pd.DataFrame()

    # Check positive prices
    non_positive_prices = (df_clean[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum()
    if non_positive_prices > 0:
        errors.append(f"Found {non_positive_prices} row(s) with price <= 0. Prices must be strictly positive.")

    # Check non-negative volume
    negative_volume = (df_clean['Volume'] < 0).sum()
    if negative_volume > 0:
        errors.append(f"Found {negative_volume} row(s) with negative Volume.")

    # Check OHLC logic sanity
    invalid_ohlc = (
        (df_clean['High'] < df_clean['Low']) |
        (df_clean['High'] < df_clean['Open']) |
        (df_clean['High'] < df_clean['Close']) |
        (df_clean['Low'] > df_clean['Open']) |
        (df_clean['Low'] > df_clean['Close'])
    ).sum()

    if invalid_ohlc > 0:
        warnings.append(f"Found {invalid_ohlc} row(s) where High/Low price logic was inconsistent. Automatically adjusted.")
        df_clean['High'] = df_clean[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df_clean['Low'] = df_clean[['Open', 'High', 'Low', 'Close']].min(axis=1)

    # Step 4: Check dataset length
    if len(df_clean) < min_rows:
        errors.append(f"Insufficient historical data: {len(df_clean)} rows found. At least {min_rows} trading days required.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, df_clean


def validate_intraday_dataframe(df: pd.DataFrame, min_bars: int = 30) -> Tuple[bool, List[str], List[str], pd.DataFrame]:
    """
    Validates an intraday candlestick dataset (e.g. 5m, 15m intervals).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return False, ["Uploaded intraday file is empty."], [], pd.DataFrame()

    df_clean = normalize_column_names(df, is_intraday=True)
    missing_cols = [col for col in REQUIRED_INTRADAY_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        errors.append(f"Missing required intraday column(s): {', '.join(missing_cols)}.")
        return False, errors, warnings, pd.DataFrame()

    # Parse Timestamp
    try:
        df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'], errors='coerce')
    except Exception as e:
        errors.append(f"Failed to parse 'Timestamp': {str(e)}")
        return False, errors, warnings, pd.DataFrame()

    if df_clean['Timestamp'].isnull().any():
        errors.append("Dataset contains unparseable Timestamp entries.")
        return False, errors, warnings, pd.DataFrame()

    # Drop duplicate timestamps and sort
    if df_clean['Timestamp'].duplicated().any():
        warnings.append("Duplicate timestamps found. Deduplicated to last observation.")
        df_clean = df_clean.drop_duplicates(subset=['Timestamp'], keep='last')

    df_clean = df_clean.sort_values('Timestamp').reset_index(drop=True)

    # Validate numbers
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()

    if (df_clean[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).any():
        errors.append("Prices must be strictly positive.")

    # High / Low sanity adjustment
    df_clean['High'] = df_clean[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df_clean['Low'] = df_clean[['Open', 'High', 'Low', 'Close']].min(axis=1)

    if len(df_clean) < min_bars:
        errors.append(f"Insufficient intraday bars ({len(df_clean)} found, {min_bars} required).")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, df_clean
