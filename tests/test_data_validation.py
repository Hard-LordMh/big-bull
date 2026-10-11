"""
test_data_validation.py
------------------------
Automated tests for data_validation.py module.
"""

import pytest
import pandas as pd
import numpy as np
from data_validation import validate_stock_dataframe, normalize_column_names
from utils import generate_synthetic_stock_data


def test_normalize_column_names():
    df = pd.DataFrame({
        'date': ['2023-01-01'],
        'OPEN': [100],
        'High Price': [105],
        'low': [95],
        'Close_Price': [102],
        'Vol': [1000]
    })
    norm_df = normalize_column_names(df)
    expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected:
        assert col in norm_df.columns


def test_validate_stock_dataframe_synthetic():
    df_synth = generate_synthetic_stock_data(num_days=100)
    is_valid, errors, warnings, df_clean = validate_stock_dataframe(df_synth, min_rows=50)
    assert is_valid is True
    assert len(errors) == 0
    assert len(df_clean) == 100
    assert list(df_clean.columns[:6]) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def test_validate_missing_columns():
    df_bad = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Open': [100],
        'Close': [105]
    })
    is_valid, errors, warnings, df_clean = validate_stock_dataframe(df_bad)
    assert is_valid is False
    assert any("Missing required column" in err for err in errors)


def test_validate_insufficient_rows():
    df_short = generate_synthetic_stock_data(num_days=20)
    is_valid, errors, warnings, df_clean = validate_stock_dataframe(df_short, min_rows=50)
    assert is_valid is False
    assert any("Insufficient historical data" in err for err in errors)


def test_validate_negative_prices():
    df_neg = generate_synthetic_stock_data(num_days=60)
    df_neg.loc[5, 'Close'] = -10.0
    is_valid, errors, warnings, df_clean = validate_stock_dataframe(df_neg)
    assert is_valid is False
    assert any("strictly positive" in err for err in errors)
