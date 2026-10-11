"""
test_feature_engineering.py
----------------------------
Automated tests for feature_engineering.py module.
"""

import pytest
import pandas as pd
import numpy as np
from utils import generate_synthetic_stock_data
from feature_engineering import build_features, create_target, prepare_datasets, calculate_rsi


def test_calculate_rsi():
    prices = pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20, 22, 24, 23, 25, 27, 26])
    rsi = calculate_rsi(prices, period=14)
    # Check RSI range is bounded [0, 100]
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_build_features_no_leakage():
    df_synth = generate_synthetic_stock_data(num_days=100)
    df_feat = build_features(df_synth)

    # Check required feature columns exist
    expected_features = [
        'Return_1D', 'Return_2D', 'Return_5D', 'Return_10D',
        'SMA_5', 'SMA_10', 'SMA_20', 'RSI_14'
    ]
    for feat in expected_features:
        assert feat in df_feat.columns

    # Verify Return_1D at index 10 only depends on Close[10] and Close[9]
    expected_ret = (df_synth.loc[10, 'Close'] / df_synth.loc[9, 'Close']) - 1.0
    assert pytest.approx(df_feat.loc[10, 'Return_1D'], rel=1e-4) == expected_ret


def test_target_creation_5d_horizon():
    df_synth = generate_synthetic_stock_data(num_days=100)
    df_target = create_target(df_synth, horizon=5)

    # Latest 5 rows must have Target as NaN (future is unobserved)
    assert df_target['Target_5D_UP'].tail(5).isna().all()

    # For historical rows, Target is 1 if Future_Close_5D > Close, else 0
    row_idx = 20
    fut_close = df_synth.loc[row_idx + 5, 'Close']
    curr_close = df_synth.loc[row_idx, 'Close']
    expected_target = 1 if fut_close > curr_close else 0
    assert df_target.loc[row_idx, 'Target_5D_UP'] == expected_target


def test_prepare_datasets_split():
    df_synth = generate_synthetic_stock_data(num_days=150)
    train_df, latest_df, feature_cols = prepare_datasets(df_synth, horizon=5)

    assert len(train_df) > 0
    assert len(latest_df) >= 1
    assert train_df['Target_5D_UP'].isna().sum() == 0
    assert len(feature_cols) > 0
