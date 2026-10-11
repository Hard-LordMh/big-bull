"""
test_intraday_model.py
----------------------
Automated tests for intraday_model.py and validate_intraday_dataframe.
"""

import pytest
import pandas as pd
from intraday_model import generate_synthetic_intraday_data, train_and_predict_intraday
from data_validation import validate_intraday_dataframe


def test_generate_and_validate_intraday():
    df_intra = generate_synthetic_intraday_data(ticker="RELIANCE.NS", num_days=3, interval_mins=15)
    assert not df_intra.empty
    assert len(df_intra) == 75  # 25 bars * 3 days
    
    is_valid, errs, warns, clean_df = validate_intraday_dataframe(df_intra, min_bars=25)
    assert is_valid is True
    assert len(errs) == 0


def test_train_and_predict_intraday():
    df_intra = generate_synthetic_intraday_data(ticker="RELIANCE.NS", num_days=4, interval_mins=15)
    res = train_and_predict_intraday(df_intra)
    assert res["valid"] is True
    assert res["predicted_direction"] in ["UP", "DOWN"]
    assert 0.0 <= res["probability_up"] <= 1.0
    assert "test_accuracy" in res
