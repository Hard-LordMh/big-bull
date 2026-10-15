"""
test_pattern_analysis.py
------------------------
Automated tests for pattern_analysis.py.
"""

import pytest
import pandas as pd
from utils import generate_synthetic_stock_data
from pattern_analysis import (
    analyze_day_of_week, analyze_month_seasonality,
    analyze_year_performance, find_similar_historical_setups,
    analyze_day_of_month_repeat, get_optimal_intraday_windows
)


def test_analyze_day_of_week():
    df = generate_synthetic_stock_data(num_days=100)
    dow_df = analyze_day_of_week(df)
    assert not dow_df.empty
    assert "Day" in dow_df.columns
    assert "Mean_Return_Pct" in dow_df.columns
    assert "Win_Rate_Pct" in dow_df.columns
    assert len(dow_df) <= 5


def test_analyze_month_seasonality():
    df = generate_synthetic_stock_data(num_days=250)
    month_df = analyze_month_seasonality(df)
    assert not month_df.empty
    assert "Month" in month_df.columns
    assert "Sample_Count" in month_df.columns
    assert "Mean_Return_Pct" in month_df.columns


def test_analyze_year_performance():
    df = generate_synthetic_stock_data(num_days=300)
    year_df = analyze_year_performance(df)
    assert not year_df.empty
    assert "Year" in year_df.columns
    assert "Total_Return_Pct" in year_df.columns


def test_find_similar_historical_setups():
    df = generate_synthetic_stock_data(num_days=150)
    res = find_similar_historical_setups(df, top_k=5, horizon=5)
    assert "matches_table" in res
    assert "summary" in res
    assert len(res["matches_table"]) <= 5
    assert "up_frequency_pct" in res["summary"]


def test_analyze_day_of_month_repeat():
    df = generate_synthetic_stock_data(num_days=400)  # Increased from 150 to ensure enough samples
    dom_df = analyze_day_of_month_repeat(df)
    assert not dom_df.empty
    assert "Day_Of_Month" in dom_df.columns
    assert "Mean_Return_Pct" in dom_df.columns


def test_get_optimal_intraday_windows():
    from intraday_model import generate_synthetic_intraday_data
    df_intra = generate_synthetic_intraday_data("RELIANCE.NS", num_days=3)
    opt_dict = get_optimal_intraday_windows(df_intra)
    assert "best_buy_window" in opt_dict
    assert "best_sell_window" in opt_dict


def test_analyze_special_events_and_festivals():
    from pattern_analysis import analyze_special_events_and_festivals, detect_current_special_event
    df = generate_synthetic_stock_data(num_days=400)
    ev_df = analyze_special_events_and_festivals(df)
    assert not ev_df.empty
    assert "Event" in ev_df.columns
    assert "Mean_Return_Pct" in ev_df.columns
    assert "Win_Rate_Pct" in ev_df.columns

    cur_ev = detect_current_special_event("2026-10-25")
    assert "name" in cur_ev
    assert "Diwali" in cur_ev["name"]


