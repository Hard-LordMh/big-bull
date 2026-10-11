"""
test_stock_model.py
-------------------
Automated tests for stock_model.py and utils export functionality.
"""

import pytest
import os
import pandas as pd
from utils import generate_synthetic_stock_data, export_pipeline_results
from feature_engineering import prepare_datasets
from stock_model import (
    train_and_evaluate_model,
    predict_latest_session,
    walk_forward_prediction_history,
    summarize_prediction_history,
    predict_next_week
)


def test_train_evaluate_and_predict():
    # 1. Generate synthetic data
    df_synth = generate_synthetic_stock_data(num_days=200, seed=42)
    train_df, latest_df, feature_cols = prepare_datasets(df_synth, horizon=5)

    # 2. Train model & run backtest
    results = train_and_evaluate_model(train_df, feature_cols)

    assert "model" in results
    assert "test_metrics" in results
    metrics = results["test_metrics"]
    assert "accuracy" in metrics
    assert "baseline_accuracy" in metrics
    assert "confusion_matrix" in metrics

    # 3. Predict latest session
    latest_pred = predict_latest_session(results["model"], latest_df, feature_cols, ticker="RELIANCE.NS")
    assert latest_pred["predicted_direction"] in ["UP", "DOWN"]
    assert 0.0 <= latest_pred["probability_up"] <= 1.0
    assert 0.0 <= latest_pred["probability_down"] <= 1.0
    assert pytest.approx(latest_pred["probability_up"] + latest_pred["probability_down"]) == 1.0

    # 4. Test output file export
    saved_files = export_pipeline_results(
        raw_df=df_synth,
        backtest_predictions=results["backtest_predictions"],
        test_metrics=results["test_metrics"],
        walk_forward_df=results["walk_forward_df"],
        feat_importance_df=results["feature_importance"],
        latest_pred_dict=latest_pred,
        model_object=results["model"],
        ticker="RELIANCE.NS"
    )

    for key, filepath in saved_files.items():
        assert os.path.exists(filepath), f"Export file missing: {filepath}"


def test_rf_detection_features():
    df_synth = generate_synthetic_stock_data(num_days=150, seed=42)
    train_df, latest_df, feature_cols = prepare_datasets(df_synth, horizon=5)
    results = train_and_evaluate_model(train_df, feature_cols)
    model = results["model"]

    # Test predict_latest_session returns full rf_detection
    latest_pred = predict_latest_session(model, latest_df, feature_cols, full_df=df_synth)
    assert "rf_detection" in latest_pred
    rf_det = latest_pred["rf_detection"]
    assert "tree_consensus" in rf_det
    assert "market_regime" in rf_det
    assert "breakout_anomaly" in rf_det

    # Test predict_5day_trajectory
    from stock_model import predict_5day_trajectory
    traj_df = predict_5day_trajectory(df_synth, feature_cols, model)
    assert not traj_df.empty
    assert len(traj_df) == 5
    assert "Predicted_Close_INR" in traj_df.columns
    assert "RF_Probability_UP" in traj_df.columns

    assert "rf_detection" in latest_pred
    rf_det = latest_pred["rf_detection"]

    # 1. Tree consensus
    assert "tree_consensus" in rf_det
    tc = rf_det["tree_consensus"]
    assert tc["total_trees"] == 100
    assert tc["trees_up"] + tc["trees_down"] == 100
    assert 50.0 <= tc["consensus_pct"] <= 100.0
    assert tc["majority_direction"] in ["UP", "DOWN"]
    assert tc["strength"] in ["HIGH_CONVICTION", "MODERATE_CONVICTION", "DIVIDED_FOREST"]

    # 2. Market regime
    assert "market_regime" in rf_det
    mr = rf_det["market_regime"]
    assert mr["regime_key"] in ["BULLISH_MOMENTUM", "BEARISH_TREND", "HIGH_VOLATILITY_EXPANSION", "CONSOLIDATION_RANGE"]
    assert "regime_name" in mr
    assert "badge" in mr

    # 3. Isolation Forest Breakout Anomalies
    assert "breakout_anomaly" in rf_det
    ba = rf_det["breakout_anomaly"]
    assert isinstance(ba["is_anomaly"], bool)
    assert "badge" in ba
    assert ba["historical_anomalies_count"] >= 0

    # 4. Driving factors
    assert "driving_factors" in rf_det
    assert isinstance(rf_det["driving_factors"], list)
    assert len(rf_det["driving_factors"]) > 0
    assert "feature" in rf_det["driving_factors"][0]


def test_walk_forward_prediction_history_and_summary():
    df_synth = generate_synthetic_stock_data(num_days=900, seed=7)

    history = walk_forward_prediction_history(df_synth, horizon=5, years=2, refit_every=45)

    assert not history.empty
    expected_cols = {
        "Signal_Date", "Year", "Fold", "Reference_Close", "Predicted_Direction",
        "Prob_UP_Pct", "Actual_Direction", "Actual_Return_Pct", "Result"
    }
    assert expected_cols.issubset(set(history.columns))
    assert history["Predicted_Direction"].isin(["UP", "DOWN"]).all()
    assert history["Actual_Direction"].isin(["UP", "DOWN"]).all()
    assert history["Result"].isin(["HIT", "MISS"]).all()
    assert history["Prob_UP_Pct"].between(0.0, 100.0).all()
    assert history["Signal_Date"].is_monotonic_increasing

    summary = summarize_prediction_history(history)
    assert summary["valid"] is True
    assert summary["signals"] == len(history)
    assert 0.0 <= summary["hit_rate_pct"] <= 100.0
    assert not summary["yearly"].empty
    assert set(summary["yearly"]["Year"]) == set(history["Year"].unique())


def test_predict_next_week_directions():
    df_synth = generate_synthetic_stock_data(num_days=600, seed=11)

    next_week = predict_next_week(df_synth, sessions=5)

    assert len(next_week) == 5
    assert next_week["Predicted_Direction"].isin(["UP", "DOWN"]).all()
    assert next_week["Prob_UP_Pct"].between(0.0, 100.0).all()
    assert (next_week["Trading_Date"] > df_synth["Date"].iloc[-1]).all()
    assert next_week["Projected_Close_INR"].gt(0).all()

    assert next_week.attrs["week_direction"] in ["UP", "DOWN"]
    mean_prob = next_week["Prob_UP_Pct"].mean()
    expected_dir = "UP" if mean_prob >= 50.0 else "DOWN"
    assert next_week.attrs["week_direction"] == expected_dir
    assert next_week.attrs["week_prob_up"] == pytest.approx(round(mean_prob, 1), abs=0.2)


def test_history_engine_handles_short_data():
    df_synth = generate_synthetic_stock_data(num_days=30, seed=3)
    assert walk_forward_prediction_history(df_synth, years=5).empty
    assert predict_next_week(df_synth, sessions=5).empty

