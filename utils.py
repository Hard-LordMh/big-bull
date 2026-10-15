"""
utils.py
--------
Utility functions for file management, Power BI export generation,
ZIP package bundling, and synthetic dataset generation for daily and intraday analysis.
"""

import os
import json
import zipfile
import io
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def ensure_directories_exist():
    """Ensures output and model persistence directories exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def export_pipeline_results(
    raw_df: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    test_metrics: Dict[str, Any],
    walk_forward_df: pd.DataFrame,
    feat_importance_df: pd.DataFrame,
    latest_pred_dict: Dict[str, Any],
    model_object: Any,
    ticker: str = "NSE_STOCK",
    timing_signal_dict: Optional[Dict[str, Any]] = None,
    pattern_summary_df: Optional[pd.DataFrame] = None,
    intraday_pred_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Saves all output CSVs, summary JSON, and trained Joblib model into 'outputs/' and 'models/'.
    Conforms to standardized schemas for direct import into Microsoft Power BI Desktop.
    """
    ensure_directories_exist()
    saved_files = {}

    # 1. historical_prices.csv
    hist_path = os.path.join(OUTPUTS_DIR, 'historical_prices.csv')
    df_hist_export = raw_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_hist_export['Date'] = pd.to_datetime(df_hist_export['Date']).dt.strftime('%Y-%m-%d')
    df_hist_export['Ticker'] = ticker
    df_hist_export.to_csv(hist_path, index=False)
    saved_files['historical_prices'] = hist_path

    # 2. backtest_predictions.csv
    backtest_path = os.path.join(OUTPUTS_DIR, 'backtest_predictions.csv')
    df_bt_export = backtest_predictions.copy()
    df_bt_export['Date'] = pd.to_datetime(df_bt_export['Date']).dt.strftime('%Y-%m-%d')
    df_bt_export['Ticker'] = ticker
    df_bt_export.to_csv(backtest_path, index=False)
    saved_files['backtest_predictions'] = backtest_path

    # 3. holdout_metrics.csv
    holdout_path = os.path.join(OUTPUTS_DIR, 'holdout_metrics.csv')
    cm = test_metrics.get("confusion_matrix", {})
    holdout_df = pd.DataFrame([{
        "Ticker": ticker,
        "Accuracy": test_metrics.get("accuracy"),
        "Balanced_Accuracy": test_metrics.get("balanced_accuracy"),
        "Precision": test_metrics.get("precision"),
        "Recall": test_metrics.get("recall"),
        "F1_Score": test_metrics.get("f1_score"),
        "ROC_AUC": test_metrics.get("roc_auc"),
        "Baseline_Accuracy": test_metrics.get("baseline_accuracy"),
        "Outperforms_Baseline": test_metrics.get("outperforms_baseline"),
        "TN": cm.get("TN"),
        "FP": cm.get("FP"),
        "FN": cm.get("FN"),
        "TP": cm.get("TP")
    }])
    holdout_df.to_csv(holdout_path, index=False)
    saved_files['holdout_metrics'] = holdout_path

    # 4. walk_forward_metrics.csv
    wf_path = os.path.join(OUTPUTS_DIR, 'walk_forward_metrics.csv')
    if not walk_forward_df.empty:
        wf_export = walk_forward_df.copy()
        wf_export['Ticker'] = ticker
        wf_export.to_csv(wf_path, index=False)
    else:
        pd.DataFrame([{"Ticker": ticker, "Status": "Single fold evaluated"}]).to_csv(wf_path, index=False)
    saved_files['walk_forward_metrics'] = wf_path

    # 5. feature_importance.csv
    feat_path = os.path.join(OUTPUTS_DIR, 'feature_importance.csv')
    df_feat_export = feat_importance_df.copy()
    df_feat_export['Ticker'] = ticker
    df_feat_export.to_csv(feat_path, index=False)
    saved_files['feature_importance'] = feat_path

    # 6. latest_prediction.csv
    latest_path = os.path.join(OUTPUTS_DIR, 'latest_prediction.csv')
    df_latest_export = pd.DataFrame([latest_pred_dict])
    df_latest_export.to_csv(latest_path, index=False)
    saved_files['latest_prediction'] = latest_path

    # 7. daily_predictions.csv (Horizon 1D)
    daily_path = os.path.join(OUTPUTS_DIR, 'daily_predictions.csv')
    daily_df = pd.DataFrame([{
        "Ticker": ticker,
        "Signal_Date": latest_pred_dict.get("signal_date"),
        "Forecast_Horizon": "1 Day",
        "Predicted_Direction": latest_pred_dict.get("predicted_direction"),
        "Prob_UP": latest_pred_dict.get("probability_up"),
        "Prob_DOWN": latest_pred_dict.get("probability_down"),
        "Reference_Close": latest_pred_dict.get("latest_close")
    }])
    daily_df.to_csv(daily_path, index=False)
    saved_files['daily_predictions'] = daily_path

    # 8. five_day_predictions.csv (Horizon 5D)
    five_day_path = os.path.join(OUTPUTS_DIR, 'five_day_predictions.csv')
    five_day_df = pd.DataFrame([{
        "Ticker": ticker,
        "Signal_Date": latest_pred_dict.get("signal_date"),
        "Forecast_Horizon": "5 Days",
        "Predicted_Direction": latest_pred_dict.get("predicted_direction"),
        "Prob_UP": latest_pred_dict.get("probability_up"),
        "Prob_DOWN": latest_pred_dict.get("probability_down"),
        "Reference_Close": latest_pred_dict.get("latest_close")
    }])
    five_day_df.to_csv(five_day_path, index=False)
    saved_files['five_day_predictions'] = five_day_path

    # 9. timing_signals.csv
    timing_path = os.path.join(OUTPUTS_DIR, 'timing_signals.csv')
    if timing_signal_dict:
        timing_df = pd.DataFrame([timing_signal_dict])
    else:
        timing_df = pd.DataFrame([{
            "Ticker": ticker,
            "Signal": "Hold / No Signal",
            "Reference_Price": latest_pred_dict.get("latest_close"),
            "Signal_Date": latest_pred_dict.get("signal_date")
        }])
    timing_df.to_csv(timing_path, index=False)
    saved_files['timing_signals'] = timing_path

    # 10. historical_patterns.csv
    patterns_path = os.path.join(OUTPUTS_DIR, 'historical_patterns.csv')
    if pattern_summary_df is not None and not pattern_summary_df.empty:
        pattern_summary_df.to_csv(patterns_path, index=False)
    else:
        pd.DataFrame([{"Ticker": ticker, "Status": "Standard Seasonality Recorded"}]).to_csv(patterns_path, index=False)
    saved_files['historical_patterns'] = patterns_path

    # 11. intraday_predictions.csv
    intraday_path = os.path.join(OUTPUTS_DIR, 'intraday_predictions.csv')
    if intraday_pred_dict:
        intraday_df = pd.DataFrame([intraday_pred_dict])
    else:
        intraday_df = pd.DataFrame([{
            "Ticker": ticker,
            "Status": "Intraday analysis available on Intraday page"
        }])
    intraday_df.to_csv(intraday_path, index=False)
    saved_files['intraday_predictions'] = intraday_path

    # 12. run_summary.json
    summary_path = os.path.join(OUTPUTS_DIR, 'run_summary.json')
    summary_data = {
        "ticker": ticker,
        "test_metrics": test_metrics,
        "latest_prediction": latest_pred_dict
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    saved_files['run_summary'] = summary_path

    # 13. nse_direction_model.joblib
    model_path = os.path.join(MODELS_DIR, 'nse_direction_model.joblib')
    model_package = {
        "model": model_object,
        "ticker": ticker,
        "feature_names": list(feat_importance_df['Feature']),
        "horizon_sessions": latest_pred_dict.get("forecast_horizon_sessions", 5),
        "trained_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "latest_prediction": latest_pred_dict
    }
    joblib.dump(model_package, model_path)
    saved_files['joblib_model'] = model_path

    return saved_files


def create_powerbi_zip_bundle() -> io.BytesIO:
    """
    Bundles all generated CSV files in `outputs/` into a single ZIP archive for one-click download.
    """
    ensure_directories_exist()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUTS_DIR):
            if fname.endswith(('.csv', '.json')):
                fpath = os.path.join(OUTPUTS_DIR, fname)
                zf.write(fpath, arcname=fname)
    zip_buffer.seek(0)
    return zip_buffer


def generate_synthetic_stock_data(
    ticker: str = "RELIANCE.NS",
    num_days: int = 350,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic daily stock dataset for testing and demonstration.
    Explicitly labeled as synthetic artificial data.
    
    WARNING: This is NOT real market data. Used only when real data is unavailable.
    """
    np.random.seed(seed)
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='B')
    
    start_price = 2500.0
    mu = 0.0005
    sigma = 0.018
    
    daily_returns = np.random.normal(mu, sigma, size=num_days)
    price_path = start_price * np.exp(np.cumsum(daily_returns))
    
    open_prices = price_path * (1.0 + np.random.uniform(-0.005, 0.005, size=num_days))
    high_prices = np.maximum(price_path, open_prices) * (1.0 + np.random.uniform(0.001, 0.015, size=num_days))
    low_prices = np.minimum(price_path, open_prices) * (1.0 - np.random.uniform(0.001, 0.015, size=num_days))
    close_prices = price_path
    
    volumes = np.random.randint(500000, 5000000, size=num_days)
    
    df = pd.DataFrame({
        'Date': date_range.strftime('%Y-%m-%d'),
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': volumes,
        'Ticker': ticker,
        'Dataset_Type': 'SYNTHETIC_ARTIFICIAL_TEST_DATA',
        'data_source': 'SYNTHETIC_FALLBACK',
        'is_fallback': True,
        'is_stale': False  # Synthetic data is always "fresh" by definition
    })
    
    return df


def fetch_real_or_fallback_stock_data(
    ticker: str = "RELIANCE.NS",
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetches real daily market data from real sources (Google Finance & NSE feeds via yfinance).
    Falls back to synthetic data only if network is offline.
    
    WARNING: When fallback is used, data is synthetic and NOT real market data.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from real_data_service import fetch_real_daily_data
        df = fetch_real_daily_data(ticker, period=period)
        if not df.empty and len(df) >= 30:
            return df
    except Exception as e:
        logger.warning(f"Failed to fetch real data for {ticker}: {e}. Using synthetic fallback.")
    
    logger.warning(f"Using SYNTHETIC FALLBACK data for {ticker} - This is NOT real market data!")
    return generate_synthetic_stock_data(ticker=ticker, num_days=350)
