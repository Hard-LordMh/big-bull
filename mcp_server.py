"""
mcp_server.py
-------------
Model Context Protocol (MCP) Server for Big Bull AI.
Exposes tools, resources, and prompt templates to AI assistants and agents (Claude Desktop, Cursor, etc.).

Tools:
- predict_stock_direction: 1-Day and 5-Day directional forecasts (UP/DOWN) with calibrated probabilities and tree consensus.
- detect_market_regime_and_anomalies: Random Forest tree consensus, market regime detection, and Isolation Forest anomalies.
- get_buy_sell_timing: Research signals (Potential Buy/Sell/Hold), ATR stop-loss, targets, and risk/reward.
- analyze_stock_patterns: Historical patterns across Day of Week, Monthly Seasonality, and Similar Setups.
- get_intraday_analysis: 15-minute candlestick trend, VWAP, EMA overlays, and next-bar predictions.
- validate_stock_data: Data sanity, chronological sorting, and OHLC integrity diagnostics.
- train_and_export_pipeline: Triggers model training, backtesting, and Power BI CSV export generation.

Resources:
- stock://metrics/holdout: Latest holdout backtest accuracy scorecard.
- stock://outputs/summary: Summary of all generated Power BI reports.

Prompts:
- analyze_stock: Structured prompt template for institutional stock evaluation.
"""

import os
import json
import datetime
from typing import Dict, Any, List
import pandas as pd

from mcp.server.mcpserver import MCPServer

from utils import fetch_real_or_fallback_stock_data, export_pipeline_results, OUTPUTS_DIR
from data_validation import validate_stock_dataframe
from feature_engineering import prepare_datasets
from stock_model import train_and_evaluate_model, predict_latest_session
from timing_strategy import evaluate_timing_signal
from pattern_analysis import (
    analyze_day_of_week,
    analyze_month_seasonality,
    analyze_year_performance,
    find_similar_historical_setups
)
from intraday_model import get_real_or_fallback_intraday_data, train_and_predict_intraday
from kronos_predictor import run_kronos_forecast
from export_powerbi_pipeline import run_complete_export_pipeline, EXPORTS_DIR

# Initialize MCP Server
mcp = MCPServer("Big Bull AI - NSE Stock Intelligence")


# -----------------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------------

@mcp.tool()
def predict_stock_direction(ticker: str = "RELIANCE.NS", horizon: int = 5) -> str:
    """
    Predict whether an NSE stock closing price will move UP or DOWN over the next 1 or 5 trading sessions.
    
    Args:
        ticker: NSE stock ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'NIFTY50').
        horizon: Prediction horizon in trading sessions (1 for next trading day, 5 for 1 week).
    
    Returns:
        JSON string containing predicted direction, calibrated probability, latest price, holdout accuracy, and baseline comparison.
    """
    horizon = 1 if horizon <= 2 else 5
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="1y")
    is_valid, errs, warns, df_clean = validate_stock_dataframe(df_raw, min_rows=40)
    
    if not is_valid:
        return json.dumps({"error": f"Data validation failed: {errs}"})

    train_df, latest_df, feature_cols = prepare_datasets(df_clean, horizon=horizon)
    results = train_and_evaluate_model(train_df, feature_cols, horizon=horizon)
    latest_pred = predict_latest_session(results["model"], latest_df, feature_cols, ticker=ticker, horizon=horizon, full_df=df_clean)
    metrics = results["test_metrics"]
    rf_det = latest_pred["rf_detection"]

    response = {
        "ticker": ticker,
        "signal_date": latest_pred["signal_date"],
        "latest_close_inr": latest_pred["latest_close"],
        "forecast_horizon_days": horizon,
        "predicted_direction": latest_pred["predicted_direction"],
        "probability_up_pct": round(latest_pred["probability_up"] * 100.0, 1),
        "probability_down_pct": round(latest_pred["probability_down"] * 100.0, 1),
        "tree_consensus": rf_det["tree_consensus"]["strength_label"],
        "market_regime": rf_det["market_regime"]["regime_name"],
        "breakout_anomaly": rf_det["breakout_anomaly"]["badge"],
        "test_accuracy_pct": round(metrics["accuracy"] * 100.0, 1),
        "baseline_accuracy_pct": round(metrics["baseline_accuracy"] * 100.0, 1),
        "outperforms_baseline": metrics["outperforms_baseline"],
        "top_features": results["feature_importance"].head(5).to_dict(orient="records")
    }
    return json.dumps(response, indent=2)


@mcp.tool()
def detect_market_regime_and_anomalies(ticker: str = "RELIANCE.NS") -> str:
    """
    Applies Random Forest and Isolation Forest detection techniques to diagnose current market regime,
    tree consensus vote breakdown (100 individual estimators), and detect institutional breakout anomalies.
    
    Args:
        ticker: NSE stock ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.NS').
        
    Returns:
        JSON string containing tree consensus votes, market regime classification, and isolation anomaly status.
    """
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="1y")
    is_valid, errs, warns, df_clean = validate_stock_dataframe(df_raw, min_rows=40)
    if not is_valid:
        return json.dumps({"error": f"Data validation failed: {errs}"})

    train_df, latest_df, feature_cols = prepare_datasets(df_clean, horizon=5)
    results = train_and_evaluate_model(train_df, feature_cols, horizon=5)
    latest_pred = predict_latest_session(results["model"], latest_df, feature_cols, ticker=ticker, horizon=5, full_df=df_clean)
    
    rf_det = latest_pred["rf_detection"]
    output = {
        "ticker": ticker,
        "signal_date": latest_pred["signal_date"],
        "latest_close_inr": latest_pred["latest_close"],
        "predicted_direction": latest_pred["predicted_direction"],
        "tree_consensus": rf_det["tree_consensus"],
        "market_regime": rf_det["market_regime"],
        "breakout_anomaly": rf_det["breakout_anomaly"],
        "driving_factors": rf_det["driving_factors"]
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def get_buy_sell_timing(ticker: str = "RELIANCE.NS", horizon: int = 5) -> str:
    """
    Evaluates research timing signals (Potential Buy, Potential Sell, Hold) with ATR stop-loss and profit target prices.
    
    Args:
        ticker: NSE stock ticker symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.NS').
        horizon: Prediction horizon in sessions (default 5).
        
    Returns:
        JSON string containing research signal, proposed entry assumption, target price, stop-loss price, risk/reward ratio, and invalidation rules.
    """
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="1y")
    _, _, _, df_clean = validate_stock_dataframe(df_raw, min_rows=40)
    
    train_df, latest_df, feature_cols = prepare_datasets(df_clean, horizon=horizon)
    results = train_and_evaluate_model(train_df, feature_cols, horizon=horizon)
    latest_pred = predict_latest_session(results["model"], latest_df, feature_cols, ticker=ticker, horizon=horizon)
    
    latest_row = df_clean.iloc[-1]
    latest_close = float(latest_row['Close'])
    atr_val = float(df_clean['ATR_14'].iloc[-1]) if 'ATR_14' in df_clean.columns else float(latest_close * 0.018)
    rsi_val = float(df_clean['RSI_14'].iloc[-1]) if 'RSI_14' in df_clean.columns else 50.0
    sma20 = float(df_clean['Close'].rolling(20).mean().iloc[-1]) if len(df_clean) >= 20 else latest_close
    close_vs_sma20 = (latest_close / sma20) - 1.0

    timing_sig = evaluate_timing_signal(
        latest_close=latest_close,
        prob_up=latest_pred['probability_up'],
        prob_down=latest_pred['probability_down'],
        atr_val=atr_val,
        rsi_val=rsi_val,
        close_vs_sma20=close_vs_sma20,
        signal_date=latest_pred['signal_date'],
        cost_bps=15.0
    )
    return json.dumps(timing_sig, indent=2)


@mcp.tool()
def analyze_stock_patterns(ticker: str = "RELIANCE.NS", dimension: str = "day_of_week") -> str:
    """
    Analyzes historical recurring market patterns for an NSE stock.
    
    Args:
        ticker: NSE stock ticker symbol.
        dimension: Pattern dimension: 'day_of_week', 'monthly_seasonality', 'annual_regimes', or 'similar_historical_setups'.
        
    Returns:
        JSON string containing statistical return distributions, win rates, and pattern metrics.
    """
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="1y")
    _, _, _, df_clean = validate_stock_dataframe(df_raw, min_rows=40)

    dim = dimension.strip().lower()
    if "day" in dim or "dow" in dim:
        res_df = analyze_day_of_week(df_clean)
        return json.dumps({"pattern_type": "Day of Week", "results": res_df.to_dict(orient="records")}, indent=2)
    elif "month" in dim or "season" in dim:
        res_df = analyze_month_seasonality(df_clean)
        return json.dumps({"pattern_type": "Monthly Seasonality", "results": res_df.to_dict(orient="records")}, indent=2)
    elif "year" in dim or "annual" in dim:
        res_df = analyze_year_performance(df_clean)
        return json.dumps({"pattern_type": "Annual Regimes", "results": res_df.to_dict(orient="records")}, indent=2)
    elif "similar" in dim or "setup" in dim:
        sim_res = find_similar_historical_setups(df_clean, top_k=6, horizon=5)
        sim_res["matches_table"] = sim_res["matches_table"].to_dict(orient="records")
        return json.dumps({"pattern_type": "Similar Historical Setups", "results": sim_res}, indent=2)
    else:
        return json.dumps({"error": f"Unknown dimension '{dimension}'. Use 'day_of_week', 'monthly_seasonality', 'annual_regimes', or 'similar_historical_setups'."})


@mcp.tool()
def get_intraday_analysis(ticker: str = "RELIANCE.NS") -> str:
    """
    Analyzes 15-minute intraday candlestick movements, VWAP, EMA overlays, and forecasts next-bar direction during NSE market hours.
    
    Args:
        ticker: NSE stock ticker symbol.
        
    Returns:
        JSON string containing latest 15m price, next-candle direction forecast, and intraday test accuracy.
    """
    df_intra = get_real_or_fallback_intraday_data(ticker=ticker, num_days=5, interval_mins=15)
    res = train_and_predict_intraday(df_intra)
    if not res.get("valid", False):
        return json.dumps({"error": res.get("reason", "Intraday modeling failed.")})

    output = {
        "ticker": ticker,
        "latest_timestamp": res["latest_timestamp"],
        "latest_close_inr": res["latest_close"],
        "predicted_direction": res["predicted_direction"],
        "probability_up_pct": round(res["probability_up"] * 100.0, 1),
        "probability_down_pct": round(res["probability_down"] * 100.0, 1),
        "intraday_test_accuracy_pct": res["test_accuracy"],
        "market_hours": "09:15 to 15:30 IST (National Stock Exchange of India)"
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def train_and_export_pipeline(ticker: str = "RELIANCE.NS", num_days: int = 500) -> str:
    """
    Runs the complete machine learning model training, walk-forward validation, and exports all standardized CSVs for Power BI.
    
    Args:
        ticker: NSE ticker symbol.
        num_days: Number of historical trading days to train on (default 500).
        
    Returns:
        JSON string containing execution summary, accuracy metrics, and list of saved files in outputs/.
    """
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="2y")
    is_valid, errs, warns, df_clean = validate_stock_dataframe(df_raw, min_rows=50)
    
    if not is_valid:
        return json.dumps({"error": f"Validation failed: {errs}"})

    train_df, latest_df, feature_cols = prepare_datasets(df_clean, horizon=5)
    results = train_and_evaluate_model(train_df, feature_cols, horizon=5)
    latest_pred = predict_latest_session(results["model"], latest_df, feature_cols, ticker=ticker, horizon=5)
    metrics = results["test_metrics"]

    saved_files = export_pipeline_results(
        raw_df=df_clean,
        backtest_predictions=results["backtest_predictions"],
        test_metrics=metrics,
        walk_forward_df=results["walk_forward_df"],
        feat_importance_df=results["feature_importance"],
        latest_pred_dict=latest_pred,
        model_object=results["model"],
        ticker=ticker
    )

    return json.dumps({
        "status": "Success",
        "ticker": ticker,
        "training_days": len(df_clean),
        "test_accuracy_pct": round(metrics["accuracy"] * 100.0, 1),
        "baseline_accuracy_pct": round(metrics["baseline_accuracy"] * 100.0, 1),
        "forecast_direction": latest_pred["predicted_direction"],
        "files_exported": list(saved_files.keys())
    }, indent=2)


@mcp.tool()
def predict_kronos_forecast(ticker: str = "RELIANCE.NS") -> str:
    """
    Run pretrained Kronos foundation model (NeoQuasar/Kronos-base) on an NSE stock.
    Generates 5-day predicted price trajectory, direction classification, and confidence bounds.
    Price levels are distinct from direction classification.
    """
    res = run_kronos_forecast(ticker)
    if not res.get("success", False):
        return json.dumps({
            "status": "Error",
            "ticker": ticker,
            "message": res.get("error", "Unknown error in Kronos forecast")
        }, indent=2)

    df_fc = res["forecast_df"]
    records = df_fc.to_dict(orient="records") if hasattr(df_fc, "to_dict") else []
    return json.dumps({
        "status": "Success",
        "ticker": res["ticker"],
        "model_version": res["model_version"],
        "last_close_inr": res["last_close"],
        "5day_expected_close_inr": res["5day_expected_close"],
        "5day_return_pct": res["5day_return_pct"],
        "5day_direction": res["5day_direction"],
        "confidence_score": res["confidence_score"],
        "daily_trajectory": records
    }, indent=2)


@mcp.tool()
def trigger_powerbi_export_pipeline(tickers: str = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS", include_kronos: bool = True) -> str:
    """
    Execute atomic Power BI export pipeline across specified NSE stocks.
    Atomically writes 8 standardized CSVs into powerbi_exports/ with full schema validation:
    1. historical_prices.csv
    2. daily_predictions.csv
    3. five_day_predictions.csv
    4. intraday_patterns.csv
    5. historical_patterns.csv
    6. timing_signals.csv
    7. backtest_results.csv
    8. model_metrics.csv
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        ticker_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    exported_files = run_complete_export_pipeline(tickers=ticker_list, include_kronos=include_kronos)
    return json.dumps({
        "status": "Success",
        "tickers_processed": ticker_list,
        "include_kronos": include_kronos,
        "exports_directory": EXPORTS_DIR,
        "files_generated": list(exported_files.keys()),
        "timestamp": datetime.datetime.now().isoformat()
    }, indent=2)


@mcp.tool()
def get_powerbi_exports_status() -> str:
    """
    Check the status, row counts, file sizes, and freshness of all 8 files in powerbi_exports/.
    """
    if not os.path.exists(EXPORTS_DIR):
        return json.dumps({"status": "Not Found", "message": f"Directory {EXPORTS_DIR} does not exist. Run export pipeline first."})

    status = {}
    expected_files = [
        "historical_prices.csv",
        "daily_predictions.csv",
        "five_day_predictions.csv",
        "intraday_patterns.csv",
        "historical_patterns.csv",
        "timing_signals.csv",
        "backtest_results.csv",
        "model_metrics.csv"
    ]
    for fname in expected_files:
        fpath = os.path.join(EXPORTS_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                mtime = os.path.getmtime(fpath)
                dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                status[fname] = {
                    "exists": True,
                    "rows": len(df),
                    "size_bytes": os.path.getsize(fpath),
                    "last_modified": dt,
                    "columns": list(df.columns)
                }
            except Exception as e:
                status[fname] = {"exists": True, "error": str(e)}
        else:
            status[fname] = {"exists": False}

    return json.dumps({"directory": EXPORTS_DIR, "files": status}, indent=2)


# -----------------------------------------------------------------------------
# MCP Resources
# -----------------------------------------------------------------------------

@mcp.resource("stock://metrics/holdout")
def get_holdout_metrics() -> str:
    """Reads latest holdout backtest metrics CSV from outputs folder."""
    path = os.path.join(OUTPUTS_DIR, "holdout_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path).to_json(orient="records", indent=2)
    return json.dumps({"status": "No holdout metrics found. Train model first."})


@mcp.resource("stock://outputs/summary")
def get_outputs_summary() -> str:
    """Returns a list of all Power BI CSV reports available in outputs/."""
    files = {}
    if os.path.exists(OUTPUTS_DIR):
        for f in os.listdir(OUTPUTS_DIR):
            fpath = os.path.join(OUTPUTS_DIR, f)
            files[f] = f"{os.path.getsize(fpath)} bytes"
    return json.dumps({"outputs_directory": OUTPUTS_DIR, "files": files}, indent=2)


@mcp.resource("stock://powerbi/exports")
def get_powerbi_exports_summary() -> str:
    """Returns summary of all 8 standardized Power BI export CSV files."""
    files = {}
    if os.path.exists(EXPORTS_DIR):
        for f in os.listdir(EXPORTS_DIR):
            if f.endswith(".csv"):
                fpath = os.path.join(EXPORTS_DIR, f)
                files[f] = f"{os.path.getsize(fpath)} bytes"
    return json.dumps({"exports_directory": EXPORTS_DIR, "files": files}, indent=2)


@mcp.resource("stock://powerbi/schema")
def get_powerbi_schema() -> str:
    """Returns the Power BI semantic model schema with dimensions, facts, and relationships."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "powerbi_model", "powerbi_model_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return json.dumps({"status": "Schema definition not found."})


# -----------------------------------------------------------------------------
# MCP Prompts
# -----------------------------------------------------------------------------

@mcp.prompt()
def analyze_stock(ticker: str = "RELIANCE.NS") -> str:
    """
    Prompt template for conducting a quantitative analysis of an NSE stock using Big Bull tools.
    """
    return f"""Please conduct a comprehensive quantitative assessment of {ticker} on the National Stock Exchange:
1. Call `predict_stock_direction` with ticker '{ticker}' (5-day horizon) to determine machine learning directional probability.
2. Call `get_buy_sell_timing` to inspect the research signal, ATR target, stop-loss, and risk/reward ratio.
3. Call `analyze_stock_patterns` for 'day_of_week' and 'similar_historical_setups' to evaluate recurring historical edge.
4. Synthesize findings into an institutional summary covering trend direction, conviction level, key risk levels, and disclaimer."""


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
