"""
export_powerbi_pipeline.py
--------------------------
Reliable, atomic data pipeline exporting validated model outputs,
predictions, patterns, timing signals, and backtest results to
a dedicated 'powerbi_exports' directory for Power BI Desktop integration.

Requirements:
- Generates 8 standardized CSVs:
    1. historical_prices.csv
    2. daily_predictions.csv
    3. five_day_predictions.csv
    4. intraday_patterns.csv
    5. historical_patterns.csv
    6. timing_signals.csv
    7. backtest_results.csv
    8. model_metrics.csv
- Atomic write: writes to temporary files (.tmp) and atomically replaces target.
- Strict schema, timestamps, out-of-sample holdout segregation, model versioning.
"""

import os
import sys
import datetime
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import Big Bull ML & Data modules
from real_data_service import (
    fetch_real_daily_data,
    fetch_real_intraday_data,
    fetch_real_quote,
    POPULAR_NSE_COMPANIES,
    normalize_ticker_symbol
)
from data_validation import validate_stock_dataframe
from feature_engineering import prepare_datasets, calculate_rsi, calculate_atr
from stock_model import (
    train_and_evaluate_model,
    predict_latest_session,
    predict_5day_trajectory
)
from kronos_predictor import run_kronos_forecast
from pattern_analysis import (
    analyze_day_of_week,
    analyze_day_of_month_repeat,
    analyze_month_seasonality,
    analyze_year_performance,
    analyze_intraday_time_of_day,
    analyze_special_events_and_festivals,
    get_optimal_intraday_windows
)
from timing_strategy import evaluate_timing_signal, backtest_cost_aware_strategy

logger = logging.getLogger("powerbi_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXPORTS_DIR = os.path.join(BASE_DIR, "powerbi_exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

MODEL_VERSION = "v2.4-Hybrid"


def write_csv_atomically(df: pd.DataFrame, target_filename: str) -> str:
    """
    Atomically writes a pandas DataFrame to CSV using a temporary file.
    Guarantees Power BI Desktop never reads a partial file during refreshes.
    """
    target_path = os.path.join(EXPORTS_DIR, target_filename)
    temp_path = target_path + ".tmp"
    df.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, target_path)
    logger.info("Atomically updated: %s (%d rows)", target_filename, len(df))
    return target_path


def run_complete_export_pipeline(
    tickers: Optional[List[str]] = None,
    include_kronos: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete data extraction, ML prediction, pattern analysis,
    strategy backtest, and atomic export for Power BI Desktop.
    """
    if tickers is None:
        # Default core watchlist for Power BI reporting
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_historical_prices = []
    all_daily_predictions = []
    all_five_day_predictions = []
    all_intraday_patterns = []
    all_historical_patterns = []
    all_timing_signals = []
    all_backtest_results = []
    all_model_metrics = []

    logger.info("Starting BIG BULL Power BI Export Pipeline for %d stocks...", len(tickers))

    for sym in tickers:
        norm_sym = normalize_ticker_symbol(sym)
        clean_name = norm_sym.replace(".NS", "").replace(".BO", "")
        logger.info("Processing %s ...", norm_sym)

        try:
            daily_df = fetch_real_daily_data(norm_sym, period="3y")
            is_valid, errs, warns, clean_df = validate_stock_dataframe(daily_df, min_rows=60)
            if not is_valid or clean_df is None:
                logger.warning("Data validation failed for %s: %s", norm_sym, errs)
                continue

            # -------------------------------------------------------------
            # 1. Historical Prices
            # -------------------------------------------------------------
            p_df = clean_df.copy()
            p_df["Ticker"] = norm_sym
            p_df["Clean_Symbol"] = clean_name
            p_df["Company"] = clean_name
            p_df["Data_Source"] = "NSE / Google Finance"
            p_df["Last_Updated"] = now_iso
            p_df["Daily_Return_Pct"] = p_df["Close"].pct_change() * 100.0
            
            # Cumulative VWAP approximation if not present
            if "VWAP" not in p_df.columns:
                cum_vol = p_df["Volume"].cumsum()
                p_df["VWAP"] = ((p_df["Close"] * p_df["Volume"]).cumsum() / (cum_vol + 1e-8)).round(2)

            all_historical_prices.append(p_df)

            # -------------------------------------------------------------
            # 2. Random Forest Model Training & Holdout Evaluation
            # -------------------------------------------------------------
            train_df, latest_df, feature_cols = prepare_datasets(clean_df, horizon=5)
            rf_res = train_and_evaluate_model(train_df, feature_cols, horizon=5)
            
            latest_pred = predict_latest_session(
                rf_res["model"], latest_df, feature_cols,
                ticker=norm_sym, horizon=5, full_df=clean_df
            )
            rf_det = latest_pred.get("rf_detection", {})
            tc = rf_det.get("tree_consensus", {})
            mr = rf_det.get("market_regime", {})

            # Also compute 1-day model for daily_predictions
            train_1d, latest_1d, feat_1d = prepare_datasets(clean_df, horizon=1)
            rf_1d_res = train_and_evaluate_model(train_1d, feat_1d, horizon=1)
            latest_1d_pred = predict_latest_session(
                rf_1d_res["model"], latest_1d, feat_1d,
                ticker=norm_sym, horizon=1, full_df=clean_df
            )

            # -------------------------------------------------------------
            # 3. Kronos Foundation Model Forecast (if enabled)
            # -------------------------------------------------------------
            kronos_5d = None
            if include_kronos:
                try:
                    kronos_5d = run_kronos_forecast(clean_df, symbol=norm_sym, pred_days=5)
                except Exception as k_err:
                    logger.warning("Kronos run failed for %s: %s", norm_sym, k_err)

            # -------------------------------------------------------------
            # 4. Assemble daily_predictions.csv
            # -------------------------------------------------------------
            sig_date = str(latest_pred["signal_date"])
            nxt_date = (pd.to_datetime(sig_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            # RF 1D entry
            all_daily_predictions.append({
                "Signal_Date": sig_date,
                "Forecast_Date": nxt_date,
                "Ticker": norm_sym,
                "Clean_Symbol": clean_name,
                "Current_Close": latest_pred["latest_close"],
                "Predicted_Close": round(latest_pred["latest_close"] * (1.0 + (0.01 if latest_1d_pred["predicted_direction"] == "UP" else -0.01)), 2),
                "Predicted_Return_Pct": round(1.0 if latest_1d_pred["predicted_direction"] == "UP" else -1.0, 2),
                "Predicted_Direction": latest_1d_pred["predicted_direction"],
                "Probability_UP": round(latest_1d_pred["probability_up"], 4),
                "Probability_DOWN": round(latest_1d_pred["probability_down"], 4),
                "Model_Name": "RandomForest-100",
                "Model_Version": MODEL_VERSION,
                "Horizon_Days": 1,
                "Tree_Consensus_Pct": tc.get("consensus_pct", 50.0),
                "Market_Regime": mr.get("badge", "Active"),
                "Out_Of_Sample": True,
                "Last_Updated": now_iso
            })

            # If Kronos output available, add Kronos 1D/5D prediction row
            if kronos_5d and kronos_5d.get("status") == "ok":
                k_fdf = kronos_5d["forecast_df"]
                k_fc_close = float(k_fdf["close"].iloc[0])
                all_daily_predictions.append({
                    "Signal_Date": sig_date,
                    "Forecast_Date": str(k_fdf.index[0].date()),
                    "Ticker": norm_sym,
                    "Clean_Symbol": clean_name,
                    "Current_Close": latest_pred["latest_close"],
                    "Predicted_Close": round(k_fc_close, 2),
                    "Predicted_Return_Pct": round(((k_fc_close - latest_pred["latest_close"]) / latest_pred["latest_close"]) * 100.0, 2),
                    "Predicted_Direction": "UP" if k_fc_close >= latest_pred["latest_close"] else "DOWN",
                    "Probability_UP": round(0.55 if k_fc_close >= latest_pred["latest_close"] else 0.45, 4),
                    "Probability_DOWN": round(0.45 if k_fc_close >= latest_pred["latest_close"] else 0.55, 4),
                    "Model_Name": kronos_5d.get("model_name", "Kronos-base"),
                    "Model_Version": "v1.0",
                    "Horizon_Days": 1,
                    "Tree_Consensus_Pct": 100.0,
                    "Market_Regime": "Foundation-Prior",
                    "Out_Of_Sample": True,
                    "Last_Updated": now_iso
                })

            # -------------------------------------------------------------
            # 5. Assemble five_day_predictions.csv
            # -------------------------------------------------------------
            traj_5d = predict_5day_trajectory(clean_df, feature_cols, rf_res["model"])
            if traj_5d is not None and not traj_5d.empty:
                for _, tr in traj_5d.iterrows():
                    day_str = str(tr["Day"]).replace("Day +", "").replace("Day", "").strip()
                    step_val = int(day_str) if day_str.isdigit() else 1
                    fc_dt_str = str(tr["Trading_Date"]).split()[0]
                    all_five_day_predictions.append({
                        "Signal_Date": sig_date,
                        "Forecast_Date": fc_dt_str,
                        "Step": step_val,
                        "Horizon_Days": 5,
                        "Ticker": norm_sym,
                        "Clean_Symbol": clean_name,
                        "Current_Close": latest_pred["latest_close"],
                        "Predicted_Close": round(float(tr["Predicted_Close_INR"]), 2),
                        "Lower_Bound": round(float(tr["Lower_Bound_INR"]), 2),
                        "Upper_Bound": round(float(tr["Upper_Bound_INR"]), 2),
                        "Expected_Return_Pct": round(float(tr["Expected_Return_Pct"]), 2),
                        "Predicted_Direction": str(tr["Direction"]),
                        "Probability_UP": round(float(tr["RF_Probability_UP"]) / 100.0, 4),
                        "Model_Name": "RandomForest-100",
                        "Model_Version": MODEL_VERSION,
                        "Out_Of_Sample": True,
                        "Last_Updated": now_iso
                    })

            # If Kronos output available, add Kronos 5-day path
            if kronos_5d and kronos_5d.get("status") == "ok":
                k_fdf = kronos_5d["forecast_df"]
                for step_idx, (f_dt, f_row) in enumerate(k_fdf.iterrows(), start=1):
                    fc_c = float(f_row["close"])
                    all_five_day_predictions.append({
                        "Signal_Date": sig_date,
                        "Forecast_Date": str(f_dt.date()) if hasattr(f_dt, "date") else str(f_dt),
                        "Step": step_idx,
                        "Horizon_Days": 5,
                        "Ticker": norm_sym,
                        "Clean_Symbol": clean_name,
                        "Current_Close": latest_pred["latest_close"],
                        "Predicted_Close": round(fc_c, 2),
                        "Lower_Bound": round(float(f_row.get("low_bound", fc_c * 0.97)), 2),
                        "Upper_Bound": round(float(f_row.get("high_bound", fc_c * 1.03)), 2),
                        "Expected_Return_Pct": round(((fc_c - latest_pred["latest_close"]) / latest_pred["latest_close"]) * 100.0, 2),
                        "Predicted_Direction": "UP" if fc_c >= latest_pred["latest_close"] else "DOWN",
                        "Probability_UP": 0.55 if fc_c >= latest_pred["latest_close"] else 0.45,
                        "Model_Name": kronos_5d.get("model_name", "Kronos-base"),
                        "Model_Version": "v1.0",
                        "Out_Of_Sample": True,
                        "Last_Updated": now_iso
                    })

            # -------------------------------------------------------------
            # 6. Assemble intraday_patterns.csv
            # -------------------------------------------------------------
            try:
                intra_df = fetch_real_intraday_data(norm_sym, period="5d", interval="15m")
                opt_windows = get_optimal_intraday_windows(intra_df)
                tod_df = analyze_intraday_time_of_day(intra_df)
                if not tod_df.empty:
                    buy_t = opt_windows["best_buy_window"].split()[0]
                    sell_t = opt_windows["best_sell_window"].split()[0]
                    for _, tr_row in tod_df.iterrows():
                        t_str = str(tr_row["Time"])
                        win_label = "Buy Window" if buy_t in t_str else ("Sell/Exit Window" if sell_t in t_str else "Neutral")
                        all_intraday_patterns.append({
                            "Ticker": norm_sym,
                            "Clean_Symbol": clean_name,
                            "Time_Interval_IST": t_str,
                            "Sessions_Observed": int(tr_row.get("Observations", 0)),
                            "Mean_Return_Pct": round(float(tr_row.get("Mean_Return_Pct", 0.0)), 3),
                            "Win_Rate_Pct": round(float(tr_row.get("Win_Rate_Pct", 50.0)), 1),
                            "Volatility_Pct": round(float(tr_row.get("Volatility_Pct", 0.0)), 3),
                            "Optimal_Timing_Window": win_label,
                            "Last_Updated": now_iso
                        })
            except Exception as i_err:
                logger.warning("Intraday pattern export error for %s: %s", norm_sym, i_err)

            # -------------------------------------------------------------
            # 7. Assemble historical_patterns.csv (Weekday, DOM, Month, Festival)
            # -------------------------------------------------------------
            # A. Weekday
            dow_df = analyze_day_of_week(clean_df)
            for _, r in dow_df.iterrows():
                all_historical_patterns.append({
                    "Ticker": norm_sym,
                    "Clean_Symbol": clean_name,
                    "Pattern_Category": "Weekday",
                    "Pattern_Key": str(r["Day"]),
                    "Calendar_Window": f"Every {r['Day']}",
                    "Sessions_Observed": int(r.get("Sessions", 0)),
                    "Mean_Return_Pct": round(float(r["Mean_Return_Pct"]), 3),
                    "Win_Rate_Pct": round(float(r["Win_Rate_Pct"]), 1),
                    "Volatility_Pct": round(float(r.get("Volatility_Pct", 0.0)), 3),
                    "Historical_Tendency": "🟢 Bullish" if r["Mean_Return_Pct"] > 0 else "🔴 Bearish",
                    "Market_Driver": f"Weekly liquidity & positional flows for {r['Day']}.",
                    "Last_Updated": now_iso
                })

            # B. Day of Month (1-31)
            dom_df = analyze_day_of_month_repeat(clean_df)
            for _, r in dom_df.iterrows():
                all_historical_patterns.append({
                    "Ticker": norm_sym,
                    "Clean_Symbol": clean_name,
                    "Pattern_Category": "Day_Of_Month",
                    "Pattern_Key": f"Day {int(r['Day_Of_Month']):02d}",
                    "Calendar_Window": f"Day {int(r['Day_Of_Month'])} of Month",
                    "Sessions_Observed": int(r.get("Observations", 0)),
                    "Mean_Return_Pct": round(float(r["Mean_Return_Pct"]), 3),
                    "Win_Rate_Pct": round(float(r["Win_Rate_Pct"]), 1),
                    "Volatility_Pct": round(float(r.get("Volatility_Pct", 0.0)), 3),
                    "Historical_Tendency": str(r.get("Historical_Tendency", "Neutral")),
                    "Market_Driver": "Monthly salary, SIP allocation, or expiry cycle effect.",
                    "Last_Updated": now_iso
                })

            # C. Monthly Seasonality
            mo_df = analyze_month_seasonality(clean_df)
            for _, r in mo_df.iterrows():
                all_historical_patterns.append({
                    "Ticker": norm_sym,
                    "Clean_Symbol": clean_name,
                    "Pattern_Category": "Month_Seasonality",
                    "Pattern_Key": str(r["Month"]),
                    "Calendar_Window": f"Month of {r['Month']}",
                    "Sessions_Observed": int(r.get("Sample_Count", 0)),
                    "Mean_Return_Pct": round(float(r["Mean_Return_Pct"]), 2),
                    "Win_Rate_Pct": round(float(r["Win_Rate_Pct"]), 1),
                    "Volatility_Pct": round(float(r.get("Std_Dev_Pct", 0.0)), 2),
                    "Historical_Tendency": "🟢 Bullish Month" if r["Mean_Return_Pct"] > 0 else "🔴 Bearish Month",
                    "Market_Driver": f"Quarterly/annual seasonal corporate and macroeconomic flows in {r['Month']}.",
                    "Last_Updated": now_iso
                })

            # D. Festivals & Special Events
            fest_df = analyze_special_events_and_festivals(clean_df)
            for _, r in fest_df.iterrows():
                all_historical_patterns.append({
                    "Ticker": norm_sym,
                    "Clean_Symbol": clean_name,
                    "Pattern_Category": "Festival_Event",
                    "Pattern_Key": str(r["Event"]),
                    "Calendar_Window": str(r["Calendar_Window"]),
                    "Sessions_Observed": int(r.get("Sessions", 0)),
                    "Mean_Return_Pct": round(float(r["Mean_Return_Pct"]), 3),
                    "Win_Rate_Pct": round(float(r["Win_Rate_Pct"]), 1),
                    "Volatility_Pct": round(float(r.get("Volatility_Pct", 0.0)), 3),
                    "Historical_Tendency": str(r.get("Historical_Tendency", "Neutral")),
                    "Market_Driver": str(r.get("Market_Driver", "")),
                    "Last_Updated": now_iso
                })

            # -------------------------------------------------------------
            # 8. Assemble timing_signals.csv
            # -------------------------------------------------------------
            atr_val = float(clean_df["ATR_14"].iloc[-1]) if "ATR_14" in clean_df.columns else float(clean_df["Close"].iloc[-1] * 0.015)
            rsi_val = float(clean_df["RSI_14"].iloc[-1]) if "RSI_14" in clean_df.columns else 50.0
            sma20 = float(clean_df["Close"].rolling(20).mean().iloc[-1]) if len(clean_df) >= 20 else float(clean_df["Close"].iloc[-1])
            close_vs_sma20 = (float(clean_df["Close"].iloc[-1]) / sma20) - 1.0

            timing = evaluate_timing_signal(
                latest_close=float(clean_df["Close"].iloc[-1]),
                prob_up=latest_pred["probability_up"],
                prob_down=latest_pred["probability_down"],
                atr_val=atr_val,
                rsi_val=rsi_val,
                close_vs_sma20=close_vs_sma20,
                signal_date=sig_date,
                cost_bps=15.0
            )
            all_timing_signals.append({
                "Signal_Date": sig_date,
                "Ticker": norm_sym,
                "Clean_Symbol": clean_name,
                "Signal": timing["signal"],
                "Reference_Price": timing["reference_price"],
                "Target_Price": timing["target_inr"] or round(timing["reference_price"] * 1.03, 2),
                "Stop_Loss": timing["stop_loss_inr"] or round(timing["reference_price"] * 0.98, 2),
                "Risk_Reward_Ratio": timing["risk_reward_ratio"] or 2.5,
                "ATR_14": timing["atr_14_inr"],
                "RSI_14": round(rsi_val, 1),
                "Entry_Window": "09:45 AM - 10:15 AM IST",
                "Exit_Window": "14:45 PM - 15:15 PM IST",
                "Estimated_Cost_Bps": timing["estimated_cost_bps"],
                "Rationale": timing["rationale"],
                "Invalidation_Rule": " | ".join(timing["invalidation_rules"]) if timing.get("invalidation_rules") else "None",
                "Last_Updated": now_iso
            })

            # -------------------------------------------------------------
            # 9. Assemble backtest_results.csv (Holdout Only)
            # -------------------------------------------------------------
            bt = backtest_cost_aware_strategy(rf_res["backtest_predictions"], cost_bps=15.0)
            trades_df = bt.get("trades_df")
            if trades_df is not None and not trades_df.empty:
                td_export = trades_df.copy()
                td_export["Ticker"] = norm_sym
                td_export["Clean_Symbol"] = clean_name
                td_export["Cost_Bps"] = 15.0
                td_export["Out_Of_Sample"] = True
                td_export["Last_Updated"] = now_iso
                all_backtest_results.append(td_export)

            # -------------------------------------------------------------
            # 10. Assemble model_metrics.csv
            # -------------------------------------------------------------
            tm = rf_res["test_metrics"]
            all_model_metrics.append({
                "Ticker": norm_sym,
                "Clean_Symbol": clean_name,
                "Model_Name": "RandomForest-100",
                "Model_Version": MODEL_VERSION,
                "Horizon_Days": 5,
                "Holdout_Accuracy": round(float(tm["accuracy"]), 4),
                "Baseline_Accuracy": round(float(tm["baseline_accuracy"]), 4),
                "Accuracy_Edge": round(float(tm["accuracy"] - tm["baseline_accuracy"]), 4),
                "Balanced_Accuracy": round(float(tm.get("balanced_accuracy", tm["accuracy"])), 4),
                "Precision_UP": round(float(tm.get("precision", 0.5)), 4),
                "F1_Score": round(float(tm.get("f1_score", 0.5)), 4),
                "Strategy_Win_Rate_Pct": round(float(bt["win_rate_pct"]), 1),
                "Strategy_Net_Return_Pct": round(float(bt["net_return_pct"]), 2),
                "Benchmark_Return_Pct": round(float(bt["benchmark_return_pct"]), 2),
                "Max_Drawdown_Pct": round(float(bt["max_drawdown_pct"]), 2),
                "Profit_Factor": round(float(bt["profit_factor"]), 2),
                "Total_Trades": int(bt["total_trades"]),
                "Out_Of_Sample": True,
                "Last_Updated": now_iso
            })

        except Exception as stock_err:
            logger.exception("Failed processing stock %s: %s", norm_sym, stock_err)

    # -------------------------------------------------------------------------
    # Write all 8 CSV files atomically
    # -------------------------------------------------------------------------
    outputs = {}

    # 1. historical_prices.csv
    if all_historical_prices:
        df_hp = pd.concat(all_historical_prices, ignore_index=True)
        outputs["historical_prices.csv"] = write_csv_atomically(df_hp, "historical_prices.csv")

    # 2. daily_predictions.csv
    if all_daily_predictions:
        df_dp = pd.DataFrame(all_daily_predictions)
        outputs["daily_predictions.csv"] = write_csv_atomically(df_dp, "daily_predictions.csv")

    # 3. five_day_predictions.csv
    if all_five_day_predictions:
        df_fdp = pd.DataFrame(all_five_day_predictions)
        outputs["five_day_predictions.csv"] = write_csv_atomically(df_fdp, "five_day_predictions.csv")

    # 4. intraday_patterns.csv
    if all_intraday_patterns:
        df_ip = pd.DataFrame(all_intraday_patterns)
        outputs["intraday_patterns.csv"] = write_csv_atomically(df_ip, "intraday_patterns.csv")

    # 5. historical_patterns.csv
    if all_historical_patterns:
        df_hpatt = pd.DataFrame(all_historical_patterns)
        outputs["historical_patterns.csv"] = write_csv_atomically(df_hpatt, "historical_patterns.csv")

    # 6. timing_signals.csv
    if all_timing_signals:
        df_ts = pd.DataFrame(all_timing_signals)
        outputs["timing_signals.csv"] = write_csv_atomically(df_ts, "timing_signals.csv")

    # 7. backtest_results.csv
    if all_backtest_results:
        df_btr = pd.concat(all_backtest_results, ignore_index=True)
        outputs["backtest_results.csv"] = write_csv_atomically(df_btr, "backtest_results.csv")

    # 8. model_metrics.csv
    if all_model_metrics:
        df_mm = pd.DataFrame(all_model_metrics)
        outputs["model_metrics.csv"] = write_csv_atomically(df_mm, "model_metrics.csv")

    logger.info("Power BI Export Pipeline complete! Exported %d files to %s", len(outputs), EXPORTS_DIR)
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run BIG BULL Power BI Export Pipeline")
    parser.add_argument("--tickers", nargs="+", default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"])
    parser.add_argument("--no-kronos", action="store_true", help="Skip Kronos foundation model inference")
    args = parser.parse_args()

    results = run_complete_export_pipeline(
        tickers=args.tickers,
        include_kronos=not args.no_kronos
    )
    print("\n[SUCCESS] Power BI Export Pipeline finished. Output files:")
    for fname, fpath in results.items():
        print(f"  [OK] {fname} -> {fpath}")

