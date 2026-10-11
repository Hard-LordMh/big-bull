"""
train_pipeline.py
-----------------
Command-line runner to train and evaluate the AI model pipeline starting today.
Trains both 1-Day and 5-Day models, evaluates performance, generates timing signals,
and exports all Power BI datasets.
"""

import sys
import os
import json
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from utils import fetch_real_or_fallback_stock_data, export_pipeline_results, OUTPUTS_DIR, MODELS_DIR
from data_validation import validate_stock_dataframe
from feature_engineering import prepare_datasets, calculate_atr, calculate_rsi
from stock_model import train_and_evaluate_model, predict_latest_session
from timing_strategy import evaluate_timing_signal, backtest_cost_aware_strategy
from pattern_analysis import analyze_day_of_week


def run_training_pipeline(ticker: str = "RELIANCE.NS", num_days: int = 500):
    print("=" * 60)
    print(f">> INITIATING BIG BULL AI MODEL TRAINING: {ticker}")
    print("=" * 60)

    # 1. Fetch Real Dataset from Google Finance / NSE Market Engine
    df_raw = fetch_real_or_fallback_stock_data(ticker=ticker, period="2y")
    is_valid, errors, warnings, df_clean = validate_stock_dataframe(df_raw, min_rows=50)

    if not is_valid:
        print(f"[ERROR] Data Validation Failed: {errors}")
        return

    start_date = df_clean['Date'].min()
    end_date = df_clean['Date'].max()
    latest_close = float(df_clean['Close'].iloc[-1])
    print(f"[DATA] Validated: {len(df_clean)} trading days ({start_date} to {end_date})")
    print(f"[PRICE] Latest Reference Price: Rs {latest_close:,.2f} on {end_date}")

    # 2. Train 5-Day Horizon Model
    print("\n[MODEL] Training 5-Day Directional Random Forest Model...")
    train_5d, latest_5d, feats_5d = prepare_datasets(df_clean, horizon=5)
    results_5d = train_and_evaluate_model(train_5d, feats_5d, horizon=5)
    latest_pred_5d = predict_latest_session(results_5d['model'], latest_5d, feats_5d, ticker=ticker, horizon=5, full_df=df_clean)

    metrics_5d = results_5d['test_metrics']
    rf_det_5d = latest_pred_5d.get('rf_detection', {})
    tc_5d = rf_det_5d.get('tree_consensus', {})
    mr_5d = rf_det_5d.get('market_regime', {})
    ba_5d = rf_det_5d.get('breakout_anomaly', {})

    print(f"   * Holdout Accuracy: {metrics_5d['accuracy']*100:.1f}% (Baseline: {metrics_5d['baseline_accuracy']*100:.1f}%)")
    if 'oob_accuracy' in metrics_5d:
        print(f"   * Out-of-Bag (OOB) Accuracy: {metrics_5d['oob_accuracy']*100:.1f}%")
    print(f"   * Balanced Accuracy: {metrics_5d['balanced_accuracy']*100:.1f}%")
    print(f"   * Precision (UP): {metrics_5d['precision']*100:.1f}% | F1-Score: {metrics_5d['f1_score']*100:.1f}%")
    print(f"   * 5-Day Forecast Signal: {latest_pred_5d['predicted_direction']} (Probability UP: {latest_pred_5d['probability_up']*100:.1f}%, DOWN: {latest_pred_5d['probability_down']*100:.1f}%)")
    print(f"   * Random Forest Tree Consensus: {tc_5d.get('trees_up', 0)}/{tc_5d.get('total_trees', 100)} Trees Vote UP ({tc_5d.get('consensus_pct', 50)}%) [{tc_5d.get('strength_label', '')}]")
    print(f"   * Detected Market Regime: {mr_5d.get('regime_name', 'Active')} - {mr_5d.get('description', '')}")
    print(f"   * Isolation Forest Anomaly Detection: {ba_5d.get('badge', 'Active')} ({ba_5d.get('description', '')})")

    # 3. Train 1-Day Horizon Model
    print("\n[MODEL] Training Next-Day (1-Day) Directional Model...")
    train_1d, latest_1d, feats_1d = prepare_datasets(df_clean, horizon=1)
    results_1d = train_and_evaluate_model(train_1d, feats_1d, horizon=1)
    latest_pred_1d = predict_latest_session(results_1d['model'], latest_1d, feats_1d, ticker=ticker, horizon=1, full_df=df_clean)

    metrics_1d = results_1d['test_metrics']
    print(f"   * Holdout Accuracy: {metrics_1d['accuracy']*100:.1f}% (Baseline: {metrics_1d['baseline_accuracy']*100:.1f}%)")
    print(f"   * Next-Day Forecast Signal: {latest_pred_1d['predicted_direction']} (Probability UP: {latest_pred_1d['probability_up']*100:.1f}%, DOWN: {latest_pred_1d['probability_down']*100:.1f}%)")

    # 4. Generate Buy/Sell Timing Analysis
    print("\n[TIMING] Evaluating Buy/Sell Timing Opportunity...")
    atr_val = float(df_clean['ATR_14'].iloc[-1]) if 'ATR_14' in df_clean.columns else float(latest_close * 0.018)
    rsi_val = float(df_clean['RSI_14'].iloc[-1]) if 'RSI_14' in df_clean.columns else 50.0
    sma20 = float(df_clean['Close'].rolling(20).mean().iloc[-1]) if len(df_clean) >= 20 else latest_close
    close_vs_sma20 = (latest_close / sma20) - 1.0

    timing_sig = evaluate_timing_signal(
        latest_close=latest_close,
        prob_up=latest_pred_5d['probability_up'],
        prob_down=latest_pred_5d['probability_down'],
        atr_val=atr_val,
        rsi_val=rsi_val,
        close_vs_sma20=close_vs_sma20,
        signal_date=str(end_date),
        cost_bps=15.0
    )
    print(f"   * Research Signal: {timing_sig['signal']}")
    if timing_sig['signal'] in ['Potential Buy', 'Potential Sell']:
        print(f"   * Entry: {timing_sig['entry_assumption']}")
        print(f"   * Target: Rs {timing_sig['target_inr']:,.2f} | Stop-Loss: Rs {timing_sig['stop_loss_inr']:,.2f}")
        print(f"   * Risk/Reward Ratio: 1 : {timing_sig['risk_reward_ratio']}")
    print(f"   * Rationale: {timing_sig['rationale']}")

    # 5. Cost-Aware Strategy Backtesting
    bt_res = backtest_cost_aware_strategy(results_5d['backtest_predictions'], cost_bps=15.0)
    print(f"\n[BACKTEST] Cost-Aware Strategy Performance (Deducting 15 bps per trade):")
    print(f"   * Total Trades: {bt_res['total_trades']}")
    print(f"   * Win Rate: {bt_res['win_rate_pct']:.1f}%")
    print(f"   * Net Return After Costs: {bt_res['net_return_pct']:+.2f}%")
    print(f"   * Max Drawdown: {bt_res['max_drawdown_pct']:.1f}%")

    # 6. Save & Export Power BI CSV Bundle
    print("\n[EXPORT] Exporting Standardized CSV Reports for Power BI...")
    dow_df = analyze_day_of_week(df_clean)
    saved_files = export_pipeline_results(
        raw_df=df_clean,
        backtest_predictions=results_5d['backtest_predictions'],
        test_metrics=metrics_5d,
        walk_forward_df=results_5d['walk_forward_df'],
        feat_importance_df=results_5d['feature_importance'],
        latest_pred_dict=latest_pred_5d,
        model_object=results_5d['model'],
        ticker=ticker,
        timing_signal_dict=timing_sig,
        pattern_summary_df=dow_df
    )
    for name, path in saved_files.items():
        print(f"   [OK] {name} -> {os.path.basename(path)}")

    print("\n" + "=" * 60)
    print(f">> MODEL TRAINING COMPLETE & SAVED TO outputs/ AND models/")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
