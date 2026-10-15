"""
timing_strategy.py
------------------
Module for researching potentially favorable Buy and Sell timing opportunities.
Includes:
- Rule-based signal gating (Potential Buy, Potential Sell, Hold/No Signal, Insufficient Evidence)
- ATR-based dynamic stop-loss and profit target estimation
- Realistic cost-aware strategy backtesting (transaction fees and slippage)
- Trade statistics: Win Rate, Net Return After Costs, Max Drawdown, Profit Factor
- Signal invalidation criteria
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def evaluate_timing_signal(
    latest_close: float,
    prob_up: float,
    prob_down: float,
    atr_val: float,
    rsi_val: float,
    close_vs_sma20: float,
    signal_date: str,
    confidence_threshold: float = 0.58,
    cost_bps: float = 15.0
) -> Dict[str, Any]:
    """
    Evaluates current market conditions to produce a disciplined research signal.
    Does not produce a signal unless strict statistical and trend filters are met.
    
    IMPORTANT DISCLAIMER: This is a research-only signal, not a trading recommendation.
    The 58% threshold is a research parameter, not a validated trading threshold.
    Past performance does not guarantee future results.
    Always conduct your own analysis before making trading decisions.
    """
    signal = "Hold / No Signal"
    reason = "Model probability or trend filters did not exceed research threshold."
    stop_loss = None
    target_price = None
    risk_reward = None
    invalidation_rules = []

    # Potential Buy Conditions (RESEARCH ONLY):
    # 1. Model UP Probability >= threshold (research parameter, not validated)
    # 2. RSI is not severely overbought (< 72)
    # 3. Price is not extended excessively above 20-day SMA (< +6%)
    if prob_up >= confidence_threshold and rsi_val < 72 and close_vs_sma20 < 0.06:
        signal = "Potential Buy (Research Signal)"
        # Proposed entry at next market open ~ current close
        stop_loss = round(latest_close - (1.5 * atr_val), 2)
        target_price = round(latest_close + (2.5 * atr_val), 2)
        risk = latest_close - stop_loss
        reward = target_price - latest_close
        risk_reward = round(reward / (risk + 1e-8), 2)
        reason = f"Model indicates {prob_up*100:.1f}% UP probability with supportive RSI ({rsi_val:.1f}) and trend alignment."
        invalidation_rules = [
            f"Daily close drops below ₹{stop_loss:,.2f} stop-loss.",
            "RSI crosses above 75 (extreme overbought).",
            "Broad market NIFTY index falls over 1.5% in a single session."
        ]

    # Potential Sell Conditions (RESEARCH ONLY):
    # 1. Model DOWN Probability >= threshold (research parameter, not validated)
    # 2. RSI is not severely oversold (> 28)
    elif prob_down >= confidence_threshold and rsi_val > 28:
        signal = "Potential Sell (Research Signal)"
        stop_loss = round(latest_close + (1.5 * atr_val), 2)
        target_price = round(latest_close - (2.5 * atr_val), 2)
        risk = stop_loss - latest_close
        reward = latest_close - target_price
        risk_reward = round(reward / (risk + 1e-8), 2)
        reason = f"Model indicates {prob_down*100:.1f}% DOWN probability with deteriorating momentum (RSI {rsi_val:.1f})."
        invalidation_rules = [
            f"Daily close rises above ₹{stop_loss:,.2f} stop-loss.",
            "RSI drops below 25 (extreme oversold bounce risk).",
            "Bullish moving average crossover occurs."
        ]

    return {
        "signal": signal,
        "signal_date": signal_date,
        "reference_price": round(latest_close, 2),
        "prob_up": round(prob_up, 3),
        "prob_down": round(prob_down, 3),
        "entry_assumption": "Next trading session opening auction (9:15 AM IST)",
        "exit_horizon": "5 trading sessions or stop-loss/target hit",
        "stop_loss_inr": stop_loss,
        "target_inr": target_price,
        "risk_reward_ratio": risk_reward,
        "atr_14_inr": round(atr_val, 2),
        "estimated_cost_bps": cost_bps,
        "rationale": reason,
        "invalidation_rules": invalidation_rules,
        "disclaimer": "RESEARCH SIGNAL ONLY - Not a trading recommendation. The 58% threshold is a research parameter without validated backtesting. Past performance does not guarantee future results."
    }


def backtest_cost_aware_strategy(
    backtest_predictions_df: pd.DataFrame,
    cost_bps: float = 15.0,
    confidence_threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Backtests a cost-aware trading strategy over holdout test data.
    Applies realistic transaction costs and slippage (default 15 bps = 0.15% per round-trip).
    """
    df = backtest_predictions_df.copy()
    if df.empty or 'Prob_UP' not in df.columns or 'Close' not in df.columns:
        return {"total_trades": 0, "win_rate_pct": 0.0, "net_return_pct": 0.0}

    # Signal generation: 1 for Buy if Prob_UP >= threshold, -1 for Sell if Prob_DOWN >= threshold, 0 otherwise
    df['Position'] = 0
    df.loc[df['Prob_UP'] >= confidence_threshold, 'Position'] = 1
    df.loc[df['Prob_DOWN'] >= confidence_threshold, 'Position'] = -1

    # 1-day market return
    df['Market_Return'] = df['Close'].pct_change().shift(-1).fillna(0.0)

    # Strategy Gross Return
    df['Strategy_Gross_Return'] = df['Position'] * df['Market_Return']

    # Cost penalty when position changes (entry and exit fee)
    position_change = df['Position'].diff().fillna(df['Position']).abs()
    cost_pct = (cost_bps / 10000.0)
    df['Cost_Deduction'] = position_change * cost_pct

    df['Strategy_Net_Return'] = df['Strategy_Gross_Return'] - df['Cost_Deduction']

    # Trade counts
    active_days = df[df['Position'] != 0]
    total_trades = int((position_change > 0).sum() // 2) + 1 if len(active_days) > 0 else 0
    winning_days = (active_days['Strategy_Net_Return'] > 0).sum()
    win_rate = (winning_days / len(active_days) * 100.0) if len(active_days) > 0 else 0.0

    # Cumulative equity curve
    df['Cumulative_Market'] = (1.0 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1.0 + df['Strategy_Net_Return']).cumprod()

    net_strategy_return = float((df['Cumulative_Strategy'].iloc[-1] - 1.0) * 100.0) if len(df) > 0 else 0.0
    benchmark_return = float((df['Cumulative_Market'].iloc[-1] - 1.0) * 100.0) if len(df) > 0 else 0.0

    # Max Drawdown
    cum_strat = df['Cumulative_Strategy']
    running_max = cum_strat.cummax()
    drawdown = (cum_strat - running_max) / running_max
    max_dd = float(drawdown.min() * 100.0)

    # Profit Factor
    gains = df.loc[df['Strategy_Net_Return'] > 0, 'Strategy_Net_Return'].sum()
    losses = df.loc[df['Strategy_Net_Return'] < 0, 'Strategy_Net_Return'].abs().sum()
    profit_factor = round(float(gains / (losses + 1e-8)), 2)

    return {
        "total_trades": max(total_trades, 1) if len(active_days) > 0 else 0,
        "win_rate_pct": round(win_rate, 1),
        "net_return_pct": round(net_strategy_return, 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": profit_factor,
        "cost_bps_applied": cost_bps,
        "trades_df": df[['Date', 'Close', 'Position', 'Strategy_Net_Return', 'Cumulative_Strategy', 'Cumulative_Market']].copy()
    }
