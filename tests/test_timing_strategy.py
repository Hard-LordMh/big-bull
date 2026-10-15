"""
test_timing_strategy.py
-----------------------
Automated tests for timing_strategy.py.
"""

import pytest
import pandas as pd
from timing_strategy import evaluate_timing_signal, backtest_cost_aware_strategy


def test_evaluate_timing_signal():
    res = evaluate_timing_signal(
        latest_close=2500.0,
        prob_up=0.65,
        prob_down=0.35,
        atr_val=45.0,
        rsi_val=55.0,
        close_vs_sma20=0.01,
        signal_date="2026-10-10",
        confidence_threshold=0.58,
        cost_bps=15.0
    )
    assert res["signal"] == "Potential Buy (Research Signal)"
    assert res["stop_loss_inr"] < 2500.0
    assert res["target_inr"] > 2500.0
    assert res["risk_reward_ratio"] > 1.0


def test_evaluate_timing_signal_hold():
    res = evaluate_timing_signal(
        latest_close=2500.0,
        prob_up=0.52,
        prob_down=0.48,
        atr_val=45.0,
        rsi_val=50.0,
        close_vs_sma20=0.01,
        signal_date="2026-10-10",
        confidence_threshold=0.58
    )
    assert res["signal"] == "Hold / No Signal"


def test_backtest_cost_aware_strategy():
    bt_data = pd.DataFrame({
        'Date': pd.date_range('2026-01-01', periods=50, freq='B'),
        'Close': [100.0 + i * 0.5 for i in range(50)],
        'Prob_UP': [0.65 if i % 2 == 0 else 0.40 for i in range(50)],
        'Prob_DOWN': [0.35 if i % 2 == 0 else 0.60 for i in range(50)]
    })
    res = backtest_cost_aware_strategy(bt_data, cost_bps=15.0)
    assert "total_trades" in res
    assert "win_rate_pct" in res
    assert "net_return_pct" in res
    assert "max_drawdown_pct" in res
