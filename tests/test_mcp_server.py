"""
test_mcp_server.py
------------------
Automated unit tests for Big Bull AI Model Context Protocol (MCP) Server.
"""

import pytest
import json
from mcp_server import (
    predict_stock_direction,
    detect_market_regime_and_anomalies,
    get_buy_sell_timing,
    analyze_stock_patterns,
    get_intraday_analysis,
    train_and_export_pipeline,
    get_outputs_summary
)


def test_mcp_predict_direction():
    res = json.loads(predict_stock_direction(ticker="RELIANCE.NS", horizon=5))
    assert "predicted_direction" in res
    assert res["predicted_direction"] in ["UP", "DOWN"]
    assert "probability_up_pct" in res
    assert "test_accuracy_pct" in res
    assert "tree_consensus" in res
    assert "market_regime" in res


def test_mcp_detect_regime_and_anomalies():
    res = json.loads(detect_market_regime_and_anomalies(ticker="RELIANCE.NS"))
    assert "tree_consensus" in res
    assert "market_regime" in res
    assert "breakout_anomaly" in res
    assert res["tree_consensus"]["total_trees"] == 100
    assert "badge" in res["market_regime"]



def test_mcp_timing_signal():
    res = json.loads(get_buy_sell_timing(ticker="RELIANCE.NS", horizon=5))
    assert "signal" in res
    assert res["signal"] in ["Potential Buy (Research Signal)", "Potential Sell (Research Signal)", "Hold / No Signal"]
    assert "reference_price" in res


def test_mcp_patterns():
    res = json.loads(analyze_stock_patterns(ticker="RELIANCE.NS", dimension="day_of_week"))
    assert res["pattern_type"] == "Day of Week"
    assert len(res["results"]) > 0


def test_mcp_intraday():
    res = json.loads(get_intraday_analysis(ticker="RELIANCE.NS"))
    assert "predicted_direction" in res
    assert "intraday_test_accuracy_pct" in res


def test_mcp_resource_summary():
    res = json.loads(get_outputs_summary())
    assert "outputs_directory" in res
    assert "files" in res
