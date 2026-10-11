---
name: big-bull-intelligence
description: "AI-Powered NSE Stock Direction Prediction, Pattern Analysis & Intraday Timing System. Triggers: stock prediction, stock direction, NSE, RELIANCE, TCS, INFY, HDFC, Buy/Sell timing, stock patterns, intraday, Power BI export."
---

# Big Bull AI - Stock Intelligence Skill

This skill enables AI agents in Antigravity to analyze NSE-listed stocks, predict 1-day and 5-day price direction, identify recurring historical patterns, evaluate disciplined Buy/Sell opportunities, and trigger model training.

## When to Use
Invoke this skill whenever a request involves:
- Predicting future direction (UP or DOWN) for an NSE stock.
- Researching Buy, Sell, or Hold timing opportunities with ATR stop-loss and target bounds.
- Finding historical recurring patterns (Day of Week, Monthly Seasonality, Annual Regimes, Similar Historical Setups).
- Inspecting 15-minute intraday candlestick movements and VWAP trends.
- Fetching live quotes from Google Finance and real daily/intraday OHLCV market data.
- Searching any company and pinning up to 3 companies for side-by-side comparison.
- Training or retraining the Random Forest classification models on real market data.
- Exporting Power BI CSV bundles.

## Available Python Modules & CLI Commands

### 1. Real Market Data Service & CLI Training
Query live quotes or run model training on real data:
```bash
python train_pipeline.py
```
```python
from real_data_service import fetch_real_quote, fetch_real_daily_data, search_companies
quote = fetch_real_quote("RELIANCE.NS")
print(quote["price"], quote["day_change_pct"], quote["google_finance_url"])
```

### 2. Querying MCP Tools Directly in Python
The MCP server tools in `mcp_server.py` can be imported and executed programmatically:

```python
import json
from mcp_server import (
    predict_stock_direction,
    detect_market_regime_and_anomalies,
    get_buy_sell_timing,
    analyze_stock_patterns,
    get_intraday_analysis,
    train_and_export_pipeline
)

# 1. 5-Day Directional Prediction
pred = json.loads(predict_stock_direction(ticker="RELIANCE.NS", horizon=5))
print(pred["predicted_direction"], pred["probability_up_pct"], pred["tree_consensus"], pred["market_regime"])

# 2. Random Forest Tree Consensus & Anomaly Detection
rf_det = json.loads(detect_market_regime_and_anomalies(ticker="RELIANCE.NS"))
print(rf_det["tree_consensus"]["strength_label"], rf_det["market_regime"]["badge"], rf_det["breakout_anomaly"]["badge"])

# 3. Disciplined Buy/Sell Timing
timing = json.loads(get_buy_sell_timing(ticker="RELIANCE.NS", horizon=5))
print(timing["signal"], timing["target_inr"], timing["stop_loss_inr"], timing["risk_reward_ratio"])

# 4. Recurring Patterns
patterns = json.loads(analyze_stock_patterns(ticker="RELIANCE.NS", dimension="day_of_week"))
print(patterns["results"])

# 5. Intraday 15m Analysis
intra = json.loads(get_intraday_analysis(ticker="RELIANCE.NS"))
print(intra["predicted_direction"], intra["latest_close_inr"])
```

### 3. Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
App opens at `http://localhost:8501`.

## Guidelines
- Never fabricate future price curves; directional predictions predict binary UP/DOWN probabilities.
- Directional probabilities should be compared against the majority-class baseline.
- All technical features strictly use data available on or before the signal date (zero lookahead leakage).
- All predictions are probabilistic estimates for educational research, not financial advice.
