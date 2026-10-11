# 📈 Big Bull · AI-Powered NSE Stock Prediction & Pattern Intelligence

A professional, beginner-friendly quantitative stock intelligence web application designed for **NSE-listed stocks**.

Predicts likely price directions (**Next Trading Day** and **Next 5 Trading Days**), identifies recurring historical patterns across dates and times, analyzes intraday 15-minute movements, and estimates disciplined Buy/Sell opportunities with dynamic risk/reward bounds.

---

## 🧭 System Architecture & Features

### 1. Modern Financial Dashboard (9 Sidebar Pages)
- **1. Home Dashboard:** High-level market overview, latest closing price, 1D % change, calibrated directional prediction, and interactive Plotly price chart with forecast horizon divider.
- **2. Upload Stock Data:** Drag-and-drop CSV / Excel upload for both Daily and Intraday datasets, with real-time data quality diagnostics (OHLC logic, non-negative volume, chronological integrity).
- **3. Predict Future:** Horizon-specific directional models (1-Day vs 5-Day) with probability, confidence ratings, and top feature drivers.
- **4. Intraday Analysis:** 15-minute candlestick charts with VWAP and EMA-9/21 overlays, next-candle direction forecast, and time-of-day return breakdown during NSE hours (09:15 to 15:30 IST).
- **5. Pattern Finder:** 
  - Day-of-week return distributions (Monday to Friday).
  - Monthly seasonality (January to December).
  - Year-over-year performance regimes.
  - Nearest Neighbors pattern matching (scans historical setups matching today's RSI, momentum, and volatility to show forward outcomes).
- **6. Buy/Sell Timing:** Disciplined research signals (Potential Buy, Potential Sell, Hold/No Signal) with ATR-based stop-loss and profit target prices, risk/reward ratios, and signal invalidation criteria.
- **7. Backtesting & Accuracy:** Chronological 80/20 holdout backtesting, 5-fold walk-forward validation with embargo gaps, confusion matrix, and cost-aware strategy equity curves (accounting for 15 bps transaction fees and slippage).
- **8. Results & Power BI:** One-click ZIP download of all standardized CSV outputs and step-by-step import instructions for Microsoft Power BI Desktop.
- **9. Settings:** Customize trader risk profile, transaction fees/slippage (bps), and model hyperparameters.

---

## 📁 Repository Structure

```
g:\The Big Bull/
│
├── app.py                   # Main 9-page Streamlit Financial Intelligence Application
├── data_validation.py       # Daily & Intraday data validation and normalization
├── feature_engineering.py   # Multi-horizon feature generation (RSI, ATR, MACD, SMAs)
├── stock_model.py           # Random Forest classifier with walk-forward validation
├── pattern_analysis.py      # Day of week, month seasonality, and similar setup matching
├── timing_strategy.py       # Buy/Sell signal research & cost-aware backtesting
├── intraday_model.py        # 15-minute intraday generator, VWAP/EMA & direction modeling
├── utils.py                 # Power BI export manager, ZIP bundler & synthetic data
├── conftest.py              # Pytest configuration
├── requirements.txt         # Core dependencies
├── README.md                # System documentation
│
├── outputs/                 # Standardized Power BI CSV exports & run summary
│   ├── historical_prices.csv
│   ├── daily_predictions.csv
│   ├── five_day_predictions.csv
│   ├── intraday_predictions.csv
│   ├── timing_signals.csv
│   ├── historical_patterns.csv
│   ├── backtest_predictions.csv
│   ├── holdout_metrics.csv
│   ├── walk_forward_metrics.csv
│   ├── feature_importance.csv
│   ├── latest_prediction.csv
│   └── run_summary.json
│
├── models/                  # Serialized Joblib model package
│   └── nse_direction_model.joblib
│
└── tests/                   # Automated unit test suite (19 test cases)
    ├── test_data_validation.py
    ├── test_feature_engineering.py
    ├── test_stock_model.py
    ├── test_pattern_analysis.py
    ├── test_timing_strategy.py
    └── test_intraday_model.py
```

---

## 🚀 Setup & Launch Commands

Run the following commands in Windows PowerShell:

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```powershell
pytest tests/ -v
```
*(All 19 test cases validate data integrity, feature engineering, models, patterns, and timing strategies).*

### 3. Launch Streamlit Application
```powershell
streamlit run app.py
```
*The application opens in your browser at `http://localhost:8501`.*

---

## 📊 Data Schema Requirements

### Daily Historical Data
- **Date:** `YYYY-MM-DD` or parseable datetime.
- **Open, High, Low, Close:** Numeric positive prices.
- **Volume:** Non-negative traded quantity.

### Intraday Candlestick Data
- **Timestamp:** Timestamp with date and time (e.g., `YYYY-MM-DD HH:MM:SS`).
- **Open, High, Low, Close, Volume:** Standard OHLCV.

---

## 📦 Microsoft Power BI Desktop Integration

All model runs automatically save standardized CSV files to the `outputs/` folder. You can also download them all at once via the **"📦 Download Complete Power BI Bundle (ZIP)"** button on page 8.

### Import Steps:
1. Open **Power BI Desktop**.
2. Select **Get Data ──► Text/CSV**.
3. Load `backtest_predictions.csv`, `timing_signals.csv`, `historical_patterns.csv`, and `holdout_metrics.csv`.
4. Create relationships on the `Date` or `Ticker` column to build institutional-grade dashboards.

---

## 🤖 Model Context Protocol (MCP) Integration

Big Bull AI implements a native **Model Context Protocol (MCP) Server** via [mcp_server.py](file:///g:/The%20Big%20Bull/mcp_server.py). This allows AI assistants (Claude Desktop, Cursor, Antigravity, etc.) to directly call Big Bull's forecasting, pattern analysis, and timing tools.

### Available MCP Tools:
1. `predict_stock_direction(ticker, horizon)`: AI directional predictions (UP/DOWN), calibrated probabilities, and holdout accuracy.
2. `get_buy_sell_timing(ticker, horizon)`: Research signals (Potential Buy/Sell/Hold), ATR targets, stop-losses, and risk/reward ratios.
3. `analyze_stock_patterns(ticker, dimension)`: Day of week, monthly seasonality, annual regimes, or similar historical setups.
4. `get_intraday_analysis(ticker)`: 15-minute candlestick trend, VWAP, EMA overlays, and next-bar predictions.
5. `train_and_export_pipeline(ticker, num_days)`: Executes model training and outputs updated Power BI CSVs.

### Available MCP Resources:
- `stock://metrics/holdout`: Latest holdout backtest accuracy scorecard.
- `stock://outputs/summary`: Summary of all generated CSV reports in `outputs/`.

### How to Connect to Claude Desktop or Cursor:
Add the configuration from [mcp_config.json](file:///g:/The%20Big%20Bull/mcp_config.json) into your Claude Desktop (`claude_desktop_config.json`) or Cursor MCP settings:

```json
{
  "mcpServers": {
    "big-bull-ai": {
      "command": "python",
      "args": ["g:/The Big Bull/mcp_server.py"],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

---

## ⚠️ Important Educational Disclaimer
This system is an AI research and decision-support tool. Model probabilities and research signals reflect historical statistical patterns and do not guarantee future market performance. It does not execute automated trades. Always practice strict risk management and stop-loss discipline.
