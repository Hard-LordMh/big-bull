# BIG BULL — Complete Technical & Model Architecture Report

**Document Version:** 2.0 (Full Benchmark Edition)  
**System Scope:** AI-Powered NSE Stock Direction Prediction, 5-Year Walk-Forward Replay, Next-Week Forecasting, and Instagram-Inspired Story UI.

---

## 1. Executive Summary

BIG BULL is a quantitative stock intelligence and machine learning system built for NSE-listed equities. It couples rigorous **supervised binary classification** (Random Forest) and **unsupervised anomaly detection** (Isolation Forest) with strict time-series leakage controls (chronological splits, horizon-matched embargo gaps) and high-impact UI presentation.

### Key Measured Findings (5-Year Walk-Forward Replay, 5 Stocks, 6,175 Signals Total)
- **1-Day Directional Accuracy:** Mean **50.51%** (Baseline 51.53%, ROC AUC **0.5046**).
- **5-Day Directional Accuracy:** Mean **50.46%** (Baseline 52.54%, ROC AUC **0.5022**).
- **Verdict:** Statistically indistinguishable from a random coin flip on price-derived technicals alone. The codebase implements correct institutional validation protocols (no lookahead, embargo, baseline comparison), which proves why technical features alone hold negligible alpha over majority-class baselines.

---

## 2. Technical Stack & Frameworks

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Language | **Python** | 3.14 | Backend, ML training, inference engine |
| UI Framework | **Streamlit** | 1.65.0 | Interactive web application & Instagram-style UI feeds |
| Data Processing | **pandas / NumPy** | 2.x / 1.24+ | OHLCV manipulation, vectorized technical indicators |
| Machine Learning | **scikit-learn** | 1.3+ | `RandomForestClassifier`, `IsolationForest`, metrics |
| Visualization | **Plotly** | 5.18+ | Candlestick charts, confidence bands, return distribution charts |
| Market Data | **yfinance** | 1.7+ | Real NSE daily and 15-minute intraday feeds |
| Timezone | **pytz** | 2023.3+ | NSE trading hours alignment (09:15–15:30 IST) |
| Agent Protocol | **MCP** (`mcp`) | 1.0+ | Model exposed as callable AI tools and resources |
| Testing | **pytest** | 7.4+ | 39 automated test cases |

---

## 3. System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION & VALIDATION (real_data_service.py, data_validation.py)  │
│    yfinance daily / intraday feeds · OHLCV sanity · ffill/bfill             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING & TARGETS (feature_engineering.py)                   │
│    22 Technical Indicators · 1D & 5D Binary Targets                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ML MODEL ENGINE (stock_model.py)                                         │
│    Random Forest (100 trees) · Isolation Forest · Walk-Forward Embargo      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 4. TIMING & PATTERN RESEARCH         │ 5. 5-YEAR REPLAY & FORECAST          │
│    timing_strategy.py                │    stock_model.py                    │
│    pattern_analysis.py               │    59 Folds · Next-Week 5-Day Call   │
└──────────────────────────────────────┴──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. USER INTERFACE & API (app.py, mcp_server.py, export_powerbi_pipeline.py) │
│    Instagram-Style Story Feed · Interactive Charts · 13 Power BI CSVs       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Engineering (22 Indicators)

All features are calculated with strictly past data (zero lookahead leakage):
1. **Returns:** `Return_1D`, `Return_2D`, `Return_5D`, `Return_10D`
2. **Prior Closes:** `Prev_Close_1D`, `Prev_Close_2D`
3. **Moving Averages & Ratios:** `SMA_5`, `SMA_10`, `SMA_20`, `Close_vs_SMA_5`, `Close_vs_SMA_10`, `Close_vs_SMA_20`
4. **Volatility & Range:** `Volatility_5D`, `Volatility_10D`, `ATR_14`, `Intraday_Range`, `Open_Gap`
5. **Volume:** `Volume_Change_1D`, `Volume_Ratio_5D`
6. **Oscillators:** `RSI_14`, `MACD`, `MACD_Signal`

---

## 5. Full Empirical Benchmark Results (5 Years, 5 Stocks)

Evaluated via expanding walk-forward replay with refit every 21 sessions and horizon-matched embargo:

### 5-Day Horizon (`Target_5D_UP`)
| Stock | Signals | Accuracy | Baseline | Edge | ROC AUC | Precision (UP) | Recall (UP) | F1 |
|---|---|---|---|---|---|---|---|---|
| **RELIANCE.NS** | 1,231 | 50.45% | 51.50% | −1.05 | 0.4977 | 53.80% | 26.81% | 35.79% |
| **TCS.NS** | 1,231 | 49.47% | 52.32% | −2.85 | 0.4955 | 46.67% | 41.74% | 44.06% |
| **HDFCBANK.NS** | 1,231 | 51.10% | 51.18% | −0.08 | 0.5114 | 53.06% | 38.57% | 44.67% |
| **INFY.NS** | 1,231 | 53.37% | 51.67% | +1.70 | 0.5277 | 55.10% | 52.67% | 53.86% |
| **SBIN.NS** | 1,231 | 47.93% | 56.05% | −8.12 | 0.4787 | 54.40% | 43.91% | 48.60% |
| **Aggregate (Mean)** | **1,231** | **50.46%** | **52.54%** | **−2.08** | **0.5022** | **52.61%** | **40.74%** | **45.40%** |

### 1-Day Horizon (`Target_1D_UP`)
- **Aggregate 1D Accuracy:** **50.51%** (Baseline **51.53%**, Mean ROC AUC **0.5046**, Mean Edge **−1.02%**).

---

## 6. Instagram-Style UI Experience ("Insta-Feed")

To make stock prediction engaging and modern, the user interface features:
- **Story Ring Avatar Cards:** Glowing gradient borders around company badges in a horizontal story feed.
- **Story Overlay Cards:** Vertical cards styled like Instagram Reels/Stories featuring bold UP/DOWN typography, neon glow accents, probability meters, and concise one-line AI takeaways.
- **Minimalist Clean Layout:** Removal of cluttered tabular noise, focusing purely on high-impact visual indicators and instant insight cards.

---

## 7. Roadmap to True Predictive Alpha
To push accuracy beyond 50% into actionable alpha:
1. **Regime Filter:** Abstained signals during range-bound chop (+1–2 pts).
2. **Cross-Sectional Features:** Incorporating NIFTY index beta and sector relative strength (+1–3 pts).
3. **Probability Calibration:** Isotonic calibration rejecting low-confidence grey-zone calls (+0.5–1 pt).
4. **Cost-Aware Optimization:** Training against net returns exceeding transaction friction (15 bps).
