# BIG BULL — Professional Power BI Report Specification

**Application:** BIG BULL · NSE Stock Prediction & Pattern Analysis  
**Theme:** Professional Dark Navy Financial Dashboard  
**Primary Color Palette:**
* Main Background: `#0F172A` (Slate Dark Navy)
* Card / Container Background: `#1E293B` (Slate Navy Accent)
* Primary Brand Blue: `#3B82F6` (Electric Blue)
* Positive Direction / Gains: `#10B981` (Emerald Green)
* Negative Direction / Losses: `#EF4444` (Coral Red)
* Warning / Neutral / Hold: `#F59E0B` (Amber)
* Subdued Typography: `#94A3B8` / Primary Text: `#F8FAFC`

---

## Global Slicers (Header of Every Page)
* **Stock Ticker Slicer:** `Dim_Stock[Clean_Symbol]` (Single-select dropdown or tile pills: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, `ICICIBANK`)
* **Date Range Slicer:** `Dim_Date[Date]` (Relative date filter: Last 1 Year, Last 3 Years, or custom range slider)
* **Model Type Slicer:** `Fact_Five_Day_Predictions[Model_Name]` (`RandomForest-100`, `NeoQuasar/Kronos-base`, `All`)

---

## Page 1: Market Overview

### Purpose
Executive summary dashboard for high-level market health, live price metrics, and real-time algorithmic positioning across the entire NSE coverage universe.

### Visual Layout
1. **Top KPI Header Cards (Row 1):**
   * **Card 1: Selected Stock & Current Close:** `[Latest Close Price]` formatted as Currency (₹#,##0.00). Delta callout comparing to `[Previous Close Price]`.
   * **Card 2: Today's Actual Return:** `[Actual Return %]` with dynamic color formatting (Green for positive, Red for negative).
   * **Card 3: 5-Day Projected Return:** `[Projected 5D Return %]` with UP/DOWN arrow indicator.
   * **Card 4: Out-of-Sample Model Accuracy:** `[Predicted Direction Accuracy %]` formatted as `##0.0%`.
   * **Card 5: Active Trading Signal:** `Fact_Timing_Signals[Signal]` (`Potential Buy`, `Potential Sell`, `Hold`).
2. **Main Price Chart (Center-Left):**
   * **Visual Type:** Line and Clustered Column Chart.
   * **X-Axis:** `Dim_Date[Date]`
   * **Line Values:** `Fact_Historical_Prices[Close]` (Navy line `#3B82F6`), `Fact_Historical_Prices[VWAP]` (Amber dotted line `#F59E0B`).
   * **Column Values:** `Fact_Historical_Prices[Volume]` on secondary axis.
3. **NSE Watchlist Comparison Matrix (Center-Right):**
   * **Visual Type:** Table / Matrix.
   * **Rows:** `Dim_Stock[Clean_Symbol]`
   * **Values:**
     * `[Latest Close Price]`
     * `[Actual Return %]` (Data bars or Green/Red font)
     * `Fact_Daily_Predictions[Predicted_Direction]` (Icon: Green Arrow Up / Red Arrow Down)
     * `Fact_Daily_Predictions[Probability_UP]` (Percentage bar)
     * `Fact_Model_Metrics[Holdout_Accuracy]`
     * `Fact_Daily_Predictions[Market_Regime]`
4. **Bottom Row — Quick Signal & Volatility Summary:**
   * **Gauge / Linear Bullet:** `Fact_Timing_Signals[RSI_14]` with reference bands (Oversold < 30, Neutral 30-70, Overbought > 70).
   * **Metric Card:** 52-Week Range Track showing current position between `[52-Week Low]` and `[52-Week High]`.

---

## Page 2: Stock Price & Forecast

### Purpose
Deep dive into historical OHLCV price action overlaid with **Random Forest** and **Kronos-base Foundation Model** predictive trajectories.

### Visual Layout
1. **Header KPIs:**
   * `Latest Price`, `Kronos 5D Forecast Close`, `Random Forest 5D Target`, `Forecast Uncertainty Band (±%)`.
2. **Primary Forecast Overlay Chart (Hero Visual, 70% Width):**
   * **Visual Type:** Line Chart with Forecast Area Envelope.
   * **X-Axis:** `Dim_Date[Date]` (Filtered to last 90 trading days + next 5 trading days).
   * **Lines:**
     * Historical Close: `Fact_Historical_Prices[Close]` (Solid Blue `#3B82F6`).
     * Kronos Predicted Path: `Fact_Five_Day_Predictions[Predicted_Close]` where `Model_Name = "NeoQuasar/Kronos-base"` (Dashed Violet `#8B5CF6`).
     * RF Predicted Path: `Fact_Five_Day_Predictions[Predicted_Close]` where `Model_Name = "RandomForest-100"` (Dashed Emerald `#10B981`).
     * Uncertainty Ribbon: High/Low bounds as shaded range using `[Forecast Lower Bound]` and `[Forecast Upper Bound]`.
3. **Session-by-Session Forecast Table (Right Panel, 30% Width):**
   * **Visual Type:** Table.
   * **Columns:**
     * `Forecast_Date` (Formatted `dd-mmm-yyyy (ddd)`)
     * `Step` (Day 1 through Day 5)
     * `Model_Name`
     * `Predicted_Close` (INR)
     * `Expected_Return_Pct`
     * `Lower_Bound`
     * `Upper_Bound`
     * `Probability_UP`

---

## Page 3: Next-Day versus Five-Day Predictions

### Purpose
Comparative analysis contrasting short-term ultra-tactical 1-Day predictions against 5-Day swing horizon projections.

### Visual Layout
1. **Dual Decision Tree Donut Charts (Row 1):**
   * **Donut 1 (1-Day Conviction):** `Fact_Daily_Predictions[Probability_UP]` vs `Probability_DOWN` with center label of predicted direction.
   * **Donut 2 (5-Day Tree Consensus):** Breakdown of Random Forest votes (Trees voting UP vs DOWN) showing forest consensus percentage.
2. **Horizon Trajectory Comparison (Hero Chart):**
   * **Visual Type:** Clustered Column & Line Chart.
   * **X-Axis:** `Dim_Stock[Clean_Symbol]`
   * **Columns:** 1-Day Expected Return % vs 5-Day Expected Return %.
   * **Lines:** Model Accuracy for 1D vs 5D.
3. **Model Concordance Matrix:**
   * Shows agreement matrix between Random Forest classifier and Kronos autoregressive model:
     * When both models agree UP: **High Conviction Bullish** (Green)
     * When both models agree DOWN: **High Conviction Bearish** (Red)
     * When models diverge: **Divergence / Consolidation** (Amber)

---

## Page 4: Intraday and Recurring Patterns

### Purpose
Surfaces statistical calendar and intraday anomalies: Time-of-Day return distribution, Day of Week, Calendar date (1–31), Monthly seasonality, and Indian Festivals.

### Visual Layout
1. **Intraday 15-Minute Bar Momentum (Top-Left):**
   * **Visual Type:** Clustered Column Chart.
   * **X-Axis:** `Fact_Intraday_Patterns[Time_Interval_IST]` (09:15 to 15:30 IST).
   * **Y-Axis:** `Fact_Intraday_Patterns[Mean_Return_Pct]`.
   * **Color Rules:** Green if > 0, Red if < 0.
   * **Callout Labels:** Best Entry Window (**09:45–10:15 IST**), Best Exit Window (**14:45–15:15 IST**).
2. **Day of Week Win Rate & Returns (Top-Right):**
   * **Visual Type:** Bar Chart.
   * **Y-Axis:** `Fact_Historical_Patterns[Pattern_Key]` (Monday through Friday).
   * **X-Axis:** `Mean_Return_Pct` and `Win_Rate_Pct`.
3. **Calendar Date Repeat Anomalies (1–31) (Bottom-Left):**
   * **Visual Type:** Column Chart.
   * **X-Axis:** Day of Month (`Day 01` to `Day 31`).
   * **Y-Axis:** `Mean_Return_Pct`.
   * **Insights:** Highlights month-start SIP inflow surge (Days 1–5) and monthly expiry volatility.
4. **Festivals & Macro Calendar Cycles Scorecard (Bottom-Right):**
   * **Visual Type:** Matrix / Cards.
   * **Rows:** Special Events (Diwali & Festive Season, Union Budget, Month-Start SIP, F&O Expiry, FY-End Rebalancing).
   * **Values:** `Sessions_Observed`, `Mean_Return_Pct`, `Win_Rate_Pct`, `Historical_Tendency` (Green/Red badge), `Market_Driver`.

---

## Page 5: Buy/Sell Strategy Backtesting

### Purpose
Validates the disciplined, cost-aware algorithmic trading strategy on holdout out-of-sample data, factoring in realistic transaction costs (15 bps) and ATR stop-loss bounds.

### Visual Layout
1. **KPI Scorecard (Row 1):**
   * `[Net Backtested Return %]` vs `[Benchmark Buy & Hold Return %]`
   * `[Strategy Alpha %]`
   * `[Strategy Win Rate %]`
   * `[Profit Factor]`
   * `[Maximum Drawdown %]`
2. **Cumulative Equity Curve (Hero Chart, Center):**
   * **Visual Type:** Line Chart.
   * **X-Axis:** `Dim_Date[Date]` (Holdout Period Only).
   * **Line 1 (Strategy Equity):** `Fact_Backtest_Results[Cumulative_Strategy]` (Solid Emerald Green `#10B981`, Line Width = 3).
   * **Line 2 (Buy & Hold Benchmark):** `Fact_Backtest_Results[Cumulative_Market]` (Muted Slate Line `#64748B`, Line Width = 2).
   * **Secondary Area (Drawdown):** Underwater drawdown curve showing distance from peak equity.
3. **Recent Executed Signals Table (Bottom Panel):**
   * **Columns:**
     * `Date`, `Ticker`, `Position` (Long / Cash / Short), `Market_Return`, `Strategy_Net_Return`, `Cost_Bps`, `Out_Of_Sample`.

---

## Page 6: Model Accuracy and Risk

### Purpose
Exclusively evaluates out-of-sample statistical rigor, walk-forward validation folds, confusion matrices, and risk guardrails.

### Visual Layout
1. **Accuracy Edge Breakdown (Top-Left):**
   * **Visual Type:** Clustered Bar Chart.
   * **Categories:** `Dim_Stock[Clean_Symbol]`.
   * **Metrics:** `Holdout_Accuracy` vs `Baseline_Accuracy`.
   * **Key Point:** Visually highlights the statistical edge generated over random guessing.
2. **Confusion Matrix Heatmap (Top-Right):**
   * **Visual Type:** Matrix with conditional background color formatting.
   * **Rows:** Actual DOWN / Actual UP.
   * **Columns:** Predicted DOWN / Predicted UP.
   * **Cells:** True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).
3. **Risk & Invalidation Criteria Panel (Bottom-Left):**
   * **Visual Type:** Multi-row Card / Table.
   * **Fields:** `Fact_Timing_Signals[Signal]`, `Stop_Loss`, `Target_Price`, `Risk_Reward_Ratio`, `Invalidation_Rule`.
4. **Model Governance & Metadata (Bottom-Right):**
   * Card displaying: Model Version (`v2.4-Hybrid`), Data Freshness (`Last_Updated`), Training Architecture (`100-Tree Random Forest + NeoQuasar/Kronos-base Foundation Model`), Validation Embargo (`Purged Walk-Forward`).
