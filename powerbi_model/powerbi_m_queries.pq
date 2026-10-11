// =============================================================================
// BIG BULL - POWER QUERY (M) INGESTION SCRIPTS
// Source Directory: g:\The Big Bull\powerbi_exports\
// =============================================================================

// Define Base Export Path Parameter (can be modified in Power BI Desktop)
let
    ExportPath = "G:\The Big Bull\powerbi_exports\"
in
    ExportPath

// --- 1. Fact_Historical_Prices ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\historical_prices.csv"),[Delimiter=",", Columns=15, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Date", type date},
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Company", type text},
        {"Open", type number},
        {"High", type number},
        {"Low", type number},
        {"Close", type number},
        {"Volume", Int64.Type},
        {"VWAP", type number},
        {"Daily_Return_Pct", type number},
        {"RSI_14", type number},
        {"ATR_14", type number},
        {"Data_Source", type text},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 2. Fact_Daily_Predictions ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\daily_predictions.csv"),[Delimiter=",", Columns=15, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Signal_Date", type date},
        {"Forecast_Date", type date},
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Current_Close", type number},
        {"Predicted_Close", type number},
        {"Predicted_Return_Pct", type number},
        {"Predicted_Direction", type text},
        {"Probability_UP", type number},
        {"Probability_DOWN", type number},
        {"Model_Name", type text},
        {"Model_Version", type text},
        {"Horizon_Days", Int64.Type},
        {"Tree_Consensus_Pct", type number},
        {"Market_Regime", type text},
        {"Out_Of_Sample", type logical},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 3. Fact_Five_Day_Predictions ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\five_day_predictions.csv"),[Delimiter=",", Columns=17, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Signal_Date", type date},
        {"Forecast_Date", type date},
        {"Step", Int64.Type},
        {"Horizon_Days", Int64.Type},
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Current_Close", type number},
        {"Predicted_Close", type number},
        {"Lower_Bound", type number},
        {"Upper_Bound", type number},
        {"Expected_Return_Pct", type number},
        {"Predicted_Direction", type text},
        {"Probability_UP", type number},
        {"Model_Name", type text},
        {"Model_Version", type text},
        {"Out_Of_Sample", type logical},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 4. Fact_Intraday_Patterns ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\intraday_patterns.csv"),[Delimiter=",", Columns=9, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Time_Interval_IST", type text},
        {"Sessions_Observed", Int64.Type},
        {"Mean_Return_Pct", type number},
        {"Win_Rate_Pct", type number},
        {"Volatility_Pct", type number},
        {"Optimal_Timing_Window", type text},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 5. Fact_Historical_Patterns ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\historical_patterns.csv"),[Delimiter=",", Columns=11, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Pattern_Category", type text},
        {"Pattern_Key", type text},
        {"Calendar_Window", type text},
        {"Sessions_Observed", Int64.Type},
        {"Mean_Return_Pct", type number},
        {"Win_Rate_Pct", type number},
        {"Volatility_Pct", type number},
        {"Historical_Tendency", type text},
        {"Market_Driver", type text},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 6. Fact_Timing_Signals ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\timing_signals.csv"),[Delimiter=",", Columns=16, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Signal_Date", type date},
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Signal", type text},
        {"Reference_Price", type number},
        {"Target_Price", type number},
        {"Stop_Loss", type number},
        {"Risk_Reward_Ratio", type number},
        {"ATR_14", type number},
        {"RSI_14", type number},
        {"Entry_Window", type text},
        {"Exit_Window", type text},
        {"Estimated_Cost_Bps", type number},
        {"Rationale", type text},
        {"Invalidation_Rule", type text},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 7. Fact_Backtest_Results ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\backtest_results.csv"),[Delimiter=",", Columns=11, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Date", type date},
        {"Close", type number},
        {"Position", Int64.Type},
        {"Strategy_Net_Return", type number},
        {"Cumulative_Strategy", type number},
        {"Cumulative_Market", type number},
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Cost_Bps", type number},
        {"Out_Of_Sample", type logical},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes

// --- 8. Fact_Model_Metrics ---
let
    Source = Csv.Document(File.Contents("G:\The Big Bull\powerbi_exports\model_metrics.csv"),[Delimiter=",", Columns=19, Encoding=65001, QuoteStyle=QuoteStyle.None]),
    PromoteHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    ChangeTypes = Table.TransformColumnTypes(PromoteHeaders,{
        {"Ticker", type text},
        {"Clean_Symbol", type text},
        {"Model_Name", type text},
        {"Model_Version", type text},
        {"Horizon_Days", Int64.Type},
        {"Holdout_Accuracy", type number},
        {"Baseline_Accuracy", type number},
        {"Accuracy_Edge", type number},
        {"Balanced_Accuracy", type number},
        {"Precision_UP", type number},
        {"F1_Score", type number},
        {"Strategy_Win_Rate_Pct", type number},
        {"Strategy_Net_Return_Pct", type number},
        {"Benchmark_Return_Pct", type number},
        {"Max_Drawdown_Pct", type number},
        {"Profit_Factor", type number},
        {"Total_Trades", Int64.Type},
        {"Out_Of_Sample", type logical},
        {"Last_Updated", type datetimezone}
    })
in
    ChangeTypes
