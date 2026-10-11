"""
pattern_analysis.py
-------------------
Module for detecting and evaluating recurring historical patterns across:
1. Day of the Week (Monday to Friday)
2. Month of the Year (Seasonality)
3. Year-by-Year Regime Comparisons
4. Time of Day (Intraday Intervals)
5. Similar Historical Setups (Nearest Neighbors Pattern Matching)
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def analyze_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical return distributions by day of the week (Monday - Friday).
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Name'] = df['Date'].dt.day_name()
    df['Day_Num'] = df['Date'].dt.dayofweek
    df['Return_1D'] = df['Close'].pct_change() * 100.0

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    valid = df.dropna(subset=['Return_1D'])

    summary = []
    for day in days_order:
        sub = valid[valid['Day_Name'] == day]
        if sub.empty:
            continue
        n_sessions = len(sub)
        avg_ret = float(sub['Return_1D'].mean())
        median_ret = float(sub['Return_1D'].median())
        win_rate = float((sub['Return_1D'] > 0).mean() * 100.0)
        vol = float(sub['Return_1D'].std())
        summary.append({
            "Day": day,
            "Sessions": n_sessions,
            "Mean_Return_Pct": round(avg_ret, 3),
            "Median_Return_Pct": round(median_ret, 3),
            "Win_Rate_Pct": round(win_rate, 1),
            "Volatility_Pct": round(vol, 3)
        })

    return pd.DataFrame(summary)


def analyze_month_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical monthly seasonality (January - December).
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month_Name'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Calculate monthly return from first to last bar in each month
    monthly_prices = df.groupby(['Year', 'Month']).agg(
        First_Close=('Close', 'first'),
        Last_Close=('Close', 'last'),
        Month_Name=('Month_Name', 'first')
    ).reset_index()

    monthly_prices['Month_Return_Pct'] = (
        (monthly_prices['Last_Close'] - monthly_prices['First_Close']) / monthly_prices['First_Close']
    ) * 100.0

    months_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]

    summary = []
    for m in months_order:
        sub = monthly_prices[monthly_prices['Month_Name'] == m]
        if sub.empty:
            continue
        count = len(sub)
        avg_ret = float(sub['Month_Return_Pct'].mean())
        win_rate = float((sub['Month_Return_Pct'] > 0).mean() * 100.0)
        vol = float(sub['Month_Return_Pct'].std()) if count > 1 else 0.0
        summary.append({
            "Month": m,
            "Sample_Count": count,
            "Mean_Return_Pct": round(avg_ret, 2),
            "Win_Rate_Pct": round(win_rate, 1),
            "Std_Dev_Pct": round(vol, 2)
        })

    return pd.DataFrame(summary)


def analyze_year_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares annual return, win rates, and volatility across years.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Return_1D'] = df['Close'].pct_change() * 100.0

    summary = []
    for year, grp in df.groupby('Year'):
        n_days = len(grp)
        first_p = grp['Close'].iloc[0]
        last_p = grp['Close'].iloc[-1]
        ann_ret = ((last_p - first_p) / first_p) * 100.0
        daily_rets = grp['Return_1D'].dropna()
        win_rate = float((daily_rets > 0).mean() * 100.0) if not daily_rets.empty else 0.0
        ann_vol = float(daily_rets.std() * np.sqrt(252)) if not daily_rets.empty else 0.0
        
        summary.append({
            "Year": int(year),
            "Trading_Days": n_days,
            "Total_Return_Pct": round(ann_ret, 2),
            "Daily_Win_Rate_Pct": round(win_rate, 1),
            "Annualized_Vol_Pct": round(ann_vol, 2)
        })

    return pd.DataFrame(summary).sort_values('Year', ascending=False).reset_index(drop=True)


def analyze_intraday_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical intraday returns by time interval (e.g. 15-minute or hourly buckets).
    """
    if 'Timestamp' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Str'] = df['Timestamp'].dt.strftime('%H:%M')
    df['Return_Bar'] = df['Close'].pct_change() * 100.0

    valid = df.dropna(subset=['Return_Bar'])
    summary = []
    for time_bucket, grp in valid.groupby('Time_Str'):
        n_bars = len(grp)
        avg_ret = float(grp['Return_Bar'].mean())
        win_rate = float((grp['Return_Bar'] > 0).mean() * 100.0)
        vol = float(grp['Return_Bar'].std())
        summary.append({
            "Time": time_bucket,
            "Observations": n_bars,
            "Mean_Return_Pct": round(avg_ret, 3),
            "Win_Rate_Pct": round(win_rate, 1),
            "Volatility_Pct": round(vol, 3)
        })

    return pd.DataFrame(summary).sort_values('Time').reset_index(drop=True)


def find_similar_historical_setups(
    df: pd.DataFrame,
    top_k: int = 10,
    horizon: int = 5
) -> Dict[str, Any]:
    """
    Searches historical data for the most similar preceding technical setups to the latest trading day.
    Uses Euclidean distance across standardized features (RSI_14, 5-day return, SMA_20 ratio, Volatility).
    Examines what happened over the subsequent `horizon` trading sessions for matched observations.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate similarity features
    df['Return_5D'] = df['Close'].pct_change(5)
    df['Return_10D'] = df['Close'].pct_change(10)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    df['SMA_20_Ratio'] = (df['Close'] / df['Close'].rolling(20).mean()) - 1.0
    df['Volatility_10D'] = df['Close'].pct_change().rolling(10).std()

    # Forward return over target horizon
    df['Forward_Return'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close'] * 100.0
    df['Forward_Direction'] = np.where(df['Forward_Return'] > 0, 'UP', 'DOWN')

    feature_cols = ['Return_5D', 'RSI_14', 'SMA_20_Ratio', 'Volatility_10D']
    clean_df = df.dropna(subset=feature_cols).copy().reset_index(drop=True)

    if len(clean_df) < top_k + 15:
        return {"matches": pd.DataFrame(), "summary": {}}

    # Latest target vector (the setup to match)
    target_vec = clean_df.iloc[-1][feature_cols].values.astype(float)
    target_date = str(clean_df.iloc[-1]['Date'].strftime('%Y-%m-%d'))
    target_close = float(clean_df.iloc[-1]['Close'])

    # Historical candidate pool (strictly exclude the last `horizon` days to ensure forward return is observed)
    historical_pool = clean_df.iloc[:-horizon].copy()

    # Standardize features for distance calculation
    mean_vals = historical_pool[feature_cols].mean().values
    std_vals = historical_pool[feature_cols].std().values + 1e-8

    norm_target = (target_vec - mean_vals) / std_vals
    norm_pool = (historical_pool[feature_cols].values - mean_vals) / std_vals

    # Euclidean distance
    distances = np.linalg.norm(norm_pool - norm_target, axis=1)
    historical_pool['Distance'] = distances

    # Top K nearest matches
    matches = historical_pool.sort_values('Distance').head(top_k).copy()
    matches['Date_Str'] = matches['Date'].dt.strftime('%Y-%m-%d')

    matches_display = matches[[
        'Date_Str', 'Close', 'RSI_14', 'Return_5D', 'Forward_Return', 'Forward_Direction', 'Distance'
    ]].copy()
    matches_display.columns = [
        'Match_Date', 'Close_INR', 'RSI', 'Prior_5D_Ret_Pct', 'Forward_5D_Ret_Pct', 'Subsequent_Direction', 'Similarity_Score'
    ]
    # Invert distance to a 0-100% similarity score
    matches_display['Similarity_Pct'] = np.round(100.0 / (1.0 + matches_display['Similarity_Score']), 1)
    matches_display['Forward_5D_Ret_Pct'] = np.round(matches_display['Forward_5D_Ret_Pct'], 2)
    matches_display['RSI'] = np.round(matches_display['RSI'], 1)

    up_count = int((matches['Forward_Direction'] == 'UP').sum())
    down_count = int((matches['Forward_Direction'] == 'DOWN').sum())
    win_rate = round((up_count / len(matches)) * 100.0, 1) if len(matches) > 0 else 0.0
    avg_fwd_ret = round(float(matches['Forward_Return'].mean()), 2)

    return {
        "target_date": target_date,
        "target_close": target_close,
        "matches_table": matches_display.drop(columns=['Similarity_Score']),
        "summary": {
            "total_matches": len(matches),
            "up_count": up_count,
            "down_count": down_count,
            "up_frequency_pct": win_rate,
            "avg_forward_return_pct": avg_fwd_ret
        }
    }


def analyze_day_of_month_repeat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical price returns across specific days of the month (1st through 31st).
    Identifies recurring calendar date anomalies (e.g., month-start rally, mid-month dip, month-end momentum).
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Of_Month'] = df['Date'].dt.day
    df['Return_1D'] = df['Close'].pct_change() * 100.0

    valid = df.dropna(subset=['Return_1D'])
    summary = []
    for dom, grp in valid.groupby('Day_Of_Month'):
        if len(grp) < 3:
            continue
        avg_ret = float(grp['Return_1D'].mean())
        win_rate = float((grp['Return_1D'] > 0).mean() * 100.0)
        vol = float(grp['Return_1D'].std())
        summary.append({
            "Day_Of_Month": int(dom),
            "Date_Label": f"Day {int(dom):02d} of Month",
            "Observations": len(grp),
            "Mean_Return_Pct": round(avg_ret, 3),
            "Win_Rate_Pct": round(win_rate, 1),
            "Volatility_Pct": round(vol, 3),
            "Historical_Tendency": "🟢 Bullish Repeat" if avg_ret > 0.1 else ("🔴 Bearish Dip" if avg_ret < -0.1 else "⚪ Neutral")
        })

    return pd.DataFrame(summary).sort_values('Day_Of_Month').reset_index(drop=True)


def get_optimal_intraday_windows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates the exact optimal time windows to BUY and SELL during NSE trading hours (09:15 to 15:30 IST)
    based on historical 15-minute bar returns and win rates.
    """
    if df is None or df.empty or 'Timestamp' not in df.columns:
        return {
            "best_buy_window": "09:45 AM - 10:15 AM IST",
            "best_sell_window": "14:45 PM - 15:15 PM IST",
            "top_buy_times": [],
            "top_sell_times": [],
            "summary_message": "Intraday analysis requires timestamped OHLCV bars."
        }

    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Str'] = df['Timestamp'].dt.strftime('%H:%M')
    df['Return_Bar'] = df['Close'].pct_change() * 100.0

    valid = df.dropna(subset=['Return_Bar'])
    summary = []
    for time_bucket, grp in valid.groupby('Time_Str'):
        if len(grp) < 2:
            continue
        avg_ret = float(grp['Return_Bar'].mean())
        win_rate = float((grp['Return_Bar'] > 0).mean() * 100.0)
        summary.append({
            "Time": time_bucket,
            "Observations": len(grp),
            "Mean_Return_Pct": round(avg_ret, 3),
            "Win_Rate_Pct": round(win_rate, 1)
        })

    sum_df = pd.DataFrame(summary)
    if sum_df.empty:
        return {
            "best_buy_window": "09:45 AM - 10:15 AM IST",
            "best_sell_window": "14:45 PM - 15:15 PM IST",
            "top_buy_times": [],
            "top_sell_times": [],
            "summary_message": "Insufficient intraday bars."
        }

    # Best BUY times: highest positive return and highest win rate
    buy_df = sum_df.sort_values(by=['Mean_Return_Pct', 'Win_Rate_Pct'], ascending=[False, False]).head(3)
    # Best SELL times: lowest return / profit taking dip window
    sell_df = sum_df.sort_values(by=['Mean_Return_Pct', 'Win_Rate_Pct'], ascending=[True, True]).head(3)

    top_buy_time = buy_df['Time'].iloc[0] if not buy_df.empty else "09:45"
    top_sell_time = sell_df['Time'].iloc[0] if not sell_df.empty else "14:45"

    return {
        "best_buy_window": f"{top_buy_time} IST (Optimal Buy Window)",
        "best_sell_window": f"{top_sell_time} IST (Optimal Profit Exit Window)",
        "top_buy_times": buy_df.to_dict(orient="records"),
        "top_sell_times": sell_df.to_dict(orient="records"),
        "summary_message": f"Historical data shows highest upward momentum around {top_buy_time} IST and profit taking around {top_sell_time} IST."
    }


def analyze_special_events_and_festivals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes historical price returns during key Indian market festivals, recurring calendar cycles,
    and macroeconomic events (Diwali & Festive season, Union Budget, Fiscal Year-End, F&O Monthly Expiry,
    Month-Start SIP Inflow, and Year-End Rally).
    """
    if df is None or df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Return_1D'] = df['Close'].pct_change() * 100.0
    valid = df.dropna(subset=['Return_1D']).copy()

    if valid.empty:
        return pd.DataFrame()

    valid['Month'] = valid['Date'].dt.month
    valid['Day'] = valid['Date'].dt.day
    valid['Year'] = valid['Date'].dt.year
    valid['DayOfWeek'] = valid['Date'].dt.dayofweek  # 3 = Thursday

    # Identify Monthly F&O Expiry dates: last Thursday of each month
    thursdays = valid[valid['DayOfWeek'] == 3]
    last_thursdays = thursdays.groupby(['Year', 'Month'])['Date'].max().values if not thursdays.empty else []

    events_definitions = [
        {
            "name": "Diwali & Festive Season",
            "window": "October 01 – November 15",
            "condition": lambda d: (d['Month'] == 10) | ((d['Month'] == 11) & (d['Day'] <= 15)),
            "driver": "Consumer demand, festive liquidity & Muhurat trading sentiment."
        },
        {
            "name": "Union Budget Window",
            "window": "January 25 – February 05",
            "condition": lambda d: ((d['Month'] == 1) & (d['Day'] >= 25)) | ((d['Month'] == 2) & (d['Day'] <= 5)),
            "driver": "Fiscal policy anticipation, sector incentives, and high volatility."
        },
        {
            "name": "Month-Start SIP Inflows",
            "window": "Days 01 – 05 of Every Month",
            "condition": lambda d: d['Day'] <= 5,
            "driver": "Systematic Investment Plan (SIP) mutual fund buying & salary allocations."
        },
        {
            "name": "Monthly F&O Expiry Day",
            "window": "Last Thursday of Every Month",
            "condition": lambda d: d['Date'].isin(last_thursdays),
            "driver": "Derivatives settlement, gamma hedging, and high intraday rollover volatility."
        },
        {
            "name": "Fiscal Year-End Adjustments",
            "window": "March 20 – March 31",
            "condition": lambda d: (d['Month'] == 3) & (d['Day'] >= 20),
            "driver": "Advance tax obligations, NAV propping, and tax-loss harvesting."
        },
        {
            "name": "New Fiscal Year Rally",
            "window": "April 01 – April 10",
            "condition": lambda d: (d['Month'] == 4) & (d['Day'] <= 10),
            "driver": "Fresh fiscal year portfolio allocations and new institutional capital deployment."
        },
        {
            "name": "Independence Day Season",
            "window": "August 01 – August 15",
            "condition": lambda d: (d['Month'] == 8) & (d['Day'] <= 15),
            "driver": "Mid-year corporate earnings review and domestic economic confidence."
        },
        {
            "name": "Year-End / Santa Rally",
            "window": "December 20 – December 31",
            "condition": lambda d: (d['Month'] == 12) & (d['Day'] >= 20),
            "driver": "Global FII book closing, lower liquidity, and calendar year mark-to-market."
        }
    ]

    summary = []
    for ev in events_definitions:
        mask = ev["condition"](valid)
        sub = valid[mask]
        count = len(sub)
        if count == 0:
            continue
        avg_ret = float(sub['Return_1D'].mean())
        win_rate = float((sub['Return_1D'] > 0).mean() * 100.0)
        vol = float(sub['Return_1D'].std()) if count > 1 else 0.0

        if avg_ret > 0.15 and win_rate >= 54.0:
            tendency = "🟢 Strong Bullish"
        elif avg_ret > 0.04:
            tendency = "🟢 Mild Bullish"
        elif avg_ret < -0.15 and win_rate < 46.0:
            tendency = "🔴 Bearish Pressure"
        elif avg_ret < -0.04:
            tendency = "🔴 Mild Bearish"
        else:
            tendency = "⚪ Mixed / Neutral"

        summary.append({
            "Event": ev["name"],
            "Calendar_Window": ev["window"],
            "Sessions": count,
            "Mean_Return_Pct": round(avg_ret, 3),
            "Win_Rate_Pct": round(win_rate, 1),
            "Volatility_Pct": round(vol, 3),
            "Historical_Tendency": tendency,
            "Market_Driver": ev["driver"]
        })

    return pd.DataFrame(summary)


def detect_current_special_event(ref_date=None) -> Dict[str, Any]:
    """
    Detects if the current date (or specified date) matches or is near any known special market window.
    """
    import datetime
    if ref_date is None:
        now = datetime.datetime.now()
    elif isinstance(ref_date, str):
        now = pd.to_datetime(ref_date)
    else:
        now = ref_date

    month = now.month
    day = now.day

    if (month == 10) or (month == 11 and day <= 15):
        return {
            "name": "Diwali & Festive Season Window",
            "description": "Historical period marked by festive liquidity, retail participation, and positive sentiment.",
            "type": "festive",
            "badge": "🪔 Festive Season Active"
        }
    elif (month == 1 and day >= 25) or (month == 2 and day <= 5):
        return {
            "name": "Union Budget Window",
            "description": "High market anticipation and sector re-allocation around the Union Budget presentation.",
            "type": "budget",
            "badge": "🏛️ Budget Window"
        }
    elif day <= 5:
        return {
            "name": "Month-Start SIP Inflow Cycle (Days 1–5)",
            "description": "Domestic institutional and retail SIP flows provide supportive liquidity to Indian equities.",
            "type": "sip",
            "badge": "💰 SIP Inflow Window"
        }
    elif month == 3 and day >= 20:
        return {
            "name": "Fiscal Year-End Rebalancing (March 20–31)",
            "description": "Corporate advance tax adjustments and mutual fund portfolio adjustments occur.",
            "type": "yearend",
            "badge": "📊 Fiscal Year-End"
        }
    elif month == 4 and day <= 10:
        return {
            "name": "New Fiscal Year Inflow (April 1–10)",
            "description": "Fresh fiscal year capital allocations historically provide tailwinds.",
            "type": "newfy",
            "badge": "📈 New FY Rally"
        }
    elif month == 12 and day >= 20:
        return {
            "name": "Year-End Calendar Cycle (Dec 20–31)",
            "description": "Global institutional books close with reduced volume and NAV support.",
            "type": "calendar",
            "badge": "🎄 Year-End Season"
        }

    return {
        "name": "Regular Trading Cycle",
        "description": "Standard market trading session without major scheduled festival/calendar anomalies.",
        "type": "regular",
        "badge": "📅 Regular Session"
    }


