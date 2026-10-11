"""
feature_engineering.py
----------------------
Module for engineering technical indicators and defining horizon-specific target variables.
All features are strictly computed using historical data up to and including the signal date,
guaranteeing zero future-data leakage.
"""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    'Return_1D', 'Return_2D', 'Return_5D', 'Return_10D',
    'Prev_Close_1D', 'Prev_Close_2D',
    'SMA_5', 'SMA_10', 'SMA_20',
    'Close_vs_SMA_5', 'Close_vs_SMA_10', 'Close_vs_SMA_20',
    'Volatility_5D', 'Volatility_10D',
    'Open_Gap', 'Intraday_Range',
    'Volume_Change_1D', 'Volume_Ratio_5D',
    'RSI_14', 'MACD', 'MACD_Signal', 'ATR_14'
]


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Average True Range (ATR) for risk/reward and stop-loss estimation.
    """
    high = df['High']
    low = df['Low']
    prev_close = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical feature columns from OHLCV data.
    Ensures NO future data is used in feature calculation.
    """
    df = df.copy()
    close = df['Close']
    open_p = df['Open']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Returns
    df['Return_1D'] = close.pct_change(1)
    df['Return_2D'] = close.pct_change(2)
    df['Return_5D'] = close.pct_change(5)
    df['Return_10D'] = close.pct_change(10)

    # Previous Closes
    df['Prev_Close_1D'] = close.shift(1)
    df['Prev_Close_2D'] = close.shift(2)

    # Moving Averages
    df['SMA_5'] = close.rolling(window=5).mean()
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()

    # Price relative to Moving Averages
    df['Close_vs_SMA_5'] = (close / df['SMA_5']) - 1.0
    df['Close_vs_SMA_10'] = (close / df['SMA_10']) - 1.0
    df['Close_vs_SMA_20'] = (close / df['SMA_20']) - 1.0

    # Volatility
    df['Volatility_5D'] = df['Return_1D'].rolling(window=5).std()
    df['Volatility_10D'] = df['Return_1D'].rolling(window=10).std()

    # Open Gap & Intraday Range
    df['Open_Gap'] = (open_p - close.shift(1)) / (close.shift(1) + 1e-10)
    df['Intraday_Range'] = (high - low) / (close + 1e-10)

    # Volume Indicators
    df['Volume_Change_1D'] = volume.pct_change(1)
    df['Volume_Ratio_5D'] = volume / (volume.rolling(window=5).mean() + 1e-8)

    # RSI
    df['RSI_14'] = calculate_rsi(close, period=14)

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    df['ATR_14'] = calculate_atr(df, period=14)

    # Clean infinite values defensively from division/zero volume
    df = df.replace([np.inf, -np.inf], 0.0)

    return df


def create_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Creates target variable for requested horizon (1 session or 5 sessions):
    UP (1) = Closing price 'horizon' sessions later > Closing price on signal date.
    DOWN (0) = Closing price 'horizon' sessions later <= Closing price on signal date.
    """
    df = df.copy()
    target_col = f"Target_{horizon}D_UP"
    future_close_col = f"Future_Close_{horizon}D"

    future_close = df['Close'].shift(-horizon)
    df[future_close_col] = future_close
    
    # 1 if future price > current close, else 0
    df[target_col] = np.where(df[future_close_col] > df['Close'], 1, 0)
    
    # NaN for latest `horizon` rows where future is unobserved
    df.loc[df[future_close_col].isna(), target_col] = np.nan

    # Backward compatibility with existing tests
    if horizon == 5:
        df['Target_5D_UP'] = df[target_col]
        df['Future_Close_5D'] = df[future_close_col]

    return df


def prepare_datasets(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepares datasets for modeling for a specified prediction horizon (1 or 5 days).
    """
    df_feat = build_features(df)
    df_full = create_target(df_feat, horizon=horizon)

    target_col = f"Target_{horizon}D_UP"
    available_features = [col for col in FEATURE_COLUMNS if col in df_full.columns]

    valid_mask = df_full[available_features].notna().all(axis=1)
    df_valid = df_full[valid_mask].copy()

    train_df = df_valid[df_valid[target_col].notna()].copy()
    train_df[target_col] = train_df[target_col].astype(int)
    train_df[available_features] = train_df[available_features].replace([np.inf, -np.inf], 0.0)

    latest_df = df_valid[df_valid[target_col].isna()].copy()
    if latest_df.empty and not df_valid.empty:
        latest_df = df_valid.iloc[[-1]].copy()
    latest_df[available_features] = latest_df[available_features].replace([np.inf, -np.inf], 0.0)

    return train_df, latest_df, available_features
