"""
kronos_predictor.py
-------------------
Kronos Foundation Model Integration for NSE Stock Forecasting.

Wraps the KronosPredictor from the local kronos_repo and exposes a clean
interface for the Big Bull Streamlit app:

  - run_kronos_forecast(daily_df, symbol, pred_days) -> dict
    Returns a dict with:
      * forecast_df  : pd.DataFrame (indexed by future trading dates, cols: open/high/low/close)
      * direction    : "UP" or "DOWN"
      * pct_change   : float (expected % move in close price)
      * uncertainty  : float (std-dev of sampled paths)
      * context_bars : int  (bars fed to model)
      * model_name   : str

Fallback: if Kronos cannot load or fails, returns a dict with status="error"
and a human-readable message so the UI degrades gracefully.

NSE Calendar Helpers:
  - generate_nse_future_dates(last_date, n_days) -> pd.DatetimeIndex
    Generates n_days of future NSE trading dates skipping weekends and
    major Indian market holidays (approximate).

Requirements (already in kronos_repo/requirements.txt):
  torch>=2.0.0, einops==0.8.1, huggingface_hub, safetensors, tqdm, pandas, numpy
"""

import sys
import os
import logging
from typing import Dict, Any, Optional
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("kronos_predictor")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Path setup: make kronos_repo importable
# ---------------------------------------------------------------------------
KRONOS_REPO_PATH = os.path.join(os.path.dirname(__file__), "kronos_repo")
if KRONOS_REPO_PATH not in sys.path:
    sys.path.insert(0, KRONOS_REPO_PATH)

# ---------------------------------------------------------------------------
# Lazy globals — loaded once and cached in memory
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_predictor = None
_load_error: Optional[str] = None
_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
_MODEL_ID = "NeoQuasar/Kronos-base"   # NeoQuasar/Kronos-base pretrained foundation model
_FALLBACK_MODEL_ID = "NeoQuasar/Kronos-small"
_MAX_CONTEXT = 400   # safe lookback for daily candles (≤ 512)
_SAMPLE_COUNT = 3    # multiple stochastic paths averaged → uncertainty estimate

# ---------------------------------------------------------------------------
# Approximate NSE holiday list (major national holidays only)
# Enough to skip them in trading day generation without a complex API.
# ---------------------------------------------------------------------------
_NSE_HOLIDAY_APPROX = set([
    # 2025
    "2025-01-26", "2025-03-14", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-06-07", "2025-08-15",
    "2025-10-02", "2025-10-03", "2025-10-24", "2025-12-25",
    # 2026
    "2026-01-26", "2026-03-03", "2026-03-30", "2026-04-02",
    "2026-04-14", "2026-04-30", "2026-05-26", "2026-08-14",
    "2026-10-02", "2026-10-22", "2026-11-14", "2026-12-25",
])


def generate_nse_future_dates(last_date: str, n_days: int = 5) -> pd.DatetimeIndex:
    """
    Generates n_days future NSE trading dates starting after last_date.
    Skips weekends (Sat/Sun) and approximate Indian national market holidays.
    """
    dates = []
    current = pd.Timestamp(last_date) + timedelta(days=1)
    while len(dates) < n_days:
        date_str = current.strftime("%Y-%m-%d")
        if current.weekday() < 5 and date_str not in _NSE_HOLIDAY_APPROX:
            dates.append(current)
        current += timedelta(days=1)
    return pd.DatetimeIndex(dates)


def load_kronos_predictor() -> bool:
    """
    Lazily loads Kronos tokenizer + NeoQuasar/Kronos-base model from Hugging Face Hub.
    Caches them in module-level globals.
    Returns True on success, False on failure.
    """
    global _tokenizer, _model, _predictor, _load_error, _MODEL_ID

    if _predictor is not None:
        return True

    if _load_error is not None:
        return False

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor

        logger.info("Loading Kronos tokenizer from %s ...", _TOKENIZER_ID)
        _tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_ID)

        try:
            logger.info("Loading Kronos model from %s ...", _MODEL_ID)
            _model = Kronos.from_pretrained(_MODEL_ID)
        except Exception as e_base:
            logger.warning("Failed to load %s (%s). Attempting fallback to %s ...", _MODEL_ID, e_base, _FALLBACK_MODEL_ID)
            _MODEL_ID = _FALLBACK_MODEL_ID
            _model = Kronos.from_pretrained(_MODEL_ID)

        _predictor = KronosPredictor(
            model=_model,
            tokenizer=_tokenizer,
            max_context=_MAX_CONTEXT,
            clip=5,
        )
        logger.info("Kronos predictor ready (%s). Device: %s", _MODEL_ID, _predictor.device)
        return True

    except Exception as exc:
        _load_error = str(exc)
        logger.warning("Kronos load failed: %s", exc)
        return False


def _prepare_kronos_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Big Bull daily_df (Date, Open, High, Low, Close, Volume, Ticker)
    into the lowercase OHLCV schema that KronosPredictor.predict() expects.
    Also converts Date column to pandas Timestamps and uses _MAX_CONTEXT rows.
    """
    df = daily_df.copy()

    # Lowercase column names
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Ensure Date is Timestamp
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Take last _MAX_CONTEXT bars
    df = df.tail(_MAX_CONTEXT).reset_index(drop=True)

    return df


def run_kronos_forecast(
    daily_df: pd.DataFrame,
    symbol: str = "RELIANCE.NS",
    pred_days: int = 5,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Main entry point — runs Kronos autoregressive forecast on the given
    historical daily OHLCV DataFrame.

    Args:
        daily_df : DataFrame with columns Date, Open, High, Low, Close, Volume
        symbol   : Ticker symbol (for labelling only)
        pred_days: Number of NSE trading sessions to forecast (1 or 5)
        temperature : Sampling temperature (default 1.0)
        top_p    : Nucleus sampling threshold (default 0.9)

    Returns:
        dict with keys:
          status         : "ok" | "error"
          message        : human-readable info string
          forecast_df    : pd.DataFrame  (columns: open/high/low/close; indexed by future dates)
          direction      : "UP" | "DOWN"
          pct_change     : float  (expected % change in close)
          uncertainty    : float  (std-dev across sample paths, expressed in %)
          context_bars   : int    (rows fed into the model)
          model_name     : str
    """
    # --- 1. Try loading model -----------------------------------------------
    model_ready = load_kronos_predictor()
    if not model_ready:
        return {
            "status": "error",
            "message": (
                f"Kronos model could not be loaded: {_load_error}. "
                "Ensure internet access and that torch, einops, huggingface_hub "
                "are installed."
            ),
            "forecast_df": None,
            "direction": None,
            "pct_change": None,
            "uncertainty": None,
            "context_bars": 0,
            "model_name": _MODEL_ID,
        }

    # --- 2. Data validation -------------------------------------------------
    if daily_df is None or len(daily_df) < 60:
        return {
            "status": "error",
            "message": "Insufficient historical data (need ≥ 60 daily bars).",
            "forecast_df": None,
            "direction": None,
            "pct_change": None,
            "uncertainty": None,
            "context_bars": 0,
            "model_name": _MODEL_ID,
        }

    try:
        # --- 3. Prepare data ------------------------------------------------
        df = _prepare_kronos_df(daily_df)
        last_date = df["date"].iloc[-1]
        last_close = float(df["close"].iloc[-1])
        context_bars = len(df)

        x_timestamp = df["date"]

        # Build future trading dates
        future_dates = generate_nse_future_dates(str(last_date.date()), n_days=pred_days)
        y_timestamp = pd.Series(future_dates)

        # Build input sub-DataFrame for Kronos (lowercase ohlcv)
        input_df = df[["open", "high", "low", "close", "volume"]].copy()

        # --- 4. Run predictions with stochastic sampling --------------------
        all_preds = []
        for _ in range(_SAMPLE_COUNT):
            pred_df_s = _predictor.predict(
                df=input_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_days,
                T=temperature,
                top_k=0,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            )
            all_preds.append(pred_df_s[["open", "high", "low", "close"]].values)

        all_preds = np.stack(all_preds, axis=0)   # (SAMPLE_COUNT, pred_days, 4)
        mean_preds = np.mean(all_preds, axis=0)   # (pred_days, 4)
        std_preds  = np.std(all_preds, axis=0)    # (pred_days, 4)

        # --- 5. Build forecast DataFrame with uncertainty envelope ---------
        forecast_df = pd.DataFrame(
            mean_preds,
            columns=["open", "high", "low", "close"],
            index=future_dates,
        )
        forecast_df["low_bound"] = np.round(forecast_df["close"] - (std_preds[:, 3] * 1.96), 2)
        forecast_df["high_bound"] = np.round(forecast_df["close"] + (std_preds[:, 3] * 1.96), 2)

        # --- 6. Compute direction and uncertainty ---------------------------
        forecast_close = float(forecast_df["close"].iloc[-1])
        pct_change = ((forecast_close - last_close) / last_close) * 100.0
        direction = "UP" if pct_change >= 0 else "DOWN"

        # Uncertainty: average std-dev of close across sample paths (as % of current price)
        close_std_abs = float(std_preds[:, 3].mean())   # column index 3 = close
        uncertainty = (close_std_abs / last_close) * 100.0

        return {
            "status": "ok",
            "message": (
                f"Kronos ({_MODEL_ID}) generated {pred_days}-day forecast "
                f"for {symbol} using {context_bars} historical bars."
            ),
            "forecast_df": forecast_df,
            "direction": direction,
            "pct_change": round(pct_change, 2),
            "uncertainty": round(uncertainty, 2),
            "context_bars": context_bars,
            "model_name": _MODEL_ID,
            "last_close": last_close,
        }

    except Exception as exc:
        logger.exception("Kronos inference error: %s", exc)
        return {
            "status": "error",
            "message": f"Kronos inference error: {exc}",
            "forecast_df": None,
            "direction": None,
            "pct_change": None,
            "uncertainty": None,
            "context_bars": 0,
            "model_name": _MODEL_ID,
        }


def kronos_status() -> Dict[str, str]:
    """Returns a short status dict describing the Kronos readiness state."""
    if _predictor is not None:
        return {
            "status": "ready",
            "model": _MODEL_ID,
            "tokenizer": _TOKENIZER_ID,
            "device": str(_predictor.device),
            "context": str(_MAX_CONTEXT),
        }
    if _load_error:
        return {"status": "error", "error": _load_error}
    return {"status": "not_loaded"}
