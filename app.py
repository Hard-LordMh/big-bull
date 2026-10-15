"""
app.py
------
BIG BULL — NSE Stock Intelligence & AI Predictions
Professional financial dashboard with dark navy theme.

Layout:
  - Left sidebar: navigation + search + watchlist
  - Main workspace: search bar + stock info + chart + prediction panel
  - All 9 feature pages with polished design
"""

import os
import io
import json
import datetime
import pytz
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom Machine Learning & Analysis Modules
from data_validation import (
    validate_stock_dataframe,
    validate_intraday_dataframe,
    normalize_column_names
)
from feature_engineering import prepare_datasets, calculate_rsi, calculate_atr, FEATURE_COLUMNS
from stock_model import (
    train_and_evaluate_model,
    predict_latest_session,
    predict_5day_trajectory,
    walk_forward_prediction_history,
    summarize_prediction_history,
    predict_next_week,
    determine_trading_horizon_tag
)
from pattern_analysis import (
    analyze_day_of_week,
    analyze_month_seasonality,
    analyze_year_performance,
    analyze_intraday_time_of_day,
    find_similar_historical_setups,
    analyze_day_of_month_repeat,
    get_optimal_intraday_windows,
    analyze_special_events_and_festivals,
    detect_current_special_event
)
from timing_strategy import evaluate_timing_signal, backtest_cost_aware_strategy
from intraday_model import (
    calculate_intraday_features,
    train_and_predict_intraday,
    get_real_or_fallback_intraday_data,
    generate_synthetic_intraday_data
)
from utils import (
    export_pipeline_results,
    create_powerbi_zip_bundle,
    fetch_real_or_fallback_stock_data,
    generate_synthetic_stock_data,
    OUTPUTS_DIR,
    MODELS_DIR
)
try:
    from export_powerbi_pipeline import EXPORTS_DIR, run_complete_export_pipeline as _run_pbi_export
except ImportError:
    EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "powerbi_exports")
    _run_pbi_export = None
import importlib
import real_data_service
try:
    importlib.reload(real_data_service)
except Exception:
    pass

# Kronos Foundation Model Integration
try:
    from kronos_predictor import run_kronos_forecast, kronos_status, load_kronos_predictor
    _KRONOS_AVAILABLE = True
except ImportError:
    _KRONOS_AVAILABLE = False
    def run_kronos_forecast(*args, **kwargs):
        return {"status": "error", "message": "kronos_predictor module not found."}
    def kronos_status():
        return {"status": "not_installed"}
    def load_kronos_predictor():
        return False

from real_data_service import (
    fetch_real_daily_data,
    fetch_real_intraday_data,
    fetch_real_quote,
    search_companies,
    fetch_all_companies_board,
    normalize_ticker_symbol,
    get_google_finance_url,
    POPULAR_NSE_COMPANIES
)

if hasattr(real_data_service, "fetch_market_indices"):
    fetch_market_indices = real_data_service.fetch_market_indices
else:
    def fetch_market_indices():
        return [
            {"name": "NIFTY 50", "price": 22520.45, "change": 288.65, "change_pct": 1.30},
            {"name": "SENSEX", "price": 72472.33, "change": 905.93, "change_pct": 1.27},
            {"name": "BANK NIFTY", "price": 55256.65, "change": 741.60, "change_pct": 1.36}
        ]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIG BULL · NSE Stock Intelligence",
    page_icon="assets/bull_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo in sidebar header
st.logo(
    "assets/bull_logo.png",
    size="large",
    link="https://github.com",
    icon_image="assets/bull_logo.png"
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — minimal targeted styles only
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main workspace spacing */
    .main .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Instagram Story / Reel UI Styles */
    .insta-story-ring {
        background: linear-gradient(45deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888);
        padding: 3px;
        border-radius: 50%;
        display: inline-block;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .insta-story-ring:hover {
        transform: scale(1.08);
    }
    .insta-story-avatar {
        background: #1E293B;
        color: #F8FAFC;
        border-radius: 50%;
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        border: 2px solid #0F172A;
    }
    .insta-reel-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px rgba(59, 130, 246, 0.1);
        position: relative;
    }
    .insta-badge-up {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .insta-badge-down {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* ── Minimal prediction UI (light theme) ───────────────── */
    .bb-hero {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.35rem 0 0.6rem;
    }
    .bb-brand { display: flex; align-items: center; gap: 0.65rem; }
    .bb-brand-mark {
        width: 40px; height: 40px; border-radius: 12px;
        background: linear-gradient(135deg, #f97316, #e11d48, #9333ea);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; box-shadow: 0 4px 14px -4px #9333ea;
    }
    .bb-brand-name { font-size: 1.3rem; font-weight: 800; letter-spacing: 0.06em; color: #0F172A; }
    .bb-brand-sub { font-size: 0.7rem; color: #64748B; letter-spacing: 0.04em; }
    .bb-status { font-size: 0.75rem; color: #64748B; }

    .bb-strip { display: flex; gap: 0.4rem; overflow-x: auto; margin-bottom: 0.9rem; }
    .bb-pill {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 20px; padding: 0.28rem 0.75rem;
        font-size: 0.72rem; white-space: nowrap; color: #334155;
    }

    .bb-section-label {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em;
        color: #94A3B8; margin: 0.4rem 0 0.55rem; font-weight: 700;
    }

    /* Story rings */
    .bb-story {
        border-radius: 16px; padding: 0.85rem 0.4rem; text-align: center;
        background: #FFFFFF; border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px -3px rgba(15, 23, 42, 0.10);
        margin-bottom: 0.35rem; transition: transform 0.15s, box-shadow 0.15s;
    }
    .bb-story:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px -10px rgba(15, 23, 42, 0.28);
    }
    .bb-ring-up {
        border-color: #10B981;
        background: linear-gradient(#FFFFFF, #ECFDF5);
        box-shadow: 0 4px 14px -6px rgba(5,150,105,0.55), 0 0 0 3px rgba(16,185,129,0.18);
    }
    .bb-ring-down {
        border-color: #EF4444;
        background: linear-gradient(#FFFFFF, #FEF2F2);
        box-shadow: 0 4px 14px -6px rgba(220,38,38,0.50), 0 0 0 3px rgba(239,68,68,0.16);
    }
    .bb-ring-unknown {
        border-color: #CBD5E1; border-style: dashed; background: #F8FAFC; box-shadow: none;
    }
    .bb-ring-empty { border-style: dashed; border-color: #CBD5E1; box-shadow: none; }
    .bb-story-badge {
        display: inline-block; margin-top: 0.3rem;
        padding: 0.1rem 0.5rem; border-radius: 20px;
        font-size: 0.66rem; font-weight: 800; letter-spacing: 0.02em;
    }

    /* Whole story card is clickable: transparent button overlays the card.
       Streamlit puts the container class .st-key-<key> on the wrapper div. */
    [class*="st-key-storycard_"] { position: relative; }
    [class*="st-key-storycard_"] button {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        width: 100%; height: 100%;
        opacity: 0;
        z-index: 5;
        cursor: pointer;
        border: none;
        background: transparent;
    }
    [class*="st-key-storycard_"] button:focus-visible {
        opacity: 0.25;
        outline: 2px solid #2563EB;
    }
    [class*="st-key-storycard_"] .bb-story { margin-bottom: 0; }

    .bb-story-active-tag {
        position: absolute; top: 6px; right: 8px; z-index: 6;
        font-size: 0.55rem; font-weight: 800; letter-spacing: 0.06em;
        color: #2563EB; background: #DBEAFE;
        padding: 0.05rem 0.35rem; border-radius: 20px;
        pointer-events: none;
    }
    .bb-story-avatar {
        width: 48px; height: 48px; margin: 0 auto 0.4rem; border-radius: 50%;
        background: linear-gradient(135deg, #F1F5F9, #E2E8F0);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.78rem; color: #334155;
        border: 2px solid #FFFFFF;
        box-shadow: 0 2px 6px -2px rgba(15,23,42,0.20);
    }
    .bb-story-plus { color: #94A3B8; font-size: 1.4rem; }
    .bb-story-sym { font-weight: 800; font-size: 0.82rem; color: #0F172A; }
    .bb-story-dir { font-size: 0.68rem; color: #64748B; margin-top: 0.2rem; font-weight: 600; }
    .bb-story-pct { font-size: 0.66rem; color: #94A3B8; margin-top: 0.1rem; }

    /* Prediction hero */
    .bb-predict {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 22px; padding: 1.5rem 1.6rem;
        box-shadow: 0 10px 30px -16px rgba(15, 23, 42, 0.35);
        transition: box-shadow 0.4s ease;
    }
    .bb-predict-top {
        display: flex; justify-content: space-between; align-items: flex-start;
        border-bottom: 1px solid #F1F5F9; padding-bottom: 1rem;
    }
    .bb-predict-sym {
        font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; color: #0F172A; line-height: 1.1;
    }
    .bb-predict-name { font-size: 0.78rem; color: #94A3B8; }
    .bb-predict-price { font-size: 1.2rem; font-weight: 800; text-align: right; color: #0F172A; }
    .bb-up { color: #059669; font-size: 0.8rem; font-weight: 700; }
    .bb-down { color: #DC2626; font-size: 0.8rem; font-weight: 700; }

    .bb-predict-dir { display: flex; align-items: center; gap: 1.1rem; padding: 1.3rem 0 1rem; }
    .bb-predict-dir > div:last-child { margin-left: auto; }
    .bb-predict-arrow { font-size: 3.2rem; font-weight: 800; line-height: 1; }
    .bb-predict-word { font-size: 2.1rem; font-weight: 800; letter-spacing: 0.01em; line-height: 1.05; }
    .bb-predict-horizon { font-size: 0.76rem; color: #94A3B8; margin-top: 0.15rem; }
    .bb-predict-prob { font-size: 2.8rem; font-weight: 800; line-height: 1; text-align: right; }
    .bb-predict-prob-sub {
        font-size: 0.6rem; font-weight: 700; text-align: right;
        margin-top: 0.15rem; opacity: 0.75;
    }

    .bb-meter {
        height: 8px; background: #F1F5F9; border-radius: 5px; overflow: hidden;
    }
    .bb-meter-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }

    .bb-predict-stats {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem; margin-top: 1.2rem;
    }
    .bb-predict-stats > div {
        background: var(--soft); border-radius: 14px;
        padding: 0.7rem 0.4rem; text-align: center;
    }
    .bb-predict-stats span { display: block; font-size: 1.1rem; font-weight: 800; color: #0F172A; }
    .bb-predict-stats small { display: block; font-size: 0.64rem; color: #94A3B8; margin-top: 0.15rem; }

    /* Pinned tiles */
    .bb-tile {
        border-radius: 14px; padding: 0.65rem 0.8rem;
        background: #FFFFFF; border: 1px solid #E2E8F0; margin-bottom: 0.3rem;
        box-shadow: 0 2px 6px -3px rgba(15,23,42,0.10);
    }
    .bb-tile-active { border-color: #93C5FD; background: #EFF6FF; }
    .bb-tile-sym { font-weight: 800; font-size: 0.86rem; color: #0F172A; }
    .bb-tile-price { font-size: 0.78rem; color: #64748B; margin-top: 0.15rem; }

    /* Range bar track */
    .bb-range-track {
        height: 5px;
        background: rgba(100, 116, 139, 0.3);
        border-radius: 3px;
        position: relative;
        margin: 0.4rem 0 0.2rem 0;
    }
    .bb-range-dot {
        width: 11px;
        height: 11px;
        background: #3B82F6;
        border: 2px solid #0F172A;
        border-radius: 50%;
        position: absolute;
        top: -3px;
        box-shadow: 0 0 6px rgba(59, 130, 246, 0.7);
    }

    /* Prediction hero cards */
    .pred-up {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .pred-down {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .pred-neutral {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Big prediction direction display */
    .direction-up {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
        line-height: 1;
    }
    .direction-down {
        font-size: 2.5rem;
        font-weight: 800;
        color: #EF4444;
        line-height: 1;
    }
    .direction-neutral {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F59E0B;
        line-height: 1;
    }

    /* Company monogram avatar */
    .co-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Sidebar watermark */
    .sidebar-watermark {
        opacity: 0.25;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
    }

    /* Quick action buttons */
    .quick-action-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: border-color 0.15s;
    }
    .quick-action-card:hover {
        border-color: #3B82F6;
    }

    /* Market status pill */
    .market-pill-open {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .market-pill-closed {
        background: rgba(100, 116, 139, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 20px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_ist_now():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.datetime.now(ist)


def is_market_open() -> bool:
    """Returns True during NSE trading hours 09:15–15:30 IST Mon–Fri."""
    now = get_ist_now()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_company_initials(name_or_symbol: str) -> str:
    sym = name_or_symbol.replace(".NS", "").replace(".BO", "")
    if len(sym) <= 3:
        return sym
    words = name_or_symbol.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return sym[:2].upper()


def apply_chart_theme(fig, height=400):
    """Transparent chart theme matching dark navy design."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="#CBD5E1"),
        margin=dict(l=10, r=10, t=30, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.6)",
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.6)",
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        hovermode="x unified",
    )
    return fig


def filter_df_by_time_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns:
        return df
    df2 = df.copy()
    df2["_dt"] = pd.to_datetime(df2["Date"])
    df2 = df2.sort_values("_dt")
    latest = df2["_dt"].max()

    cutoffs = {
        "1M":  latest - pd.Timedelta(days=32),
        "3M":  latest - pd.Timedelta(days=93),
        "6M":  latest - pd.Timedelta(days=185),
        "1Y":  latest - pd.Timedelta(days=366),
        "3Y":  latest - pd.Timedelta(days=1098),
        "All": None,
        # Legacy keys
        "1D":  latest - pd.Timedelta(days=2),
        "5D":  latest - pd.Timedelta(days=7),
        "YTD": pd.Timestamp(year=latest.year, month=1, day=1),
        "5Y":  None,
        "Max": None,
    }
    cutoff = cutoffs.get(range_key)
    if cutoff is None:
        result = df2
    else:
        result = df2[df2["_dt"] >= cutoff]
    result = result.drop(columns=["_dt"], errors="ignore")
    return result if len(result) >= 2 else df2.drop(columns=["_dt"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_real_daily_data(symbol: str) -> pd.DataFrame:
    try:
        return fetch_real_daily_data(symbol, period="3y")
    except Exception:
        return generate_synthetic_stock_data(symbol, 750)


@st.cache_data(ttl=300, show_spinner=False)
def load_real_intraday_data(symbol: str) -> pd.DataFrame:
    try:
        return fetch_real_intraday_data(symbol, period="5d", interval="15m")
    except Exception:
        return generate_synthetic_intraday_data(symbol, 5, 15)


@st.cache_data(ttl=1800, show_spinner=False)
def load_long_history_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Deep history used for the multi-year walk-forward prediction replay.
    Two extra years are pulled so the replay window always has a training runway
    before the first predicted signal date.
    """
    best = pd.DataFrame()
    for period in (f"{years + 2}y", "10y", "max"):
        try:
            df = fetch_real_daily_data(symbol, period=period)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        span_years = (pd.to_datetime(df["Date"]).max() - pd.to_datetime(df["Date"]).min()).days / 365.25
        if span_years >= years:
            best = df
            break
        best = df if best.empty else best
    if best.empty:
        return best
    cutoff = pd.to_datetime(best["Date"]).max() - pd.DateOffset(years=years + 2)
    best = best[pd.to_datetime(best["Date"]) >= cutoff].reset_index(drop=True)
    return best


@st.cache_data(ttl=60, show_spinner=False)
def load_real_quote(symbol: str) -> dict:
    return fetch_real_quote(symbol)


@st.cache_data(ttl=120, show_spinner=False)
def load_all_companies_board() -> pd.DataFrame:
    return fetch_all_companies_board()


@st.cache_data(ttl=120, show_spinner=False)
def load_market_indices() -> list:
    return fetch_market_indices()


def render_market_indices_ribbon():
    """Compact live index strip (NIFTY 50 / SENSEX / BANK NIFTY)."""
    indices = load_market_indices()
    chips = []
    for idx in indices:
        change = float(idx.get("change", 0.0))
        color = "#10B981" if change >= 0 else "#EF4444"
        chips.append(f"""
        <div style="display:inline-flex; align-items:center; gap:0.4rem;
                    background:rgba(30,41,59,0.7); border:1px solid #334155;
                    border-radius:20px; padding:0.3rem 0.7rem; margin-right:0.4rem;">
            <span style="font-size:0.72rem; color:#94A3B8;">{idx.get('name', '')}</span>
            <span style="font-size:0.78rem; font-weight:700;">{idx.get('price', 0):,.2f}</span>
            <span style="font-size:0.72rem; color:{color}; font-weight:600;">
                {change:+.2f} ({idx.get('change_pct', 0.0):+.2f}%)
            </span>
        </div>
        """)
    st.markdown(f"<div style='margin:0.2rem 0 0.6rem;'>{''.join(chips)}</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIN PERSISTENCE
# Pinned stocks survive refreshes and are never re-seeded once the user changes
# them. State lives in the URL query string (?pins=TCS.NS,HDFCBANK.NS), which
# persists across reloads and works on both localhost and Streamlit Cloud.
# The sentinel value "none" means "user unpinned everything".
# ─────────────────────────────────────────────────────────────────────────────
PINS_PARAM = "pins"
PINS_NONE = "none"
SEED_PINS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
MAX_PINS = 3


def read_pins_from_url():
    """Returns the pinned list stored in the URL, or None if the user has never pinned."""
    try:
        raw = st.query_params.get(PINS_PARAM)
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    raw = str(raw).strip()
    if not raw or raw == PINS_NONE:
        return []
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def write_pins_to_url(pins):
    """Persists the pinned list into the URL so it survives a refresh."""
    try:
        st.query_params[PINS_PARAM] = ",".join(pins) if pins else PINS_NONE
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_url_pins = read_pins_from_url()

_defaults = {
    "ticker": "RELIANCE.NS",
    "pinned_companies": list(SEED_PINS),
    "favourites": list(SEED_PINS),
    "horizon": 5,
    "cost_bps": 15.0,
    "risk_profile": "Moderate",
    "chart_time_filter": "1Y",
    "chart_type": "Candlestick",
    "model_cache": {},
    "pred_mode": "5 Days",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# URL always wins, so an unpin is never undone by the seed defaults.
if _url_pins is not None:
    st.session_state["pinned_companies"] = _url_pins
    st.session_state["favourites"] = _url_pins or st.session_state["favourites"]

if "daily_df" not in st.session_state:
    st.session_state["daily_df"] = load_real_daily_data(st.session_state["ticker"])
if "intraday_df" not in st.session_state:
    st.session_state["intraday_df"] = load_real_intraday_data(st.session_state["ticker"])


def set_active_stock(symbol: str):
    norm = normalize_ticker_symbol(symbol)
    st.session_state["ticker"] = norm
    with st.spinner(f"Loading {norm}…"):
        st.session_state["daily_df"] = load_real_daily_data(norm)
        st.session_state["intraday_df"] = load_real_intraday_data(norm)
        
        # Check for stale data
        if not st.session_state["daily_df"].empty and 'is_stale' in st.session_state["daily_df"].columns:
            if st.session_state["daily_df"]["is_stale"].iloc[-1]:
                st.warning("⚠️ Market data is stale (last data point is > 2 trading days old). Predictions may not reflect current market conditions.", icon=":material/warning:")
    st.session_state["model_cache"] = {}


def toggle_favourite(symbol: str):
    norm = normalize_ticker_symbol(symbol)
    favs = list(st.session_state["favourites"])
    pinned = list(st.session_state["pinned_companies"])
    if norm in favs:
        favs.remove(norm)
        if norm in pinned:
            pinned.remove(norm)
        st.toast(f"Removed {norm} from favourites", icon=":material/bookmark_remove:")
    else:
        favs.append(norm)
        if norm not in pinned and len(pinned) < 3:
            pinned.append(norm)
        elif norm not in pinned:
            pinned.pop(0)
            pinned.append(norm)
        st.toast(f"Saved {norm} to favourites", icon=":material/bookmark_added:")
    st.session_state["favourites"] = favs
    st.session_state["pinned_companies"] = pinned
    write_pins_to_url(pinned)


def toggle_pin_stock(symbol: str):
    norm = normalize_ticker_symbol(symbol)
    pinned = list(st.session_state["pinned_companies"])
    if norm in pinned:
        pinned.remove(norm)
        st.toast(f"Unpinned {norm}", icon=":material/pin_drop:")
    else:
        if len(pinned) < MAX_PINS:
            pinned.append(norm)
            st.toast(f"Pinned {norm}", icon=":material/push_pin:")
        else:
            old = pinned.pop(0)
            pinned.append(norm)
            st.toast(f"Replaced {old} → {norm}", icon=":material/push_pin:")
    st.session_state["pinned_companies"] = pinned
    write_pins_to_url(pinned)


# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def get_daily_model_pipeline(ticker_symbol: str, df: pd.DataFrame, horizon: int = 5):
    cache_key = f"{ticker_symbol}_{horizon}_{len(df)}"
    if cache_key in st.session_state["model_cache"]:
        return st.session_state["model_cache"][cache_key]

    is_valid, errs, warns, clean_df = validate_stock_dataframe(df, min_rows=40)
    if not is_valid or clean_df is None:
        return {"valid": False, "error": errs[0] if errs else "Validation error"}

    train_df, latest_df, feature_cols = prepare_datasets(clean_df, horizon=horizon)
    results = train_and_evaluate_model(train_df, feature_cols, horizon=horizon)
    latest_pred = predict_latest_session(
        results["model"], latest_df, feature_cols,
        ticker=ticker_symbol, horizon=horizon, full_df=clean_df
    )
    trajectory_5d = predict_5day_trajectory(clean_df, feature_cols, results["model"])

    last_row = clean_df.iloc[-1]
    atr_val = float(clean_df["ATR_14"].iloc[-1]) if "ATR_14" in clean_df.columns else float(last_row["Close"] * 0.015)
    rsi_val = float(clean_df["RSI_14"].iloc[-1]) if "RSI_14" in clean_df.columns else 50.0
    sma20 = float(clean_df["Close"].rolling(20).mean().iloc[-1]) if len(clean_df) >= 20 else float(last_row["Close"])
    close_vs_sma20 = (float(last_row["Close"]) / sma20) - 1.0

    timing_sig = evaluate_timing_signal(
        latest_close=float(last_row["Close"]),
        prob_up=latest_pred["probability_up"],
        prob_down=latest_pred["probability_down"],
        atr_val=atr_val,
        rsi_val=rsi_val,
        close_vs_sma20=close_vs_sma20,
        signal_date=latest_pred["signal_date"],
        cost_bps=st.session_state["cost_bps"]
    )

    dow_df = analyze_day_of_week(clean_df)
    try:
        export_pipeline_results(
            raw_df=clean_df,
            backtest_predictions=results["backtest_predictions"],
            test_metrics=results["test_metrics"],
            walk_forward_df=results["walk_forward_df"],
            feat_importance_df=results["feature_importance"],
            latest_pred_dict=latest_pred,
            model_object=results["model"],
            ticker=ticker_symbol,
            timing_signal_dict=timing_sig,
            pattern_summary_df=dow_df
        )
    except OSError:
        # Read-only container filesystems (cloud deploys) must not break the dashboard.
        pass

    bundle = {
        "valid": True,
        "clean_df": clean_df,
        "results": results,
        "latest_pred": latest_pred,
        "timing_sig": timing_sig,
        "trajectory_5d": trajectory_5d,
        "feature_cols": feature_cols,
    }
    st.session_state["model_cache"][cache_key] = bundle
    return bundle


@st.cache_data(ttl=1800, show_spinner=False)
def get_history_prediction_bundle(ticker_symbol: str, horizon: int = 5, years: int = 5) -> dict:
    """
    Runs the expanding walk-forward replay over the last `years` of history and the
    forward-looking next-week UP/DOWN forecast for the given stock.
    """
    hist_cache_key = f"hist_{ticker_symbol}_{horizon}_{years}"
    if hist_cache_key in st.session_state["model_cache"]:
        return st.session_state["model_cache"][hist_cache_key]

    long_df = load_long_history_data(ticker_symbol, years=years)
    if long_df is None or long_df.empty:
        out = {"valid": False, "error": "Historical data unavailable"}
        st.session_state["model_cache"][hist_cache_key] = out
        return out

    is_valid, errs, warns, clean_df = validate_stock_dataframe(long_df, min_rows=120)
    if not is_valid or clean_df is None or clean_df.empty:
        out = {"valid": False, "error": errs[0] if errs else "Validation error"}
        st.session_state["model_cache"][hist_cache_key] = out
        return out

    available_features = [c for c in FEATURE_COLUMNS if c in clean_df.columns] or None

    history = walk_forward_prediction_history(
        clean_df, feature_cols=available_features, horizon=horizon, years=years
    )
    summary = summarize_prediction_history(history)
    next_week = predict_next_week(clean_df, feature_cols=available_features, sessions=5)

    out = {
        "valid": bool(summary.get("valid", False)) or not next_week.empty,
        "history": history,
        "summary": summary,
        "next_week": next_week,
        "feature_cols": available_features,
        "history_span": f"{clean_df['Date'].min()} → {clean_df['Date'].max()}",
        "history_rows": len(clean_df),
    }
    st.session_state["model_cache"][hist_cache_key] = out
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def get_next_week_bias(symbol: str) -> dict:
    """Single source of truth for direction + probability.

    Uses the SAME dataset as the main dashboard pipeline (load_real_daily_data, 3y)
    so the story cards and the prediction card can never disagree.
    """
    try:
        df = load_real_daily_data(symbol)
        if df is None or df.empty or len(df) < 120:
            return {"ok": False}
        is_valid, _, _, clean_df = validate_stock_dataframe(df, min_rows=120)
        if not is_valid:
            return {"ok": False}

        # predict_next_week() requires the canonical feature list; it builds the
        # indicator columns internally from the raw OHLCV frame.
        wk = predict_next_week(clean_df, list(FEATURE_COLUMNS), sessions=5)
        if wk is None or wk.empty:
            return {"ok": False}

        prob_up = float(wk["Prob_UP_Pct"].mean())
        direction = "UP" if prob_up >= 50 else "DOWN"
        return {
            "ok": True,
            "direction": direction,
            "prob_up": round(prob_up, 1),
            "prob_down": round(100.0 - prob_up, 1),
            # Probability of the direction actually being shown, so the badge
            # number and the arrow always agree.
            "prob_direction": round(prob_up if direction == "UP" else 100.0 - prob_up, 1),
            "reference_close": wk.attrs.get("reference_close"),
            "reference_date": wk.attrs.get("reference_date", ""),
            "days_up": int((wk["Predicted_Direction"] == "UP").sum()),
            "sessions": int(len(wk)),
        }
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_candlestick_chart(df: pd.DataFrame, height=450, show_volume=True, forecast_df=None) -> go.Figure:
    """Builds professional candlestick chart with optional volume panel and forecast overlay."""
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(None,)
    )

    dates = pd.to_datetime(df["Date"])
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing=dict(line=dict(color="#10B981", width=1), fillcolor="#10B981"),
        decreasing=dict(line=dict(color="#EF4444", width=1), fillcolor="#EF4444"),
        showlegend=True,
        hovertext=[
            f"O: ₹{o:,.2f}  H: ₹{h:,.2f}  L: ₹{l:,.2f}  C: ₹{c:,.2f}"
            for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty:
        fc_dates = forecast_df.index if isinstance(forecast_df.index, pd.DatetimeIndex) else pd.to_datetime(forecast_df.index)
        # Uncertainty band
        if "high" in forecast_df.columns and "low" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=list(fc_dates) + list(fc_dates[::-1]),
                y=list(forecast_df["high"]) + list(forecast_df["low"][::-1]),
                fill="toself",
                fillcolor="rgba(139, 92, 246, 0.12)",
                line=dict(width=0),
                name="Forecast uncertainty",
                showlegend=True,
                hoverinfo="skip",
            ), row=1, col=1)
        # Dashed forecast line
        fc_col = "close" if "close" in forecast_df.columns else forecast_df.columns[0]
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=forecast_df[fc_col],
            mode="lines+markers",
            name="Kronos forecast",
            line=dict(color="#8B5CF6", width=2, dash="dash"),
            marker=dict(size=6, color="#8B5CF6"),
            hovertemplate="<b>Forecast:</b> ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        # Vertical separator
        sep_date = fc_dates[0]
        fig.add_vline(
            x=sep_date,
            line_width=1.5,
            line_dash="dot",
            line_color="rgba(139, 92, 246, 0.6)",
            row=1, col=1
        )
        fig.add_annotation(
            x=sep_date, y=1.02, yref="paper",
            text="Forecast →", showarrow=False,
            font=dict(color="#8B5CF6", size=11),
            xanchor="left"
        )

    # Volume bars
    if show_volume and "Volume" in df.columns:
        vol_colors = [
            "#10B981" if c >= o else "#EF4444"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=dates,
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.7,
            showlegend=True,
            hovertemplate="<b>Volume:</b> %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # Style
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="#CBD5E1"),
        margin=dict(l=10, r=10, t=10, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11)
        ),
        xaxis_rangeslider_visible=False,
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout({
            ax: dict(
                showgrid=True,
                gridcolor="rgba(51, 65, 85, 0.5)",
                zeroline=False,
                showline=False,
            )
        })
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1, title_font=dict(size=11))
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1, title_font=dict(size=11))
    return fig


def build_line_chart(df: pd.DataFrame, height=380) -> go.Figure:
    """Simple area line chart for quick overview."""
    dates = pd.to_datetime(df["Date"])
    period_pos = df["Close"].iloc[-1] >= df["Close"].iloc[0] if len(df) >= 2 else True
    line_col = "#10B981" if period_pos else "#EF4444"
    fill_col = "rgba(16,185,129,0.08)" if period_pos else "rgba(239,68,68,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=df["Close"],
        mode="lines",
        name="Close price",
        line=dict(color=line_col, width=2.5),
        fill="tozeroy",
        fillcolor=fill_col,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>₹%{y:,.2f}<extra></extra>",
    ))
    return apply_chart_theme(fig, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE COMPONENT RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_market_header():
    """Renders the top header with BIG BULL brand, market status, festival/calendar cycle, time."""
    ist_now = get_ist_now()
    market_open = is_market_open()
    status_html = (
        '<span class="market-pill-open">● NSE Open</span>'
        if market_open else
        '<span class="market-pill-closed">● NSE Closed</span>'
    )
    time_str = ist_now.strftime("%H:%M IST · %d %b %Y")
    cur_event = detect_current_special_event(ist_now)
    event_badge = cur_event.get("badge", "📅 Regular Session")
    event_desc = cur_event.get("description", "")
    event_html = f'<span style="background:rgba(245,158,11,0.12); color:#F59E0B; border:1px solid rgba(245,158,11,0.35); border-radius:20px; padding:0.15rem 0.65rem; font-size:0.78rem; font-weight:600;" title="{event_desc}">{event_badge}</span>'

    col_brand, col_status = st.columns([3, 2])
    with col_brand:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.75rem; padding:0.5rem 0;">
            <div>
                <div style="font-size:1.5rem; font-weight:800; letter-spacing:-0.5px; line-height:1;">
                    BIG BULL
                </div>
                <div style="font-size:0.78rem; color:#64748B; letter-spacing:0.02em;">
                    Understand the Market. Predict with Data.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_status:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:flex-end; gap:0.75rem; padding:0.5rem 0; flex-wrap:wrap;">
            {event_html}
            {status_html}
            <span style="font-size:0.8rem; color:#64748B;">{time_str}</span>
        </div>
        """, unsafe_allow_html=True)


def render_stock_info_card(quote: dict):
    """Displays the selected stock's key information."""
    sym = quote["clean_symbol"]
    name = quote["name"] or sym
    price = quote["price"]
    chg = quote["day_change"]
    pct = quote["day_change_pct"]
    is_up = chg >= 0
    sign = "+" if is_up else ""
    chg_color = "#10B981" if is_up else "#EF4444"
    arrow = "▲" if is_up else "▼"
    is_fav = st.session_state["ticker"] in st.session_state["favourites"]
    avatar = get_company_initials(name)
    data_note = "Latest available close"
    
    # Check for fallback or stale data
    is_fallback = quote.get("is_fallback", False)
    data_source = quote.get("source", "Unknown")

    with st.container(border=True):
        # Show warning if using fallback data
        if is_fallback:
            st.warning("⚠️ Using synthetic fallback data - Real market data unavailable. This is NOT real market data.", icon=":material/warning:")
        
        # Show data source info
        st.caption(f"Data source: {data_source}", help="yfinance provides delayed data, not real-time exchange feeds")
        h1, h2, h3 = st.columns([3, 3, 2])

        with h1:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div class="co-avatar">{avatar}</div>
                <div>
                    <div style="display:flex; align-items:center; gap:0.4rem;">
                        <span style="font-size:1.3rem; font-weight:700;">{name}</span>
                        :blue-badge[NSE]
                    </div>
                    <div style="font-size:0.8rem; color:#64748B;">{sym} · {quote.get('sector','Market')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            st.markdown(f"""
            <div style="margin-top:0.2rem;">
                <div style="font-size:1.9rem; font-weight:800; letter-spacing:-0.5px;">₹{price:,.2f}</div>
                <div style="font-size:0.95rem; color:{chg_color}; font-weight:600;">
                    {arrow} {sign}{chg:,.2f} ({sign}{pct:.2f}%) Today
                </div>
                <div style="font-size:0.72rem; color:#64748B; margin-top:0.15rem;">{data_note}</div>
            </div>
            """, unsafe_allow_html=True)

        with h3:
            st.write("")
            btn_row = st.container(horizontal=True)
            with btn_row:
                pin_label = "Unpin" if st.session_state["ticker"] in st.session_state["pinned_companies"] else "Pin"
                if st.button(pin_label, icon=":material/push_pin:", key="stock_pin_btn"):
                    toggle_pin_stock(st.session_state["ticker"])
                    st.rerun()
                fav_icon = ":material/bookmark:" if is_fav else ":material/bookmark_border:"
                if st.button("" , icon=fav_icon, key="stock_fav_btn", help="Save to favourites"):
                    toggle_favourite(st.session_state["ticker"])
                    st.rerun()
                if st.button("", icon=":material/sync:", key="stock_sync_btn", help="Sync market data"):
                    st.cache_data.clear()
                    st.session_state["daily_df"] = load_real_daily_data(st.session_state["ticker"])
                    st.session_state["intraday_df"] = load_real_intraday_data(st.session_state["ticker"])
                    st.session_state["model_cache"] = {}
                    st.rerun()
                st.link_button("", url=quote["google_finance_url"], icon=":material/open_in_new:", help="Open in Google Finance")

        # OHLCV stats row
        st.divider() if False else None  # no heavy divider
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        with s1:
            st.metric("Open", f"₹{quote.get('day_low', price):,.2f}", border=True)
        with s2:
            st.metric("High", f"₹{quote['day_high']:,.2f}", border=True)
        with s3:
            st.metric("Low", f"₹{quote['day_low']:,.2f}", border=True)
        with s4:
            st.metric("Prev. close", f"₹{quote['prev_close']:,.2f}", border=True)
        with s5:
            vol_str = f"{quote['volume']:,.0f}" if quote["volume"] else "—"
            st.metric("Volume", vol_str, border=True)
        with s6:
            st.metric("Exchange", "NSE India", border=True)


def render_range_bars(quote: dict):
    """Compact day range + 52-week range bars."""
    lp = quote["price"]
    dl = quote["day_low"]
    dh = quote["day_high"]
    yl = quote["fifty_two_low"] or dl
    yh = quote["fifty_two_high"] or dh

    def pct_pos(val, lo, hi):
        if hi <= lo:
            return 50.0
        return min(max(((val - lo) / (hi - lo)) * 100.0, 2.0), 98.0)

    dp = pct_pos(lp, dl, dh)
    yp = pct_pos(lp, yl, yh)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#94A3B8; margin-bottom:0.25rem;">Day range</div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
            <span>₹{dl:,.2f}</span><span>₹{dh:,.2f}</span>
        </div>
        <div class="bb-range-track">
            <div class="bb-range-dot" style="left:calc({dp}% - 5px);"></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#94A3B8; margin-bottom:0.25rem;">52-week range</div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
            <span>₹{yl:,.2f}</span><span>₹{yh:,.2f}</span>
        </div>
        <div class="bb-range-track">
            <div class="bb-range-dot" style="left:calc({yp}% - 5px);"></div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_control_bar():
    """Horizontal prediction control bar: mode selector."""
    with st.container(border=True):
        c_mode, c_range = st.columns([3, 2])
        with c_mode:
            mode = st.segmented_control(
                "Prediction mode",
                ["Next Day", "5 Days", "Intraday", "Pattern Analysis"],
                default=st.session_state["pred_mode"],
                key="pred_mode_ctrl",
                label_visibility="collapsed"
            )
            if mode:
                st.session_state["pred_mode"] = mode
                if mode == "Next Day":
                    st.session_state["horizon"] = 1
                elif mode == "5 Days":
                    st.session_state["horizon"] = 5

        with c_range:
            chart_range = st.segmented_control(
                "Date range",
                ["1M", "3M", "6M", "1Y", "3Y", "All"],
                default=st.session_state.get("chart_time_filter", "1Y"),
                key="chart_range_ctrl",
                label_visibility="collapsed"
            )
            if chart_range:
                st.session_state["chart_time_filter"] = chart_range
    return st.session_state.get("pred_mode", "5 Days")


def render_prediction_panel(latest_pred: dict, results: dict, quote: dict):
    """Right-side prediction summary card."""
    p_dir = latest_pred["predicted_direction"]
    p_up = latest_pred["probability_up"] * 100.0
    p_down = latest_pred["probability_down"] * 100.0
    metrics = results["test_metrics"]
    acc = metrics["accuracy"] * 100.0
    baseline = metrics["baseline_accuracy"] * 100.0
    horizon = st.session_state["horizon"]

    is_up = p_dir == "UP"
    is_down = p_dir == "DOWN"
    card_cls = "pred-up" if is_up else ("pred-down" if is_down else "pred-neutral")
    dir_cls = "direction-up" if is_up else ("direction-down" if is_down else "direction-neutral")
    dir_sym = "↑" if is_up else ("↓" if is_down else "→")
    dir_label = p_dir
    prob = p_up if is_up else p_down

    with st.container(border=True):
        st.subheader(f"AI prediction · {horizon}D", icon=":material/auto_awesome:")

        st.markdown(f"""
        <div class="{card_cls}">
            <div class="{dir_cls}">{dir_sym} {dir_label}</div>
            <div style="font-size:0.85rem; margin-top:0.4rem; opacity:0.85;">
                Probability: <b>{prob:.1f}%</b>
            </div>
            <div style="font-size:0.78rem; margin-top:0.3rem; color:#94A3B8;">
                Horizon: {horizon} trading session(s)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown(f"""
        <div style="margin:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:0.2rem;">
                <span style="color:#10B981;">▲ UP {p_up:.1f}%</span>
                <span style="color:#EF4444;">▼ DOWN {p_down:.1f}%</span>
            </div>
            <div style="height:6px; background:#334155; border-radius:3px; overflow:hidden;">
                <div style="height:100%; width:{p_up:.1f}%; background:linear-gradient(90deg,#10B981,#34D399); border-radius:3px; transition:width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key stats
        st.caption(f"Reference close: ₹{latest_pred['latest_close']:,.2f}")
        st.caption(f"Signal date: {latest_pred['signal_date']}")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Model accuracy", f"{acc:.1f}%", f"+{acc-baseline:.1f}% vs baseline", border=True)
        with m2:
            st.metric("Baseline", f"{baseline:.1f}%", border=True)

        rf_det = latest_pred.get("rf_detection", {})
        tc = rf_det.get("tree_consensus", {})
        mr = rf_det.get("market_regime", {})
        st.caption(f"Tree consensus: {tc.get('consensus_pct', 50)}% · {tc.get('strength_label', '')}")
        st.caption(f"Regime: {mr.get('badge', 'Active')}")
        st.caption("Model: Random Forest (100 trees) · Walk-forward validated")


def render_key_info_card(latest_pred: dict, results: dict, df: pd.DataFrame):
    """Key information summary card."""
    metrics = results["test_metrics"]
    with st.container(border=True):
        st.subheader("Key information", icon=":material/info:")
        rows = [
            ("Latest close", f"₹{latest_pred['latest_close']:,.2f}"),
            ("Predicted direction", latest_pred["predicted_direction"]),
            ("Model accuracy", f"{metrics['accuracy']*100:.1f}%"),
            ("Baseline accuracy", f"{metrics['baseline_accuracy']*100:.1f}%"),
            ("F1-Score", f"{metrics['f1_score']*100:.1f}%"),
            ("Dataset rows", f"{len(df):,}"),
            ("Date range", f"{df['Date'].min()} → {df['Date'].max()}" if "Date" in df.columns else "—"),
            ("Data source", "NSE · Google Finance"),
            ("Frequency", "Daily (EOD)"),
        ]
        for label, val in rows:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.caption(label)
            with c2:
                st.markdown(f"**{val}**")


def render_intraday_compact_card(intraday_df: pd.DataFrame):
    """Compact intraday card for dashboard."""
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        with h1:
            st.subheader("Intraday (15m)", icon=":material/candlestick_chart:")
        with h2:
            st.write("")
            if st.button("Deep dive →", key="intra_more_btn", use_container_width=True):
                st.session_state["expand_deep_dive"] = True
                st.rerun()

        intra_res = train_and_predict_intraday(intraday_df)
        if intra_res.get("valid", False):
            df_feat = intra_res["df_with_features"]
            fig_i = go.Figure()
            fig_i.add_trace(go.Candlestick(
                x=pd.to_datetime(df_feat["Timestamp"]),
                open=df_feat["Open"], high=df_feat["High"],
                low=df_feat["Low"], close=df_feat["Close"],
                name="15m candle",
                increasing=dict(line=dict(color="#10B981"), fillcolor="#10B981"),
                decreasing=dict(line=dict(color="#EF4444"), fillcolor="#EF4444"),
                showlegend=False,
            ))
            if "VWAP" in df_feat.columns:
                fig_i.add_trace(go.Scatter(
                    x=pd.to_datetime(df_feat["Timestamp"]),
                    y=df_feat["VWAP"], mode="lines",
                    name="VWAP", line=dict(color="#F59E0B", width=1.5),
                ))
            fig_i.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=200, margin=dict(l=5, r=5, t=5, b=5),
                xaxis_rangeslider_visible=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.5)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.5)"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.05, font=dict(size=10)),
            )
            st.plotly_chart(fig_i, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Latest close", f"₹{intra_res['latest_close']:,.2f}", border=True)
            with m2:
                st.metric("Next candle", intra_res["predicted_direction"],
                          f"{intra_res['probability_up']*100:.0f}% UP", border=True)
            with m3:
                st.metric("Test accuracy", f"{intra_res['test_accuracy']:.1f}%", border=True)
        else:
            st.caption("Intraday data unavailable for this stock. NSE intraday feeds have limited lookback.")


def render_pattern_compact_card(daily_df: pd.DataFrame):
    """Compact historical pattern card with weekday/month/year tabs."""
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        with h1:
            st.subheader("Historical patterns", icon=":material/calendar_month:")
        with h2:
            st.write("")
            if st.button("Deep dive →", key="pattern_more_btn", use_container_width=True):
                st.session_state["expand_deep_dive"] = True
                st.rerun()

        tab_wd, tab_mo, tab_yr, tab_fest = st.tabs(["Weekday", "Month", "Year", "Festivals"])

        with tab_wd:
            try:
                dow_df = analyze_day_of_week(daily_df)
                if not dow_df.empty and "Mean_Return_Pct" in dow_df.columns:
                    for _, row in dow_df.iterrows():
                        ret = row["Mean_Return_Pct"]
                        obs = int(row.get("Sessions", row.get("Count", 0)))
                        day_name = row.get("Day", "")
                        col_ret = "#10B981" if ret >= 0 else "#EF4444"
                        sign = "+" if ret >= 0 else ""
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center;
                                    padding:0.3rem 0; border-bottom:1px solid rgba(51,65,85,0.4);">
                            <span style="font-size:0.85rem;">{day_name}</span>
                            <span style="font-size:0.85rem; font-weight:600; color:{col_ret};">
                                {sign}{ret:.2f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    n_obs = int(dow_df["Sessions"].sum()) if "Sessions" in dow_df.columns else len(daily_df)
                    st.caption(f"{n_obs} trading sessions · {daily_df['Date'].min()} → {daily_df['Date'].max()}")
            except Exception as e:
                st.caption(f"Pattern data unavailable: {e}")

        with tab_mo:
            try:
                month_df = analyze_month_seasonality(daily_df)
                if not month_df.empty and "Mean_Return_Pct" in month_df.columns:
                    fig_m = px.bar(
                        month_df, x="Month", y="Mean_Return_Pct",
                        color="Mean_Return_Pct",
                        color_continuous_scale=[[0, "#EF4444"], [0.5, "#94A3B8"], [1, "#10B981"]],
                    )
                    fig_m.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=170, margin=dict(l=5, r=5, t=5, b=5),
                        coloraxis_showscale=False,
                        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                        yaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.4)"),
                    )
                    st.plotly_chart(fig_m, use_container_width=True)
            except Exception:
                st.caption("Monthly patterns unavailable.")

        with tab_yr:
            try:
                yr_df = analyze_year_performance(daily_df)
                if not yr_df.empty:
                    display_cols = [c for c in ["Year", "Total_Return_Pct", "Daily_Win_Rate_Pct", "Trading_Days"] if c in yr_df.columns]
                    st.dataframe(yr_df[display_cols].tail(8), hide_index=True, use_container_width=True)
            except Exception:
                st.caption("Annual data unavailable.")

        with tab_fest:
            try:
                fest_df = analyze_special_events_and_festivals(daily_df)
                if not fest_df.empty:
                    for _, row in fest_df.head(4).iterrows():
                        ret = row["Mean_Return_Pct"]
                        ev_name = row["Event"]
                        col_ret = "#10B981" if ret >= 0 else "#EF4444"
                        sign = "+" if ret >= 0 else ""
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center;
                                    padding:0.3rem 0; border-bottom:1px solid rgba(51,65,85,0.4);">
                            <span style="font-size:0.8rem;">{ev_name}</span>
                            <span style="font-size:0.8rem; font-weight:600; color:{col_ret};">
                                {sign}{ret:.2f}% ({row['Win_Rate_Pct']:.0f}%)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.caption("Festivals, Union Budget & monthly F&O expiry")
            except Exception:
                st.caption("Festival patterns unavailable.")


def render_history_and_nextweek_card(ticker_symbol: str, years: int = 5):
    """Multi-year walk-forward prediction replay + explicit next-week UP/DOWN call."""
    with st.spinner(f"Replaying {years}-year prediction history for {ticker_symbol}…"):
        pack = get_history_prediction_bundle(ticker_symbol, st.session_state["horizon"], years)

    if not pack.get("valid", False):
        with st.container(border=True):
            st.subheader(f"{years}-year model prediction history", icon=":material/history:")
            st.caption(f"Prediction history unavailable: {pack.get('error', 'insufficient data')}")
        return

    history = pack["history"]
    summary = pack["summary"]
    next_week = pack["next_week"]
    
    # Determine trading horizon tag based on 5-year performance
    horizon_tag = determine_trading_horizon_tag(history, summary)

    with st.container(border=True):
        st.subheader(f"{years}-year model prediction history", icon=":material/history:")
        st.caption(
            f"Expanding walk-forward replay · {summary.get('signals', 0):,} model calls · "
            f"{summary.get('start_date', '—')} → {summary.get('end_date', '—')} · zero lookahead"
        )

        if not history.empty:
            hit_rate = summary.get("hit_rate_pct", 0.0)
            baseline = summary.get("baseline_pct", 0.0)
            edge = summary.get("edge_pct", 0.0)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Historical hit rate", f"{hit_rate:.1f}%",
                          f"{edge:+.1f}% vs baseline", border=True,
                          delta_color="normal")
            with m2:
                st.metric("Majority baseline", f"{baseline:.1f}%", border=True)
            with m3:
                st.metric("Model calls replayed", f"{summary.get('signals', 0):,}", border=True)
            with m4:
                st.metric("Avg actual move", f"{summary.get('avg_return_pct', 0.0):+.2f}%", border=True)
            
            # Display trading horizon tag near accuracy metrics
            st.markdown(
                f"""
                <div style="margin-top: 1rem; padding: 0.75rem; background-color: {horizon_tag['color']}15; 
                     border-left: 4px solid {horizon_tag['color']}; border-radius: 4px;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-weight: 700; color: {horizon_tag['color']}; font-size: 1.1rem;">
                            {horizon_tag['tag']}
                        </span>
                        <span style="font-size: 0.8rem; color: #64748B; margin-left: auto;">
                            Confidence: {horizon_tag['confidence']}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; margin-top: 0.25rem;">
                        {horizon_tag['rationale']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            tab_log, tab_year, tab_chart = st.tabs(
                ["Signal log", "Year-by-year", "Accuracy trend"]
            )

            with tab_log:
                log_cols = ["Signal_Date", "Reference_Close", "Predicted_Direction",
                            "Prob_UP_Pct", "Actual_Direction", "Actual_Return_Pct", "Result"]
                log_df = history[log_cols].copy()
                log_df.columns = ["Signal date", "Reference close (₹)", "Predicted",
                                  "P(UP) %", "Actual", "Actual return (%)", "Result"]
                show_n = int(st.slider("Signals shown", 20, min(400, len(log_df)), 60, step=20,
                                       key=f"log_rows_{ticker_symbol}"))
                st.dataframe(log_df.tail(show_n).iloc[::-1], hide_index=True, use_container_width=True)
                st.download_button(
                    "Download full prediction log (CSV)",
                    history.to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker_symbol}_prediction_history_{years}y.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key=f"dl_predhist_{ticker_symbol}",
                )

            with tab_year:
                yearly = summary["yearly"].copy()
                yearly["Edge vs Baseline %"] = (yearly["Hit_Rate_Pct"] - yearly["Baseline_Pct"]).round(1)
                yearly = yearly[["Year", "Signals", "Hits", "Hit_Rate_Pct", "Baseline_Pct",
                                 "Edge vs Baseline %", "Avg_Return_When_Right_Pct",
                                 "Avg_Return_When_Wrong_Pct"]]
                yearly.columns = ["Year", "Signals", "Hits", "Hit rate (%)", "Baseline (%)",
                                  "Edge (%)", "Avg return when right (%)", "Avg return when wrong (%)"]
                st.dataframe(yearly, hide_index=True, use_container_width=True)

            with tab_chart:
                fig_h = px.bar(
                    history, x="Signal_Date", y="Actual_Return_Pct",
                    color="Result",
                    color_discrete_map={"HIT": "#10B981", "MISS": "#EF4444"},
                    hover_data={"Predicted_Direction": True, "Prob_UP_Pct": True,
                                "Actual_Return_Pct": ":.2f"},
                )
                fig_h.add_hline(y=0, line_color="#64748B", line_width=1)
                fig_h.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=5, r=5, t=10, b=5),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.4)", title="Actual move %"),
                    legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                )
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption("Green bars: model called the direction correctly. Red bars: model was wrong.")

        st.divider()

        # ── Next week UP / DOWN ────────────────────────────────────────────
        st.markdown("**Next week prediction (UP or DOWN)**")
        if next_week is None or next_week.empty:
            st.caption("Next-week forecast unavailable for this stock.")
            return

        week_dir = next_week.attrs.get("week_direction", "UP")
        ref_close = next_week.attrs.get("reference_close", 0.0)
        wk_days_up = int((next_week["Predicted_Direction"] == "UP").sum())

        up = week_dir == "UP"
        card_cls = "pred-up" if up else "pred-down"
        dir_sym = "↑" if up else "↓"
        accent = "#10B981" if up else "#EF4444"

        st.markdown(f"""
        <div class="{card_cls}">
            <div class="{'direction-up' if up else 'direction-down'}">{dir_sym} {week_dir}</div>
            <div style="font-size:0.85rem; margin-top:0.4rem; opacity:0.85;">
                Next 5 sessions average probability: <b>{next_week.attrs.get('week_prob_up', 50.0):.1f}% UP</b>
            </div>
            <div style="font-size:0.78rem; margin-top:0.3rem; color:#94A3B8;">
                {wk_days_up}/5 sessions predicted UP · reference close ₹{ref_close:,.2f} ({next_week.attrs.get('reference_date', '')})
            </div>
        </div>
        """, unsafe_allow_html=True)

        wk_display = next_week[[
            "Trading_Day", "Trading_Date", "Weekday", "Predicted_Direction",
            "Prob_UP_Pct", "Prob_DOWN_Pct", "Reference_Close_INR", "Note"
        ]].copy()
        wk_display.columns = ["Session", "Date", "Day", "Prediction", "P(UP) %",
                              "P(DOWN) %", "Reference close (₹)", "Note"]
        st.dataframe(
            wk_display,
            hide_index=True,
            use_container_width=True,
        )

        up_days = next_week[next_week["Predicted_Direction"] == "UP"]
        dn_days = next_week[next_week["Predicted_Direction"] == "DOWN"]
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                f"<div style='color:{accent};font-weight:700;'>▲ {len(up_days)} of 5 sessions predicted UP</div>",
                unsafe_allow_html=True)
        with s2:
            st.markdown(
                f"<div style='color:#EF4444;font-weight:700;text-align:right;'>▼ {len(dn_days)} of 5 sessions predicted DOWN</div>",
                unsafe_allow_html=True)

        st.caption(
            "Probabilistic model output for research only. Each session is projected bar-by-bar "
            "with ATR bounds and re-scored by the Random Forest ensemble — not financial advice."
        )


def render_future_prediction_graph(sym: str, lookback_days: int = 180):
    """History chart + next-5-session direction probability bars for one stock.

    Deliberately shows DIRECTION PROBABILITIES only — no synthetic price path,
    matching predict_5day_trajectory()'s "no validated price forecast" policy.
    """
    with st.spinner(f"Building {sym} forecast chart…"):
        df = load_real_daily_data(sym)
        is_valid, errs, _, clean_df = validate_stock_dataframe(df, min_rows=120)
        if not is_valid or clean_df is None or len(clean_df) < 60:
            st.error(f"Chart data unavailable for {sym}: {errs[0] if errs else 'insufficient history'}")
            return

        from feature_engineering import FEATURE_COLUMNS
        traj = predict_5day_trajectory(clean_df, list(FEATURE_COLUMNS), None)

    if traj is None or traj.empty:
        st.info(f"No forward projection available for {sym}.")
        return

    q = load_real_quote(sym)
    last_close = float(clean_df["Close"].iloc[-1])

    probs = traj["RF_Probability_UP"].astype(float).values
    dirs = traj["Direction"].tolist()
    labels = [d.replace("(", " ").replace(")", "") for d in traj["Trading_Date"].tolist()]
    up_days = sum(1 for d in dirs if d == "UP")
    avg_prob = float(probs.mean())
    net_tilt = (avg_prob - 50.0) * 2.0
    up = net_tilt >= 0
    accent = "#059669" if up else "#DC2626"
    bar_colors = ["#059669" if d == "UP" else "#DC2626" for d in dirs]

    # ── Panel 1: factual price history (no forecast overlay) ────────────────
    hist = clean_df.tail(lookback_days).reset_index(drop=True)
    fig_hist = build_candlestick_chart(hist, height=250, show_volume=False)
    fig_hist.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=5, r=5, t=28, b=5),
        title=dict(
            text=f"{sym.replace('.NS', '')} · last {len(hist)} sessions",
            font=dict(size=13, color="#64748B"), x=0.01, xanchor="left",
        ),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Panel 2: next 5 sessions direction probability ──────────────────────
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=labels, y=probs,
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{p:.0f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=12, weight="bold", color="#0F172A"),
        hovertemplate="<b>%{x}</b><br>P(UP) %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig_p.add_hline(y=50, line_color="#94A3B8", line_width=1.5, line_dash="dot")
    fig_p.add_annotation(
        x=1, y=50, xref="paper", yref="y", text="coin flip 50%",
        showarrow=False, xanchor="right", yshift=10,
        font=dict(size=10, color="#94A3B8"),
    )
    fig_p.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=5, r=5, t=42, b=5),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#F1F5F9",
                   title="Probability UP (%)", ticksuffix="%"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        title=dict(
            text=f"Next 5 sessions · predicted {net_tilt:+.0f}% tilt",
            font=dict(size=13, color=accent), x=0.01, xanchor="left",
        ),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    st.html(f"""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.6rem;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                    padding:0.55rem 0.9rem;font-size:0.78rem;">
            <b style="color:#0F172A;">Week tilt</b>
            <span style="color:{accent};font-weight:800;">{'UP' if up else 'DOWN'} {net_tilt:+.0f}%</span>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                    padding:0.55rem 0.9rem;font-size:0.78rem;">
            <b style="color:#0F172A;">Avg P(UP)</b>
            <span style="color:#0F172A;font-weight:800;">{avg_prob:.1f}%</span>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                    padding:0.55rem 0.9rem;font-size:0.78rem;">
            <b style="color:#0F172A;">Sessions UP</b>
            <span style="color:#0F172A;font-weight:800;">{up_days}/5</span>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                    padding:0.55rem 0.9rem;font-size:0.78rem;">
            <b style="color:#0F172A;">Reference close</b>
            <span style="color:#0F172A;font-weight:800;">₹{q.get('price') or last_close:,.2f}</span>
        </div>
    </div>
    """)
    st.caption(
        "Bars are the Random Forest's probability of an UP close for each session "
        "(decayed for horizon uncertainty). Direction probabilities only — no price "
        "forecast is produced. Statistical output for research, not advice."
    )


def render_minimal_header():
    """Compact brand header with a single live-market strip."""
    q_indices = load_market_indices()
    pills = "".join(
        f"<span class='bb-pill'><b>{i['name']}</b> {i['price']:,.0f} "
        f"<span style='color:#10B981'>+{i['change_pct']:.2f}%</span></span>"
        for i in q_indices
    )
    status = "🟢 Market open" if is_market_open() else "🔴 Market closed"
    st.html(f"""
    <div class="bb-hero">
        <div class="bb-brand">
            <span class="bb-brand-mark">🐂</span>
            <div>
                <div class="bb-brand-name">BIG BULL</div>
                <div class="bb-brand-sub">AI direction engine · NSE</div>
            </div>
        </div>
        <div class="bb-status">{status}</div>
    </div>
    <div class="bb-strip">{pills}</div>
    """)


def render_minimal_search():
    """Search bar with realtime pin / unpin for the selected stock."""
    sym = st.session_state["ticker"]
    pinned = st.session_state["pinned_companies"]
    is_pinned = sym in pinned

    col_search, col_pin = st.columns([5, 1])
    with col_search:
        search_input = st.text_input(
            "Search", value="", placeholder="Search any NSE stock…  (e.g. RELIANCE, TCS, INFY)",
            label_visibility="collapsed", key="minimal_search"
        )
    with col_pin:
        label = "Unpin" if is_pinned else "Pin"
        icon = ":material/push_pin:" if not is_pinned else ":material/pin_drop:"
        if st.button(f"{label} {sym}", icon=icon, use_container_width=True, key="minimal_pin_btn"):
            toggle_pin_stock(sym)
            st.rerun()

    if search_input and len(search_input.strip()) >= 2:
        matches = search_companies(search_input.strip())
        if matches:
            st.html("<div class='bb-section-label'>Results</div>")
            cols = st.columns(min(len(matches), 5))
            for i, m in enumerate(matches[:5]):
                with cols[i]:
                    m_sym = m.get("symbol", "")
                    is_pin = m_sym in pinned
                    if st.button(
                        f"{'📌 ' if is_pin else ''}{m_sym.replace('.NS', '')}",
                        use_container_width=True, key=f"srch_{m_sym}",
                        help=f"{m.get('name', '')} — {'Unpin' if is_pin else 'Pin'}",
                    ):
                        set_active_stock(m_sym)
                        toggle_pin_stock(m_sym)
                        st.rerun()


def render_stories_row():
    """Instagram-style story rings: live prediction for every pinned stock.

    Each card has: Open (select stock), Graph (show its future-prediction chart)
    and Unpin. The graph target is held in session state so it survives reruns.
    """
    pinned = st.session_state["pinned_companies"]
    if not pinned:
        return

    st.html("<div class='bb-section-label'>Your stories</div>")
    cols = st.columns(min(len(pinned) + 1, 6))

    for i, sym in enumerate(pinned[:5]):
        with cols[i]:
            q = load_real_quote(sym)
            bias = get_next_week_bias(sym)
            known = bool(bias.get("ok"))
            up = known and bias.get("direction") == "UP"

            if known:
                arrow = "↑" if up else "↓"
                ring_cls = "bb-ring-up" if up else "bb-ring-down"
                badge_bg = "#ECFDF5" if up else "#FEF2F2"
                badge_fg = "#047857" if up else "#B91C1C"
                dir_text = bias.get("direction", "—")
                # Show the probability of the direction on the badge, so the arrow,
                # the word and the number can never disagree.
                prob_text = f"{bias.get('prob_direction', 50):.0f}% {dir_text}"
            else:
                arrow = "•"
                ring_cls = "bb-ring-unknown"
                badge_bg = "#F1F5F9"
                badge_fg = "#64748B"
                dir_text = "No signal"
                prob_text = "—"

            initials = get_company_initials(q.get("name") or sym)
            pct = f"{q['day_change_pct']:+.2f}%" if q.get("day_change_pct") is not None else ""
            is_chart_open = st.session_state.get("graph_ticker") == sym
            is_active = st.session_state["ticker"] == sym
            card_key = "storycard_" + sym.replace(".", "_").replace("-", "_")

            # Whole card is clickable: a transparent button is laid over it via CSS,
            # so tapping anywhere on the story loads that stock on the dashboard.
            with st.container(key=card_key):
                if is_active:
                    st.html(f'<div class="bb-story-active-tag">● viewing</div>')
                st.html(f"""
                <div class="bb-story {ring_cls}">
                    <div class="bb-story-avatar">{initials}</div>
                    <div class="bb-story-sym">{sym.replace('.NS', '')}</div>
                    <div class="bb-story-badge"
                         style="background:{badge_bg};color:{badge_fg};">
                        {arrow} {dir_text}
                    </div>
                    <div class="bb-story-pct">{prob_text}</div>
                    <div class="bb-story-pct">{pct}</div>
                </div>
                """)
                if st.button(
                    f"Load {sym.replace('.NS', '')} prediction",
                    key=f"story_select_{sym}",
                    help=f"Show {sym} on the dashboard",
                ):
                    if not is_active:
                        set_active_stock(sym)
                    st.rerun()

            ga, gc2 = st.columns([3, 1])
            with ga:
                if st.button(
                    "Close graph" if is_chart_open else "Graph",
                    key=f"story_graph_{sym}",
                    use_container_width=True,
                    type="primary" if is_chart_open else "secondary",
                ):
                    st.session_state["graph_ticker"] = None if is_chart_open else sym
                    st.rerun()
            with gc2:
                if st.button("✕", key=f"story_unpin_{sym}", help=f"Unpin {sym}",
                             use_container_width=True):
                    toggle_pin_stock(sym)
                    st.rerun()

    with cols[len(pinned[:5])]:
        st.html("""
        <div class="bb-story bb-ring-empty">
            <div class="bb-story-avatar bb-story-plus">+</div>
            <div class="bb-story-sym">Add</div>
            <div class="bb-story-dir">—</div>
            <div class="bb-story-pct">search above</div>
        </div>
        """)


def render_minimal_prediction(latest_pred: dict, results: dict, quote: dict, next_week_pack: dict):
    """The single prediction card.

    Direction and probability come from ONE engine (get_next_week_bias) — the same
    one the story cards use — so the two can never contradict each other.
    The big number is the probability of the direction being shown, and it is
    labelled P(UP) or P(DOWN) so the word and the figure always agree.
    """
    horizon = st.session_state["horizon"]
    sessions = next_week_pack.get("sessions", horizon) if next_week_pack.get("ok") else horizon

    if next_week_pack.get("ok"):
        wk_dir = next_week_pack["direction"]
        prob_display = next_week_pack["prob_direction"]
        days_up = next_week_pack.get("days_up", 0)
    else:
        wk_dir = latest_pred["predicted_direction"]
        prob_display = (latest_pred["probability_up"] if wk_dir == "UP"
                        else latest_pred["probability_down"]) * 100.0
        days_up = 0

    wk_up = wk_dir == "UP"
    prob_label = "P(UP)" if wk_up else "P(DOWN)"

    metrics = results["test_metrics"]
    acc = metrics["accuracy"] * 100.0
    edge = acc - metrics["baseline_accuracy"] * 100.0

    accent = "#059669" if wk_up else "#DC2626"
    soft = "#ECFDF5" if wk_up else "#FEF2F2"
    glow = ("0 14px 34px -18px rgba(5,150,105,0.65)" if wk_up
            else "0 14px 34px -18px rgba(220,38,38,0.55)")
    price = quote.get("price") or latest_pred["latest_close"]
    change = quote.get("day_change_pct")

    st.html(f"""
    <div class="bb-predict" style="--accent:{accent};--soft:{soft};box-shadow:{glow};">
        <div class="bb-predict-top">
            <div>
                <div class="bb-predict-sym">{quote.get('clean_symbol', st.session_state['ticker'])}</div>
                <div class="bb-predict-name">{quote.get('name', '')}</div>
            </div>
            <div class="bb-predict-price">
                ₹{price:,.2f}
                <span class="{'bb-up' if (change or 0) >= 0 else 'bb-down'}">
                    {(f'{change:+.2f}%' if change is not None else '')}
                </span>
            </div>
        </div>

        <div class="bb-predict-dir">
            <span class="bb-predict-arrow">{'↑' if wk_up else '↓'}</span>
            <div>
                <div class="bb-predict-word" style="color:{accent};">
                    {'BULLISH' if wk_up else 'BEARISH'}
                </div>
                <div class="bb-predict-horizon">{prob_label} · next {sessions} sessions</div>
            </div>
            <div>
                <div class="bb-predict-prob" style="color:{accent};">{prob_display:.0f}%</div>
                <div class="bb-predict-prob-sub" style="color:{accent};">
                    {prob_label} probability
                </div>
            </div>
        </div>

        <div class="bb-meter">
            <div class="bb-meter-fill" style="width:{prob_display:.1f}%;background:linear-gradient(90deg,{accent}80,{accent});"></div>
        </div>

        <div class="bb-predict-stats">
            <div><span>{days_up}/{sessions}</span><small>Sessions UP</small></div>
            <div><span>{acc:.0f}%</span><small>Accuracy</small></div>
            <div><span class="{'bb-up' if edge >= 0 else 'bb-down'}">{edge:+.0f}%</span><small>vs baseline</small></div>
            <div><span>{results['walk_forward_metrics']['mean_accuracy']*100:.0f}%</span><small>Walk-fwd</small></div>
        </div>
    </div>
    """)
    st.caption(
        f"Probability shown is of the stated direction ({prob_label}), from a single Random "
        f"Forest fit on {sessions} sessions of {st.session_state['ticker']} daily data. "
        "Accuracy / vs baseline / walk-fwd describe historical model quality, not this call."
    )


def render_minimal_pinned_strip():
    """Pinned stocks with realtime pin/unpin toggles."""
    pinned = st.session_state["pinned_companies"]
    st.html("<div class='bb-section-label'>Pinned</div>")

    if not pinned:
        st.caption("Nothing pinned yet — use the Pin button above.")
        return

    for row_start in range(0, len(pinned), 4):
        cols = st.columns(4)
        for j, sym in enumerate(pinned[row_start:row_start + 4]):
            with cols[j]:
                q = load_real_quote(sym)
                is_active = sym == st.session_state["ticker"]
                chg = q.get("day_change_pct") or 0.0
                cls = "bb-tile-active" if is_active else ""
                st.html(f"""
                <div class="bb-tile {cls}">
                    <div class="bb-tile-sym">{sym.replace('.NS', '')}</div>
                    <div class="bb-tile-price">₹{q.get('price', 0):,.0f}
                        <span class="{'bb-up' if chg >= 0 else 'bb-down'}">{chg:+.2f}%</span>
                    </div>
                </div>
                """)
                b1, b2 = st.columns([3, 1])
                with b1:
                    if st.button("Open", key=f"tile_open_{sym}", use_container_width=True,
                                 disabled=is_active):
                        set_active_stock(sym)
                        st.rerun()
                with b2:
                    if st.button("✕", key=f"tile_unpin_{sym}", use_container_width=True,
                                 help=f"Unpin {sym}"):
                        toggle_pin_stock(sym)
                        st.rerun()


def render_watchlist_cards():
    """3 pinned watchlist cards side-by-side."""
    pinned = st.session_state["pinned_companies"]
    if not pinned:
        return
    with st.container(border=True):
        st.markdown("##### :material/push_pin: Pinned watchlist")
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                if i < len(pinned):
                    sym = pinned[i]
                    is_active = sym == st.session_state["ticker"]
                    q = load_real_quote(sym)
                    is_up = q["day_change"] >= 0
                    sign = "+" if is_up else ""
                    avatar = get_company_initials(q["name"] or sym)
                    with st.container(border=True):
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
                            <div class="co-avatar" style="width:34px;height:34px;font-size:0.8rem;">{avatar}</div>
                            <div>
                                <div style="font-weight:700; font-size:0.9rem;">{q['clean_symbol']}</div>
                                <div style="font-size:0.72rem; color:#64748B; max-width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{q['name']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.metric(
                            "Price", f"₹{q['price']:,.2f}",
                            f"{sign}{q['day_change_pct']:.2f}%",
                            label_visibility="collapsed"
                        )

                        bias = get_next_week_bias(sym)
                        if bias.get("ok", False):
                            b_up = bias["direction"] == "UP"
                            b_col = "#10B981" if b_up else "#EF4444"
                            b_sym = "↑" if b_up else "↓"
                            st.markdown(
                                f"<div style='font-size:0.78rem;color:{b_col};font-weight:700;'>"
                                f"{b_sym} Next week {bias['direction']} · {bias['prob_up']:.0f}% UP "
                                f"({bias['days_up']}/5 UP)</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div style='font-size:0.78rem;color:#64748B;'>Next week: —</div>",
                                unsafe_allow_html=True,
                            )

                        b1, b2 = st.columns([2, 1])
                        with b1:
                            lbl = "Viewing" if is_active else "Open"
                            if st.button(lbl, key=f"wl_open_{sym}_{i}", use_container_width=True):
                                if not is_active:
                                    set_active_stock(sym)
                                    st.rerun()
                        with b2:
                            if st.button("✕", key=f"wl_rm_{sym}_{i}", help=f"Unpin {sym}"):
                                toggle_pin_stock(sym)
                                st.rerun()
                else:
                    with st.container(border=True):
                        st.markdown("""
                        <div style="text-align:center; padding:1rem 0.5rem; color:#64748B;">
                            <div style="font-size:1.2rem;">+</div>
                            <div style="font-size:0.78rem;">Pin a stock</div>
                        </div>
                        """, unsafe_allow_html=True)


def render_quick_actions():
    """Right-side quick actions panel."""
    with st.container(border=True):
        st.subheader("Quick actions", icon=":material/flash_on:")

        if st.button("Deep-dive analytics & models", icon=":material/analytics:", key="qa_deep_dive", use_container_width=True):
            st.session_state["expand_deep_dive"] = True
            st.rerun()

        # Bookmark toggle
        sym = st.session_state["ticker"]
        favs = st.session_state.get("favourites", [])
        is_fav = sym in favs
        fav_label = "Remove bookmark" if is_fav else "Bookmark stock"
        fav_icon = ":material/bookmark_remove:" if is_fav else ":material/bookmark_add:"
        if st.button(fav_label, icon=fav_icon, key="qa_bookmark_toggle", use_container_width=True):
            if is_fav:
                favs.remove(sym)
                st.toast(f"Removed {sym} from bookmarks.", icon=":material/delete:")
            else:
                favs.append(sym)
                st.toast(f"Added {sym} to bookmarks!", icon=":material/bookmark:")
            st.session_state["favourites"] = favs
            st.rerun()

        # Power BI export trigger
        if st.button("Export to Power BI (8 CSVs)", icon=":material/sync:", key="qa_trigger_pbi", use_container_width=True):
            with st.spinner("Generating atomic Power BI CSV exports..."):
                try:
                    if _run_pbi_export is not None:
                        res = _run_pbi_export(tickers=[sym])
                        st.toast(f"Exported {len(res)} files to powerbi_exports!", icon=":material/check_circle:")
                    else:
                        st.toast("Power BI export module not available.", icon=":material/error:")
                except Exception as ex:
                    st.toast(f"Export failed: {ex}", icon=":material/error:")

        st.write("")
        # CSV download
        ticker = st.session_state["ticker"]
        df_export = st.session_state.get("daily_df")
        if df_export is not None and not df_export.empty:
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download historical data (CSV)",
                csv_bytes,
                file_name=f"{ticker}_historical.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
                key="qa_download_csv"
            )


def render_5day_trajectory_card(clean_df, feature_cols, model):
    """5-day projected path card."""
    traj_df = predict_5day_trajectory(clean_df, feature_cols, model)
    if traj_df is None or traj_df.empty:
        return
    with st.container(border=True):
        st.subheader("5-day projected path", icon=":material/show_chart:")
        st.caption("Projected trajectory from Random Forest probabilities + ATR risk bounds.")

        last_row = clean_df.iloc[-1]
        cur_price = float(last_row["Close"])
        cur_dt = pd.to_datetime(last_row["Date"]).strftime("%Y-%m-%d")

        plot_dates = [cur_dt] + list(traj_df["Trading_Date"])
        plot_prices = [cur_price] + list(traj_df["Predicted_Close_INR"])
        lowers = [cur_price] + list(traj_df["Lower_Bound_INR"])
        uppers = [cur_price] + list(traj_df["Upper_Bound_INR"])

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=plot_dates, y=uppers, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_t.add_trace(go.Scatter(
            x=plot_dates, y=lowers, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(59,130,246,0.12)", name="ATR risk bound"
        ))
        fig_t.add_trace(go.Scatter(
            x=plot_dates, y=plot_prices, mode="lines+markers",
            name="Projected close",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=7, color="#3B82F6"),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>",
        ))
        apply_chart_theme(fig_t, height=260)
        st.plotly_chart(fig_t, use_container_width=True)

        display_df = traj_df[[
            "Day", "Trading_Date", "Predicted_Close_INR",
            "Expected_Return_Pct", "Direction", "RF_Probability_UP"
        ]].copy()
        display_df.columns = ["Horizon", "Date", "Projected Close (₹)", "Return (%)", "Direction", "P(UP) %"]
        st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_optimal_timing_card(intraday_df):
    """Buy/sell timing windows."""
    opt = get_optimal_intraday_windows(intraday_df)
    with st.container(border=True):
        st.subheader("Optimal intraday timing & price windows", icon=":material/schedule:")
        st.caption("15-minute historical bar analysis · NSE 09:15–15:30 IST")
        tc1, tc2 = st.columns(2)
        with tc1:
            with st.container(border=True):
                st.markdown(":green-badge[🟢 Best buy window]")
                st.markdown(f"### {opt['best_buy_window']}")
                st.caption("Historically highest upward momentum & bullish win rate.")
                if opt.get("top_buy_times"):
                    for bt in opt["top_buy_times"][:3]:
                        st.markdown(f"- **{bt['Time']} IST**: `+{bt['Mean_Return_Pct']:.2f}%` avg return ({bt['Win_Rate_Pct']:.0f}% win rate)")
        with tc2:
            with st.container(border=True):
                st.markdown(":red-badge[🔴 Best sell / exit window]")
                st.markdown(f"### {opt['best_sell_window']}")
                st.caption("Statistical profit-taking, intraday pullback, or end-of-day square-off.")
                if opt.get("top_sell_times"):
                    for st_item in opt["top_sell_times"][:3]:
                        st.markdown(f"- **{st_item['Time']} IST**: `{st_item['Mean_Return_Pct']:+.2f}%` avg return ({st_item['Win_Rate_Pct']:.0f}% win rate)")


def render_timing_confluence_playbook(daily_df: pd.DataFrame, intraday_df: pd.DataFrame):
    """Synthesizes Intraday Hour, Day of Week, Day of Month repeat, and Festival cycles."""
    with st.container(border=True):
        st.subheader("Date & time confluence playbook", icon=":material/psychology:")
        st.caption("Cross-analyzes historical time-of-day, weekday, recurring calendar date (1–31), and active festival cycles.")

        c1, c2, c3, c4 = st.columns(4)

        opt_intra = get_optimal_intraday_windows(intraday_df)
        with c1:
            with st.container(border=True):
                st.markdown("**1. Time of Day**")
                st.markdown(f":green-badge[{opt_intra['best_buy_window'].split('(')[0].strip()}]")
                st.caption("Optimal entry window. Exit: " + opt_intra['best_sell_window'].split('(')[0].strip())

        dow_df = analyze_day_of_week(daily_df)
        best_day = "Wednesday"
        best_day_ret = 0.0
        if not dow_df.empty:
            top_dow = dow_df.sort_values(by="Mean_Return_Pct", ascending=False).iloc[0]
            best_day = top_dow["Day"]
            best_day_ret = top_dow["Mean_Return_Pct"]
        with c2:
            with st.container(border=True):
                st.markdown("**2. Day of Week**")
                st.markdown(f":blue-badge[{best_day}]")
                st.caption(f"Historically strongest weekday ({best_day_ret:+.2f}% avg return).")

        dom_df = analyze_day_of_month_repeat(daily_df)
        ist_now = get_ist_now()
        cur_dom = ist_now.day
        dom_match = dom_df[dom_df["Day_Of_Month"] == cur_dom] if not dom_df.empty else pd.DataFrame()
        dom_tendency = dom_match["Historical_Tendency"].iloc[0] if not dom_match.empty else "⚪ Neutral"
        dom_ret = dom_match["Mean_Return_Pct"].iloc[0] if not dom_match.empty else 0.0
        with c3:
            with st.container(border=True):
                st.markdown(f"**3. Calendar Date (#{cur_dom})**")
                st.markdown(f"**{dom_tendency}**")
                st.caption(f"Day {cur_dom} historical tendency ({dom_ret:+.2f}% avg return).")

        cur_ev = detect_current_special_event(ist_now)
        with c4:
            with st.container(border=True):
                st.markdown("**4. Seasonal Festival**")
                st.markdown(f":orange-badge[{cur_ev.get('badge', '📅 Regular')}]")
                st.caption(cur_ev.get("name", "Standard Cycle"))




def render_performance_stats(quote: dict):
    """Day range, 52-week range, and key stats."""
    with st.container(border=True):
        st.subheader("Performance & key stats", icon=":material/equalizer:")
        render_range_bars(quote)
        st.write("")
        s1, s2, s3, s4 = st.columns(4)
        yh = quote["fifty_two_high"] or quote["day_high"]
        yl = quote["fifty_two_low"] or quote["day_low"]
        with s1:
            st.metric("52-wk high", f"₹{yh:,.2f}", border=True)
            st.metric("52-wk low", f"₹{yl:,.2f}", border=True)
        with s2:
            st.metric("Prev. close", f"₹{quote['prev_close']:,.2f}", border=True)
            st.metric("Volume", f"{quote['volume']:,}", border=True)
        with s3:
            st.metric("Sector", quote.get("sector", "NSE Market"), border=True)
            st.metric("Exchange", "NSE India", border=True)
        with s4:
            st.metric("Data feed", "Real-time market", border=True)
            st.metric("Source", "Google Finance", border=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Header
    st.markdown("""
    <div style="padding:0.75rem 0 0.5rem 0;">
        <div style="font-size:1.4rem; font-weight:800; letter-spacing:-0.5px;">BIG BULL</div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:0.15rem;">
            NSE Stock Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("Popular stocks")
    popular_syms = [c["symbol"] for c in POPULAR_NSE_COMPANIES]
    cur_tick = st.session_state["ticker"]
    # No widget key: the dropdown is a pure view of st.session_state["ticker"], so a
    # story-card / search switch is never snapped back by a stale widget value.
    sel_idx = popular_syms.index(cur_tick) if cur_tick in popular_syms else 0
    chosen_pop = st.selectbox(
        "Active stock",
        popular_syms,
        index=sel_idx,
        label_visibility="visible"
    )
    if chosen_pop != st.session_state["ticker"]:
        set_active_stock(chosen_pop)
        st.rerun()

    # Pin toggle
    is_pinned = st.session_state["ticker"] in st.session_state["pinned_companies"]
    if st.button(
        "Unpin" if is_pinned else f"Pin ({len(st.session_state['pinned_companies'])}/3)",
        icon=":material/push_pin:",
        key="sb_pin_toggle",
        use_container_width=True
    ):
        toggle_pin_stock(st.session_state["ticker"])
        st.rerun()

    st.write("")
    st.caption(f"Pinned watchlist ({len(st.session_state['pinned_companies'])}/3)")
    for p_sym in st.session_state["pinned_companies"]:
        ca, cb = st.columns([3, 1])
        with ca:
            is_act = p_sym == st.session_state["ticker"]
            icon = ":material/radio_button_checked:" if is_act else ":material/radio_button_unchecked:"
            if st.button(p_sym.replace(".NS", ""), icon=icon, key=f"sb_p_{p_sym}"):
                set_active_stock(p_sym)
                st.rerun()
        with cb:
            if st.button("✕", key=f"sb_unp_{p_sym}"):
                toggle_pin_stock(p_sym)
                st.rerun()

    st.write("")
    hor_choice = st.segmented_control(
        "Forecast horizon",
        ["1 Day", "5 Days"],
        default="5 Days" if st.session_state["horizon"] == 5 else "1 Day",
        key="sb_horizon"
    )
    st.session_state["horizon"] = 5 if hor_choice == "5 Days" else 1

    if st.button("Sync data", icon=":material/sync:", key="sb_sync", use_container_width=True):
        st.cache_data.clear()
        st.session_state["daily_df"] = load_real_daily_data(st.session_state["ticker"])
        st.session_state["intraday_df"] = load_real_intraday_data(st.session_state["ticker"])
        st.session_state["model_cache"] = {}
        st.toast("Data synchronized!", icon=":material/check_circle:")
        st.rerun()

    # Watermark
    st.markdown("""
    <div class="sidebar-watermark">
        🐂<br>
        <b>BIG BULL</b><br>
        Better Insights. Smarter Decisions.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN PAGE (Home/Dashboard - navigation removed)
# =============================================================================
selected_page = "1. Home"  # Always show Home page since navigation is removed

if selected_page == "1. Home":
    # Search / pin / unpin live in the sidebar — main canvas shows prediction only.
    render_stories_row()
    st.write("")

    active_q = load_real_quote(st.session_state["ticker"])

    # ── Prediction only ────────────────────────────────────────────────
    with st.spinner("Predicting…"):
        bundle = get_daily_model_pipeline(
            st.session_state["ticker"],
            st.session_state["daily_df"],
            st.session_state["horizon"],
        )
    next_week_pack = get_next_week_bias(st.session_state["ticker"])

    if not bundle.get("valid", False):
        st.error(f"Prediction unavailable: {bundle.get('error')}")
    else:
        render_minimal_prediction(bundle["latest_pred"], bundle["results"], active_q, next_week_pack)

    # ── Future prediction graph for the story the user opened ───────────────
    graph_sym = st.session_state.get("graph_ticker")
    if graph_sym:
        st.write("")
        with st.container(border=True):
            gh, gc2 = st.columns([5, 1])
            with gh:
                st.subheader(f"Future prediction · {graph_sym.replace('.NS', '')}", icon=":material/show_chart:")
            with gc2:
                if st.button("✕ Close", key="close_graph_btn", use_container_width=True):
                    st.session_state["graph_ticker"] = None
                    st.rerun()
            render_future_prediction_graph(graph_sym)


# =============================================================================
# PAGE 2: SAVED STOCKS (removed - navigation disabled)
# =============================================================================
# elif selected_page == "2. Saved Stocks":
#     st.title("Saved stocks", icon=":material/bookmark:")
#     st.caption("Your bookmarked stocks and pinned watchlist. Click any row to open full analysis.")
# 
#     favs = st.session_state.get("favourites", [])
#     pinned = st.session_state.get("pinned_companies", [])
# 
#     tab_fav, tab_watchlist, tab_all = st.tabs([
#         "★ Favourites",
#         "📌 Pinned watchlist",
#         "🌐 All stocks board"
#     ])
# 
#     with tab_fav:
#         st.subheader("Bookmarked stocks", icon=":material/bookmark:")
#         if not favs:
#             st.info("No favourites saved yet.", icon=":material/info:")
#         else:
#             fav_rows = []
#             for sym in favs:
#                 q = load_real_quote(sym)
#                 sign = "+" if q["day_change"] >= 0 else ""
#                 fav_rows.append({
#                     "Symbol": q["clean_symbol"],
#                     "Company": q["name"],
#                     "Price (₹)": f"₹{q['price']:,.2f}",
#                     "Day Change": f"{sign}{q['day_change_pct']:.2f}%",
#                     "Day High": f"₹{q['day_high']:,.2f}",
#                     "Day Low": f"₹{q['day_low']:,.2f}",
#                 })
#             fav_df = pd.DataFrame(fav_rows)
#             st.dataframe(fav_df, hide_index=True, use_container_width=True)
# 
#             st.write("")
#             sel_fav, btn_open, btn_rm = st.columns([3, 1, 1])
#             with sel_fav:
#                 open_sym = st.selectbox("Select to open or remove", favs, key="fav_sel")
#             with btn_open:
#                 st.write("")
#                 if st.button("Open stock", type="primary", key="fav_open", icon=":material/open_in_browser:"):
#                     set_active_stock(open_sym)
#                     st.session_state["_forced_page"] = "1. Home"
#                     st.rerun()
#             with btn_rm:
#                 st.write("")
#                 if st.button("Remove", key="fav_remove", icon=":material/bookmark_remove:"):
#                     toggle_favourite(open_sym)
#                     st.rerun()
# 
#     with tab_watchlist:
#         st.subheader("Pinned watchlist comparison", icon=":material/compare_arrows:")
#         st.caption("Side-by-side ML conviction, returns, and performance for pinned stocks.")
#         if not pinned:
#             st.info("Pin stocks from the sidebar to start comparing.", icon=":material/info:")
#         else:
#             compare_rows = []
#             hist_series = {}
#             for sym in pinned:
#                 q = load_real_quote(sym)
#                 try:
#                     df_sym = load_real_daily_data(sym)
#                     pb = get_daily_model_pipeline(sym, df_sym, st.session_state["horizon"])
#                     if pb.get("valid"):
#                         pp = pb["latest_pred"]
#                         res = pb["results"]
#                         rf = pp.get("rf_detection", {})
#                         reg = rf.get("market_regime", {}).get("badge", "Active")
#                         dir_str = f"▲ UP ({pp['probability_up']*100:.1f}%)" if pp["predicted_direction"] == "UP" else f"▼ DOWN ({pp['probability_down']*100:.1f}%)"
#                         acc_str = f"{res['test_metrics']['accuracy']*100:.1f}%"
#                     else:
#                         dir_str, acc_str, reg = "N/A", "N/A", "—"
#                     if not df_sym.empty:
#                         hist_series[sym] = df_sym.set_index("Date")["Close"]
#                 except Exception:
#                     dir_str, acc_str, reg = "N/A", "N/A", "—"
# 
#                 sign = "+" if q["day_change"] >= 0 else ""
#                 compare_rows.append({
#                     "Stock": sym.replace(".NS", ""),
#                     "Company": q["name"],
#                     "Price (₹)": f"₹{q['price']:,.2f}",
#                     "Day Change": f"{sign}{q['day_change_pct']:.2f}%",
#                     f"{st.session_state['horizon']}D Forecast": dir_str,
#                     "Accuracy": acc_str,
#                     "Regime": reg,
#                 })
# 
#             st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
# 
#             if len(hist_series) > 1:
#                 with st.container(border=True):
#                     st.subheader("Normalized return comparison", icon=":material/multiline_chart:")
#                     comp_df = pd.DataFrame(hist_series).dropna()
#                     if not comp_df.empty:
#                         norm_df = (comp_df / comp_df.iloc[0]) * 100.0 - 100.0
#                         fig_n = go.Figure()
#                         palette = ["#3B82F6", "#10B981", "#8B5CF6"]
#                         for idx, col in enumerate(norm_df.columns):
#                             fig_n.add_trace(go.Scatter(
#                                 x=pd.to_datetime(norm_df.index), y=norm_df[col],
#                                 mode="lines", name=col.replace(".NS", ""),
#                                 line=dict(color=palette[idx % len(palette)], width=2)
#                             ))
#                         apply_chart_theme(fig_n, height=320)
#                         st.plotly_chart(fig_n, use_container_width=True)
# 
#     with tab_all:
#         st.subheader("Market board — all stocks", icon=":material/grid_on:")
#         board_df = load_all_companies_board()
#         if not board_df.empty:
#             st.dataframe(board_df, hide_index=True, use_container_width=True)
#             sel_col, open_col, pin_col = st.columns([3, 1, 1])
#             with sel_col:
#                 board_sym = st.selectbox("Select stock", list(board_df["Ticker"]), key="board_sel")
#             with open_col:
#                 st.write("")
#                 if st.button("Open", type="primary", key="board_open", icon=":material/open_in_browser:"):
#                     set_active_stock(board_sym)
#                     st.session_state["_forced_page"] = "1. Home"
#                     st.rerun()
#             with pin_col:
#                 st.write("")
#                 bp = board_sym in st.session_state["pinned_companies"]
#                 if st.button("Unpin" if bp else "Pin", key="board_pin", icon=":material/push_pin:"):
#                     toggle_pin_stock(board_sym)
#                     st.rerun()


# =============================================================================
# PAGE 3: SETTINGS (removed - navigation disabled)
# =============================================================================
elif selected_page == "_removed_predict_future":
    active_q = load_real_quote(st.session_state["ticker"])
    st.title(f"Directional forecast · {active_q['name']}", icon=":material/auto_awesome:")
    st.caption("Dual-engine: Random Forest technical classifier + Kronos financial foundation model.")

    render_watchlist_cards()
    st.write("")

    pred_hor = st.segmented_control(
        "Forecast horizon",
        ["Next trading day (1 session)", "Next 5 trading days (1 week)"],
        default="Next 5 trading days (1 week)" if st.session_state["horizon"] == 5 else "Next trading day (1 session)",
        key="predict_hor_ctrl"
    )
    horizon_val = 5 if "5" in pred_hor else 1
    st.session_state["horizon"] = horizon_val

    tab_rf, tab_kronos = st.tabs([
        "🌲 Random forest · Technical AI",
        "🔬 Kronos · Foundation model (OHLCV)"
    ])

    with tab_rf:
        with st.spinner("Running Random Forest pipeline…"):
            bundle = get_daily_model_pipeline(
                st.session_state["ticker"],
                st.session_state["daily_df"],
                horizon_val
            )

        if bundle.get("valid", False):
            lp = bundle["latest_pred"]
            res = bundle["results"]
            metrics = res["test_metrics"]
            clean_df = bundle["clean_df"]
            feature_cols = bundle["feature_cols"]

            p_dir = lp["predicted_direction"]
            p_up = lp["probability_up"] * 100.0
            p_down = lp["probability_down"] * 100.0
            is_up = p_dir == "UP"
            card_cls = "pred-up" if is_up else "pred-down"
            dir_col = "#10B981" if is_up else "#EF4444"

            col_fc1, col_fc2 = st.columns([3, 2])
            with col_fc1:
                st.markdown(f"""
                <div class="{card_cls}">
                    <div style="font-size:0.85rem; font-weight:700; color:{dir_col};">
                        {"▲" if is_up else "▼"} PREDICTED DIRECTION: {p_dir}
                    </div>
                    <h2 style="margin:0.3rem 0 0.2rem; color:{dir_col};">
                        Probability: {p_up if is_up else p_down:.1f}% {"UP" if is_up else "DOWN"}
                    </h2>
                    <p style="margin:0; opacity:0.85; font-size:0.9rem;">
                        Reference close: ₹{lp['latest_close']:,.2f} · Signal date: {lp['signal_date']}
                    </p>
                    <p style="margin:0.4rem 0 0; font-size:0.82rem; color:#94A3B8;">
                        Horizon: {horizon_val} trading session(s)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                render_5day_trajectory_card(clean_df, feature_cols, res["model"])
                render_optimal_timing_card(st.session_state["intraday_df"])

            with col_fc2:
                with st.container(border=True):
                    st.subheader("Forecast summary", icon=":material/summarize:")
                    st.metric("Model accuracy (holdout)", f"{metrics['accuracy']*100:.1f}%",
                              f"+{(metrics['accuracy']-metrics['baseline_accuracy'])*100:.1f}% vs baseline", border=True)
                    st.metric("Balanced accuracy", f"{metrics['balanced_accuracy']*100:.1f}%", border=True)
                    st.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%", border=True)
                    st.metric("Precision (UP)", f"{metrics['precision']*100:.1f}%", border=True)

                    rf_det = lp.get("rf_detection", {})
                    tc = rf_det.get("tree_consensus", {})
                    mr = rf_det.get("market_regime", {})
                    st.write("")
                    st.caption(f"Tree consensus: {tc.get('consensus_pct', 50)}% · {tc.get('strength_label', '')}")
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=["UP trees", "DOWN trees"],
                        values=[tc.get("trees_up", 50), tc.get("trees_down", 50)],
                        hole=0.65,
                        marker=dict(colors=["#10B981", "#EF4444"]),
                        textinfo="label+value",
                    )])
                    apply_chart_theme(fig_donut, height=220)
                    st.plotly_chart(fig_donut, use_container_width=True)

                    st.caption(f"Regime: {mr.get('badge', 'Active')} · {mr.get('description', '')}")

            with st.container(border=True):
                st.subheader("Top driving technical features", icon=":material/analytics:")
                feat_df = res["feature_importance"].head(8)
                fig_imp = px.bar(
                    feat_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues"
                )
                apply_chart_theme(fig_imp, height=280)
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.error(f"Pipeline error: {bundle.get('error')}")

    with tab_kronos:
        with st.container(border=True):
            st.subheader("Kronos financial foundation model", icon=":material/science:")
            st.caption("Decoder-only foundation model pre-trained on K-line candlestick data from 45+ global exchanges.")

        k_c1, k_c2 = st.columns([2, 1])
        with k_c1:
            kronos_hor = st.segmented_control(
                "Kronos horizon", ["1 trading day", "5 trading days"],
                default="5 trading days" if horizon_val == 5 else "1 trading day",
                key="kronos_hor_ctrl"
            )
            kdays = 5 if "5" in kronos_hor else 1
        with k_c2:
            st.write("")
            run_k = st.button("Run Kronos forecast", type="primary", icon=":material/bolt:", key="kronos_run")

        ks = kronos_status()
        if ks["status"] == "ready":
            st.caption(f"✅ Kronos ready · {ks.get('model', '')} · {ks.get('device', 'cpu')}")
        elif ks["status"] == "error":
            st.error(f"Kronos error: {ks.get('error')}", icon=":material/error:")
        else:
            st.info("Kronos is ready. Press **Run Kronos forecast** to generate predictions.", icon=":material/info:")

        for k in ["kronos_result", "kronos_key"]:
            if k not in st.session_state:
                st.session_state[k] = None if k == "kronos_result" else ""

        cur_k_key = f"{st.session_state['ticker']}_{len(st.session_state['daily_df'])}_{kdays}"

        if run_k or (st.session_state["kronos_key"] == cur_k_key and st.session_state["kronos_result"] is not None):
            if run_k:
                with st.spinner(f"Running Kronos for {st.session_state['ticker']} ({kdays}D)…"):
                    k_result = run_kronos_forecast(
                        st.session_state["daily_df"],
                        symbol=st.session_state["ticker"],
                        pred_days=kdays,
                    )
                    st.session_state["kronos_result"] = k_result
                    st.session_state["kronos_key"] = cur_k_key
            else:
                k_result = st.session_state["kronos_result"]

            if k_result["status"] == "ok":
                k_dir = k_result["direction"]
                k_pct = k_result["pct_change"]
                k_unc = k_result["uncertainty"]
                k_fdf = k_result["forecast_df"]
                k_last = k_result["last_close"]

                is_kup = k_dir == "UP"
                k_sign = "+" if is_kup else ""
                k_cls = "pred-up" if is_kup else "pred-down"
                k_col = "#10B981" if is_kup else "#EF4444"

                st.markdown(f"""
                <div class="{k_cls}">
                    <span style="font-size:0.85rem; font-weight:700; color:{k_col};">
                        KRONOS FORECAST: {"▲" if is_kup else "▼"} {k_dir} ({k_sign}{k_pct:.2f}%)
                    </span>
                    <h2 style="margin:0.3rem 0 0.1rem; font-weight:700;">
                        Projected close: ₹{k_fdf['close'].iloc[-1]:,.2f}
                        <span style="font-size:1rem; font-weight:400;"> vs current ₹{k_last:,.2f}</span>
                    </h2>
                    <p style="margin:0; font-size:0.85rem; opacity:0.85;">
                        Uncertainty: ±{k_unc:.2f}% · Horizon: {kdays} session(s)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                km1, km2, km3, km4 = st.columns(4)
                with km1:
                    st.metric("Projected open", f"₹{k_fdf['open'].iloc[-1]:,.2f}", border=True)
                with km2:
                    st.metric("Projected high", f"₹{k_fdf['high'].iloc[-1]:,.2f}", border=True)
                with km3:
                    st.metric("Projected low", f"₹{k_fdf['low'].iloc[-1]:,.2f}", border=True)
                with km4:
                    st.metric("Projected close", f"₹{k_fdf['close'].iloc[-1]:,.2f}", f"{k_sign}{k_pct:.2f}%", border=True)

                with st.container(border=True):
                    st.subheader("Historical + Kronos forecast chart", icon=":material/stacked_line_chart:")
                    daily_plot = st.session_state["daily_df"].tail(120).copy()
                    fig_k = build_candlestick_chart(daily_plot, height=440, show_volume=False, forecast_df=k_fdf)
                    st.plotly_chart(fig_k, use_container_width=True)

                with st.container(border=True):
                    st.subheader("Session-by-session forecast table", icon=":material/table_chart:")
                    k_tbl = k_fdf.copy().reset_index()
                    k_tbl.columns = ["Trading Date", "Open", "High", "Low", "Close"]
                    k_tbl["Trading Date"] = k_tbl["Trading Date"].dt.strftime("%a %d %b %Y")
                    for c in ["Open", "High", "Low", "Close"]:
                        k_tbl[c] = k_tbl[c].apply(lambda x: f"₹{x:,.2f}")
                    st.dataframe(k_tbl, hide_index=True, use_container_width=True)
            else:
                st.error(f"Kronos failed: {k_result.get('message', 'Unknown error')}", icon=":material/error:")


# PAGE 4: INTRADAY (removed from sidebar)
elif selected_page == "_removed_intraday":
    active_q = load_real_quote(st.session_state["ticker"])
    st.title(f"Intraday analysis · {active_q['name']}", icon=":material/candlestick_chart:")
    st.caption("15-minute candlestick analysis, VWAP, EMA, and recurring time-of-day patterns (09:15–15:30 IST).")

    intra_df = st.session_state["intraday_df"]
    intra_res = train_and_predict_intraday(intra_df)

    if intra_res.get("valid", False):
        df_feat = intra_res["df_with_features"]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Latest 15m close", f"₹{intra_res['latest_close']:,.2f}", border=True)
        with m2:
            st.metric("Next candle", intra_res["predicted_direction"],
                      f"{intra_res['probability_up']*100:.1f}% UP", border=True)
        with m3:
            st.metric("Intraday test accuracy", f"{intra_res['test_accuracy']:.1f}%", border=True)
        with m4:
            st.metric("Bars analyzed", f"{len(df_feat)}", border=True)

        render_optimal_timing_card(intra_df)

        with st.container(border=True):
            st.subheader("15-minute candlestick with VWAP & EMA overlays", icon=":material/candlestick_chart:")
            fig_intra = go.Figure()
            fig_intra.add_trace(go.Candlestick(
                x=pd.to_datetime(df_feat["Timestamp"]),
                open=df_feat["Open"], high=df_feat["High"],
                low=df_feat["Low"], close=df_feat["Close"],
                name="Candlestick",
                increasing=dict(line=dict(color="#10B981"), fillcolor="#10B981"),
                decreasing=dict(line=dict(color="#EF4444"), fillcolor="#EF4444"),
            ))
            if "VWAP" in df_feat.columns:
                fig_intra.add_trace(go.Scatter(
                    x=pd.to_datetime(df_feat["Timestamp"]), y=df_feat["VWAP"],
                    mode="lines", name="VWAP", line=dict(color="#F59E0B", width=2)
                ))
            if "EMA_9" in df_feat.columns:
                fig_intra.add_trace(go.Scatter(
                    x=pd.to_datetime(df_feat["Timestamp"]), y=df_feat["EMA_9"],
                    mode="lines", name="EMA 9", line=dict(color="#3B82F6", width=1.5)
                ))
            apply_chart_theme(fig_intra, height=420)
            fig_intra.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_intra, use_container_width=True)

        with st.container(border=True):
            st.subheader("Time-of-day return distribution (15m intervals)", icon=":material/bar_chart:")
            tod_df = analyze_intraday_time_of_day(df_feat)
            if not tod_df.empty and "Mean_Return_Pct" in tod_df.columns:
                fig_tod = px.bar(
                    tod_df, x="Time", y="Mean_Return_Pct",
                    color="Mean_Return_Pct",
                    color_continuous_scale=[[0, "#EF4444"], [0.5, "#64748B"], [1, "#10B981"]],
                    title="Average 15m bar return across NSE trading hours"
                )
                apply_chart_theme(fig_tod, height=280)
                fig_tod.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_tod, use_container_width=True)
    else:
        st.warning(
            "Intraday data is not available for this stock. "
            "NSE intraday feeds have limited historical lookback (typically 5–7 trading days). "
            "Try a large-cap liquid stock like RELIANCE or TCS.",
            icon=":material/warning:"
        )


# PAGE 5: PATTERN FINDER (removed from sidebar)
elif selected_page == "_removed_pattern_finder":
    active_q = load_real_quote(st.session_state["ticker"])
    st.title(f"Pattern finder · {active_q['name']}", icon=":material/calendar_month:")
    st.caption("Historical recurring patterns: day of week, calendar date, monthly seasonality, annual regimes, and similar historical setups.")

    daily_df = st.session_state["daily_df"]
    tab_dow, tab_dom, tab_month, tab_year, tab_fest, tab_setups = st.tabs([
        "Weekday",
        "Calendar date (1–31)",
        "Monthly seasonality",
        "Annual regimes",
        "🪔 Festivals & Events",
        "Similar setups"
    ])

    with tab_dow:
        with st.container(border=True):
            st.subheader("Day of week return distribution (Mon–Fri)", icon=":material/today:")
            dow_df = analyze_day_of_week(daily_df)
            c1, c2 = st.columns([3, 2])
            with c1:
                fig_dow = px.bar(
                    dow_df, x="Day", y="Mean_Return_Pct",
                    color="Mean_Return_Pct",
                    color_continuous_scale=[[0, "#EF4444"], [0.5, "#64748B"], [1, "#10B981"]],
                )
                apply_chart_theme(fig_dow, height=300)
                fig_dow.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_dow, use_container_width=True)
            with c2:
                st.dataframe(dow_df, hide_index=True, use_container_width=True)

    with tab_dom:
        with st.container(border=True):
            st.subheader("Calendar date repeat anomalies (1–31 of month)", icon=":material/date_range:")
            dom_df = analyze_day_of_month_repeat(daily_df)
            if not dom_df.empty:
                fig_dom = px.bar(
                    dom_df, x="Date_Label", y="Mean_Return_Pct",
                    color="Win_Rate_Pct",
                    color_continuous_scale="Blues",
                )
                apply_chart_theme(fig_dom, height=300)
                st.plotly_chart(fig_dom, use_container_width=True)
                st.dataframe(dom_df, hide_index=True, use_container_width=True)

    with tab_month:
        with st.container(border=True):
            st.subheader("Monthly seasonality (Jan–Dec)", icon=":material/calendar_view_month:")
            month_df = analyze_month_seasonality(daily_df)
            if not month_df.empty:
                fig_m = px.bar(
                    month_df, x="Month", y="Mean_Return_Pct",
                    color="Win_Rate_Pct", color_continuous_scale="Blues",
                )
                apply_chart_theme(fig_m, height=300)
                st.plotly_chart(fig_m, use_container_width=True)
                st.dataframe(month_df, hide_index=True, use_container_width=True)

    with tab_year:
        with st.container(border=True):
            st.subheader("Year-over-year performance", icon=":material/history:")
            yr_df = analyze_year_performance(daily_df)
            st.dataframe(yr_df, hide_index=True, use_container_width=True)

    with tab_fest:
        with st.container(border=True):
            st.subheader("Festivals, budget & recurring market cycles", icon=":material/celebration:")
            st.caption("How Indian festivals (Diwali), Union Budget, monthly F&O expiry, and FY-end rebalancing impact this stock historically.")

            cur_ev = detect_current_special_event()
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3); border-radius:8px; padding:0.8rem 1rem; margin-bottom:1rem;">
                <span style="font-weight:700; color:#F59E0B;">Current Cycle: {cur_ev.get('badge', '')} — {cur_ev.get('name', '')}</span>
                <p style="margin:0.2rem 0 0; font-size:0.85rem; color:#CBD5E1;">{cur_ev.get('description', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            fest_df = analyze_special_events_and_festivals(daily_df)
            if not fest_df.empty:
                f_c1, f_c2 = st.columns([3, 2])
                with f_c1:
                    fig_fest = px.bar(
                        fest_df, x="Event", y="Mean_Return_Pct",
                        color="Mean_Return_Pct",
                        color_continuous_scale=[[0, "#EF4444"], [0.5, "#64748B"], [1, "#10B981"]],
                        title="Average daily return during special events (%)"
                    )
                    apply_chart_theme(fig_fest, height=320)
                    fig_fest.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_fest, use_container_width=True)
                with f_c2:
                    st.markdown("##### Key insights & tendencies")
                    for _, row in fest_df.iterrows():
                        ev_tend = row["Historical_Tendency"]
                        st.markdown(f"**{row['Event']}** ({row['Calendar_Window']})")
                        st.caption(f"{ev_tend} · Avg: `{row['Mean_Return_Pct']:+.2f}%` · Win: `{row['Win_Rate_Pct']:.0f}%` ({row['Sessions']} sessions)")
                        st.caption(f"_{row['Market_Driver']}_")
                        st.markdown("---")

                st.subheader("Historical event performance scorecard")
                st.dataframe(fest_df, hide_index=True, use_container_width=True)
            else:
                st.info("Insufficient historical data for festival and event analysis.")

    with tab_setups:
        with st.container(border=True):
            st.subheader("Nearest-neighbor historical setup matching", icon=":material/find_replace:")
            st.caption("Finds historical periods with similar technical indicators and reports what happened next.")
            similar = find_similar_historical_setups(daily_df, top_k=8, horizon=st.session_state["horizon"])
            if "summary" in similar and similar["summary"]:
                sd = similar["summary"]
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Matching setups", sd["total_matches"], border=True)
                with s2:
                    st.metric("UP win rate", f"{sd['up_frequency_pct']:.1f}%", border=True)
                with s3:
                    st.metric("Avg forward return", f"{sd['avg_forward_return_pct']:+.2f}%", border=True)
                st.write("")
                st.dataframe(similar["matches_table"], hide_index=True, use_container_width=True)


# PAGE 6: BUY/SELL TIMING (removed from sidebar)
elif selected_page == "_removed_buysell":
    active_q = load_real_quote(st.session_state["ticker"])
    st.title(f"Buy/sell timing research · {active_q['name']}", icon=":material/timer:")
    st.caption("Disciplined ATR-based profit targets and stop-loss bounds on real market prices. Educational research only.")

    with st.spinner("Running timing analysis…"):
        bundle = get_daily_model_pipeline(
            st.session_state["ticker"],
            st.session_state["daily_df"],
            st.session_state["horizon"]
        )

    if bundle.get("valid", False):
        timing = bundle["timing_sig"]
        res = bundle["results"]
        clean_df = bundle["clean_df"]
        feature_cols = bundle["feature_cols"]
        sig = timing["signal"]

        if sig == "Potential Buy":
            card_cls, sig_col, arrow = "pred-up", "#10B981", "▲"
        elif sig == "Potential Sell":
            card_cls, sig_col, arrow = "pred-down", "#EF4444", "▼"
        else:
            card_cls, sig_col, arrow = "pred-neutral", "#F59E0B", "→"

        st.markdown(f"""
        <div class="{card_cls}">
            <span style="font-size:0.85rem; font-weight:700; color:{sig_col};">
                {arrow} {sig.upper()}
            </span>
            <h2 style="margin:0.3rem 0 0.2rem; color:{sig_col};">
                Reference price: ₹{timing['reference_price']:,.2f}
            </h2>
            <p style="margin:0; font-size:0.9rem;">
                Entry: {timing['entry_assumption']} ·
                Target: ₹{timing['target_inr']:,.2f} ·
                Stop-loss: ₹{timing['stop_loss_inr']:,.2f}
            </p>
            <p style="margin-top:0.3rem; font-size:0.82rem; opacity:0.85;">
                Risk / Reward: 1 : {timing['risk_reward_ratio']} · {timing['rationale']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Signal", sig, border=True)
        with m2:
            st.metric("Target (₹)", f"₹{timing['target_inr']:,.2f}", border=True)
        with m3:
            st.metric("Stop-loss (₹)", f"₹{timing['stop_loss_inr']:,.2f}", border=True)
        with m4:
            st.metric("Risk / Reward", f"1 : {timing['risk_reward_ratio']}", border=True)

        render_timing_confluence_playbook(clean_df, st.session_state["intraday_df"])
        render_optimal_timing_card(st.session_state["intraday_df"])
        render_5day_trajectory_card(clean_df, feature_cols, res["model"])

        with st.container(border=True):
            st.subheader("Cost-aware strategy backtest (holdout)", icon=":material/receipt_long:")
            bt = backtest_cost_aware_strategy(res["backtest_predictions"], cost_bps=st.session_state["cost_bps"])
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total trades", bt["total_trades"], border=True)
            with b2:
                st.metric("Win rate", f"{bt['win_rate_pct']:.1f}%", border=True)
            with b3:
                st.metric("Net return", f"{bt['net_return_pct']:+.2f}%",
                          f"{bt['net_return_pct'] - bt['benchmark_return_pct']:+.2f}% vs buy & hold", border=True)
            with b4:
                st.metric("Max drawdown", f"{bt['max_drawdown_pct']:.1f}%", border=True)

        st.info(
            "⚠ All signals are for educational research only. Past strategy performance does not guarantee future results. "
            "Transaction costs, taxes, and slippage are approximated.",
            icon=":material/warning:"
        )


# PAGE 7: BACKTESTING (removed from sidebar)
elif selected_page == "_removed_backtesting":
    active_q = load_real_quote(st.session_state["ticker"])
    st.title(f"Backtesting & accuracy · {active_q['name']}", icon=":material/history:")
    st.caption("Chronological holdout (80/20) and expanding walk-forward validation. All predictions are out-of-sample.")

    with st.spinner("Computing walk-forward backtest…"):
        bundle = get_daily_model_pipeline(
            st.session_state["ticker"],
            st.session_state["daily_df"],
            st.session_state["horizon"]
        )

    if bundle.get("valid", False):
        res = bundle["results"]
        metrics = res["test_metrics"]
        wf_df = res["walk_forward_df"]

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Holdout accuracy", f"{metrics['accuracy']*100:.1f}%",
                      f"+{(metrics['accuracy']-metrics['baseline_accuracy'])*100:.1f}% vs baseline", border=True)
        with m2:
            st.metric("Baseline accuracy", f"{metrics['baseline_accuracy']*100:.1f}%", border=True)
        with m3:
            st.metric("Balanced accuracy", f"{metrics['balanced_accuracy']*100:.1f}%", border=True)
        with m4:
            st.metric("Precision (UP)", f"{metrics['precision']*100:.1f}%", border=True)
        with m5:
            st.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%", border=True)

        col_cm, col_wf = st.columns(2)
        with col_cm:
            with st.container(border=True):
                st.subheader("Confusion matrix", icon=":material/grid_on:")
                cm = metrics["confusion_matrix"]
                cm_df = pd.DataFrame(
                    [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                    index=["Actual DOWN", "Actual UP"],
                    columns=["Pred DOWN", "Pred UP"]
                )
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
                apply_chart_theme(fig_cm, height=260)
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_wf:
            with st.container(border=True):
                st.subheader("Walk-forward validation folds", icon=":material/view_week:")
                st.dataframe(wf_df, hide_index=True, use_container_width=True)

        with st.container(border=True):
            st.subheader("Holdout prediction log", icon=":material/table_chart:")
            bp_df = res["backtest_predictions"]
            if bp_df is not None and not bp_df.empty:
                st.dataframe(bp_df.tail(60), hide_index=True, use_container_width=True)


# PAGE 8: RESULTS (removed from sidebar)
elif selected_page == "_removed_results":
    st.title("Results export & Power BI integration", icon=":material/bar_chart:")
    st.caption("Download standardized CSV datasets formatted for Power BI Desktop.")

    with st.container(border=True):
        st.subheader("One-click Power BI bundle", icon=":material/folder_zip:")
        st.caption("Download all CSVs in a single ZIP file ready to import into Power BI.")

    try:
        zip_data = create_powerbi_zip_bundle()
        st.download_button(
            "Download Power BI CSV bundle (.zip)",
            zip_data,
            file_name=f"bigbull_powerbi_{st.session_state['ticker']}.zip",
            mime="application/zip",
            type="primary",
            icon=":material/download:",
        )
    except OSError:
        st.info("No export files found yet. Run a prediction to generate them.")

    with st.container(border=True):
        st.subheader("Individual output datasets", icon=":material/description:")
        file_list = [
            ("historical_prices.csv", "Real historical daily OHLCV prices"),
            ("daily_predictions.csv", "1-day directional forecast"),
            ("five_day_predictions.csv", "5-day directional forecast"),
            ("intraday_predictions.csv", "Intraday direction predictions"),
            ("timing_signals.csv", "Buy/Sell timing research signals"),
            ("historical_patterns.csv", "Day of week & monthly seasonality"),
            ("backtest_predictions.csv", "Holdout predictions & actual outcomes"),
            ("holdout_metrics.csv", "Model vs baseline accuracy scorecard"),
            ("feature_importance.csv", "Ranked Random Forest technical features"),
        ]
        for fname, desc in file_list:
            fpath = os.path.join(OUTPUTS_DIR, fname)
            c_a, c_b = st.columns([4, 1])
            with c_a:
                exists = os.path.exists(fpath)
                badge = ":green-badge[Available]" if exists else ":orange-badge[Run prediction first]"
                st.markdown(f"**`{fname}`** {badge}  \n*{desc}*")
            with c_b:
                if os.path.exists(fpath):
                    with open(fpath, "rb") as f:
                        st.download_button(
                            "Download",
                            f.read(), fname, "text/csv",
                            key=f"dl_{fname}",
                            icon=":material/download:"
                        )


# =============================================================================
# PAGE 3: SETTINGS (removed - navigation disabled)
# =============================================================================
# elif selected_page == "3. Settings":
#     st.title("Settings & preferences", icon=":material/settings:")
# 
#     with st.container(border=True):
#         st.subheader("Data feeds & cache", icon=":material/database:")
#         st.markdown("- **Primary data engine:** Google Finance & NSE")
#         st.markdown(f"- **Active stock:** `{st.session_state['ticker']}`")
#         st.markdown(f"- **Pinned watchlist:** {', '.join(st.session_state['pinned_companies'])}")
#         st.markdown(f"- **Favourites:** {', '.join(st.session_state['favourites']) or 'None'}")
# 
#         if st.button("Force clear cache & resync data", icon=":material/refresh:", key="settings_clear_cache"):
#             st.cache_data.clear()
#             st.session_state["daily_df"] = load_real_daily_data(st.session_state["ticker"])
#             st.session_state["intraday_df"] = load_real_intraday_data(st.session_state["ticker"])
#             st.session_state["model_cache"] = {}
#             st.toast("Data resynchronized!", icon=":material/check_circle:")
#             st.rerun()
# 
#     with st.container(border=True):
#         st.subheader("Risk & cost parameters", icon=":material/tune:")
#         cost_input = st.number_input(
#             "Transaction cost & slippage (basis points):",
#             min_value=0.0, max_value=100.0,
#             value=float(st.session_state["cost_bps"]), step=1.0,
#             key="settings_cost_bps"
#         )
#         if cost_input != st.session_state["cost_bps"]:
#             st.session_state["cost_bps"] = cost_input
#             st.session_state["model_cache"] = {}
#             st.toast("Cost parameters updated.", icon=":material/check:")
# 
#     with st.container(border=True):
#         st.subheader("Chart preferences", icon=":material/stacked_line_chart:")
#         chart_pref = st.segmented_control(
#             "Default chart type",
#             ["Candlestick", "Line"],
#             default=st.session_state.get("chart_type", "Candlestick"),
#             key="settings_chart_type"
#         )
#         if chart_pref:
#             st.session_state["chart_type"] = chart_pref
# 
#     with st.container(border=True):
#         st.subheader("Model & data information", icon=":material/info:")
#         k_status = kronos_status()
#         st.markdown(f"- **Kronos model status:** `{k_status['status']}`")
#         if k_status.get("model"):
#             st.markdown(f"- **Kronos model:** `{k_status['model']}`")
#         st.markdown("- **Random Forest:** 100 trees · Walk-forward validated")
#         st.markdown("- **Data source:** yFinance / Google Finance (free, real historical data)")
#         st.markdown("- **All predictions are probabilistic estimates for educational research only.**")
#         st.markdown("- **No financial advice is provided or implied.**")
# 
#     with st.container(border=True):
#         st.subheader("Upload custom stock data", icon=":material/upload_file:")
#         tab_d, tab_i = st.tabs(["Daily historical data", "Intraday data"])
#         with tab_d:
#             up_file = st.file_uploader("Upload (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="settings_upload_daily")
#             if up_file is not None:
#                 try:
#                     df_up = pd.read_excel(up_file) if up_file.name.endswith((".xlsx", ".xls")) else pd.read_csv(up_file)
#                     is_valid, errs, warns, clean_df = validate_stock_dataframe(df_up, min_rows=40)
#                     if is_valid:
#                         st.success(f"Loaded {len(clean_df)} sessions from '{up_file.name}'.", icon=":material/check_circle:")
#                         st.session_state["daily_df"] = clean_df
#                         st.session_state["model_cache"] = {}
#                     else:
#                         for err in errs:
#                             st.error(err, icon=":material/error:")
#                 except Exception as e:
#                     st.error(f"Failed to read file: {e}")
#         with tab_i:
#             up_intra = st.file_uploader("Upload intraday (.csv)", type=["csv"], key="settings_upload_intra")
#             if up_intra is not None:
#                 try:
#                     df_in = pd.read_csv(up_intra)
#                     is_v, errs, warns, clean_in = validate_intraday_dataframe(df_in, min_bars=25)
#                     if is_v:
#                         st.success(f"Loaded {len(clean_in)} intraday bars.", icon=":material/check_circle:")
#                         st.session_state["intraday_df"] = clean_in
#                     else:
#                         for err in errs:
#                             st.error(err, icon=":material/error:")
#                 except Exception as e:
#                     st.error(f"Failed to load: {e}")
