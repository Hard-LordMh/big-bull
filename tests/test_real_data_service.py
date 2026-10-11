"""
test_real_data_service.py
-------------------------
Unit tests for real_data_service module.
Tests ticker normalization, Google Finance URL generation, stock search,
live quote retrieval, and real daily/intraday dataset fetching.
"""

import pytest
import pandas as pd
from real_data_service import (
    normalize_ticker_symbol,
    get_google_finance_url,
    search_companies,
    fetch_real_quote,
    fetch_real_daily_data,
    fetch_real_intraday_data,
    fetch_all_companies_board,
    POPULAR_NSE_COMPANIES
)


def test_normalize_ticker_symbol():
    assert normalize_ticker_symbol("RELIANCE") == "RELIANCE.NS"
    assert normalize_ticker_symbol("tcs") == "TCS.NS"
    assert normalize_ticker_symbol("INFY.NS") == "INFY.NS"
    assert normalize_ticker_symbol("AAPL") == "AAPL"


def test_get_google_finance_url():
    url_rel = get_google_finance_url("RELIANCE.NS")
    assert "google.com/finance/quote/RELIANCE:NSE" in url_rel
    url_tcs = get_google_finance_url("TCS")
    assert "google.com/finance/quote/TCS:NSE" in url_tcs


def test_search_companies():
    # Test catalog search
    tata_res = search_companies("tata")
    assert len(tata_res) > 0
    symbols = [r["symbol"] for r in tata_res]
    assert any("TCS.NS" in s or "TATAMOTORS" in s for s in symbols)

    # Empty query returns popular companies
    top_res = search_companies("")
    assert len(top_res) > 0


def test_fetch_real_quote():
    quote = fetch_real_quote("RELIANCE.NS")
    assert "symbol" in quote
    assert quote["symbol"] == "RELIANCE.NS"
    assert "price" in quote
    assert quote["price"] > 0
    assert "day_change_pct" in quote
    assert "google_finance_url" in quote
    assert "google.com/finance" in quote["google_finance_url"]


def test_fetch_real_daily_data():
    df = fetch_real_daily_data("RELIANCE.NS", period="3mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 30
    for col in ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]:
        assert col in df.columns
    # Check sorted ascending
    assert df["Date"].iloc[0] < df["Date"].iloc[-1]


def test_fetch_real_intraday_data():
    df = fetch_real_intraday_data("RELIANCE.NS", period="5d", interval="15m")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ["Timestamp", "Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns


def test_fetch_all_companies_board():
    board = fetch_all_companies_board(["RELIANCE.NS", "TCS.NS"])
    assert isinstance(board, pd.DataFrame)
    assert len(board) == 2
    assert "Ticker" in board.columns
    assert "Real Price (INR)" in board.columns
    assert "Google Finance" in board.columns
