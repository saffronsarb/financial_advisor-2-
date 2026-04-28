"""
data_fetcher.py — Fetches stock data via yfinance with caching and validation.
"""

import time

import yfinance as yf

# Simple in-memory cache: {ticker: (timestamp, data)}
_cache: dict = {}
CACHE_TTL = 300  # seconds


def fetch_company_data(ticker: str) -> dict:
    """
    Fetch financial data for a given ticker symbol.
    Results are cached for CACHE_TTL seconds to avoid redundant API calls.

    Returns a dict with keys: info, financials, balance_sheet, history.
    Raises ValueError for unknown tickers, Exception for network issues.
    """
    ticker = ticker.strip().upper()

    # Return cached result if fresh
    if ticker in _cache:
        ts, cached_data = _cache[ticker]
        if time.time() - ts < CACHE_TTL:
            return cached_data

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info or {}

        # Detect invalid tickers (yfinance returns minimal info for bad tickers)
        if not info or (
            info.get("regularMarketPrice") is None
            and info.get("currentPrice") is None
        ):
            raise ValueError(
                f"'{ticker}' doesn't look like a valid ticker. "
                "Try formats like AAPL, MSFT, RELIANCE.NS, TCS.NS."
            )

        history = _safe_df(stock.history(period="1y"))

        data = {
            "info":          info,
            "financials":    _safe_df(stock.financials),
            "balance_sheet": _safe_df(stock.balance_sheet),
            "history":       history,
        }

        _cache[ticker] = (time.time(), data)
        return data

    except ValueError:
        raise
    except Exception as exc:
        raise Exception(f"Could not fetch data for '{ticker}': {exc}") from exc


def _safe_df(df):
    """Return the DataFrame or None if it's None/empty."""
    try:
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def clear_cache():
    """Utility: wipe the in-memory cache (useful for testing)."""
    _cache.clear()