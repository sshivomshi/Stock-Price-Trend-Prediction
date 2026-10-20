"""Data loading module for fetching and saving historical OHLCV stock data via yfinance.
Addresses Experiment 2: Dataset Details.
"""

import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from src.config import DEFAULT_TICKERS, PRIMARY_TICKER, RAW_DATA_DIR, START_DATE, END_DATE


def fetch_ticker_data(
    ticker: str,
    start: str = START_DATE,
    end: Optional[str] = END_DATE,
    save_raw: bool = True
) -> pd.DataFrame:
    """Download daily historical OHLCV data for a single ticker via Yahoo Finance."""
    print(f"[DataLoader] Fetching data for ticker: {ticker} (Start: {start}, End: {end or 'Latest'})...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Please check symbol or network.")

    # Flatten MultiIndex columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    # Standardize column names
    df.columns = [col.strip().title() for col in df.columns]
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})

    # Ensure Date is datetime and sort temporally
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Ticker"] = ticker

    # Validate required OHLCV columns
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from downloaded data for {ticker}.")

    if save_raw:
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        csv_path = RAW_DATA_DIR / f"{safe_ticker}_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"[DataLoader] Saved {len(df)} records to {csv_path}")

    return df


def fetch_all_tickers(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = START_DATE,
    end: Optional[str] = END_DATE
) -> Dict[str, pd.DataFrame]:
    """Fetch and cache raw data for all tickers in the basket."""
    ticker_dfs = {}
    for ticker in tickers:
        try:
            df = fetch_ticker_data(ticker, start=start, end=end, save_raw=True)
            ticker_dfs[ticker] = df
        except Exception as e:
            print(f"[DataLoader] Warning: Failed to fetch {ticker}: {e}")

    # Combine all into a single consolidated raw dataset
    if ticker_dfs:
        combined_df = pd.concat(ticker_dfs.values(), ignore_index=True)
        combined_csv = RAW_DATA_DIR / "combined_raw_stocks.csv"
        combined_parquet = RAW_DATA_DIR / "combined_raw_stocks.parquet"
        combined_df.to_csv(combined_csv, index=False)
        combined_df.to_parquet(combined_parquet, index=False)
        print(f"[DataLoader] Consolidated raw dataset saved: {len(combined_df)} total rows across {len(ticker_dfs)} tickers.")

    return ticker_dfs


def get_dataset_metadata(df: pd.DataFrame, ticker: str = PRIMARY_TICKER) -> dict:
    """Generate comprehensive dataset metadata for Experiment 2 documentation."""
    metadata = {
        "ticker": ticker,
        "source": "Yahoo Finance (via yfinance package)",
        "license": "Public financial market data for educational & quantitative research",
        "date_range": {
            "start": str(df["Date"].min().date()),
            "end": str(df["Date"].max().date()),
            "total_trading_days": int(len(df))
        },
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
        "features_dictionary": {
            "Date": "Trading session date (YYYY-MM-DD)",
            "Open": "Opening price of the stock in market currency",
            "High": "Highest price reached during regular trading hours",
            "Low": "Lowest price reached during regular trading hours",
            "Close": "Unadjusted closing price for the trading session",
            "AdjClose": "Closing price adjusted for splits and dividend distributions",
            "Volume": "Total number of shares traded during the trading session",
            "Ticker": "Stock ticker identifier symbol"
        },
        "limitations": [
            "Survivorship bias: Evaluates current liquid constituents rather than historical index membership.",
            "Market closures: Non-trading holidays and weekends cause discrete step intervals.",
            "High frequency noise: Daily aggregation obscures intraday microsecond price dynamics.",
            "Unannounced corporate actions: Earnings announcements and splits can introduce non-stationarity."
        ]
    }
    return metadata


if __name__ == "__main__":
    dfs = fetch_all_tickers()
    if PRIMARY_TICKER in dfs:
        meta = get_dataset_metadata(dfs[PRIMARY_TICKER])
        print("\nDataset Metadata for Primary Ticker:")
        print(json.dumps(meta, indent=2))
