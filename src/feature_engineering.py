"""Feature engineering module for calculating financial technical indicators and target labels.
Addresses Experiment 3: Data Pre-processing and Feature Engineering.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import TECHNICAL_INDICATORS


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Use exponential moving average (Wilder's RSI)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD), Signal, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (Upper, Lower, Width)."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    band_width = (upper_band - lower_band) / (rolling_mean + 1e-9)
    return upper_band, lower_band, band_width


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators, price geometry, volume dynamics, and targets.
    
    Target definition:
      - 'Target': 1 if Close[t+1] > Close[t], else 0 (Binary Directional Trend)
      - 'Target_Return': (Close[t+1] - Close[t]) / Close[t] (Next-day return for backtesting)
    """
    data = df.copy()
    data = data.sort_values("Date").reset_index(drop=True)
    cfg = TECHNICAL_INDICATORS

    # 1. Moving Averages
    data["SMA_10"] = data["Close"].rolling(window=cfg["sma_short"]).mean()
    data["SMA_50"] = data["Close"].rolling(window=cfg["sma_long"]).mean()
    data["EMA_20"] = data["Close"].ewm(span=cfg["ema_short"], adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=cfg["ema_long"], adjust=False).mean()

    # Moving Average Distance Ratios (stationary indicators)
    data["Dist_SMA_10"] = (data["Close"] - data["SMA_10"]) / data["SMA_10"]
    data["Dist_SMA_50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"]
    data["SMA_10_50_Ratio"] = data["SMA_10"] / data["SMA_50"]

    # 2. Momentum & Oscillators
    data["RSI_14"] = calculate_rsi(data["Close"], period=cfg["rsi_period"])
    macd_line, macd_signal, macd_hist = calculate_macd(
        data["Close"],
        fast=cfg["macd_fast"],
        slow=cfg["macd_slow"],
        signal=cfg["macd_signal"]
    )
    data["MACD"] = macd_line
    data["MACD_Signal"] = macd_signal
    data["MACD_Hist"] = macd_hist

    # 3. Volatility Indicators
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(
        data["Close"],
        window=cfg["bollinger_window"],
        num_std=cfg["bollinger_std"]
    )
    data["BB_Upper"] = bb_upper
    data["BB_Lower"] = bb_lower
    data["BB_Width"] = bb_width
    data["BB_Pct"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    data["ATR_14"] = calculate_atr(data, period=cfg["atr_period"])
    data["ATR_Pct"] = data["ATR_14"] / data["Close"]

    # Returns & Volatility
    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(periods=5)
    data["Volatility_10"] = data["Return_1d"].rolling(window=cfg["volatility_window"]).std()

    # 4. Price Spread Geometry
    data["High_Low_Spread"] = (data["High"] - data["Low"]) / data["Close"]
    data["Close_Open_Spread"] = (data["Close"] - data["Open"]) / data["Open"]

    # 5. Volume Dynamics
    data["Volume_Change_1d"] = data["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    data["Volume_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / (data["Volume_SMA_20"] + 1e-9)

    # 6. Target Variables
    # Next day close price
    data["Next_Close"] = data["Close"].shift(-1)
    data["Target_Return"] = (data["Next_Close"] - data["Close"]) / data["Close"]
    data["Target"] = (data["Target_Return"] > 0).astype(int)

    # Drop NaNs created by rolling windows and target shift
    data = data.dropna().reset_index(drop=True)

    return data


def get_feature_column_names() -> List[str]:
    """Return canonical list of numeric engineered features for machine learning models."""
    return [
        "Return_1d",
        "Return_5d",
        "Volatility_10",
        "Dist_SMA_10",
        "Dist_SMA_50",
        "SMA_10_50_Ratio",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Width",
        "BB_Pct",
        "ATR_Pct",
        "High_Low_Spread",
        "Close_Open_Spread",
        "Volume_Change_1d",
        "Volume_Ratio"
    ]
