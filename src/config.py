"""Configuration settings for Stock Price Trend Prediction project.
Adheres to CSE4007L Lab Manual requirements.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, NOTEBOOKS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Tickers basket: Liquid NASDAQ & NIFTY 50 stocks covering >= 5 years of daily trading history
DEFAULT_TICKERS = [
    "AAPL",        # Apple Inc. (NASDAQ)
    "MSFT",        # Microsoft Corp. (NASDAQ)
    "GOOGL",       # Alphabet Inc. (NASDAQ)
    "NVDA",        # NVIDIA Corp. (NASDAQ)
    "AMZN",        # Amazon.com Inc. (NASDAQ)
    "RELIANCE.NS", # Reliance Industries Ltd (NIFTY 50)
    "TCS.NS",      # Tata Consultancy Services (NIFTY 50)
    "INFY.NS"      # Infosys Ltd (NIFTY 50)
]

PRIMARY_TICKER = "AAPL"

# Date Range (at least 5 years of historical trading data)
START_DATE = "2019-01-01"
END_DATE = None  # None pulls up to the latest available trading date

# Feature Engineering Parameters
TECHNICAL_INDICATORS = {
    "sma_short": 10,
    "sma_long": 50,
    "ema_short": 20,
    "ema_long": 50,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "bollinger_std": 2,
    "atr_period": 14,
    "volatility_window": 10
}

# Dataset Splitting (Time-series aware: 70% Train, 15% Validation, 15% Test)
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# LSTM Hyperparameters
LSTM_PARAMS = {
    "sequence_length": 20,    # 20 trading days lookback window
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 35,
    "early_stopping_patience": 7
}

# Drift Monitoring Thresholds
DRIFT_PARAMS = {
    "rolling_window": 20,
    "accuracy_alert_threshold": 0.50,  # Alert if rolling accuracy falls below naive baseline
    "ks_significance_level": 0.05
}
