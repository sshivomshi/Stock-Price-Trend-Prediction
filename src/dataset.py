import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_sample_stock_data(n_days: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic stock price time series for baseline testing."""
    np.random.seed(random_state)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    returns = np.random.normal(0.0005, 0.02, size=n_days)
    price_series = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "Date": dates,
        "Close": price_series,
        "Volume": np.random.randint(100000, 5000000, size=n_days)
    }).set_index("Date")
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer technical indicators and target trend label (1 = Up, 0 = Down)."""
    data = df.copy()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["Return_1d"] = data["Close"].pct_change()
    data["Volatility_10"] = data["Return_1d"].rolling(window=10).std()
    
    # Target: 1 if next day's close is higher than today's, else 0
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    # Drop NaNs caused by rolling calculations
    data = data.dropna()
    return data


def get_stock_dataset(test_size: float = 0.2):
    """Return train/test splits for stock trend classification."""
    df = create_sample_stock_data()
    featured = extract_features(df)
    
    feature_cols = ["Close", "Volume", "SMA_10", "SMA_50", "Return_1d", "Volatility_10"]
    X = featured[feature_cols]
    y = featured["Target"]
    
    # Time-series aware split (no shuffle to prevent data leakage)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test
