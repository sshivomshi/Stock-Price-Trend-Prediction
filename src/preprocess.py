"""Preprocessing and data splitting module for stock trend prediction.
Addresses Experiment 3: Data Pre-processing and Pipeline Persistence.
"""

from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.config import MODELS_DIR, PRIMARY_TICKER, PROCESSED_DATA_DIR, RAW_DATA_DIR, SPLIT_RATIOS
from src.data_loader import fetch_ticker_data
from src.feature_engineering import engineer_features, get_feature_column_names


def load_and_preprocess_data(
    ticker: str = PRIMARY_TICKER,
    save_artifacts: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Execute feature engineering, temporal train/val/test split, scaling, and artifact persistence.
    
    Returns:
        train_df, val_df, test_df, fitted_scaler
    """
    raw_csv = RAW_DATA_DIR / f"{ticker.replace('^', '').replace('.', '_')}_raw.csv"
    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    else:
        df_raw = fetch_ticker_data(ticker)

    # 1. Feature Engineering
    df_featured = engineer_features(df_raw)
    feature_cols = get_feature_column_names()

    # 2. Strict Chronological Split (No lookahead bias / data leakage)
    n = len(df_featured)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = int(n * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]))

    train_df = df_featured.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_featured.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_featured.iloc[val_end:].copy().reset_index(drop=True)

    # 3. Fitting Scaler strictly on Training Data
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    # Transform features in-place or into new scaled columns
    for df in [train_df, val_df, test_df]:
        scaled_features = scaler.transform(df[feature_cols])
        scaled_col_names = [f"{col}_scaled" for col in feature_cols]
        scaled_df = pd.DataFrame(scaled_features, columns=scaled_col_names, index=df.index)
        for col in scaled_col_names:
            df[col] = scaled_df[col]

    # Verify Class Balances
    print(f"\n[Preprocessing] Class Balance Verification for {ticker}:")
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        up_pct = (subset["Target"] == 1).mean() * 100
        down_pct = 100.0 - up_pct
        print(f"  {name:5s}: Total={len(subset):4d} | Up={up_pct:.1f}% | Down={down_pct:.1f}%")

    # 4. Persist Artifacts
    if save_artifacts:
        scaler_path = MODELS_DIR / f"{ticker}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        # Also persist default global pipeline scaler
        joblib.dump(scaler, MODELS_DIR / "preprocessor_pipeline.joblib")

        train_df.to_parquet(PROCESSED_DATA_DIR / f"{ticker}_train.parquet", index=False)
        val_df.to_parquet(PROCESSED_DATA_DIR / f"{ticker}_val.parquet", index=False)
        test_df.to_parquet(PROCESSED_DATA_DIR / f"{ticker}_test.parquet", index=False)

        train_df.to_csv(PROCESSED_DATA_DIR / f"{ticker}_train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / f"{ticker}_val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / f"{ticker}_test.csv", index=False)

        print(f"[Preprocessing] Preprocessing pipeline saved to {scaler_path}")
        print(f"[Preprocessing] Processed datasets saved to {PROCESSED_DATA_DIR}")

    return train_df, val_df, test_df, scaler


def get_preprocessed_matrices(
    ticker: str = PRIMARY_TICKER
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function returning (X_train, y_train, X_val, y_val, X_test, y_test)."""
    train_df, val_df, test_df, _ = load_and_preprocess_data(ticker=ticker, save_artifacts=False)
    feature_cols = get_feature_column_names()
    scaled_cols = [f"{c}_scaled" for c in feature_cols]

    X_train = train_df[scaled_cols].values
    y_train = train_df["Target"].values

    X_val = val_df[scaled_cols].values
    y_val = val_df["Target"].values

    X_test = test_df[scaled_cols].values
    y_test = test_df["Target"].values

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    train_df, val_df, test_df, scaler = load_and_preprocess_data()
