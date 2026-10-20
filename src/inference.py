"""Inference and scoring engine for real-time stock trend prediction.
Addresses Experiment 7: Deployment, Inference Script, and Production Logging.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.config import BASE_DIR, LOGS_DIR, MODELS_DIR, PRIMARY_TICKER, RAW_DATA_DIR
from src.data_loader import fetch_ticker_data
from src.feature_engineering import engineer_features, get_feature_column_names

MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "preprocessor_pipeline.joblib"
LOG_FILE = LOGS_DIR / "prediction_logs.jsonl"


class StockTrendPredictor:
    """Production inference engine for directional stock movement scoring."""

    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}. Run preprocessing first.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = get_feature_column_names()

    def predict(
        self,
        ticker: str = PRIMARY_TICKER,
        custom_df: Optional[pd.DataFrame] = None,
        log_prediction: bool = True
    ) -> Dict:
        """Score next-day directional probability for the given ticker or custom OHLCV data."""
        if custom_df is not None:
            df = custom_df.copy()
        else:
            # Check local raw cache first, or fetch live
            raw_csv = RAW_DATA_DIR / f"{ticker.replace('^', '').replace('.', '_')}_raw.csv"
            if raw_csv.exists():
                df = pd.read_csv(raw_csv)
                df["Date"] = pd.to_datetime(df["Date"])
            else:
                df = fetch_ticker_data(ticker)

        # Feature extraction
        featured_df = engineer_features(df)
        if featured_df.empty:
            raise ValueError(f"Insufficient historical data to calculate technical indicators for {ticker}.")

        latest_row = featured_df.iloc[-1:]
        as_of_date = str(latest_row["Date"].values[0])[:10]
        current_close = float(latest_row["Close"].values[0])

        # Extract features and scale
        X_raw = latest_row[self.feature_cols]
        X_scaled = self.scaler.transform(X_raw)

        # Predict
        prob_up = float(self.model.predict_proba(X_scaled)[0, 1])
        prob_down = 1.0 - prob_up
        target_code = 1 if prob_up >= 0.50 else 0
        direction = "BULLISH" if target_code == 1 else "BEARISH"

        # Determine confidence grade
        max_prob = max(prob_up, prob_down)
        if max_prob >= 0.65:
            confidence = "High"
        elif max_prob >= 0.55:
            confidence = "Moderate"
        else:
            confidence = "Neutral / Low"

        action = "ENTER_LONG" if direction == "BULLISH" and confidence in ["High", "Moderate"] else (
            "ENTER_SHORT / CASH" if direction == "BEARISH" and confidence in ["High", "Moderate"] else "HOLD_CASH"
        )

        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "as_of_date": as_of_date,
            "current_price": round(current_close, 2),
            "prediction": {
                "direction": direction,
                "target_code": target_code,
                "probability_up": round(prob_up, 4),
                "probability_down": round(prob_down, 4),
                "confidence_level": confidence
            },
            "key_indicators": {
                "RSI_14": round(float(latest_row["RSI_14"].values[0]), 2),
                "MACD": round(float(latest_row["MACD"].values[0]), 4),
                "MACD_Signal": round(float(latest_row["MACD_Signal"].values[0]), 4),
                "SMA_10": round(float(latest_row["SMA_10"].values[0]), 2),
                "SMA_50": round(float(latest_row["SMA_50"].values[0]), 2),
                "Return_1d": round(float(latest_row["Return_1d"].values[0]), 4),
                "Volatility_10": round(float(latest_row["Volatility_10"].values[0]), 4)
            },
            "suggested_action": action
        }

        # Log prediction for Experiment 8 monitoring & auditing
        if log_prediction:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(response) + "\n")

        return response


def run_sample_inference(ticker: str = PRIMARY_TICKER) -> Dict:
    """Convenience runner demonstrating production inference."""
    predictor = StockTrendPredictor()
    result = predictor.predict(ticker=ticker)
    print(f"\n[Inference] Successfully generated real-time directional prediction for {ticker}:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_sample_inference()
