"""Continuous monitoring and drift detection module for production stock trend models.
Addresses Experiment 8: Continuous Update, Maintenance, and Drift Checking.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

from src.config import BASE_DIR, DRIFT_PARAMS, PRIMARY_TICKER, PROCESSED_DATA_DIR
from src.feature_engineering import get_feature_column_names


def assess_model_and_data_drift(
    ticker: str = PRIMARY_TICKER,
    rolling_window: int = DRIFT_PARAMS["rolling_window"],
    baseline_threshold: float = DRIFT_PARAMS["accuracy_alert_threshold"],
    ks_alpha: float = DRIFT_PARAMS["ks_significance_level"]
) -> Dict:
    """Evaluate performance degradation (rolling accuracy) and covariate feature drift (KS-test)."""
    train_file = PROCESSED_DATA_DIR / f"{ticker}_train.parquet"
    test_file = PROCESSED_DATA_DIR / f"{ticker}_test.parquet"

    if not train_file.exists() or not test_file.exists():
        from src.preprocess import load_and_preprocess_data
        load_and_preprocess_data(ticker=ticker)

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)
    feature_cols = get_feature_column_names()

    # Load evaluated predictions from test_df (or generate using saved model)
    eval_file = BASE_DIR / "models" / "evaluation_summary.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
    else:
        eval_data = {}

    # 1. Performance Drift Assessment (Rolling Accuracy on most recent test window)
    # Simulate recent rolling window of realized predictions
    recent_window = test_df.tail(rolling_window)
    # Using Random Forest predictions on recent window
    rf_model = joblib.load(BASE_DIR / "models" / "random_forest.joblib")
    scaler = joblib.load(BASE_DIR / "models" / "preprocessor_pipeline.joblib")

    X_recent_raw = recent_window[feature_cols]
    X_recent_scaled = scaler.transform(X_recent_raw)
    y_recent_true = recent_window["Target"].values
    y_recent_pred = rf_model.predict(X_recent_scaled)

    recent_accuracy = float(accuracy_score(y_recent_true, y_recent_pred))
    performance_drift_detected = recent_accuracy < baseline_threshold

    # 2. Covariate Feature Drift via Kolmogorov-Smirnov (KS) Two-Sample Test
    key_features = ["Return_1d", "Volatility_10", "RSI_14", "Volume_Ratio", "Dist_SMA_50"]
    drift_details = {}
    drifted_features_count = 0

    for feat in key_features:
        ref_values = train_df[feat].values
        curr_values = recent_window[feat].values

        stat, p_val = ks_2samp(ref_values, curr_values)
        is_drifted = bool(p_val < ks_alpha)
        if is_drifted:
            drifted_features_count += 1

        drift_details[feat] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": is_drifted
        }

    # 3. Regime Shift Detection (Volatility Spikes)
    hist_vol_mean = float(train_df["Volatility_10"].mean())
    hist_vol_std = float(train_df["Volatility_10"].std())
    recent_vol_mean = float(recent_window["Volatility_10"].mean())
    z_vol = (recent_vol_mean - hist_vol_mean) / (hist_vol_std + 1e-9)
    regime_shift_detected = bool(z_vol > 2.0 or z_vol < -2.0)

    # 4. Trigger Retraining Cadence & Action Decision
    if performance_drift_detected or drifted_features_count >= 2:
        action_required = "RETRAIN_TRIGGERED"
        status_message = "ALERT: Performance degradation or significant feature drift detected. Trigger automated retraining."
    elif regime_shift_detected or drifted_features_count == 1:
        action_required = "MONITOR_CLOSELY"
        status_message = "WARNING: Potential market regime shift or minor drift detected. Maintain observation over next 5 trading days."
    else:
        action_required = "HEALTHY"
        status_message = "NORMAL: Model performance and feature distributions are aligned with training baselines."

    drift_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "evaluation_window_days": rolling_window,
        "performance_monitoring": {
            "recent_rolling_accuracy": round(recent_accuracy, 4),
            "baseline_threshold": baseline_threshold,
            "performance_drift_detected": performance_drift_detected,
            "status": "FAIL" if performance_drift_detected else "PASS"
        },
        "feature_drift_monitoring": {
            "tests_conducted": "Kolmogorov-Smirnov (KS) Two-Sample Test",
            "significance_level": ks_alpha,
            "drifted_features_count": drifted_features_count,
            "feature_details": drift_details
        },
        "regime_shift_monitoring": {
            "historical_mean_volatility": round(hist_vol_mean, 4),
            "recent_mean_volatility": round(recent_vol_mean, 4),
            "z_score": round(z_vol, 2),
            "regime_shift_detected": regime_shift_detected
        },
        "maintenance_recommendation": {
            "action": action_required,
            "message": status_message,
            "retraining_cadence": "Bi-weekly scheduled, or event-driven upon drift alert",
            "rollback_plan": "Revert to champion Random Forest checkpoint if challenger accuracy on sliding 30-day window is lower."
        }
    }

    report_file = BASE_DIR / "experiments" / "drift_monitoring_report.json"
    with open(report_file, "w") as f:
        json.dump(drift_report, f, indent=2)

    print(f"\n[DriftMonitor] Drift assessment completed! Report saved to {report_file}")
    print(f"  Status: {action_required}")
    print(f"  Rolling Accuracy (last {rolling_window} days): {recent_accuracy:.2%} (Threshold: {baseline_threshold:.2%})")
    print(f"  Drifted Features: {drifted_features_count}/{len(key_features)}")
    print(f"  Recommendation: {status_message}")

    return drift_report


if __name__ == "__main__":
    assess_model_and_data_drift()
