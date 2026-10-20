"""Model training module implementing Baseline, Logistic Regression, Random Forest, and PyTorch LSTM.
Addresses Experiment 5: Model Training and Cross-Validation.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LSTM_PARAMS, MODELS_DIR, PRIMARY_TICKER, PROCESSED_DATA_DIR
from src.feature_engineering import get_feature_column_names


# -------------------------------------------------------------
# 1. PyTorch LSTM Architecture
# -------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Take the output of the final time step
        last_step = out[:, -1, :]
        logits = self.fc(last_step).squeeze(-1)
        return logits


def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for recurrent LSTM training."""
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i : i + seq_len])
        y_seq.append(targets[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# -------------------------------------------------------------
# 2. Training Pipelines
# -------------------------------------------------------------
def train_all_models(ticker: str = PRIMARY_TICKER) -> Dict:
    """Train Baseline, Logistic Regression, Random Forest, and LSTM with TimeSeriesSplit."""
    train_file = PROCESSED_DATA_DIR / f"{ticker}_train.parquet"
    val_file = PROCESSED_DATA_DIR / f"{ticker}_val.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"Processed files not found. Run preprocessing first.")

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)

    feature_cols = get_feature_column_names()
    scaled_cols = [f"{c}_scaled" for c in feature_cols]

    X_train = train_df[scaled_cols].values
    y_train = train_df["Target"].values
    X_val = val_df[scaled_cols].values
    y_val = val_df["Target"].values

    training_logs = {
        "ticker": ticker,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "models": {}
    }

    # ---------------------------------------------------------
    # Baseline: Naive Persistence & Majority Class
    # ---------------------------------------------------------
    majority_class = int(np.round(np.mean(y_train)))
    y_val_pred_majority = np.full_like(y_val, majority_class)
    majority_acc = float(accuracy_score(y_val, y_val_pred_majority))
    majority_f1 = float(f1_score(y_val, y_val_pred_majority, zero_division=0))

    # Persistence baseline (today's direction predicts tomorrow)
    y_val_pred_persistence = np.roll(y_val, 1)
    y_val_pred_persistence[0] = majority_class
    persist_acc = float(accuracy_score(y_val, y_val_pred_persistence))
    persist_f1 = float(f1_score(y_val, y_val_pred_persistence, zero_division=0))

    training_logs["models"]["baseline_majority"] = {
        "val_accuracy": round(majority_acc, 4),
        "val_f1": round(majority_f1, 4),
        "description": "Predicts majority class of training data"
    }
    training_logs["models"]["baseline_persistence"] = {
        "val_accuracy": round(persist_acc, 4),
        "val_f1": round(persist_f1, 4),
        "description": "Naive persistence baseline (today direction predicts tomorrow)"
    }
    print(f"[Training] Baseline Majority Val Acc: {majority_acc:.4f} | Persistence Val Acc: {persist_acc:.4f}")

    # ---------------------------------------------------------
    # TimeSeriesSplit Cross Validation Setup
    # ---------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)

    # ---------------------------------------------------------
    # Model 1: Logistic Regression
    # ---------------------------------------------------------
    print("\n[Training] Training Logistic Regression with L2 Regularization & TimeSeriesSplit CV...")
    lr_cv_scores = []
    for train_idx, cv_idx in tscv.split(X_train):
        lr_fold = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        lr_fold.fit(X_train[train_idx], y_train[train_idx])
        fold_acc = lr_fold.score(X_train[cv_idx], y_train[cv_idx])
        lr_cv_scores.append(fold_acc)

    lr_model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    lr_model.fit(X_train, y_train)

    lr_val_preds = lr_model.predict(X_val)
    lr_val_probs = lr_model.predict_proba(X_val)[:, 1]
    lr_acc = float(accuracy_score(y_val, lr_val_preds))
    lr_f1 = float(f1_score(y_val, lr_val_preds, zero_division=0))
    lr_auc = float(roc_auc_score(y_val, lr_val_probs))

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    training_logs["models"]["logistic_regression"] = {
        "cv_mean_accuracy": round(float(np.mean(lr_cv_scores)), 4),
        "cv_std_accuracy": round(float(np.std(lr_cv_scores)), 4),
        "val_accuracy": round(lr_acc, 4),
        "val_f1": round(lr_f1, 4),
        "val_roc_auc": round(lr_auc, 4),
        "artifact": "models/logistic_regression.joblib"
    }
    print(f"[Training] Logistic Regression -> CV Mean Acc: {np.mean(lr_cv_scores):.4f} | Val Acc: {lr_acc:.4f} | Val F1: {lr_f1:.4f}")

    # ---------------------------------------------------------
    # Model 2: Random Forest Classifier
    # ---------------------------------------------------------
    print("\n[Training] Training Random Forest Classifier with TimeSeriesSplit CV...")
    rf_cv_scores = []
    for train_idx, cv_idx in tscv.split(X_train):
        rf_fold = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_fold.fit(X_train[train_idx], y_train[train_idx])
        fold_acc = rf_fold.score(X_train[cv_idx], y_train[cv_idx])
        rf_cv_scores.append(fold_acc)

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_val_preds = rf_model.predict(X_val)
    rf_val_probs = rf_model.predict_proba(X_val)[:, 1]
    rf_acc = float(accuracy_score(y_val, rf_val_preds))
    rf_f1 = float(f1_score(y_val, rf_val_preds, zero_division=0))
    rf_auc = float(roc_auc_score(y_val, rf_val_probs))

    feature_importances = {
        feature_cols[i]: round(float(rf_model.feature_importances_[i]), 4)
        for i in np.argsort(rf_model.feature_importances_)[::-1]
    }

    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")
    training_logs["models"]["random_forest"] = {
        "cv_mean_accuracy": round(float(np.mean(rf_cv_scores)), 4),
        "cv_std_accuracy": round(float(np.std(rf_cv_scores)), 4),
        "val_accuracy": round(rf_acc, 4),
        "val_f1": round(rf_f1, 4),
        "val_roc_auc": round(rf_auc, 4),
        "feature_importances": feature_importances,
        "artifact": "models/random_forest.joblib"
    }
    print(f"[Training] Random Forest -> CV Mean Acc: {np.mean(rf_cv_scores):.4f} | Val Acc: {rf_acc:.4f} | Val F1: {rf_f1:.4f}")

    # ---------------------------------------------------------
    # Model 3: Deep Learning LSTM (PyTorch)
    # ---------------------------------------------------------
    print("\n[Training] Training Deep Learning LSTM Network on 20-day Sliding Windows...")
    seq_len = LSTM_PARAMS["sequence_length"]
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len=seq_len)

    train_dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
    val_dataset = TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq))

    train_loader = DataLoader(train_dataset, batch_size=LSTM_PARAMS["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_PARAMS["batch_size"], shuffle=False)

    torch.manual_seed(42)
    device = torch.device("cpu")
    lstm_model = StockLSTM(
        input_dim=X_train.shape[1],
        hidden_dim=LSTM_PARAMS["hidden_dim"],
        num_layers=LSTM_PARAMS["num_layers"],
        dropout=LSTM_PARAMS["dropout"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LSTM_PARAMS["learning_rate"], weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(1, LSTM_PARAMS["epochs"] + 1):
        lstm_model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = lstm_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_dataset)

        # Validation Step
        lstm_model.eval()
        val_loss = 0.0
        val_preds_all, val_probs_all, val_true_all = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = lstm_model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

                val_probs_all.extend(probs.cpu().numpy())
                val_preds_all.extend(preds.cpu().numpy())
                val_true_all.extend(batch_y.cpu().numpy())

        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = lstm_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= LSTM_PARAMS["early_stopping_patience"]:
                print(f"  [LSTM] Early stopping triggered at epoch {epoch}")
                break

    if best_weights is not None:
        lstm_model.load_state_dict(best_weights)

    lstm_val_acc = float(accuracy_score(val_true_all, val_preds_all))
    lstm_val_f1 = float(f1_score(val_true_all, val_preds_all, zero_division=0))
    lstm_val_auc = float(roc_auc_score(val_true_all, val_probs_all))

    torch.save(lstm_model.state_dict(), MODELS_DIR / "lstm_model.pt")
    training_logs["models"]["lstm"] = {
        "val_loss": round(float(best_val_loss), 4),
        "val_accuracy": round(lstm_val_acc, 4),
        "val_f1": round(lstm_val_f1, 4),
        "val_roc_auc": round(lstm_val_auc, 4),
        "artifact": "models/lstm_model.pt"
    }
    print(f"[Training] PyTorch LSTM -> Val Loss: {best_val_loss:.4f} | Val Acc: {lstm_val_acc:.4f} | Val F1: {lstm_val_f1:.4f}")

    # Save summary log
    log_file = MODELS_DIR / "training_logs.json"
    with open(log_file, "w") as f:
        json.dump(training_logs, f, indent=2)
    print(f"\n[Training] All models trained and training logs saved to {log_file}")

    return training_logs


if __name__ == "__main__":
    train_all_models()
