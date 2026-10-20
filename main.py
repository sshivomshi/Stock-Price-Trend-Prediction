"""Main pipeline orchestrator for Stock Price Trend Prediction.
Executes the complete machine learning lifecycle corresponding to Experiments 1-8.
Course: CSE4007L - Applications of Machine Learning in Industry
"""

import argparse
import sys

from src.config import PRIMARY_TICKER
from src.data_loader import fetch_all_tickers, fetch_ticker_data, get_dataset_metadata
from src.drift_monitor import assess_model_and_data_drift
from src.eda import run_exploratory_analysis
from src.evaluate import evaluate_all_models
from src.inference import run_sample_inference
from src.models_training import train_all_models
from src.preprocess import load_and_preprocess_data


def run_full_pipeline(ticker: str = PRIMARY_TICKER):
    """Run all 8 experiment pipelines sequentially."""
    print("=" * 70)
    print("  CSE4007L: Applications of Machine Learning in Industry")
    print("  Fin-Project 3: Stock Price Trend Prediction Pipeline")
    print(f"  Target Asset: {ticker} | Prediction Horizon: 1-Day Direction")
    print("=" * 70)

    print("\n>>> [Step 1/7 - Exp 2] Fetching Historical OHLCV Data (>= 5 Years)...")
    df_raw = fetch_ticker_data(ticker=ticker)
    meta = get_dataset_metadata(df_raw, ticker=ticker)
    print(f"    Loaded {meta['shape']['rows']} sessions from {meta['date_range']['start']} to {meta['date_range']['end']}.")

    print("\n>>> [Step 2/7 - Exp 3] Running Feature Engineering & Temporal Preprocessing...")
    train_df, val_df, test_df, scaler = load_and_preprocess_data(ticker=ticker)
    print(f"    Engineered train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)}) sets.")

    print("\n>>> [Step 3/7 - Exp 4] Generating Exploratory Data Analysis & Visualizations...")
    eda_summary = run_exploratory_analysis(ticker=ticker)
    print(f"    Saved {len(eda_summary['figures'])} plots to experiments/figures/")

    print("\n>>> [Step 4/7 - Exp 5] Training Candidate Models (Baseline, LR, Random Forest, PyTorch LSTM)...")
    train_logs = train_all_models(ticker=ticker)
    print(f"    Models trained. Saved checkpoints in models/")

    print("\n>>> [Step 5/7 - Exp 6] Evaluating Models on Held-Out Test Window & Strategy Backtest...")
    eval_results = evaluate_all_models(ticker=ticker)
    best_model = eval_results["final_model_selection"]["selected_model"]
    print(f"    Selected Final Model: {best_model}")

    print("\n>>> [Step 6/7 - Exp 7] Running Production Real-Time Inference...")
    sample_pred = run_sample_inference(ticker=ticker)

    print("\n>>> [Step 7/7 - Exp 8] Executing Continuous Drift & Performance Monitoring...")
    drift_report = assess_model_and_data_drift(ticker=ticker)

    print("\n" + "=" * 70)
    print("  Pipeline Execution Complete!")
    print(f"  Final Model: {best_model}")
    print(f"  Test Accuracy: {eval_results['models']['random_forest']['accuracy']:.2%}")
    print(f"  Backtest Sharpe: {eval_results['models']['random_forest']['backtest']['sharpe_ratio']:.2f}")
    print(f"  Drift Status: {drift_report['maintenance_recommendation']['action']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stock Price Trend Prediction ML Pipeline (CSE4007L)")
    parser.add_argument("--ticker", type=str, default=PRIMARY_TICKER, help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--fetch-all", action="store_true", help="Download raw data for all portfolio basket stocks")
    args = parser.parse_args()

    if args.fetch_all:
        print("[Pipeline] Downloading data for full basket of stocks...")
        fetch_all_tickers()

    run_full_pipeline(ticker=args.ticker)


if __name__ == "__main__":
    main()
