"""Script to generate all 8 Jupyter Notebooks for college submission (CSE4007L)."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def generate_all():
    # ---------------------------------------------------------
    # Notebook 1: Objective Definition
    # ---------------------------------------------------------
    nb1 = make_notebook([
        md_cell("""# Experiment 1: Objective Definition
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 1.5 Hours

### 1. Case Brief
> "A retail trading-signals startup wants a model that predicts next-day price direction for a basket of liquid stocks to feed into its alert system."

### 2. Mathematical Target Definition
$$Y_t = \\mathbb{I}(\\text{Close}_{t+1} > \\text{Close}_t)$$
Where:
- $Y_t = 1$: Next-day price direction is UP (Bullish)
- $Y_t = 0$: Next-day price direction is DOWN or FLAT (Bearish)"""),
        code_cell("""# Display Problem Formulation Parameters
problem_def = {
    "project_name": "Stock Price Trend Prediction",
    "ml_type": "Supervised Binary Classification",
    "horizon": "1 Trading Day Ahead",
    "primary_asset": "AAPL (Apple Inc.)",
    "basket": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
    "primary_metrics": ["Directional Accuracy (%)", "F1-Score (Up Class)", "Sharpe Ratio"],
    "benchmark": "Naive Persistence Baseline & Majority Class Baseline"
}
import json
print(json.dumps(problem_def, indent=2))""")
    ])
    with open(NOTEBOOKS_DIR / "01_objective_definition.ipynb", "w") as f:
        json.dump(nb1, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 2: Dataset Details
    # ---------------------------------------------------------
    nb2 = make_notebook([
        md_cell("""# Experiment 2: Dataset Details
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Ingest and inspect historical daily OHLCV data covering >= 5 years for liquid stocks via Yahoo Finance API."""),
        code_cell("""import pandas as pd
from src.data_loader import fetch_ticker_data, get_dataset_metadata

# Fetch historical data for primary ticker
df_aapl = fetch_ticker_data("AAPL")
df_aapl.head()"""),
        code_cell("""# Inspect Schema and Metadata
metadata = get_dataset_metadata(df_aapl, "AAPL")
import json
print(json.dumps(metadata, indent=2))"""),
        code_cell("""# Check Class Balance
df_aapl['Target'] = (df_aapl['Close'].shift(-1) > df_aapl['Close']).astype(int)
print("Target Class Distribution:")
print(df_aapl['Target'].value_counts(normalize=True).rename({1: "Up (1)", 0: "Down/Flat (0)"}))""")
    ])
    with open(NOTEBOOKS_DIR / "02_dataset_details.ipynb", "w") as f:
        json.dump(nb2, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 3: Data Preprocessing
    # ---------------------------------------------------------
    nb3 = make_notebook([
        md_cell("""# Experiment 3: Data Pre-processing
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Technical feature engineering, chronological train/val/test splitting, robust scaling, and pipeline persistence."""),
        code_cell("""import pandas as pd
from src.preprocess import load_and_preprocess_data

train_df, val_df, test_df, scaler = load_and_preprocess_data("AAPL")
print(f"Train Shape: {train_df.shape} | Val Shape: {val_df.shape} | Test Shape: {test_df.shape}")"""),
        code_cell("""# Display Engineered Technical Features
from src.feature_engineering import get_feature_column_names
features = get_feature_column_names()
print(f"Total Engineered Features: {len(features)}")
print(train_df[['Date', 'Close', 'SMA_10', 'SMA_50', 'RSI_14', 'MACD', 'Volatility_10', 'Target']].head())""")
    ])
    with open(NOTEBOOKS_DIR / "03_data_preprocessing.ipynb", "w") as f:
        json.dump(nb3, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 4: Exploratory Data Analysis
    # ---------------------------------------------------------
    nb4 = make_notebook([
        md_cell("""# Experiment 4: Data Exploration (EDA)
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Univariate/bivariate analysis, correlation study, moving average overlays, and key quantitative insights."""),
        code_cell("""from src.eda import run_exploratory_analysis
import json

summary = run_exploratory_analysis("AAPL")
print("Key Quantitative Insights:")
for idx, insight in enumerate(summary["insights"], 1):
    print(f"{idx}. {insight}")"""),
        code_cell("""# Display Generated Visualization Files
from IPython.display import Image, display
for fig_path in summary["figures"][:2]:
    print(f"Displaying: {fig_path}")
    display(Image(filename=fig_path))""")
    ])
    with open(NOTEBOOKS_DIR / "04_exploratory_data_analysis.ipynb", "w") as f:
        json.dump(nb4, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 5: Model Training
    # ---------------------------------------------------------
    nb5 = make_notebook([
        md_cell("""# Experiment 5: Model Training
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Train baseline persistence, regularized Logistic Regression, Random Forest, and PyTorch LSTM using TimeSeriesSplit cross validation."""),
        code_cell("""from src.models_training import train_all_models
import json

training_logs = train_all_models("AAPL")
print(json.dumps(training_logs, indent=2))""")
    ])
    with open(NOTEBOOKS_DIR / "05_model_training.ipynb", "w") as f:
        json.dump(nb5, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 6: Model Evaluation & Backtest
    # ---------------------------------------------------------
    nb6 = make_notebook([
        md_cell("""# Experiment 6: Model Evaluation & Financial Backtesting
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Evaluate models on unseen held-out test data, perform strategy backtest vs Buy & Hold, and justify final model selection."""),
        code_cell("""from src.evaluate import evaluate_all_models
import json

eval_results = evaluate_all_models("AAPL")
print(json.dumps(eval_results["final_model_selection"], indent=2))"""),
        code_cell("""# Display Evaluation Metrics Table
models_dict = eval_results["models"]
table_data = []
for m_name, m_data in models_dict.items():
    table_data.append({
        "Model": m_name,
        "Accuracy (%)": f"{m_data.get('accuracy', 0)*100:.2f}%",
        "F1 (Up)": m_data.get("f1", 0),
        "ROC-AUC": m_data.get("roc_auc", 0),
        "Sharpe Ratio": round(m_data.get("backtest", {}).get("sharpe_ratio", 0), 2)
    })
import pandas as pd
pd.DataFrame(table_data)"""),
        code_cell("""# Display ROC Curve and Backtest Returns
from IPython.display import Image, display
display(Image(filename="experiments/figures/AAPL_backtest_cumulative_returns.png"))
display(Image(filename="experiments/figures/AAPL_test_roc_curves.png"))""")
    ])
    with open(NOTEBOOKS_DIR / "06_model_evaluation_and_backtest.ipynb", "w") as f:
        json.dump(nb6, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 7: Deployment & Inference
    # ---------------------------------------------------------
    nb7 = make_notebook([
        md_cell("""# Experiment 7: Deployment & Real-Time Scoring
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Test production inference scoring engine and explore Streamlit dashboard deployment."""),
        code_cell("""from src.inference import StockTrendPredictor
import json

predictor = StockTrendPredictor()
prediction = predictor.predict(ticker="AAPL")
print("Production Real-Time Inference Response:")
print(json.dumps(prediction, indent=2))"""),
        code_cell("""# Instructions to Run Streamlit Dashboard:
print("To launch the interactive dashboard, run the following command in terminal:")
print("streamlit run app.py")""")
    ])
    with open(NOTEBOOKS_DIR / "07_deployment_and_inference.ipynb", "w") as f:
        json.dump(nb7, f, indent=2)

    # ---------------------------------------------------------
    # Notebook 8: Continuous Monitoring & Maintenance
    # ---------------------------------------------------------
    nb8 = make_notebook([
        md_cell("""# Experiment 8: Continuous Update & Maintenance
**Course**: CSE4007L - Applications of Machine Learning in Industry  
**Objective**: Monitor rolling performance degradation, assess covariate feature drift (KS-Test), and execute retraining policies."""),
        code_cell("""from src.drift_monitor import assess_model_and_data_drift
import json

drift_report = assess_model_and_data_drift("AAPL")
print(json.dumps(drift_report, indent=2))""")
    ])
    with open(NOTEBOOKS_DIR / "08_drift_monitoring_maintenance.ipynb", "w") as f:
        json.dump(nb8, f, indent=2)

    print(f"Generated all 8 notebooks successfully in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    generate_all()
