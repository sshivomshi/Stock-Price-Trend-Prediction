# Stock Price Trend Prediction

A machine learning project designed to analyze historical financial data and predict stock price trends.

## Project Structure

```text
Ml Project/
├── data/               # Historical stock prices (raw & processed)
│   ├── raw/
│   └── processed/
├── models/             # Exported model weights and pipeline artifacts
├── notebooks/          # Exploratory Data Analysis & backtesting notebooks
├── src/                # Modular ML pipeline
│   ├── __init__.py
│   ├── dataset.py      # Stock data extraction & feature engineering (indicators, moving averages)
│   └── model.py        # Trend classification/regression models & evaluation
├── .gitignore          # Git ignore rules for Python & ML artifacts
├── requirements.txt    # Project dependencies
├── main.py             # Main entry point to run the trend prediction pipeline
└── README.md           # Project documentation
```

## Getting Started

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```
