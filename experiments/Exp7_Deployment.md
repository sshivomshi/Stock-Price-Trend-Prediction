# Experiment 7: Deployment

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 2.5 Hours  
**Document Type**: Deployment Report, API Documentation & User Interface Guide  

---

## 1. Deployment Architecture Overview
To operationalize the final selected Random Forest model into the startup's alert infrastructure, a two-tier serving layer was developed:
1. **Interactive Streamlit Web Dashboard (`app.py`)**: A production-grade visual UI for quantitative analysts and retail users.
2. **Deterministic Inference Scoring Module (`src/inference.py`)**: A low-latency Python/REST programmatic scoring engine with schema validation and automatic audit logging.

```
+-------------------------------------------------------------+
|                     Data Ingestion Tier                     |
|  (Yahoo Finance API / Automated Daily OHLCV Feed / Cache)   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Feature Engineering & Transform               |
|      (17 Technical Indicators + RobustScaler Pipeline)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Champion Inference Model                   |
|          (Random Forest Classifier / models/*.joblib)       |
+------------------------------+------------------------------+
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
+-------------------------+         +-------------------------+
| Streamlit Web Dashboard |         | Programmatic Inference  |
|         (app.py)        |         |    (src/inference.py)   |
| - Live Candlestick/MA   |         | - Structured JSON API   |
| - Prediction Badges     |         | - Audit Log (logs/*.json|
| - Strategy Backtest     |         | - Latency: < 5 ms       |
| - Drift Status Monitor  |         |                         |
+-------------------------+         +-------------------------+
```

---

## 2. Interactive Streamlit Dashboard Features (`app.py`)
- **Asset Selector**: Dropdown supporting both NASDAQ liquid equities (`AAPL`, `MSFT`, `GOOGL`, `NVDA`, `AMZN`) and NIFTY 50 equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`), with custom symbol input.
- **Top KPI Cards**:
  - Latest Closing Price with 1-day percentage movement.
  - Directional Prediction Badge: `🟢 BULLISH (UP)` or `🔴 BEARISH (DOWN)`.
  - Directional Probability $P(\text{Up})$ with user-configurable decision threshold (default $0.52$).
  - Recommended Action: `ENTER LONG`, `ENTER SHORT / CASH`, or `HOLD_CASH`.
- **5 Multi-Function Tabs**:
  1. *Technical Analysis & Indicators*: Dual-axis charts displaying Close price, 10/50 SMAs, Bollinger Bands, and RSI oscillators.
  2. *Strategy Backtest vs Buy & Hold*: Interactive cumulative performance curve showcasing risk-adjusted outperformance and maximum drawdown.
  3. *Model Explainability*: Real-time bar chart of Gini feature importances.
  4. *Continuous Drift & Maintenance*: Live display of rolling 20-day accuracy and KS-test feature drift.
  5. *API Documentation & Audit Logs*: View the live prediction audit log from `logs/prediction_logs.jsonl`.

### How to Launch the Dashboard:
```bash
streamlit run app.py
```

---

## 3. Programmatic API Specification

### Endpoint: `POST /api/v1/predict` (Simulated via `StockTrendPredictor`)

#### Request Schema:
```json
{
  "ticker": "AAPL",
  "as_of_date": "optional, defaults to latest session"
}
```

#### Production Response Schema:
```json
{
  "status": "success",
  "timestamp": "2026-10-20T17:47:45.575098+00:00",
  "ticker": "AAPL",
  "as_of_date": "2026-10-19",
  "current_price": 342.94,
  "prediction": {
    "direction": "BULLISH",
    "target_code": 1,
    "probability_up": 0.5445,
    "probability_down": 0.4555,
    "confidence_level": "Neutral / Low"
  },
  "key_indicators": {
    "RSI_14": 61.2,
    "MACD": 4.1295,
    "MACD_Signal": 3.9818,
    "SMA_10": 340.1,
    "SMA_50": 326.29,
    "Return_1d": -0.0015,
    "Volatility_10": 0.0072
  },
  "suggested_action": "HOLD_CASH"
}
```

---

## 4. Production Audit Logging
Every inference event is appended to `logs/prediction_logs.jsonl` with full feature context, prediction probabilities, and timestamp. This creates an immutable audit trail required for regulatory compliance and downstream drift analysis.
