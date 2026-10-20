# Experiment 1: Objective Definition

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 1.5 Hours  
**Document Type**: 1-Page Problem Definition Document  

---

## 1. Case Brief
> *"A retail trading-signals startup wants a model that predicts next-day price direction for a basket of liquid stocks to feed into its alert system."*

In modern financial markets, automated algorithmic signal engines assist retail traders and systematic funds in directional positioning, risk mitigation, and hedging. Retail investors often suffer from emotional biases and information latency. The startup's objective is to ingest daily trading data and emit high-probability directional alerts before market open.

---

## 2. Problem Statement
Given historical daily OHLCV (Open, High, Low, Close, Volume) market data up to trading day $t$ for a designated liquid equity asset (e.g., Apple Inc. `AAPL` or Reliance Industries `RELIANCE.NS`), predict whether the asset's closing price on the next trading day $t+1$ will move **Up** relative to day $t$'s close.

$$\text{Target Variable } Y_t = \begin{cases} 1, & \text{if } \text{Close}_{t+1} > \text{Close}_t \quad (\text{Bullish / Up}) \\ 0, & \text{if } \text{Close}_{t+1} \le \text{Close}_t \quad (\text{Bearish / Down or Flat}) \end{cases}$$

---

## 3. ML Formulation & Justification
- **Formulation**: **Supervised Binary Classification** on engineered time-series features and technical indicators.
- **Why Classification instead of Regression?**
  - Continuous return forecasting $(\hat{r}_{t+1})$ introduces substantial noise due to high market micro-volatility and leptokurtic distribution.
  - Retail trading alerts are fundamentally action-oriented: traders require a clear directional recommendation (`LONG` vs `CASH/SHORT`) accompanied by a calibrated confidence probability $P(Y_t = 1 \mid X_t)$, rather than an uncertain point estimate of fractional returns.
  - Binary classification allows direct threshold tuning to trade off precision against alert volume.

---

## 4. Key Performance Indicators (KPIs) & Success Metrics
1. **Primary KPI: Directional Accuracy (%)**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Class-Specific Metric: F1-Score on the 'Up' Class (Class 1)**:
   $$F_1 = 2 \times \frac{\text{Precision}_1 \times \text{Recall}_1}{\text{Precision}_1 + \text{Recall}_1}$$
   - *Precision on Up*: Ensures alerts do not trigger false long entries leading to capital drawdown.
   - *Recall on Up*: Captures a meaningful proportion of bullish momentum opportunities.
3. **Financial Backtest Metric: Sharpe Ratio & Maximum Drawdown**:
   - Out-of-sample cumulative strategy return when entering long on predicted Up days compared against the passive Buy & Hold benchmark.
4. **Benchmark Baselines**:
   - **Naive Persistence Baseline**: Predicts tomorrow's direction equals today's direction ($Y_t = Y_{t-1}$).
   - **Majority Class Baseline**: Predicts the unconditional historical upward drift of equities (~53% for major indices).
   - *Target Criterion*: Candidate model must statistically beat both persistence and majority baselines on unseen out-of-sample test data.

---

## 5. Stakeholders & Consumers
- **Retail End Users / Subscribers**: Receive daily actionable trade alerts before market session open.
- **Quantitative Research Team**: Develops, audits, and backtests feature representations and model architectures.
- **MLOps / Platform Engineers**: Maintains continuous serving, latency SLAs, model registry, and automated drift alerts.
- **Risk & Compliance Officers**: Audits model drawdowns, position exposures, and regulatory constraints.

---

## 6. Constraints & Risks
- **Temporal Integrity (No Lookahead Bias)**: Features at time $t$ must only use information available at or before session close $t$. Normalization scalers must be fit strictly on historical training splits.
- **Inference Latency SLA**: End-to-end feature extraction and scoring must execute in $<100\text{ ms}$ per ticker.
- **Market Regime Shifts**: Geopolitical events, central bank rate announcements, and black-swan shocks can render historical patterns non-stationary.
- **Transaction Costs & Slippage**: Practical profitability requires signals to overcome typical execution friction (5-10 bps).
