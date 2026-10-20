# Experiment 3: Data Pre-processing

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 2.0 Hours  
**Document Type**: Pre-processing Pipeline & Data Splitting Report  

---

## 1. Objectives & Challenges
Financial time series violate the Independent and Identically Distributed (i.i.d.) assumption. Standard machine learning pre-processing steps (such as randomized k-fold splits or whole-dataset global normalization) introduce catastrophic **data leakage (lookahead bias)**, where future market information informs past training instances.

Experiment 3 establishes a mathematically rigorous, leakage-free pipeline:
1. Handling missing data and timestamp discontinuities.
2. Deriving stationary, scale-invariant technical indicators.
3. Enforcing a strict chronological Train / Validation / Test split.
4. Robust feature scaling fitted strictly on training data.
5. Serializing the pipeline artifacts for deployment.

---

## 2. Technical Feature Engineering
Raw asset prices ($P_t$) are non-stationary and exhibit a random walk with drift. To make features stationary, 17 technical indicators spanning trend, momentum, volatility, volume, and geometry were engineered:

### A. Trend & Moving Averages
- **Simple Moving Averages (SMA)**:
  $$\text{SMA}_{k}(t) = \frac{1}{k}\sum_{i=0}^{k-1} \text{Close}_{t-i} \quad \text{for } k \in \{10, 50\}$$
- **Exponential Moving Averages (EMA)**:
  $$\text{EMA}_{\alpha}(t) = \alpha \cdot \text{Close}_t + (1 - \alpha) \cdot \text{EMA}_t \quad (\alpha = 2/(N+1), N \in \{20, 50\})$$
- **Normalized Moving Average Distance**:
  $$\text{Dist\_SMA\_10}_t = \frac{\text{Close}_t - \text{SMA}_{10}(t)}{\text{SMA}_{10}(t)}, \quad \text{SMA\_10\_50\_Ratio}_t = \frac{\text{SMA}_{10}(t)}{\text{SMA}_{50}(t)}$$

### B. Momentum & Oscillators
- **Relative Strength Index (RSI - 14 Days)**:
  $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}, \quad \text{RS} = \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}$$
- **Moving Average Convergence Divergence (MACD)**:
  $$\text{MACD} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})$$
  $$\text{Signal} = \text{EMA}_{9}(\text{MACD}), \quad \text{Hist} = \text{MACD} - \text{Signal}$$

### C. Volatility & Price Geometry
- **Bollinger Bands ($20\text{ days}, 2\sigma$)**:
  $$\text{Upper} = \text{SMA}_{20} + 2\sigma_{20}, \quad \text{Lower} = \text{SMA}_{20} - 2\sigma_{20}$$
  $$\text{BB\_Pct} = \frac{\text{Close} - \text{Lower}}{\text{Upper} - \text{Lower}}, \quad \text{BB\_Width} = \frac{\text{Upper} - \text{Lower}}{\text{SMA}_{20}}$$
- **Average True Range (ATR - 14 Days)**:
  $$\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
- **Rolling Volatility**: Standard deviation of 1-day logarithmic returns over 10 trading sessions.

### D. Volume Dynamics
- **Volume Ratio**: Ratio of current session volume relative to its 20-day simple moving average:
  $$\text{Volume\_Ratio}_t = \frac{\text{Volume}_t}{\text{SMA}_{20}(\text{Volume}_t)}$$

---

## 3. Handling NaN Values & Warm-Up Window
Rolling window computations require up to 50 historical periods to initialize (e.g., 50-day SMA). All leading rows with `NaN` values were dropped systematically. Target calculation ($\text{Close}_{t+1}$) shifts the target back by 1 day, removing the final session from the training set.

---

## 4. Chronological Train / Validation / Test Splitting
Random shuffling was strictly forbidden. The dataset was segmented chronologically using a **70% / 15% / 15%** proportion:

```
[================= TRAIN (70%) =================] [=== VAL (15%) ===] [=== TEST (15%) ===]
2019-03-18 -------------------------> 2024-07-02   2024-07-03 -> 2025-08-27   2025-08-28 -> 2026-10-20
             (1,337 Sessions)                           (287 Sessions)             (287 Sessions)
```

### Class Balance Across Splits:
- **Train Set**: 1,337 sessions | **Up: 53.2%** (711) | **Down: 46.8%** (626)
- **Validation Set**: 287 sessions | **Up: 55.1%** (158) | **Down: 44.9%** (129)
- **Held-Out Test Set**: 287 sessions | **Up: 53.7%** (154) | **Down: 46.3%** (133)

---

## 5. Normalization Pipeline & Artifact Persistence
Financial returns and indicator spreads exhibit fat-tailed (leptokurtic) distributions with extreme outliers. A `RobustScaler` (scaling by median and Interquartile Range $IQR = Q_3 - Q_1$) was utilized:
$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{IQR(x)}$$

- **Fitted strictly on `X_train`**; applied to transform `X_val` and `X_test`.
- Persisted artifact: `models/preprocessor_pipeline.joblib` and `models/AAPL_scaler.joblib`.
- Output datasets: Saved to `data/processed/{ticker}_train.parquet`, `data/processed/{ticker}_val.parquet`, and `data/processed/{ticker}_test.parquet`.
