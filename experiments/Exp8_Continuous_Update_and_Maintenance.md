# Experiment 8: Continuous Update & Maintenance

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 1.5 Hours  
**Document Type**: Monitoring, Maintenance Plan & Drift Detection Document  

---

## 1. Overview & Problem Definition
Financial markets are non-stationary dynamic systems. Models trained on historical data inevitably experience **concept drift** (shifts in the statistical relationship between technical features and next-day returns, $P(Y \mid X)$) and **data drift** (shifts in covariate distributions $P(X)$ driven by changes in volatility or volume regimes).

Experiment 8 defines an automated monitoring and continuous maintenance framework to detect degradation and trigger retraining before trading capital is compromised.

---

## 2. Retraining Triggers & Drift Detection Framework
An automated health-check script (`src/drift_monitor.py`) evaluates three layers of telemetry on a daily cadence:

### A. Performance Degradation Trigger (Rolling Accuracy)
- **Mechanism**: Maintains a rolling 20-trading-day evaluation window of realized directional accuracy:
  $$\text{Acc}_{\text{rolling}} = \frac{1}{20}\sum_{i=1}^{20} \mathbb{I}(\hat{Y}_{t-i} == Y_{t-i})$$
- **Hard Threshold**: If $\text{Acc}_{\text{rolling}} < 50.0\%$ (falling below the naive random benchmark), a `RETRAIN_TRIGGERED` alarm is fired.

### B. Covariate Feature Drift (Kolmogorov-Smirnov Test)
- **Mechanism**: Compares reference training feature distributions ($X_{\text{train}}$) against recent test/inference distributions ($X_{\text{recent}}$) using the non-parametric **Kolmogorov-Smirnov (KS) Two-Sample Test**:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{recent}}(x)|$$
- **Significance Level**: $\alpha = 0.05$. If $p\text{-value} < 0.05$, the feature is flagged as statistically drifted.
- **Trigger**: If $\ge 2$ core features (`Return_1d`, `Volatility_10`, `RSI_14`, `Volume_Ratio`) drift simultaneously, automated model review is initiated.

### C. Market Volatility Regime Shift (Z-Score)
- **Mechanism**: Calculates standardized volatility z-score:
  $$Z_{\text{vol}} = \frac{\mu_{\text{recent\_vol}} - \mu_{\text{hist\_vol}}}{\sigma_{\text{hist\_vol}}}$$
- **Trigger**: $|Z_{\text{vol}}| > 2.0$ indicates an extreme regime shift (e.g. liquidity shock), pausing aggressive long alerts and switching to `HOLD_CASH`.

---

## 3. Empirical Drift Assessment Output
Running `python -m src.drift_monitor` yielded the following audit report:

```json
{
  "timestamp": "2026-10-20T17:47:46.033614+00:00",
  "ticker": "AAPL",
  "evaluation_window_days": 20,
  "performance_monitoring": {
    "recent_rolling_accuracy": 0.5000,
    "baseline_threshold": 0.50,
    "performance_drift_detected": false,
    "status": "PASS"
  },
  "feature_drift_monitoring": {
    "tests_conducted": "Kolmogorov-Smirnov (KS) Two-Sample Test",
    "significance_level": 0.05,
    "drifted_features_count": 3,
    "feature_details": {
      "Return_1d": { "ks_statistic": 0.2850, "p_value": 0.0780, "drift_detected": false },
      "Volatility_10": { "ks_statistic": 0.4420, "p_value": 0.0006, "drift_detected": true },
      "RSI_14": { "ks_statistic": 0.3200, "p_value": 0.0350, "drift_detected": true },
      "Volume_Ratio": { "ks_statistic": 0.3800, "p_value": 0.0050, "drift_detected": true },
      "Dist_SMA_50": { "ks_statistic": 0.2400, "p_value": 0.1980, "drift_detected": false }
    }
  },
  "maintenance_recommendation": {
    "action": "RETRAIN_TRIGGERED",
    "message": "ALERT: Significant feature drift detected in volatility and volume. Trigger automated retraining."
  }
}
```

---

## 4. Retraining Cadence & Champion-Challenger Rollback Plan

### Retraining Cadence:
1. **Scheduled Retraining**: Executed bi-weekly on alternate Friday evenings after market close, incorporating the latest 10 trading sessions into the expanding training window.
2. **Event-Driven Retraining**: Automatically triggered when `src/drift_monitor.py` flags `RETRAIN_TRIGGERED`.

### Champion-Challenger Model Rollback Protocol:
```
+--------------------------+
|  Incoming Data Retrain   |
+------------+-------------+
             |
             v
+--------------------------+
| Candidate Challenger RF  |
+------------+-------------+
             |
             v
+--------------------------+
|  30-Day Sliding Backtest |
|  Evaluation vs Champion  |
+------------+-------------+
             |
     +-------+-------+
     |               |
 [Acc > Champ]  [Acc <= Champ]
     |               |
     v               v
+------------+  +-------------+
|  Promote   |  | Rollback &  |
| Challenger |  | Retain Old  |
| to Prod    |  | Champion    |
+------------+  +-------------+
```
- A candidate model is only promoted to production if its out-of-sample directional accuracy on a 30-day sliding validation window exceeds the current champion by $\ge 1.0\%$ and maintains a Sharpe ratio $\ge 1.20$.
- If candidate fails, the system automatically rolls back to the previous champion checkpoint stored in `models/`.

---

## 5. Ownership, SLAs & Governance Matrix

| Role | Primary Responsibility | SLA / Response Time |
| :--- | :--- | :--- |
| **MLOps Engineer** | Automated pipeline health, latency monitoring, daily drift checks | Alert acknowledgment $< 15\text{ mins}$ |
| **Quantitative Researcher** | Model architecture upgrades, feature selection, backtesting audit | Retraining review $< 24\text{ hours}$ |
| **Risk Manager** | Drawdown auditing, position sizing, volatility kill-switch activation | Immediate intraday halt if MDD $> 15\%$ |
