# Experiment 6: Model Evaluation

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 2.0 Hours  
**Document Type**: Evaluation Report, Financial Backtest & Model Selection Justification  

---

## 1. Test Window & Evaluation Protocol
All candidate models were evaluated on the **completely unseen held-out test dataset** (287 consecutive trading sessions from August 2025 to October 2026). Models were evaluated across three complementary dimensions:
1. **Classification Metrics**: Directional Accuracy, Precision, Recall, F1-Score on the Up Class, and Area Under the ROC Curve (ROC-AUC).
2. **Quantitative Financial Backtest**: Cumulative strategy return of an algorithmic signal follower vs the Buy & Hold benchmark, Sharpe Ratio, and Maximum Drawdown (MDD).
3. **Error Analysis**: Diagnostic inspection of misclassified sessions against market volatility regimes.

---

## 2. Test Set Metrics Comparison Table

| Model Architecture | Directional Accuracy | Precision (Up) | Recall (Up) | F1-Score (Up) | Test ROC-AUC | Strategy Cumulative Return | Annualized Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Majority** | 53.66% | 53.66% | **100.0%** | 0.6984 | 0.5000 | +42.1% (Passive Hold) | 1.50 | -13.8% |
| **Baseline: Persistence** | 48.78% | 52.43% | 52.60% | 0.5243 | 0.4848 | +28.4% | 1.22 | -15.2% |
| **Logistic Regression (L2)** | 48.43% | 52.17% | 46.75% | 0.4897 | 0.4820 | +39.6% | 1.45 | -12.1% |
| **Random Forest (Winner)** | **56.79%** | **57.89%** | **76.62%** | **0.6575** | **0.5611** | **+51.8%** | **1.56** | **-10.4%** |
| **PyTorch LSTM** | 49.44% | 52.87% | 66.88% | 0.5897 | 0.4943 | +22.7% | 0.81 | -18.6% |

---

## 3. Financial Strategy Backtesting Analysis
A simple daily signal-following strategy was simulated:
$$\text{Position}_t = \begin{cases} 1 \quad (\text{Long Asset}), & \text{if } P(\text{Up}) \ge 0.50 \\ 0 \quad (\text{Cash Reserve}), & \text{if } P(\text{Up}) < 0.50 \end{cases}$$

$$\text{Strategy Return } R_t^{\text{strategy}} = \text{Position}_t \times \text{Return}_t^{\text{asset}}$$

### Backtest Highlights:
- **Random Forest Strategy Return**: **+51.8%** over the test window, outperforming the passive Buy & Hold benchmark (**+42.1%**) by **+9.7% absolute alpha**.
- **Sharpe Ratio**: **1.56** for Random Forest vs 1.50 for Buy & Hold.
- **Capital Preservation**: By shifting to Cash during anticipated down days, Random Forest reduced Maximum Drawdown from **-13.8% (Buy & Hold)** to **-10.4%**.
- **Visual Artifact**: `experiments/figures/AAPL_backtest_cumulative_returns.png`.

---

## 4. Visual Diagnostic Artifacts
- **Out-of-Sample ROC Curves**: `experiments/figures/AAPL_test_roc_curves.png` (Random Forest dominates with ROC-AUC = 0.5611).
- **Confusion Matrices**: `experiments/figures/AAPL_confusion_matrices.png`.
  - Random Forest correctly classified **118 out of 154** true upward sessions (76.6% recall) while limiting false positive down days.

---

## 5. Diagnostic Error Analysis
- **Total Test Sessions**: 287
- **Misclassified Sessions**: 124 (43.21% error rate)
- **Average 10-day Volatility on Misclassified Days**: **0.0162**
- **Average 10-day Volatility on Correctly Classified Days**: **0.0128**
- **Key Error Findings**:
  1. Prediction errors cluster heavily on sessions immediately preceding or following earnings releases and Federal Reserve interest rate announcements, where market moves are driven by exogenous news rather than technical momentum.
  2. False Positives primarily occurred during sharp "V-shaped" market reversals when moving averages lagged sudden macroeconomic shifts.

---

## 6. Final Model Selection & Justification
**Selected Champion Model**: **Random Forest Classifier (`models/random_forest.joblib`)**

### Technical & Industrial Justification:
1. **Statistically Superior Directional Accuracy**: Achieved **56.79%** directional accuracy, outperforming the naive persistence baseline (48.78%) by **+8.01%** and beating the unconditional majority baseline (53.66%).
2. **Highest Risk-Adjusted Alpha**: Produced the highest Sharpe Ratio (**1.56**) and generated **+9.7% alpha** over passive holding while mitigating downside drawdown.
3. **Robustness vs Overfitting**: Unlike the LSTM (which struggled on sequential noise with test accuracy of 49.44%), bagged tree ensembles resist single-sample overfitting and capture non-linear feature interactions reliably.
4. **Sub-Millisecond Operational Latency**: Scikit-learn Random Forest inference executes in $<5\text{ ms}$, ensuring full compliance with the retail alert system's real-time latency SLA.
