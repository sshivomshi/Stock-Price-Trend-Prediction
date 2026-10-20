# Experiment 4: Data Exploration (EDA)

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 2.0 Hours  
**Document Type**: EDA Report & Key Quantitative Insights  

---

## 1. Overview & Objectives
Exploratory Data Analysis was performed on the cleaned pre-processed historical training partition (`AAPL`, 1,337 sessions). The analysis encompassed univariate distribution profiling, bivariate correlation mapping, oscillator behavior analysis, and outlier detection.

---

## 2. Visualizations Generated

| Visualization Artifact | Description & Key Analytical Focus |
| :--- | :--- |
| `experiments/figures/AAPL_price_and_moving_averages.png` | Historical daily close price plotted alongside 10-day SMA, 50-day SMA, and 20-day $2\sigma$ Bollinger Bands. |
| `experiments/figures/AAPL_oscillators_rsi_macd.png` | 14-day RSI (overbought $>70$, oversold $<30$) and MACD line, signal line, and histogram divergence. |
| `experiments/figures/AAPL_feature_correlation_heatmap.png` | Pearson correlation matrix evaluating linear dependencies between all 17 indicators, next-day return, and binary target. |
| `experiments/figures/AAPL_return_distribution.png` | Empirical density distribution of 1-day percentage returns featuring 1st and 99th percentile extreme risk thresholds. |
| `experiments/figures/AAPL_volume_by_direction.png` | Boxplot comparing relative volume ratio distributions across Up sessions vs Down sessions. |

---

## 3. Written List of 6 Key Quantitative Insights

### Insight 1: Positive Market Drift Sets the Baseline
- Across 1,337 training sessions, **53.2% (711)** were Up days and **46.8% (626)** were Down/Flat days.
- *Implication for Modeling*: A naive majority model achieves 53.2% accuracy. Any candidate machine learning model must statistically outperform this baseline on out-of-sample data to claim positive alpha.

### Insight 2: Fat-Tailed (Leptokurtic) Return Distribution
- Daily returns exhibit an empirical **kurtosis of 5.13** (far exceeding the Gaussian normal distribution benchmark of 3.0) with slight positive skewness (0.07).
- *Implication for Modeling*: Extreme daily swings ($> 4\%$) occur more frequently than predicted by normal distribution assumptions. Standard z-score standardization would be distorted by these outliers, justifying the adoption of `RobustScaler`.

### Insight 3: Nonlinear Momentum Regime Switching
- Up days display a mean RSI of **55.6**, whereas Down days show a mean RSI of **55.7**.
- *Implication for Modeling*: The unconditional mean difference in RSI is minimal, because RSI exhibits dual behaviors: strong momentum continuation in moderate regimes ($50 < \text{RSI} < 65$) but sharp mean-reversion pullbacks in overbought regimes ($\text{RSI} > 70$). Linear classifiers struggle to disentangle this without non-linear decision boundaries or tree splits.

### Insight 4: Trend Persistence via Moving Average Ratios
- The ratio $\text{SMA}_{10} / \text{SMA}_{50}$ and the distance $\text{Dist\_SMA\_50}$ serve as reliable proxies for macro regime direction.
- *Implication for Modeling*: When $\text{SMA}_{10} > \text{SMA}_{50}$ (Golden Cross territory), multi-day runs of positive returns cluster, whereas Death Cross environments exhibit heightened volatility and negative drift.

### Insight 5: Volume Ratio as Directional Conviction
- Sessions where $\text{Volume\_Ratio} > 1.5$ (abnormal institutional participation) precede larger next-day absolute returns. High relative volume coupled with a strong positive close ($(\text{Close}-\text{Open})/\text{Open} > 1\%$) strongly favors short-term bullish continuation.

### Insight 6: Need for Ensembles & Recurrent Models
- Individual linear correlation coefficients between raw technical indicators and the next-day binary target are low ($|r| < 0.12$).
- *Implication for Modeling*: Market prices are semi-strong efficient; individual indicators alone do not possess linear predictive power. High-order interaction effects (captured by Random Forest ensembles) and multi-day temporal sequence patterns (captured by LSTMs) are mandatory to extract faint signal from market noise.
