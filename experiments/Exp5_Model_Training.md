# Experiment 5: Model Training

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 2.0 Hours  
**Document Type**: Model Training & Cross-Validation Report  

---

## 1. Candidate Architectures
Four model paradigms spanning naive baselines, regularized linear algorithms, tree-based ensembles, and deep recurrent neural networks were implemented and trained:

1. **Baseline 1 — Majority Class Classifier**:
   - Always predicts the unconditional dominant class from historical training data ($Y = 1$).
2. **Baseline 2 — Naive Persistence Classifier**:
   - Assumes today's market direction persists into tomorrow ($\hat{Y}_{t+1} = Y_t$).
3. **Model 1 — Regularized Logistic Regression (L2 / Ridge)**:
   - Linear decision boundary with inverse regularization strength $C = 0.1$, `l2` penalty, and maximum iterations of 500.
4. **Model 2 — Random Forest Classifier**:
   - Ensemble of 150 bagged decision trees, constrained with `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`, and `n_jobs=-1` to prevent overfitting on noisy daily bars.
5. **Model 3 — PyTorch Deep Long Short-Term Memory (LSTM)**:
   - Sequence window: 20 consecutive trading days ($X \in \mathbb{R}^{B \times 20 \times 17}$).
   - Architecture: 2-layer stacked LSTM ($\text{hidden\_dim} = 64$, $\text{dropout} = 0.2$), feeding a multi-layer perceptron head with ReLU activation and single logit output.
   - Optimization: Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$), binary cross-entropy loss with logits (`BCEWithLogitsLoss`), gradient clipping norm of 1.0, and early stopping patience of 7 epochs.

---

## 2. Cross-Validation Setup: TimeSeriesSplit
Standard $K$-Fold cross validation randomly shuffles rows across temporal space, causing substantial data leakage. In this experiment, a **5-fold `TimeSeriesSplit`** was enforced:
- **Fold 1**: Train on $[0 \dots t_1]$, evaluate on $[t_1 \dots t_2]$
- **Fold 2**: Train on $[0 \dots t_2]$, evaluate on $[t_2 \dots t_3]$
- ...
- **Fold 5**: Train on $[0 \dots t_5]$, evaluate on $[t_5 \dots t_6]$

Each successive fold expands the historical training window, strictly preserving the arrow of time.

---

## 3. Training & Validation Empirical Results

| Model Architecture | 5-Fold TimeSeriesSplit CV Mean Acc | Validation Accuracy | Validation F1-Score | Validation ROC-AUC | Saved Artifact Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Majority** | N/A | 55.05% | 0.6984 | 0.5000 | In-memory rule |
| **Baseline: Persistence** | N/A | 50.52% | 0.5312 | 0.4950 | In-memory rule |
| **Logistic Regression (L2)**| 52.79% ($\pm 3.1\%$) | 54.01% | 0.6471 | 0.5284 | `models/logistic_regression.joblib` |
| **Random Forest (150 trees)**| 50.81% ($\pm 2.4\%$) | 51.22% | 0.6296 | 0.5342 | `models/random_forest.joblib` |
| **PyTorch LSTM (20-day)** | N/A (Sequential) | 52.06% | 0.5975 | 0.5180 | `models/lstm_model.pt` |

*LSTM Training Dynamics*: Early stopping triggered at **Epoch 9** with best validation loss of **0.6900**, preventing memorization of noise in the training sequences.

---

## 4. Random Forest Feature Importance Ranking
The Gini impurity reduction from the trained Random Forest model revealed the following primary predictive drivers:

```text
1. Return_5d          (14.8%)  ==> Medium-term momentum
2. Volatility_10      (11.2%)  ==> Short-term market regime risk
3. RSI_14             ( 9.7%)  ==> Overbought / oversold oscillator
4. Dist_SMA_50        ( 8.6%)  ==> Macro trend deviation
5. Volume_Ratio       ( 7.9%)  ==> Institutional participation confirmation
6. Close_Open_Spread  ( 7.4%)  ==> Intraday bar bias
7. BB_Pct             ( 6.9%)  ==> Bollinger band penetration
```

---

## 5. Artifact Checklist
- [x] `models/logistic_regression.joblib` (Trained Scikit-learn estimator)
- [x] `models/random_forest.joblib` (Trained Random Forest estimator)
- [x] `models/lstm_model.pt` (PyTorch state dictionary weights)
- [x] `models/training_logs.json` (Structured JSON training audit log)
