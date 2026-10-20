"""Streamlit interactive web dashboard for real-time Stock Price Trend Prediction.
Addresses Experiment 7: Deployment and Production Serving.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import BASE_DIR, DEFAULT_TICKERS, LOGS_DIR, MODELS_DIR, RAW_DATA_DIR
from src.data_loader import fetch_ticker_data
from src.drift_monitor import assess_model_and_data_drift
from src.feature_engineering import engineer_features, get_feature_column_names
from src.inference import StockTrendPredictor

# Set page configuration
st.set_page_config(
    page_title="Stock Trend Prediction AI | CSE4007L",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge-bullish {
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-bearish {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
    .title-banner {
        padding: 12px 0 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<div class='title-banner'>", unsafe_allow_html=True)
st.title("📈 Stock Price Trend Prediction — AI Trading Signal Dashboard")
st.caption("CSE4007L: Applications of Machine Learning in Industry | Fin-Project 3 — Production Deployment & Signal Engine")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🕹️ Control Panel")
selected_ticker = st.sidebar.selectbox("Select Asset Ticker", DEFAULT_TICKERS, index=0)
custom_ticker = st.sidebar.text_input("Or enter custom ticker symbol:", "")
if custom_ticker.strip():
    selected_ticker = custom_ticker.strip().upper()

model_choice = st.sidebar.selectbox(
    "Active Model Architecture",
    ["Random Forest (Production Champion)", "Logistic Regression (L2)", "PyTorch LSTM (Sequence)"]
)

threshold = st.sidebar.slider("Signal Probability Threshold", min_value=0.50, max_value=0.75, value=0.52, step=0.01)

refresh_btn = st.sidebar.button("🔄 Refresh Market Data & Predict")


# Load Data and Run Inference
@st.cache_data(ttl=3600)
def get_stock_data(ticker: str):
    raw_csv = RAW_DATA_DIR / f"{ticker.replace('^', '').replace('.', '_')}_raw.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = fetch_ticker_data(ticker)
    return df


try:
    df_raw = get_stock_data(selected_ticker)
    df_featured = engineer_features(df_raw)

    predictor = StockTrendPredictor()
    inference_result = predictor.predict(ticker=selected_ticker, custom_df=df_raw, log_prediction=True)

    pred_data = inference_result["prediction"]
    prob_up = pred_data["probability_up"]
    is_bullish = prob_up >= threshold
    direction_label = "BULLISH (UP)" if is_bullish else "BEARISH (DOWN)"
    badge_class = "badge-bullish" if is_bullish else "badge-bearish"

    # KPI Header Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        latest_close = inference_result["current_price"]
        ret_1d = inference_result["key_indicators"]["Return_1d"] * 100
        st.metric("Latest Closing Price", f"${latest_close:.2f}", f"{ret_1d:+.2f}%")

    with c2:
        st.metric("Next-Day Direction Signal", direction_label)

    with c3:
        st.metric("Signal Probability P(Up)", f"{prob_up:.1%}", f"Threshold: {threshold:.0%}")

    with c4:
        action = "ENTER LONG" if is_bullish else "ENTER SHORT / CASH"
        st.metric("Recommended Action", action, f"Confidence: {pred_data['confidence_level']}")

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Technical Analysis & Indicators",
        "📈 Strategy Backtest vs Buy & Hold",
        "🧠 Model Explainability",
        "🛡️ Continuous Drift & Maintenance",
        "🔌 API Documentation & Audit Logs"
    ])

    # Tab 1: Charts
    with tab1:
        st.subheader(f"Historical Trends and Oscillators — {selected_ticker}")
        recent_history = df_featured.tail(250)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        # Price + MAs + Bollinger Bands
        ax1.plot(recent_history["Date"], recent_history["Close"], label="Close Price", color="#1f77b4", linewidth=1.5)
        ax1.plot(recent_history["Date"], recent_history["SMA_10"], label="10-day SMA", color="#ff7f0e", linestyle="--")
        ax1.plot(recent_history["Date"], recent_history["SMA_50"], label="50-day SMA", color="#2ca02c", linestyle="-.")
        ax1.fill_between(recent_history["Date"], recent_history["BB_Upper"], recent_history["BB_Lower"], color="gray", alpha=0.15, label="Bollinger Bands (20, 2σ)")
        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{selected_ticker} Daily Price Action & Moving Averages (Last 250 Sessions)")
        ax1.legend(loc="upper left")

        # RSI Oscillator
        ax2.plot(recent_history["Date"], recent_history["RSI_14"], color="#9467bd", linewidth=1.2, label="RSI (14)")
        ax2.axhline(70, color="crimson", linestyle=":", label="Overbought (70)")
        ax2.axhline(30, color="forestgreen", linestyle=":", label="Oversold (30)")
        ax2.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Date")
        ax2.set_ylim(10, 90)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tab 2: Strategy Backtesting
    with tab2:
        st.subheader("Quantitative Strategy Backtesting on Held-Out Test Window")
        eval_path = MODELS_DIR / "evaluation_summary.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                eval_data = json.load(f)

            rf_eval = eval_data["models"]["random_forest"]
            backtest_data = rf_eval["backtest"]

            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                st.metric("Strategy Cumulative Return", f"{backtest_data['cumulative_return_strategy']:.1%}")
            with bc2:
                st.metric("Benchmark (Buy & Hold)", f"{backtest_data['cumulative_return_benchmark']:.1%}")
            with bc3:
                st.metric("Annualized Sharpe Ratio", f"{backtest_data['sharpe_ratio']:.2f}")
            with bc4:
                st.metric("Maximum Drawdown", f"{backtest_data['max_drawdown']:.1%}")

            # Plot Backtest
            fig_b, ax_b = plt.subplots(figsize=(11, 4.5))
            cum_strat = np.array(backtest_data["cum_strategy_series"]) * 100
            cum_bench = np.array(backtest_data["cum_benchmark_series"]) * 100
            ax_b.plot(cum_strat, label=f"Random Forest Trading Strategy ({cum_strat[-1]:.1f}%)", color="#2ca02c", linewidth=2)
            ax_b.plot(cum_bench, label=f"Buy & Hold Benchmark ({cum_bench[-1]:.1f}%)", color="black", linestyle="--", linewidth=1.5)
            ax_b.set_title("Out-of-Sample Cumulative Returns Comparison (%)", fontweight="bold")
            ax_b.set_ylabel("Cumulative Return (%)")
            ax_b.set_xlabel("Held-Out Test Days")
            ax_b.legend(loc="upper left")
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close()

    # Tab 3: Explainability
    with tab3:
        st.subheader("Model Explainability: Feature Importance Ranking")
        train_log_file = MODELS_DIR / "training_logs.json"
        if train_log_file.exists():
            with open(train_log_file, "r") as f:
                tlogs = json.load(f)
            fi = tlogs["models"]["random_forest"].get("feature_importances", {})
            if fi:
                fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
                ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color="#1f77b4")
                ax_fi.set_title("Random Forest Gini Feature Importance", fontweight="bold")
                ax_fi.set_xlabel("Relative Importance Score")
                plt.tight_layout()
                st.pyplot(fig_fi)
                plt.close()
                st.info("💡 Technical indicators like Return_5d, Volatility_10, and RSI_14 exhibit the highest information gain when forecasting direction.")

    # Tab 4: Drift Monitoring
    with tab4:
        st.subheader("Experiment 8: Continuous Update & Automated Drift Monitoring")
        drift_report = assess_model_and_data_drift(ticker=selected_ticker)

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("Monitoring Status", drift_report["maintenance_recommendation"]["action"])
        with dc2:
            st.metric("Rolling Accuracy (20 Sessions)", f"{drift_report['performance_monitoring']['recent_rolling_accuracy']:.1%}", "Threshold: 50%")
        with dc3:
            st.metric("Drifted Features Detected", f"{drift_report['feature_drift_monitoring']['drifted_features_count']} Features")

        st.markdown(f"**Automated Decision:** `{drift_report['maintenance_recommendation']['message']}`")

        st.write("#### Feature Drift Details (Kolmogorov-Smirnov Two-Sample Test)")
        feat_drift_df = pd.DataFrame.from_dict(drift_report["feature_drift_monitoring"]["feature_details"], orient="index")
        st.dataframe(feat_drift_df, use_container_width=True)

    # Tab 5: API Documentation
    with tab5:
        st.subheader("Production REST / CLI Scoring Specification")
        st.markdown("""
        ### Endpoint: `POST /api/v1/predict`
        Returns next-day directional trend classification for a given stock symbol.

        #### Sample Python Client Request:
        ```python
        from src.inference import StockTrendPredictor

        predictor = StockTrendPredictor()
        result = predictor.predict(ticker="AAPL")
        print(result)
        ```

        #### Sample JSON Output Schema:
        ```json
        {
          "status": "success",
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
            "SMA_10": 340.1,
            "SMA_50": 326.29
          },
          "suggested_action": "HOLD_CASH"
        }
        ```
        """)

        st.write("#### Recent Prediction Audit Logs (`logs/prediction_logs.jsonl`)")
        log_file = LOGS_DIR / "prediction_logs.jsonl"
        if log_file.exists():
            with open(log_file, "r") as f:
                lines = [json.loads(line) for line in f.readlines()[-10:]]
            if lines:
                log_df = pd.DataFrame(lines)
                st.dataframe(log_df[["timestamp", "ticker", "as_of_date", "current_price", "suggested_action"]], use_container_width=True)

except Exception as err:
    st.error(f"Error loading dashboard: {err}")
    st.exception(err)
