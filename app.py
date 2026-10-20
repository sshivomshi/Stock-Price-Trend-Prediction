"""Streamlit interactive web dashboard for real-time Stock Price Trend Prediction.
Includes AI Chart Vision Analyzer (image & CSV upload) and multi-model technical analysis.
Addresses Experiment 7: Deployment and Production Serving.
"""

from io import BytesIO
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.chart_analyzer import analyze_chart_image, analyze_uploaded_csv
from src.config import BASE_DIR, DEFAULT_TICKERS, LOGS_DIR, MODELS_DIR, RAW_DATA_DIR
from src.data_loader import fetch_ticker_data
from src.drift_monitor import assess_model_and_data_drift
from src.feature_engineering import engineer_features, get_feature_column_names
from src.inference import StockTrendPredictor

# Page configuration
st.set_page_config(
    page_title="Stock Trend Prediction AI | CSE4007L",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-bearish {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
    }
    .title-banner {
        padding: 8px 0 16px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<div class='title-banner'>", unsafe_allow_html=True)
st.title("📈 Stock Price Trend Prediction — AI Trading Signal Dashboard")
st.caption("CSE4007L: Applications of Machine Learning in Industry | Fin-Project 3 — Production Deployment & Signal Engine")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🕹️ Analysis Mode")
input_mode = st.sidebar.radio(
    "Choose Input Source:",
    ["🖼️ Upload Chart Screenshot", "📁 Upload Custom CSV Data", "📊 Market Basket Stocks"]
)

st.sidebar.markdown("---")

if input_mode == "📊 Market Basket Stocks":
    st.sidebar.subheader("Asset Selection")
    selected_ticker = st.sidebar.selectbox("Select Asset Ticker", DEFAULT_TICKERS, index=0)
    custom_ticker = st.sidebar.text_input("Or enter custom symbol (e.g. TSLA, NVDA):", "")
    if custom_ticker.strip():
        selected_ticker = custom_ticker.strip().upper()

    model_choice = st.sidebar.selectbox(
        "Active Model Architecture",
        ["Random Forest (Production Champion)", "Logistic Regression (L2)", "PyTorch LSTM (Sequence)"]
    )
    threshold = st.sidebar.slider("Signal Probability Threshold", min_value=0.50, max_value=0.75, value=0.52, step=0.01)
    refresh_btn = st.sidebar.button("🔄 Refresh Market Data & Predict")
else:
    selected_ticker = "AAPL"
    threshold = 0.52


# ==============================================================================
# MODE 1: UPLOAD CHART SCREENSHOT / IMAGE
# ==============================================================================
if input_mode == "🖼️ Upload Chart Screenshot":
    st.subheader("🖼️ Upload Any Stock Chart Screenshot for AI Trend Prediction")
    st.write(
        "Upload a screenshot of any stock chart (candlestick or line chart from TradingView, Zerodha, Yahoo Finance, or your broker). "
        "The AI Vision engine analyzes candlestick buying/selling pressure, momentum trajectory, and support/resistance breakout levels to predict whether the stock will move **UP** or **DOWN**."
    )

    col_u1, col_u2 = st.columns([1, 1])

    with col_u1:
        uploaded_image = st.file_uploader(
            "Upload Chart Image (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a candlestick or line chart screenshot from any platform."
        )

        st.markdown("**Or test with a sample historical chart:**")
        sample_col1, sample_col2 = st.columns(2)
        sample_path = None
        with sample_col1:
            if st.button("📊 Sample Chart 1 (Price & MAs)"):
                sample_path = BASE_DIR / "experiments" / "figures" / "AAPL_price_and_moving_averages.png"
        with sample_col2:
            if st.button("📉 Sample Chart 2 (Oscillators)"):
                sample_path = BASE_DIR / "experiments" / "figures" / "AAPL_oscillators_rsi_macd.png"

    image_to_process = None
    if uploaded_image is not None:
        image_to_process = Image.open(uploaded_image)
    elif sample_path is not None and sample_path.exists():
        image_to_process = Image.open(sample_path)

    if image_to_process is not None:
        with col_u1:
            st.image(image_to_process, caption="Input Chart Image", use_container_width=True)

        with col_u2:
            with st.spinner("🔍 Analyzing chart structure, candlestick density, and momentum trajectory..."):
                analysis_res = analyze_chart_image(image_to_process, min_confidence=threshold)

            direction = analysis_res["direction"]
            prob_up = analysis_res["probability_up"]
            prob_down = analysis_res["probability_down"]
            is_up = "UP" in direction
            badge_class = "badge-bullish" if is_up else "badge-bearish"

            st.markdown(f"### Predicted Direction: <span class='{badge_class}'>{direction}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence**: `{analysis_res['confidence_level']}` | **P(Up)**: `{prob_up:.1%}` | **P(Down)**: `{prob_down:.1%}`")
            st.markdown(f"**Recommended Action**: **`{analysis_res['recommended_action']}`**")

            # Progress Gauge
            st.progress(prob_up, text=f"Bullish Probability Score: {prob_up:.1%}")

            st.markdown("#### 🔍 Technical Pattern Breakdown")
            for pt in analysis_res["analysis_points"]:
                st.markdown(f"- {pt}")

            st.markdown("#### 🎯 Annotated Chart Vision Overlay")
            st.image(
                analysis_res["annotated_image"],
                caption="AI Vision Overlay: Support (Green), Resistance (Red), Direction Vector",
                use_container_width=True
            )

    else:
        st.info("👆 Please upload a chart image or click a sample chart button above to generate directional analysis.")


# ==============================================================================
# MODE 2: UPLOAD CUSTOM CSV DATA
# ==============================================================================
elif input_mode == "📁 Upload Custom CSV Data":
    st.subheader("📁 Upload Custom Historical Price Data (CSV)")
    st.write("Upload a CSV file containing columns: `Open`, `High`, `Low`, `Close`, `Volume` (and optionally `Date`).")

    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_csv is not None:
        try:
            user_df = pd.read_csv(uploaded_csv)
            st.write("#### Data Preview (Latest 5 Rows):")
            st.dataframe(user_df.tail(5), use_container_width=True)

            with st.spinner("Calculating 17 technical indicators and generating model inference..."):
                res_csv = analyze_uploaded_csv(user_df)

            direction = res_csv["direction"]
            is_bullish = "UP" in direction
            badge_class = "badge-bullish" if is_bullish else "badge-bearish"

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Latest Close", f"${res_csv['latest_close']:.2f}")
            with m2:
                st.markdown(f"**Predicted Direction:**<br><span class='{badge_class}'>{direction}</span>", unsafe_allow_html=True)
            with m3:
                st.metric("Probability P(Up)", f"{res_csv['probability_up']:.1%}")
            with m4:
                st.metric("Confidence", res_csv["confidence_level"])

            # Technical Indicator Summary
            st.markdown("#### Key Technical Indicators:")
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("RSI (14)", f"{res_csv['rsi_14']}")
            ic2.metric("MACD", f"{res_csv['macd']}")
            ic3.metric("SMA (10)", f"${res_csv['sma_10']}")
            ic4.metric("SMA (50)", f"${res_csv['sma_50']}")

            # Plot Candlestick/Close from uploaded data
            st.markdown("#### Chart from Uploaded Data:")
            f_df = res_csv["featured_df"]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(f_df["Date"], f_df["Close"], label="Close Price", color="#1f77b4")
            ax.plot(f_df["Date"], f_df["SMA_10"], label="10-day SMA", color="#ff7f0e", linestyle="--")
            ax.plot(f_df["Date"], f_df["SMA_50"], label="50-day SMA", color="#2ca02c", linestyle="-.")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
    else:
        st.info("👆 Please upload a CSV file with financial time-series data to run predictions.")


# ==============================================================================
# MODE 3: MARKET BASKET STOCKS (FULL LAB SUITE)
# ==============================================================================
else:
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
            st.markdown(f"**Next-Day Direction:**<br><span class='{badge_class}'>{direction_label}</span>", unsafe_allow_html=True)

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
            st.subheader(f"Quantitative Strategy Backtesting on Held-Out Test Window — {selected_ticker}")
            try:
                test_file = BASE_DIR / "data" / "processed" / f"{selected_ticker}_test.parquet"
                if not test_file.exists():
                    from src.preprocess import load_and_preprocess_data
                    load_and_preprocess_data(ticker=selected_ticker)

                test_df_sel = pd.read_parquet(test_file)
                from src.evaluate import calculate_backtest_metrics
                import joblib

                rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
                scaler = joblib.load(MODELS_DIR / "preprocessor_pipeline.joblib")
                feat_cols = get_feature_column_names()

                X_scaled_sel = scaler.transform(test_df_sel[feat_cols])
                preds_sel = rf_model.predict(X_scaled_sel)
                returns_sel = test_df_sel["Target_Return"].values

                backtest_data = calculate_backtest_metrics(returns_sel, preds_sel)

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
                ax_b.set_title(f"{selected_ticker} Out-of-Sample Cumulative Returns Comparison (%)", fontweight="bold")
                ax_b.set_ylabel("Cumulative Return (%)")
                ax_b.set_xlabel("Held-Out Test Days")
                ax_b.legend(loc="upper left")
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close()
            except Exception as e:
                st.warning(f"Could not compute backtest for {selected_ticker}: {e}")

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
            try:
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
            except Exception as e:
                st.warning(f"Could not compute drift report for {selected_ticker}: {e}")

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
