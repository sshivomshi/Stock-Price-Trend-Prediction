"""GOD Trade AI - Industry-Standard AI Stock Prediction and Chart Analysis System.
Integrates Computer Vision, Native OCR, and Trained Machine Learning Models (Random Forest, PyTorch LSTM,
and Technical Indicators) to deliver calibrated, transparent stock movement predictions from chart screenshots.
"""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.chart_analyzer import analyze_stock_chart, analyze_uploaded_csv
from src.config import BASE_DIR, DEFAULT_TICKERS, MODELS_DIR, RAW_DATA_DIR
from src.data_loader import fetch_ticker_data
from src.feature_engineering import engineer_features, get_feature_column_names
from src.image_validator import validate_chart_image
from src.inference import StockTrendPredictor
from src.ocr_engine import extract_raw_text, parse_chart_metadata

# Set page config
st.set_page_config(
    page_title="GOD Trade | AI Stock Prediction & Chart Scanner",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for scan history
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Custom CSS: Restrained Financial Terminal Design with Subtle Background Drift Motion
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-dark: #08080a;
        --card-dark: #101014;
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-green: #00ffa3;
        --accent-red: #ef4444;
        --accent-amber: #f59e0b;
        --accent-blue: #38bdf8;
        --text-dim: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        background-color: var(--bg-dark);
        color: #f1f5f9;
    }

    .stApp {
        background-color: transparent;
    }

    /* -------------------------------------------------------------
       CONTINUOUS AMBIENT BACKGROUND MOTION
       ------------------------------------------------------------- */
    .bg-motion-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: -1;
        pointer-events: none;
        background: #08080a;
    }

    .grid-drift {
        position: absolute;
        top: 0;
        left: 0;
        width: 200%;
        height: 200%;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 64px 64px;
        animation: drift-animation 50s linear infinite;
    }

    @keyframes drift-animation {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-64px, -64px); }
    }

    .glow-orb-1 {
        position: absolute;
        width: 580px;
        height: 580px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0, 255, 163, 0.12), transparent 70%);
        filter: blur(110px);
        top: -160px;
        left: -120px;
        animation: orb-move-1 25s ease-in-out infinite alternate;
    }

    .glow-orb-2 {
        position: absolute;
        width: 620px;
        height: 620px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.10), transparent 70%);
        filter: blur(130px);
        bottom: -220px;
        right: -120px;
        animation: orb-move-2 32s ease-in-out infinite alternate;
    }

    @keyframes orb-move-1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(90px, 70px) scale(1.1); }
        100% { transform: translate(-40px, 110px) scale(0.95); }
    }

    @keyframes orb-move-2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-100px, -50px) scale(1.08); }
        100% { transform: translate(30px, -80px) scale(0.92); }
    }

    /* -------------------------------------------------------------
       NAVIGATION BAR & BRAND
       ------------------------------------------------------------- */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 28px;
        background: rgba(16, 16, 20, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        margin-bottom: 22px;
    }

    .top-brand {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .brand-accent {
        color: #00ffa3;
    }

    .brand-sub {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 10px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .engine-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #00ffa3;
        background: rgba(0, 255, 163, 0.08);
        border: 1px solid rgba(0, 255, 163, 0.25);
        padding: 5px 14px;
        border-radius: 20px;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #00ffa3;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ffa3;
        animation: pulse-ring 2s infinite;
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.9); opacity: 0.8; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.8; }
    }

    /* -------------------------------------------------------------
       SECTION CONTAINERS
       ------------------------------------------------------------- */
    .glass-card {
        background: rgba(16, 16, 20, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 700;
        color: #00ffa3;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 6px;
        display: block;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
    }

    .section-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* -------------------------------------------------------------
       PREDICTION VERDICT CARDS
       ------------------------------------------------------------- */
    .verdict-banner {
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
        border: 1px solid;
    }

    .verdict-upward {
        background: linear-gradient(135deg, rgba(0, 255, 163, 0.12), rgba(16, 24, 39, 0.90));
        border-color: rgba(0, 255, 163, 0.45);
        box-shadow: 0 0 35px rgba(0, 255, 163, 0.16);
    }

    .verdict-downward {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(16, 24, 39, 0.90));
        border-color: rgba(239, 68, 68, 0.45);
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.16);
    }

    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(16, 24, 39, 0.90));
        border-color: rgba(245, 158, 11, 0.45);
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.16);
    }

    .verdict-pill {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-upward {
        background: #00ffa3;
        color: #000000;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.4);
    }

    .pill-downward {
        background: #ef4444;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    .pill-uncertain {
        background: #f59e0b;
        color: #000000;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
    }

    /* Structured Metric Tiles */
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin: 18px 0;
    }

    .metric-tile {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 16px 18px;
    }

    .metric-tile-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .metric-tile-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.01em;
    }

    .metric-tile-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.80rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-top: 26px;
    }
</style>

<!-- BACKGROUND MOTION -->
<div class="bg-motion-container">
    <div class="grid-drift"></div>
    <div class="glow-orb-1"></div>
    <div class="glow-orb-2"></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. TOP HEADER & BRAND BAR
# ==============================================================================
st.markdown("""
<div class="top-header">
    <div class="top-brand">
        GOD<span class="brand-accent">TRADE</span>
        <span class="brand-sub">AI STOCK PREDICTION & CHART ANALYSIS SYSTEM</span>
    </div>
    <div class="engine-badge">
        <span class="pulse-dot"></span>
        CV & ML HYBRID FORECAST ENGINE ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. IMAGE INPUT & SPECIFICATION INTERFACE
# ==============================================================================
st.markdown("""
<div class="glass-card">
    <span class="section-eyebrow">01 // CHART INPUT & RECOGNITION</span>
    <div class="section-title">Upload Stock Chart Screenshot</div>
    <div class="section-desc">
        Accepts candlestick charts, line charts, and technical indicator views (RSI, MACD, Moving Averages).
        The system executes OCR to identify tickers, timeframe, and price boundaries, then combines computer vision
        with trained time-series machine learning models to produce calibrated predictions.
    </div>
</div>
""", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs([
    "Upload Chart Screenshot (PNG / JPG / JPEG)",
    "Live Stock Ticker Scanner"
])

uploaded_img = None
chart_source_label = ""
auto_detected_ticker = ""
auto_detected_tf = "1D (Daily)"
validation_status = None

# ------------------------------------------------------------------------------
# TAB 1: SCREENSHOT UPLOAD
# ------------------------------------------------------------------------------
with input_tab1:
    up_col1, up_col2 = st.columns([1.1, 1])

    with up_col1:
        file_input = st.file_uploader(
            "Drop chart screenshot here",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        st.caption("Or test with verified reference market charts:")
        s_c1, s_c2 = st.columns(2)
        sample_path = None
        with s_c1:
            if st.button("Sample Trendline Chart"):
                sample_path = BASE_DIR / "experiments" / "figures" / "AAPL_price_and_moving_averages.png"
        with s_c2:
            if st.button("Sample Oscillator Chart"):
                sample_path = BASE_DIR / "experiments" / "figures" / "AAPL_oscillators_rsi_macd.png"

        if file_input is not None:
            uploaded_img = Image.open(file_input)
            chart_source_label = "Uploaded Screenshot"
        elif sample_path is not None and sample_path.exists():
            uploaded_img = Image.open(sample_path)
            chart_source_label = f"Sample: {sample_path.stem}"

    with up_col2:
        if uploaded_img is not None:
            # Validate image
            validation_status = validate_chart_image(uploaded_img)
            if validation_status["is_valid"]:
                st.success("Chart validated: " + validation_status["diagnostic_summary"])
            else:
                st.error("Validation Error: " + validation_status["diagnostic_summary"])

            st.image(uploaded_img, caption="Target Stock Chart", use_container_width=True)

            # Auto-extract OCR metadata for pre-filling
            with st.spinner("Extracting text and ticker markers via OCR..."):
                ocr_text = extract_raw_text(uploaded_img)
                ocr_info = parse_chart_metadata(ocr_text)
                if ocr_info.get("detected_ticker"):
                    auto_detected_ticker = ocr_info["detected_ticker"]
                if ocr_info.get("detected_timeframe"):
                    auto_detected_tf = ocr_info["detected_timeframe"]
        else:
            st.info("Upload a chart image on the left or select a sample chart to begin technical analysis.")

    # Manual confirmation / correction form
    if uploaded_img is not None and validation_status and validation_status["is_valid"]:
        st.markdown("<div style='margin-top: 14px; font-weight: 700; color: #fff;'>Confirm or Correct Detected Chart Metadata:</div>", unsafe_allow_html=True)
        meta_c1, meta_c2, meta_c3 = st.columns(3)
        with meta_c1:
            input_ticker = st.text_input("Stock Symbol / Ticker", value=auto_detected_ticker or "AAPL", help="Identified via OCR. You can correct it here.")
        with meta_c2:
            tf_options = ["1D (Daily)", "4H (4 Hours)", "1H (1 Hour)", "15m (15 Minutes)", "5m (5 Minutes)"]
            tf_index = 0
            if auto_detected_tf in tf_options:
                tf_index = tf_options.index(auto_detected_tf)
            input_timeframe = st.selectbox("Timeframe", tf_options, index=tf_index)
        with meta_c3:
            market_options = ["US Equities (NASDAQ / NYSE)", "Indian Equities (NSE / BSE)", "Crypto / Futures", "Forex"]
            market_default_idx = 1 if input_ticker.endswith(".NS") else 0
            input_market = st.selectbox("Market / Exchange", market_options, index=market_default_idx)

        run_scan_btn = st.button("Run Full AI Stock Analysis", type="primary")
        if run_scan_btn:
            with st.spinner("Running Computer Vision geometry, quantitative ML inference, and risk modeling..."):
                analysis_output = analyze_stock_chart(
                    image_input=uploaded_img,
                    user_ticker=input_ticker,
                    user_timeframe=input_timeframe,
                    user_market=input_market
                )
                st.session_state.current_analysis = analysis_output
                # Save to history
                st.session_state.scan_history.insert(0, {
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                    "ticker": analysis_output["detected_stock"]["ticker"],
                    "direction": analysis_output["predicted_direction"],
                    "confidence": analysis_output["model_confidence"]["calibrated_score"],
                    "result": analysis_output
                })


# ------------------------------------------------------------------------------
# TAB 2: LIVE STOCK TICKER SCANNER
# ------------------------------------------------------------------------------
with input_tab2:
    live_c1, live_c2 = st.columns([1, 2])
    with live_c1:
        live_ticker = st.selectbox("Select Asset Ticker", DEFAULT_TICKERS, index=0)
        live_custom = st.text_input("Or enter any ticker (e.g. TSLA, NVDA):", "")
        if live_custom.strip():
            live_ticker = live_custom.strip().upper()
        live_btn = st.button("Generate Live Chart & Run Analysis")

    if live_btn or (uploaded_img is None and st.session_state.current_analysis is None):
        with live_c2:
            with st.spinner(f"Fetching verified OHLCV market data for {live_ticker}..."):
                raw_path = RAW_DATA_DIR / f"{live_ticker.replace('^', '').replace('.', '_')}_raw.csv"
                if raw_path.exists():
                    df_live = pd.read_csv(raw_path)
                    df_live["Date"] = pd.to_datetime(df_live["Date"])
                else:
                    df_live = fetch_ticker_data(live_ticker)

                df_feat = engineer_features(df_live)
                recent_bars = df_feat.tail(120)

                # Render clean chart
                fig, ax = plt.subplots(figsize=(10, 4.2))
                fig.patch.set_facecolor("#101014")
                ax.set_facecolor("#08080a")
                ax.plot(recent_bars["Date"], recent_bars["Close"], color="#00ffa3", linewidth=2, label=f"{live_ticker} Close")
                ax.plot(recent_bars["Date"], recent_bars["SMA_10"], color="#f59e0b", linestyle="--", label="10-day SMA")
                ax.plot(recent_bars["Date"], recent_bars["SMA_50"], color="#8b5cf6", linestyle="-.", label="50-day SMA")
                ax.fill_between(recent_bars["Date"], recent_bars["BB_Upper"], recent_bars["BB_Lower"], color="gray", alpha=0.10, label="Bollinger Bands")
                ax.tick_params(colors="#64748b")
                ax.grid(color="#ffffff", alpha=0.04)
                ax.legend(loc="upper left", facecolor="#141417", edgecolor="none", labelcolor="#fff")
                plt.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=200)
                buf.seek(0)
                plt.close()

                live_img = Image.open(buf)
                live_res = analyze_stock_chart(
                    image_input=live_img,
                    user_ticker=live_ticker,
                    user_timeframe="1D (Daily)",
                    user_market="US Equities" if not live_ticker.endswith(".NS") else "NSE / BSE"
                )
                st.session_state.current_analysis = live_res


# ==============================================================================
# 3. STRUCTURED PREDICTION OUTPUT & VERDICT
# ==============================================================================
res = st.session_state.current_analysis
if res is not None and res.get("status") == "success":
    st.markdown("---")

    direction = res["predicted_direction"]
    is_upward = direction == "Upward"
    is_uncertain = "Uncertain" in direction

    banner_class = "verdict-uncertain" if is_uncertain else ("verdict-upward" if is_upward else "verdict-downward")
    pill_class = "pill-uncertain" if is_uncertain else ("pill-upward" if is_upward else "pill-downward")
    color_accent = "#f59e0b" if is_uncertain else ("#00ffa3" if is_upward else "#ef4444")

    stock_info = res["detected_stock"]
    conf = res["model_confidence"]
    action = res["recommended_action"]
    levels = res["key_price_levels"]
    exp_range = res["expected_price_range"]

    # MAIN EXECUTIVE VERDICT
    st.markdown(f"""
    <div class="verdict-banner {banner_class}">
        <span class="section-eyebrow">02 // EXECUTIVE PREDICTION VERDICT</span>
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin: 12px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 18px;">
                <span class="verdict-pill {pill_class}">{direction}</span>
                <div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #fff;">
                        {stock_info['company_name']} ({stock_info['ticker']})
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                        Source: {stock_info['source']} | Timeframe: {stock_info['timeframe']} | Horizon: {res['forecast_horizon']}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">CALIBRATED CONFIDENCE</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {color_accent};">
                    {conf['calibrated_score']:.1%}
                </div>
                <div style="font-size: 0.72rem; color: #64748b;">Benchmark: {conf['test_accuracy_benchmark']}</div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 10px 18px; border-radius: 10px;">
            <div style="font-size: 0.90rem; color: #e2e8f0;">
                <b>Market Trend:</b> <span style="color: {color_accent}; font-weight: 700;">{res['market_trend'].upper()}</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.90rem; font-weight: 700;">
                RECOMMENDED ACTION: <span style="color: {color_accent};">{action}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. STRUCTURED PREDICTION METRICS (SECTION 3 OF PROMPT)
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-card">
        <span class="section-eyebrow">03 // STRUCTURED ANALYSIS REPORT</span>
        <div class="section-title">Key Price Levels & Probabilistic Ranges</div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">ESTIMATED SUPPORT</div>
            <div class="metric-tile-val" style="color: #00ffa3;">{levels['estimated_support']}</div>
            <div class="metric-tile-sub">Scale: {levels['scale_type']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">ESTIMATED RESISTANCE</div>
            <div class="metric-tile-val" style="color: #ef4444;">{levels['estimated_resistance']}</div>
            <div class="metric-tile-sub">Scale: {levels['scale_type']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">POTENTIAL BREAKOUT</div>
            <div class="metric-tile-val" style="color: #38bdf8;">{levels['potential_breakout']}</div>
            <div class="metric-tile-sub">Upper Volatility Boundary</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        if exp_range:
            range_val = f"{exp_range['low_estimate']} - {exp_range['high_estimate']}"
            range_sub = f"Current: {exp_range['current_price']} (1.5x ATR)"
        else:
            range_val = "Requires Numerical Data"
            range_sub = "Upload lacks verified price feed"

        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">EXPECTED PRICE RANGE</div>
            <div class="metric-tile-val">{range_val}</div>
            <div class="metric-tile-sub">{range_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. TECHNICAL SIGNALS, RISK FACTORS & EXPLANATION
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-card">
        <span class="section-eyebrow">04 // EVIDENCE & JUSTIFICATION</span>
        <div class="section-title">Technical Signals and Risk Assessment</div>
    """, unsafe_allow_html=True)

    ev_c1, ev_c2 = st.columns([1.2, 1])
    with ev_c1:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 8px;'>Key Supporting Signals:</div>", unsafe_allow_html=True)
        for sig in res["key_signals"]:
            st.markdown(f"- {sig}")

        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #fff; margin: 18px 0 8px 0;'>Model Explanation:</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;'>{res['explanation']}</div>", unsafe_allow_html=True)

    with ev_c2:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #ef4444; margin-bottom: 8px;'>Risk Factors & Conflicting Indicators:</div>", unsafe_allow_html=True)
        for risk in res["risk_factors"]:
            st.markdown(f"- {risk}")

        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin: 18px 0 8px 0;'>Observed Values vs Model Estimates:</div>", unsafe_allow_html=True)
        obs = res["observed_vs_estimated"]["observed_from_chart"]
        est = res["observed_vs_estimated"]["model_generated_estimates"]

        comp_df = pd.DataFrame([
            {"Type": "Observed", "Parameter": "Candle Green Ratio", "Value": f"{obs['green_candle_pixel_ratio']:.1%}"},
            {"Type": "Observed", "Parameter": "Candle Red Ratio", "Value": f"{obs['red_candle_pixel_ratio']:.1%}"},
            {"Type": "Observed", "Parameter": "Recent Trajectory Slope", "Value": obs['recent_slope_direction']},
            {"Type": "Model Estimate", "Parameter": "Predicted Direction", "Value": est['predicted_direction']},
            {"Type": "Model Estimate", "Parameter": "Calibrated Confidence", "Value": f"{est['calibrated_confidence']:.1%}"},
            {"Type": "Model Estimate", "Parameter": "Action Recommendation", "Value": est['action_recommendation']}
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 6. ANNOTATED COMPUTER VISION OVERLAY
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-card">
        <span class="section-eyebrow">05 // COMPUTER VISION ANNOTATION</span>
        <div class="section-title">Support, Resistance & Directional Vector Overlay</div>
        <div class="section-desc">
            The computer vision module plots calculated resistance ceilings (top red line), support floors (bottom green line),
            and the directional velocity trajectory vector directly over the target chart image.
        </div>
    """, unsafe_allow_html=True)

    st.image(res["annotated_image"], caption="Computer Vision Detection with Dynamic Boundary Annotation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 7. HISTORICAL MODEL PERFORMANCE & REPORT EXPORT
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-card">
        <span class="section-eyebrow">06 // MODEL VERIFICATION & EXPORT</span>
        <div class="section-title">Historical Backtest Evaluation & Downloadable Report</div>
    """, unsafe_allow_html=True)

    rep_c1, rep_c2 = st.columns([1.2, 1])
    with rep_c1:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 8px;'>Out-of-Sample Model Benchmarks:</div>", unsafe_allow_html=True)
        eval_metrics = [
            {"Model": "Random Forest (Champion)", "Directional Accuracy": "56.79%", "F1 Score": "0.6575", "Sharpe Ratio": "1.56", "Max Drawdown": "-11.01%"},
            {"Model": "Baseline (Majority Class)", "Directional Accuracy": "53.66%", "F1 Score": "0.6984", "Sharpe Ratio": "1.50", "Max Drawdown": "-13.80%"},
            {"Model": "Logistic Regression Baseline", "Directional Accuracy": "48.43%", "F1 Score": "0.4897", "Sharpe Ratio": "1.45", "Max Drawdown": "-11.07%"},
            {"Model": "PyTorch 2-Layer LSTM", "Directional Accuracy": "49.44%", "F1 Score": "0.5897", "Sharpe Ratio": "0.81", "Max Drawdown": "-14.43%"}
        ]
        st.dataframe(pd.DataFrame(eval_metrics), hide_index=True, use_container_width=True)
        st.caption("Evaluated on strict chronological test splits with transaction-cost-aware walk-forward simulation (zero lookahead bias).")

    with rep_c2:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 8px;'>Download Analysis Report:</div>", unsafe_allow_html=True)
        
        # Prepare Markdown Report String
        report_md = f"""# GOD Trade AI - Stock Analysis Report
- **Asset**: {stock_info['company_name']} ({stock_info['ticker']})
- **Market**: {stock_info['market']}
- **Timeframe**: {stock_info['timeframe']}
- **Forecast Horizon**: {res['forecast_horizon']}
- **Timestamp**: {res['timestamp']}

## Executive Verdict
- **Market Trend**: {res['market_trend']}
- **Predicted Direction**: {res['predicted_direction']}
- **Calibrated Confidence**: {conf['calibrated_score']:.1%}
- **Recommended Action**: {action}

## Key Price Levels
- **Estimated Support**: {levels['estimated_support']}
- **Estimated Resistance**: {levels['estimated_resistance']}
- **Breakout Level**: {levels['potential_breakout']}
- **Scale Basis**: {levels['scale_type']}

## Supporting Signals
{chr(10).join(['- ' + s for s in res['key_signals']])}

## Risk Factors
{chr(10).join(['- ' + r for r in res['risk_factors']])}

## Explanation
{res['explanation']}

## Disclaimer
This report is strictly for informational analysis. Financial market predictions are probabilistic and not guaranteed.
"""
        st.download_button(
            label="Download Report (Markdown .md)",
            data=report_md,
            file_name=f"{stock_info['ticker']}_AI_Analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

        # JSON Download
        export_dict = {k: v for k, v in res.items() if k != "annotated_image"}
        st.download_button(
            label="Download Full Data (JSON .json)",
            data=json.dumps(export_dict, indent=2),
            file_name=f"{stock_info['ticker']}_AI_Analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 8. SCAN HISTORY (COMPARE MULTIPLE CHARTS)
# ==============================================================================
if len(st.session_state.scan_history) > 1:
    st.markdown("""
    <div class="glass-card">
        <span class="section-eyebrow">07 // SCAN SESSION HISTORY</span>
        <div class="section-title">Compare Previously Scanned Charts</div>
    """, unsafe_allow_html=True)

    hist_rows = []
    for idx, item in enumerate(st.session_state.scan_history):
        hist_rows.append({
            "Scan #": idx + 1,
            "Timestamp": item["time"],
            "Ticker": item["ticker"],
            "Predicted Direction": item["direction"],
            "Confidence": f"{item['confidence']:.1%}",
            "Action": item["result"]["recommended_action"]
        })
    st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 9. RELIABILITY & SAFETY DISCLAIMER (SECTION 7 OF PROMPT)
# ==============================================================================
st.markdown("""
<div class="disclaimer-box">
    <b>Regulatory and Safety Notice:</b> GOD Trade AI operates as an informational machine learning and computer vision analytics system. 
    Stock market forecasts are probabilistic estimates subject to systemic market risk, macroeconomic regime shifts, and unexpected volatility. 
    Predictions never constitute personalized investment advice, guaranteed returns, or brokerage solicitation. 
    Always apply rigorous risk management, position sizing, and capital preservation protocols.
</div>
""", unsafe_allow_html=True)
