"""GOD Trade AI - Stock Trend Prediction & Chart Scanner.
Pure, focused AI chart scanning engine with animated background motion.
Takes user chart input (screenshot or live ticker), predicts next-day direction with high clarity,
and provides deep, step-by-step technical and mathematical justifications.
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
from src.config import BASE_DIR, DEFAULT_TICKERS, LOGS_DIR, RAW_DATA_DIR
from src.data_loader import fetch_ticker_data
from src.feature_engineering import engineer_features, get_feature_column_names
from src.inference import StockTrendPredictor

# Page configuration
st.set_page_config(
    page_title="GOD Trade | AI Stock Chart Scanner",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-End Design: Continuous Background Motion, Dark Glass Cards, High Contrast
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-dark: #08080a;
        --card-dark: #111115;
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-green: #00ffa3;
        --accent-red: #ef4444;
        --accent-blue: #38bdf8;
        --text-dim: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: var(--bg-dark);
        color: #f8fafc;
    }

    .stApp {
        background-color: transparent;
    }

    /* -------------------------------------------------------------
       CONTINUOUS BACKGROUND MOTION ANIMATION
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
            linear-gradient(rgba(255, 255, 255, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
        background-size: 60px 60px;
        animation: drift-animation 45s linear infinite;
    }

    @keyframes drift-animation {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-60px, -60px); }
    }

    .glow-orb-1 {
        position: absolute;
        width: 550px;
        height: 550px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0, 255, 163, 0.18), transparent 70%);
        filter: blur(100px);
        top: -150px;
        left: -100px;
        animation: orb-move-1 22s ease-in-out infinite alternate;
    }

    .glow-orb-2 {
        position: absolute;
        width: 650px;
        height: 650px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.16), transparent 70%);
        filter: blur(120px);
        bottom: -200px;
        right: -100px;
        animation: orb-move-2 28s ease-in-out infinite alternate;
    }

    @keyframes orb-move-1 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(100px, 80px) scale(1.15); }
        100% { transform: translate(-30px, 120px) scale(0.95); }
    }

    @keyframes orb-move-2 {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(-120px, -60px) scale(1.1); }
        100% { transform: translate(40px, -100px) scale(0.9); }
    }

    /* -------------------------------------------------------------
       CLEAN TOP BAR
       ------------------------------------------------------------- */
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 28px;
        background: rgba(15, 15, 18, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        margin-bottom: 24px;
    }

    .top-brand {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .brand-accent {
        color: #00ffa3;
    }

    .engine-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #00ffa3;
        background: rgba(0, 255, 163, 0.1);
        border: 1px solid rgba(0, 255, 163, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
    }

    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #00ffa3;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ffa3;
        animation: pulse-ring 1.8s infinite;
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.9); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.8; }
    }

    /* -------------------------------------------------------------
       SECTION SYSTEM
       ------------------------------------------------------------- */
    .glass-section {
        background: rgba(17, 17, 21, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4);
    }

    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #00ffa3;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 8px;
        display: block;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 20px;
    }

    /* -------------------------------------------------------------
       PREDICTION VERDICT CARDS (HIGH CLARITY)
       ------------------------------------------------------------- */
    .verdict-box {
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
        border: 1px solid;
    }

    .verdict-bullish {
        background: linear-gradient(135deg, rgba(0, 255, 163, 0.12), rgba(15, 23, 42, 0.85));
        border-color: rgba(0, 255, 163, 0.5);
        box-shadow: 0 0 35px rgba(0, 255, 163, 0.18);
    }

    .verdict-bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.85));
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.18);
    }

    .verdict-tag {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 8px;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .tag-bullish {
        background: #00ffa3;
        color: #000000;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.45);
    }

    .tag-bearish {
        background: #ef4444;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.45);
    }

    /* Justification Cards Grid */
    .justification-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        margin-top: 18px;
    }

    .just-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px;
    }

    .just-card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #00ffa3;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }

    .just-card-val {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .just-card-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.45;
    }
</style>

<!-- BACKGROUND IN CONTINUOUS MOTION -->
<div class="bg-motion-container">
    <div class="grid-drift"></div>
    <div class="glow-orb-1"></div>
    <div class="glow-orb-2"></div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 1. TOP BRAND BAR
# ------------------------------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div class="top-brand">
        GOD<span class="brand-accent">TRADE</span>
        <span style="font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-left: 10px; font-weight: 500;">AI CHART SCANNER</span>
    </div>
    <div class="engine-status">
        <span class="status-pulse"></span>
        COMPUTER VISION ENGINE READY
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. INPUT SELECTION (SCREENSHOT UPLOAD VS LIVE TICKER)
# ------------------------------------------------------------------------------
st.markdown("""
<div class="glass-section">
    <span class="section-label">01 // INPUT CHART SOURCE</span>
    <div class="hero-title">Input Stock Chart & Predict Next-Day Direction</div>
    <div class="hero-subtitle">
        Upload a candlestick chart screenshot from TradingView, Zerodha, or any broker, or select a live stock ticker. 
        The vision model analyzes candlestick buying/selling pressure, recent momentum gradients, and support/resistance breakouts.
    </div>
</div>
""", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs([
    "Upload Chart Screenshot (Image)",
    "Live Stock Ticker Scanner"
])

scanned_result = None
image_to_display = None
chart_source_title = ""

# ==============================================================================
# TAB 1: UPLOAD CHART SCREENSHOT
# ==============================================================================
with input_tab1:
    up_c1, up_c2 = st.columns([1, 1.2])

    with up_c1:
        st.markdown('<div class="section-label">DROP CHART IMAGE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Screenshot (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        st.caption("Or test with pre-rendered technical charts:")
        btn1, btn2 = st.columns(2)
        sample_img = None
        with btn1:
            if st.button("Sample Trend Chart"):
                sample_img = BASE_DIR / "experiments" / "figures" / "AAPL_price_and_moving_averages.png"
        with btn2:
            if st.button("Sample Oscillator Chart"):
                sample_img = BASE_DIR / "experiments" / "figures" / "AAPL_oscillators_rsi_macd.png"

        target_img = None
        if uploaded_file is not None:
            target_img = Image.open(uploaded_file)
            chart_source_title = "Uploaded Screenshot"
        elif sample_img is not None and sample_img.exists():
            target_img = Image.open(sample_img)
            chart_source_title = "Sample Technical Chart"

        if target_img is not None:
            st.image(target_img, caption="Input Target Chart", use_container_width=True)

    with up_c2:
        if target_img is not None:
            with st.spinner("Scanning pixel geometry, candlestick volume pressure, and trendline velocity..."):
                scanned_result = analyze_chart_image(target_img)
            image_to_display = scanned_result["annotated_image"]
        else:
            st.info("Drop a chart image on the left or click a sample chart button to trigger instant scanning.")


# ==============================================================================
# TAB 2: LIVE STOCK TICKER SCANNER
# ==============================================================================
with input_tab2:
    tk_c1, tk_c2 = st.columns([1, 2])

    with tk_c1:
        st.markdown('<div class="section-label">SELECT TICKER SYMBOL</div>', unsafe_allow_html=True)
        selected_stock = st.selectbox("Stock Asset", DEFAULT_TICKERS, index=0)
        custom_input = st.text_input("Or enter custom symbol (e.g. TSLA, NVDA):", "")
        if custom_input.strip():
            selected_stock = custom_input.strip().upper()

        scan_btn = st.button("Fetch Live Chart & Scan")

    if scan_btn or selected_stock:
        with tk_c2:
            with st.spinner(f"Fetching live OHLCV market chart for {selected_stock}..."):
                raw_csv = RAW_DATA_DIR / f"{selected_stock.replace('^', '').replace('.', '_')}_raw.csv"
                if raw_csv.exists():
                    df_sym = pd.read_csv(raw_csv)
                    df_sym["Date"] = pd.to_datetime(df_sym["Date"])
                else:
                    df_sym = fetch_ticker_data(selected_stock)

                df_feat_sym = engineer_features(df_sym)
                pred_obj = StockTrendPredictor()
                stock_res = pred_obj.predict(ticker=selected_stock, custom_df=df_sym, log_prediction=True)

            recent_df = df_feat_sym.tail(120)
            fig_t, ax_t = plt.subplots(figsize=(10, 4.5))
            fig_t.patch.set_facecolor("#111115")
            ax_t.set_facecolor("#09090b")

            ax_t.plot(recent_df["Date"], recent_df["Close"], color="#00ffa3", linewidth=2, label=f"{selected_stock} Close")
            ax_t.plot(recent_df["Date"], recent_df["SMA_10"], color="#f59e0b", linestyle="--", label="10-day SMA")
            ax_t.plot(recent_df["Date"], recent_df["SMA_50"], color="#8b5cf6", linestyle="-.", label="50-day SMA")
            ax_t.fill_between(recent_df["Date"], recent_df["BB_Upper"], recent_df["BB_Lower"], color="gray", alpha=0.12, label="Bollinger Bands")
            ax_t.tick_params(colors="#64748b")
            ax_t.grid(color="#ffffff", alpha=0.04)
            ax_t.legend(loc="upper left", facecolor="#141417", edgecolor="none", labelcolor="#fff")
            plt.tight_layout()

            # Save chart to memory buffer to pass through computer vision scanner
            buf = BytesIO()
            fig_t.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            plt.close()

            ticker_img = Image.open(buf)
            chart_source_title = f"{selected_stock} Real-Time Chart"
            scanned_result = analyze_chart_image(ticker_img)
            image_to_display = scanned_result["annotated_image"]


# ------------------------------------------------------------------------------
# 3. HIGH-CLARITY PREDICTION & VERDICT SECTION
# ------------------------------------------------------------------------------
if scanned_result is not None:
    st.markdown("---")

    direction = scanned_result["direction"]
    is_bullish = "UP" in direction
    prob_up = scanned_result["probability_up"]
    prob_down = scanned_result["probability_down"]
    dominant_prob = max(prob_up, prob_down)
    conf_level = scanned_result["confidence_level"]
    action = scanned_result["recommended_action"]

    card_class = "verdict-bullish" if is_bullish else "verdict-bearish"
    tag_class = "tag-bullish" if is_bullish else "tag-bearish"

    # MAIN VERDICT BOX
    st.markdown(f"""
    <div class="verdict-box {card_class}">
        <span class="section-label">02 // PREDICTION VERDICT ({chart_source_title})</span>
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin: 12px 0 18px 0;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span class="verdict-tag {tag_class}">{direction}</span>
                <span style="font-size: 1.1rem; color: #e2e8f0; font-weight: 700;">
                    {dominant_prob:.1%} Directional Probability
                </span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; color: #fff; background: rgba(0,0,0,0.4); padding: 8px 16px; border-radius: 8px;">
                ACTION: <span style="color: {'#00ffa3' if is_bullish else '#ef4444'};">{action}</span>
            </div>
        </div>
        <div style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.5;">
            <b>Direct Summary:</b> Based on algorithmic pattern scanning, this asset exhibits <b>{conf_level}</b> 
            for an immediate <b>{direction}</b> move. The recommended risk-adjusted execution is <b>{action}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. DEEP TECHNICAL & MATHEMATICAL JUSTIFICATION SECTION
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-section">
        <span class="section-label">03 // TECHNICAL & MATHEMATICAL JUSTIFICATION</span>
        <div style="font-size: 1.25rem; font-weight: 800; color: #fff; margin-bottom: 6px;">Why This Prediction Was Made</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">Detailed breakdown of the mathematical parameters, volume pressures, and geometric triggers that generated the signal.</div>
    """, unsafe_allow_html=True)

    # 4 Structured Justification Tiles
    green_pct = scanned_result["green_ratio"] * 100
    red_pct = scanned_result["red_ratio"] * 100
    slope_status = "Positive Upward Velocity" if is_bullish else "Negative Downward Drag"

    st.markdown(f"""
    <div class="justification-grid">
        <div class="just-card">
            <div class="just-card-title">1. CANDLESTICK VOLUME DENSITY</div>
            <div class="just-card-val" style="color: {'#00ffa3' if green_pct >= 50 else '#ef4444'};">
                {green_pct:.1f}% Green vs {red_pct:.1f}% Red
            </div>
            <div class="just-card-desc">
                Measures buying vs selling volume across all detected candle bodies. Dominance above 50% indicates net institutional accumulation.
            </div>
        </div>
        <div class="just-card">
            <div class="just-card-title">2. PRICE MOMENTUM GRADIENT</div>
            <div class="just-card-val">{slope_status}</div>
            <div class="just-card-desc">
                Calculates the horizontal velocity slope across the rightmost 30% of the chart. Detects whether price is printing higher highs or lower lows.
            </div>
        </div>
        <div class="just-card">
            <div class="just-card-title">3. BREAKOUT & BOUNDARY CHANNELS</div>
            <div class="just-card-val">Key Levels Mapped</div>
            <div class="just-card-desc">
                Support (green line) and resistance (red line) are dynamically plotted. Price positioning relative to these levels confirms breakout probability.
            </div>
        </div>
        <div class="just-card">
            <div class="just-card-title">4. RISK MANAGEMENT RULE</div>
            <div class="just-card-val">{action}</div>
            <div class="just-card-desc">
                Stop-loss should be positioned strictly below the detected dynamic support boundary to limit adverse tail-risk exposure.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='font-size: 1rem; color: #fff; margin-top: 22px;'>Step-by-Step Analytical Log:</h4>", unsafe_allow_html=True)
    for pt in scanned_result["analysis_points"]:
        st.markdown(f"- {pt}")

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. ANNOTATED VISION OVERLAY CHART
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="glass-section">
        <span class="section-label">04 // ANNOTATED VISION OVERLAY</span>
        <div style="font-size: 1.15rem; font-weight: 800; color: #fff; margin-bottom: 4px;">Detected Price Levels & Trajectory Vector</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 16px;">
            The computer vision engine draws the calculated Resistance boundary (Top Red Line), Support boundary (Bottom Green Line), 
            and the Directional Momentum Arrow directly over the chart.
        </div>
    """, unsafe_allow_html=True)

    st.image(image_to_display, caption="AI Vision Output with Automated Boundary Annotation", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
