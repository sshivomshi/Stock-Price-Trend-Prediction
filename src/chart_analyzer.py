"""Chart Vision and Technical Pattern Analysis Engine.
Analyzes user-uploaded stock chart screenshots (candlestick/line charts) or custom time-series data
to determine next-day directional trend (UP vs DOWN) with complete technical breakdown.
"""

from io import BytesIO
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def analyze_chart_image(
    image_input: Union[str, bytes, Image.Image],
    min_confidence: float = 0.50
) -> Dict:
    """Analyze a stock chart screenshot using computer vision and pattern recognition.
    
    Extracts:
    1. Bullish (green) vs Bearish (red) candlestick pixel ratio
    2. Recent price trajectory slope (time-series gradient from left to right)
    3. Support and resistance levels
    4. Price breakout / breakdown patterns
    5. Annotated visual feedback image
    """
    if isinstance(image_input, (str, bytes)):
        img = Image.open(image_input if isinstance(image_input, str) else BytesIO(image_input))
    else:
        img = image_input

    img_rgb = img.convert("RGB")
    width, height = img_rgb.size

    # Convert to NumPy array
    arr = np.array(img_rgb, dtype=np.int32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Mask out typical white or dark background
    brightness = (r + g + b) / 3.0
    is_not_bg = (brightness > 25) & (brightness < 240)

    # 1. Color-based Candlestick Sentiment Extraction
    # Bullish candles: Green dominates
    green_mask = is_not_bg & (g > r + 15) & (g > b + 10)
    # Bearish candles: Red dominates
    red_mask = is_not_bg & (r > g + 15) & (r > b + 10)

    green_pixel_count = int(np.sum(green_mask))
    red_pixel_count = int(np.sum(red_mask))
    total_colored_pixels = green_pixel_count + red_pixel_count + 1e-6

    green_ratio = green_pixel_count / total_colored_pixels
    red_ratio = red_pixel_count / total_colored_pixels

    # 2. Extract Price Curve Across Horizontal Axis (Time Axis)
    # Candlestick or line pixels
    chart_content_mask = green_mask | red_mask
    if np.sum(chart_content_mask) < 200:
        # Fallback to general colored/darker foreground pixels if line chart
        chart_content_mask = is_not_bg & (np.abs(r - b) > 10) | (np.abs(g - b) > 10)

    # Profile horizontal columns from left to right
    x_indices = []
    y_centers = []

    # Exclude axis edges (5% left, 5% right, 5% top, 10% bottom for volume/date labels)
    x_start = int(width * 0.05)
    x_end = int(width * 0.95)
    y_min = int(height * 0.05)
    y_max = int(height * 0.85)

    step = max(1, width // 100)
    for x in range(x_start, x_end, step):
        col_pixels = np.where(chart_content_mask[y_min:y_max, x])[0]
        if len(col_pixels) > 0:
            # Average vertical position in column
            # In image coordinates, y=0 is TOP (high price) and y=height is BOTTOM (low price)
            y_avg = float(np.median(col_pixels)) + y_min
            x_indices.append(x)
            y_centers.append(y_avg)

    # Calculate Trend Slope
    if len(x_indices) >= 5:
        x_arr = np.array(x_indices)
        # Invert y so higher value = higher price
        price_arr = height - np.array(y_centers)

        # Full chart trend slope
        full_slope, _ = np.polyfit(x_arr, price_arr, 1)

        # Recent 30% trend (latest price momentum)
        recent_cutoff = int(len(x_arr) * 0.70)
        recent_x = x_arr[recent_cutoff:]
        recent_price = price_arr[recent_cutoff:]

        if len(recent_x) >= 3:
            recent_slope, _ = np.polyfit(recent_x, recent_price, 1)
        else:
            recent_slope = full_slope

        # Highs and Lows in recent segment
        recent_high = float(np.max(recent_price))
        recent_low = float(np.min(recent_price))
        last_price = float(recent_price[-1])
        first_recent_price = float(recent_price[0])
        recent_change_pct = (last_price - first_recent_price) / (first_recent_price + 1e-6)

        # Resistance and Support detection
        resistance_y = height - recent_high
        support_y = height - recent_low
    else:
        full_slope = 0.0
        recent_slope = 0.0
        recent_change_pct = 0.0
        resistance_y = height * 0.3
        support_y = height * 0.7

    # 3. Decision Logic & Signal Weighting
    # Score between 0.0 (Strongly Bearish) and 1.0 (Strongly Bullish)
    sentiment_score = 0.50

    # Weight 1: Candlestick color balance (35% weight)
    # 50% is neutral; >50% green is bullish
    color_score = green_ratio
    sentiment_score += (color_score - 0.50) * 0.40

    # Weight 2: Recent momentum slope (40% weight)
    # Standardize slope relative to image height
    norm_recent_slope = recent_slope / (height * 0.005 + 1e-6)
    slope_contribution = np.clip(norm_recent_slope * 0.15, -0.25, 0.25)
    sentiment_score += slope_contribution

    # Weight 3: Price relative to Support/Resistance (25% weight)
    if len(x_indices) >= 5:
        range_span = (recent_high - recent_low) + 1e-6
        position_in_range = (last_price - recent_low) / range_span  # 0 to 1
        # Close near resistance with upward momentum indicates breakout potential
        if position_in_range > 0.70 and recent_slope > 0:
            sentiment_score += 0.10
        elif position_in_range < 0.30 and recent_slope < 0:
            sentiment_score -= 0.10

    # Bound probability
    prob_up = float(np.clip(sentiment_score, 0.10, 0.90))
    prob_down = 1.0 - prob_up

    is_bullish = prob_up >= min_confidence
    direction = "BULLISH (UP)" if is_bullish else "BEARISH (DOWN)"
    target_code = 1 if is_bullish else 0

    # Pattern Identification
    patterns = []
    if green_ratio > 0.60:
        patterns.append(f"Dominant Bullish Buying Pressure ({green_ratio:.1%} green candle volume)")
    elif red_ratio > 0.60:
        patterns.append(f"Heavy Bearish Selling Pressure ({red_ratio:.1%} red candle volume)")

    if recent_slope > 0.15:
        patterns.append("Ascending Higher Highs / Bullish Trendline Momentum")
    elif recent_slope < -0.15:
        patterns.append("Descending Lower Lows / Bearish Downward Momentum")
    else:
        patterns.append("Horizontal Consolidation / Equilibrium Range")

    if recent_change_pct > 0.02:
        patterns.append(f"Recent Right-Side Price Expansion (+{recent_change_pct:.1%})")
    elif recent_change_pct < -0.02:
        patterns.append(f"Recent Right-Side Price Contraction ({recent_change_pct:.1%})")

    # Determine confidence grade
    dominant_prob = max(prob_up, prob_down)
    if dominant_prob >= 0.68:
        conf_label = "High Confidence"
    elif dominant_prob >= 0.56:
        conf_label = "Moderate Confidence"
    else:
        conf_label = "Neutral / Low Conviction"

    action = "BUY / ENTER LONG" if is_bullish and dominant_prob >= 0.55 else (
        "SELL / ENTER SHORT" if not is_bullish and dominant_prob >= 0.55 else "WAIT / HOLD CASH"
    )

    # 4. Create Visual Annotation Overlay on Chart
    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)

    # Draw Support and Resistance lines
    draw.line([(0, int(resistance_y)), (width, int(resistance_y))], fill=(220, 53, 69, 200), width=2)
    draw.line([(0, int(support_y)), (width, int(support_y))], fill=(40, 167, 69, 200), width=2)

    # Draw Recent Direction Vector Arrow
    arrow_color = (40, 167, 69) if is_bullish else (220, 53, 69)
    p_start_x = int(width * 0.75)
    p_end_x = int(width * 0.95)

    if len(x_indices) >= 5:
        p_start_y = int(height - recent_price[0])
        p_end_y = int(height - recent_price[-1])
    else:
        p_start_y = int(height * 0.5)
        p_end_y = int(height * 0.4 if is_bullish else height * 0.6)

    draw.line([(p_start_x, p_start_y), (p_end_x, p_end_y)], fill=arrow_color, width=4)

    # Add text labels on image
    draw.text((15, 15), f"AI Signal: {direction} ({dominant_prob:.1%})", fill=arrow_color)
    draw.text((15, 35), f"Resistance: Top Line | Support: Bottom Line", fill=(100, 100, 100))

    analysis_points = [
        f"**Candlestick Sentiment**: {green_ratio:.1%} Bullish (Green) vs {red_ratio:.1%} Bearish (Red) pixels detected across chart body.",
        f"**Recent Momentum Gradient**: Recent slope is {'positive (upward trajectory)' if recent_slope > 0 else 'negative (downward pull)'}.",
        f"**Key Patterns Detected**: {', '.join(patterns)}.",
        f"**Directional Recommendation**: Model forecasts stock moving **{direction}** with **{dominant_prob:.1%}** probability.",
        f"**Risk Management Advice**: Recommended execution is **{action}** with stops placed below the support level."
    ]

    return {
        "status": "success",
        "direction": direction,
        "target_code": target_code,
        "probability_up": round(prob_up, 4),
        "probability_down": round(prob_down, 4),
        "confidence_level": conf_label,
        "recommended_action": action,
        "green_ratio": round(green_ratio, 3),
        "red_ratio": round(red_ratio, 3),
        "recent_change_pct": round(recent_change_pct, 4),
        "detected_patterns": patterns,
        "analysis_points": analysis_points,
        "annotated_image": annotated
    }


def analyze_uploaded_csv(df_uploaded: pd.DataFrame) -> Dict:
    """Run full trained machine learning pipeline on custom user-uploaded OHLCV CSV."""
    from src.feature_engineering import engineer_features, get_feature_column_names
    import joblib
    from src.config import MODELS_DIR

    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    scaler = joblib.load(MODELS_DIR / "preprocessor_pipeline.joblib")
    feature_cols = get_feature_column_names()

    # Standardize column names
    col_map = {c.strip().title(): c for c in df_uploaded.columns}
    df = df_uploaded.copy()
    df.columns = [c.strip().title() for c in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Uploaded CSV must contain '{c}' column. Found: {list(df.columns)}")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="B")

    df = df.sort_values("Date").reset_index(drop=True)
    df_featured = engineer_features(df)

    latest_row = df_featured.iloc[-1:]
    X_raw = latest_row[feature_cols]
    X_scaled = scaler.transform(X_raw)

    prob_up = float(rf_model.predict_proba(X_scaled)[0, 1])
    prob_down = 1.0 - prob_up
    is_bullish = prob_up >= 0.50
    direction = "BULLISH (UP)" if is_bullish else "BEARISH (DOWN)"

    dominant_prob = max(prob_up, prob_down)
    confidence = "High Confidence" if dominant_prob >= 0.65 else ("Moderate Confidence" if dominant_prob >= 0.55 else "Low / Neutral")

    return {
        "status": "success",
        "direction": direction,
        "probability_up": round(prob_up, 4),
        "probability_down": round(prob_down, 4),
        "confidence_level": confidence,
        "latest_close": float(latest_row["Close"].values[0]),
        "rsi_14": round(float(latest_row["RSI_14"].values[0]), 2),
        "macd": round(float(latest_row["MACD"].values[0]), 4),
        "sma_10": round(float(latest_row["SMA_10"].values[0]), 2),
        "sma_50": round(float(latest_row["SMA_50"].values[0]), 2),
        "featured_df": df_featured
    }
