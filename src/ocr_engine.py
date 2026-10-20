"""Optical Character Recognition and Financial Chart Metadata Extractor.
Uses native Windows Runtime OCR (Windows.Media.Ocr) for zero-dependency,
high-precision text, ticker, indicator, and price-scale extraction from stock chart images.
"""

from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Dict, List, Optional
from PIL import Image

from src.config import BASE_DIR, DEFAULT_TICKERS

SCRIPTS_DIR = BASE_DIR / "scripts"
OCR_PS_SCRIPT = SCRIPTS_DIR / "ocr_extractor.ps1"

# Known ticker symbol map for company name resolution
TICKER_NAME_MAP = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "GOOG": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NFLX": "Netflix Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq 100)",
    "RELIANCE": "Reliance Industries Ltd.",
    "RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS": "Tata Consultancy Services",
    "TCS.NS": "Tata Consultancy Services",
    "INFY": "Infosys Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "BTC": "Bitcoin / USD",
    "BTCUSDT": "Bitcoin Tether",
    "ETH": "Ethereum / USD",
    "^GSPC": "S&P 500 Index"
}


def extract_raw_text(image: Image.Image) -> str:
    """Extract raw text from a PIL image using native Windows OCR."""
    if not OCR_PS_SCRIPT.exists():
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            image.convert("RGB").save(temp_path, format="PNG")

        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(OCR_PS_SCRIPT),
            "-ImagePath",
            temp_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if res.returncode == 0:
            return res.stdout.strip()
        else:
            return ""
    except Exception as exc:
        return ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def parse_chart_metadata(raw_text: str) -> Dict:
    """Parse financial chart metadata from OCR extracted text.
    
    Identifies:
    - Ticker symbol & Company name
    - Timeframe (1D, 1H, 15m, etc.)
    - Visible technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
    - Price scale levels
    - Date / Year markers
    - OCR confidence / reliability flag
    """
    cleaned_text = raw_text.replace("\n", " ").replace("\r", " ")
    tokens = re.findall(r"[A-Za-z0-9\.\^\-]+", cleaned_text)

    # 1. Detect Ticker Symbol
    detected_ticker = None
    detected_name = None

    # Check known tickers first
    for token in tokens:
        clean_token = token.upper().strip(".,;:()[]")
        if clean_token in TICKER_NAME_MAP:
            detected_ticker = clean_token
            detected_name = TICKER_NAME_MAP[clean_token]
            break

    # If no known ticker, search for typical 2-5 letter uppercase ticker pattern
    if not detected_ticker:
        ticker_candidates = re.findall(r"\b[A-Z]{2,6}(?:\.[A-Z]{2})?\b", cleaned_text)
        # Filter out common indicator acronyms that look like tickers
        stopwords = {"RSI", "MACD", "SMA", "EMA", "ATR", "USD", "VOL", "DATE", "HIGH", "LOW", "OPEN", "CLOSE"}
        valid_candidates = [c for c in ticker_candidates if c not in stopwords]
        if valid_candidates:
            detected_ticker = valid_candidates[0]
            detected_name = f"{detected_ticker} (Detected from chart)"

    # 2. Detect Timeframe
    detected_timeframe = "1D (Daily)"
    tf_patterns = [
        (r"\b1[dD]\b|\bDaily\b", "1D (Daily)"),
        (r"\b1[wW]\b|\bWeekly\b", "1W (Weekly)"),
        (r"\b4[hH]\b", "4H (4 Hours)"),
        (r"\b1[hH]\b|\b60[mM]\b", "1H (1 Hour)"),
        (r"\b15[mM]\b", "15m (15 Minutes)"),
        (r"\b5[mM]\b", "5m (5 Minutes)"),
        (r"\b1[mM]\b", "1m (1 Minute)")
    ]
    for pattern, label in tf_patterns:
        if re.search(pattern, cleaned_text, re.IGNORECASE):
            detected_timeframe = label
            break

    # 3. Detect Technical Indicators
    visible_indicators = []
    text_upper = cleaned_text.upper()
    if "RSI" in text_upper:
        visible_indicators.append("Relative Strength Index (RSI)")
    if "MACD" in text_upper:
        visible_indicators.append("Moving Average Convergence Divergence (MACD)")
    if "SMA" in text_upper or "SIMPLE MOVING" in text_upper:
        visible_indicators.append("Simple Moving Average (SMA)")
    if "EMA" in text_upper or "EXPONENTIAL" in text_upper:
        visible_indicators.append("Exponential Moving Average (EMA)")
    if "BOLLINGER" in text_upper or "BB" in text_upper:
        visible_indicators.append("Bollinger Bands")
    if "VOLUME" in text_upper or "VOL" in text_upper:
        visible_indicators.append("Volume Oscillator / Histogram")
    if "HISTOGRAM" in text_upper and "Moving Average Convergence Divergence (MACD)" not in visible_indicators:
        visible_indicators.append("Momentum Histogram")

    # 4. Detect Price Axis Values
    # Find sequence of numbers that represent prices
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_text)
    float_nums = []
    for n in numbers:
        try:
            val = float(n)
            # Filter reasonable stock price values (between 1 and 100,000, excluding years 2010-2030)
            if (val < 1990 or val > 2035) and 1 <= val <= 200000:
                float_nums.append(val)
        except ValueError:
            pass

    price_levels = sorted(list(set(float_nums)))

    # 5. Detect Date Markers
    years = re.findall(r"\b(201\d|202\d|2030)\b", cleaned_text)
    unique_years = sorted(list(set(years)))

    # 6. Overall Reliability Assessment
    has_ticker = detected_ticker is not None
    has_indicators = len(visible_indicators) > 0
    has_prices = len(price_levels) >= 2
    has_dates = len(unique_years) >= 1

    chart_readability_score = sum([has_ticker, has_indicators, has_prices, has_dates]) / 4.0

    return {
        "raw_text": cleaned_text,
        "detected_ticker": detected_ticker,
        "detected_company_name": detected_name or "Unknown / Manual Input Required",
        "detected_timeframe": detected_timeframe,
        "visible_indicators": visible_indicators,
        "detected_price_levels": price_levels[-6:] if price_levels else [],
        "detected_years": unique_years,
        "readability_score": round(chart_readability_score, 2),
        "is_reliable": chart_readability_score >= 0.25
    }
