"""Image Validation Engine for Stock Market Chart Uploads.
Ensures uploaded files conform to financial chart structural criteria,
detects corrupted or non-chart uploads, and provides clear diagnostics.
"""

from typing import Dict, Tuple
import numpy as np
from PIL import Image


def validate_chart_image(img: Image.Image) -> Dict:
    """Validate whether an uploaded image contains viable stock chart content.
    
    Checks:
    1. Image dimensions & aspect ratio
    2. Color and luminance variance (rejects blank, black, or flat images)
    3. Edge and line structure density (detects candlesticks, grid lines, or price curves)
    4. Contrast ratio across canvas
    """
    errors = []
    warnings = []

    width, height = img.size

    # 1. Dimension checks
    if width < 150 or height < 150:
        errors.append(f"Image resolution too low ({width}x{height} px). Minimum required is 150x150 px.")
    elif width < 300 or height < 200:
        warnings.append("Low resolution chart: price labels or small candlesticks may have reduced detection accuracy.")

    aspect_ratio = width / max(1, height)
    if aspect_ratio > 5.0 or aspect_ratio < 0.2:
        warnings.append(f"Unusual aspect ratio ({aspect_ratio:.2f}:1). Standard trading charts are typically between 1:1 and 3:1.")

    # Convert to RGB numpy array for structural analysis
    try:
        rgb_img = img.convert("RGB")
        arr = np.array(rgb_img, dtype=np.float32)
    except Exception as exc:
        return {
            "is_valid": False,
            "status": "REJECT",
            "errors": [f"Failed to decode image data: {str(exc)}"],
            "warnings": warnings,
            "resolution": (width, height),
            "diagnostic_summary": "Corrupted or unsupported image encoding."
        }

    # 2. Luminance & Variance checks
    luminance = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
    lum_std = float(np.std(luminance))
    lum_mean = float(np.mean(luminance))

    if lum_std < 8.0:
        errors.append("Image is nearly uniform or blank. No chart, candlesticks, or price lines detected.")

    if lum_mean < 10.0:
        errors.append("Image is almost completely black. Please upload a visible chart screenshot.")
    elif lum_mean > 250.0 and lum_std < 12.0:
        errors.append("Image is solid white canvas with no discernible market data.")

    # 3. Edge and Line Structure Gradient Check
    # Financial charts have high horizontal and vertical edge density (grids, candles, wicks)
    dx = np.abs(luminance[:, 1:] - luminance[:, :-1])
    dy = np.abs(luminance[1:, :] - luminance[:-1, :])
    edge_density = float(np.mean(dx > 20) + np.mean(dy > 20)) / 2.0

    if edge_density < 0.005 and lum_std < 15.0:
        errors.append("Lacks chart coordinate grid, price trendlines, or candlestick structures.")

    # 4. Color Distribution (Green / Red / Neutral)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    brightness = (r + g + b) / 3.0
    fg_mask = (brightness > 25) & (brightness < 240)
    green_candles = fg_mask & (g > r + 15) & (g > b + 10)
    red_candles = fg_mask & (r > g + 15) & (r > b + 10)
    colored_ratio = float(np.sum(green_candles | red_candles)) / float(max(1, width * height))

    if colored_ratio < 0.0005:
        warnings.append("Monochrome or grayscale chart detected. Analysis will rely on price curve geometry rather than green/red candle color balance.")

    is_valid = len(errors) == 0
    status = "PASS" if is_valid and len(warnings) == 0 else ("WARNING" if is_valid else "REJECT")

    if not is_valid:
        diagnostic = "Chart verification failed: " + "; ".join(errors)
    elif warnings:
        diagnostic = "Chart validated with advisories: " + "; ".join(warnings)
    else:
        diagnostic = "Chart format and structural validation successful. Ready for high-precision technical scanning."

    return {
        "is_valid": is_valid,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "resolution": (width, height),
        "lum_std": round(lum_std, 2),
        "edge_density": round(edge_density, 4),
        "colored_candle_ratio": round(colored_ratio, 4),
        "diagnostic_summary": diagnostic
    }
