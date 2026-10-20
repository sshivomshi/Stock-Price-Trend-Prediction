# Experiment 2: Dataset Details

**Course Code**: CSE4007L - Applications of Machine Learning in Industry  
**Project**: Fin-Project 3: Stock Price Trend Prediction  
**Duration**: 1.5 Hours  
**Document Type**: Dataset Description Report  

---

## 1. Data Source & Licensing
- **Source**: Yahoo Finance API accessed via the open-source Python library `yfinance` (`v1.7.0+`).
- **Asset Universe**: Basket of liquid large-cap equities across major global and domestic indices:
  - **NASDAQ**: Apple Inc. (`AAPL`), Microsoft Corp. (`MSFT`), Alphabet Inc. (`GOOGL`), NVIDIA Corp. (`NVDA`), Amazon.com Inc. (`AMZN`).
  - **NIFTY 50 (India)**: Reliance Industries Ltd (`RELIANCE.NS`), Tata Consultancy Services (`TCS.NS`), Infosys Ltd (`INFY.NS`).
- **License & Usage Rights**: Free and open market data retrieved for educational, academic evaluation, and non-commercial algorithmic research under the Yahoo Finance Terms of Service.

---

## 2. Dataset Size & Temporal Coverage
- **Historical Period**: January 2, 2019 to Present (> 5 years of daily trading history, exceeding curriculum requirements).
- **Sampling Frequency**: Daily trading sessions (1 trading day per record).
- **Row & Column Dimensions**:
  - Primary Evaluation Asset (`AAPL`): **1,961 trading sessions**, 7 raw columns.
  - Consolidated Multi-Asset Portfolio: **15,586 total records** across 8 stocks.
  - File Formats: Cached in `data/raw/` as both CSV and Apache Parquet for high-throughput serialization.

---

## 3. Feature Dictionary (Raw Schema)

| Feature Name | Data Type | Physical / Financial Meaning | Null Count | Unit / Range |
| :--- | :--- | :--- | :--- | :--- |
| `Date` | `datetime64[ns]` | Trading session date (YYYY-MM-DD) | 0 | 2019-01-02 to 2026-10-20 |
| `Open` | `float64` | Opening price at the bell | 0 | Asset currency (USD / INR) |
| `High` | `float64` | Maximum price attained during regular hours | 0 | Asset currency |
| `Low` | `float64` | Minimum price attained during regular hours | 0 | Asset currency |
| `Close` | `float64` | Unadjusted closing price at 16:00 EST / 15:30 IST | 0 | Asset currency |
| `Adj Close` | `float64` | Closing price adjusted for stock splits and dividend cash payouts | 0 | Asset currency |
| `Volume` | `int64` | Total volume of shares exchanged during session | 0 | Integer shares (>100,000) |
| `Ticker` | `string` | Ticker identifier symbol (e.g. `AAPL`, `RELIANCE.NS`) | 0 | Categorical symbol |

---

## 4. Target Distribution & Class Balance Check
For the directional prediction objective, the target is calculated as:
$$\text{Target}_t = \mathbb{I}(\text{Close}_{t+1} > \text{Close}_t)$$

- **Total Historical Observations (`AAPL`)**: 1,961
- **Up Sessions ($\text{Target} = 1$)**: 1,043 sessions (**53.19%**)
- **Down / Flat Sessions ($\text{Target} = 0$)**: 918 sessions (**46.81%**)
- **Imbalance Ratio**: $\approx 1.14 : 1$ (Naturally balanced market distribution).
- **Significance**: Unlike extreme imbalance problems (such as fraud detection), financial directional prediction operates near a 50/50 prior with a slight positive drift reflecting long-term equity growth.

---

## 5. Known Limitations & Constraints
1. **Survivorship Bias**:
   - Selecting top constituents as of the present day inherently conditions on companies that did not undergo bankruptcy or index delisting between 2019 and 2026.
2. **Weekend & Holiday Gaps**:
   - Equities do not trade on weekends or statutory market holidays. While time is discrete, calendar distance varies between Friday-to-Monday sessions.
3. **Macroeconomic Shocks & Structural Breaks**:
   - The 2019-2026 window encompasses the COVID-19 pandemic shock (March 2020), subsequent inflation rate hiking cycles (2022-2023), and AI technology sector expansions, creating multi-regime distribution shifts.
4. **Intraday Execution Friction**:
   - Daily OHLC bars compress all intra-session market micro-structure, spread variance, and order-book depth.
