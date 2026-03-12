<div align="center">

# ₿ Bitcoin AI — Price Prediction & Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-006600?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

An end-to-end **Machine Learning pipeline** that predicts next-day Bitcoin prices by combining a deep-learning **LSTM** network, gradient-boosted **XGBoost** trees, **9 technical indicators**, and real-time **news sentiment analysis** — all served through an interactive **Streamlit** dashboard.

[Features](#-features) · [Demo](#-dashboard-preview) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works) · [Metrics](#-model-evaluation-metrics) · [License](#-license)

</div>

---

## ✨ Features

| Category | Details |
|---|---|
| **Data Pipeline** | Auto-downloads BTC-USD / BTC-INR OHLCV history via Yahoo Finance; caches locally as CSV |
| **Live Price** | Real-time spot price from Binance public REST API (no API key needed) |
| **Technical Indicators** | RSI · EMA-20 · MACD (line, signal, histogram) · Bollinger Bands · ATR · OBV |
| **Deep Learning (LSTM)** | 3-layer LSTM (128 → 64 → 32 units) with Dropout + BatchNorm, trained on 60-day sliding windows |
| **Gradient Boosting (XGBoost)** | 500 estimators, learning rate 0.03, max depth 6 — captures non-linear patterns |
| **Ensemble** | Weighted blend: **70 % LSTM + 30 % XGBoost** for robust predictions |
| **News Sentiment** | VADER NLP on live headlines from CoinDesk, CryptoPanic, Decrypt RSS feeds (+ optional NewsAPI) |
| **Trading Signals** | Sentiment-aware BUY / SELL / HOLD with confidence levels, stop-loss & take-profit targets |
| **Interactive Dashboard** | Streamlit + Plotly: candlestick charts, RSI subplot, sentiment bar chart, prediction overlay |
| **Multi-Currency** | Full support for **USD** and **INR** (auto-converted via live exchange rate) |
| **Comprehensive Metrics** | MAE, RMSE, MAPE, R², Directional Accuracy, Pearson / Spearman correlation, Theil's U, and more |

---

## 🖥 Dashboard Preview

The Streamlit dashboard combines all modules into a single interactive interface:

- **Candlestick chart** with Bollinger Bands & EMA-20 overlay
- **RSI subplot** with overbought / oversold zones (70 / 30)
- **Sentiment panel** — scored headlines with color-coded bar chart
- **Prediction panel** — Actual vs Predicted prices, next-day forecast, and trading signal with stop-loss / take-profit

> Launch the dashboard with `streamlit run dashboard/app.py` after training the models.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** (comes with Python)
- Internet connection (for data download & live price / news)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Bitcoin-price-predication-using-ML.git
cd Bitcoin-price-predication-using-ML
```

### 2. Create & Activate a Virtual Environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>📦 Full dependency list</strong></summary>

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation & numerical computing |
| `matplotlib` | Static plotting & metric visualizations |
| `scikit-learn` | MinMaxScaler, regression metrics |
| `tensorflow` | LSTM deep-learning model |
| `xgboost` | Gradient-boosted tree model |
| `ta` | Technical analysis indicators (RSI, MACD, BB, etc.) |
| `yfinance` | Yahoo Finance historical OHLCV data |
| `python-binance` | Binance live price API |
| `newsapi-python` | NewsAPI client (optional — for richer news) |
| `vaderSentiment` | VADER sentiment scoring engine |
| `streamlit` | Interactive web dashboard framework |
| `plotly` | Candlestick & interactive charts |
| `requests` | HTTP requests (RSS feeds, APIs) |
| `joblib` | Model serialization for XGBoost |
| `lxml` | XML / RSS feed parsing |
| `scipy` | Pearson & Spearman correlation metrics |

</details>

### 4. Train the Models

```bash
python main.py
```

This single command executes the **entire pipeline**:

1. Downloads historical BTC-USD data (from 2017 onward)
2. Computes all 9 technical indicators
3. Scales features with MinMaxScaler and builds 60-day sliding windows
4. Trains the **LSTM** neural network (with EarlyStopping & model checkpointing)
5. Trains the **XGBoost** regressor (500 boosted trees)
6. Evaluates all three models (LSTM, XGBoost, Ensemble) with comprehensive metrics
7. Fetches live news headlines and computes aggregate sentiment
8. Generates a **next-day price prediction** and **trading signal**

**Optional flags:**

```bash
# Use INR instead of USD
python main.py --currency INR

# Provide a NewsAPI key for broader news coverage
python main.py --newsapi YOUR_API_KEY

# Combine both
python main.py --currency INR --newsapi YOUR_API_KEY
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens automatically at **http://localhost:8501**. The sidebar lets you:

- Switch between **USD** and **INR**
- Set the historical data start date
- Enter an optional NewsAPI key
- Toggle Bollinger Bands & EMA-20 overlays
- Adjust the forecast window (30–180 days)
- Refresh sentiment on demand

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                 │
│  Yahoo Finance (OHLCV) ──► Technical Indicators (ta library)         │
│  Binance REST API ──► Live Spot Price                                │
│  RSS / NewsAPI ──► Raw Headlines                                     │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                 │
│  MinMaxScaler (0–1) ──► 60-day Sliding Windows ──► Train/Test Split  │
│  VADER Sentiment ──► Compound Score [-1, +1]                         │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐ ┌──────────────────┐
│   LSTM Model     │ │  XGBoost Model   │
│  3 LSTM layers   │ │  500 estimators  │
│  128 → 64 → 32   │ │  lr=0.03, d=6    │
│  Dropout + BN    │ │  subsample=0.8   │
│  Huber loss      │ │  colsample=0.8   │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ENSEMBLE PREDICTOR                                │
│               70% LSTM  +  30% XGBoost                               │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SIGNAL GENERATOR                                  │
│  Price prediction  +  Sentiment score  ──►  BUY / SELL / HOLD        │
│  Confidence level  ·  Stop-loss  ·  Take-profit  ·  Reason           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
bitcoin-ai-project/
│
├── data/                        # Auto-generated cached CSV files
│   ├── BTC_USD.csv              #   Historical BTC-USD OHLCV data
│   └── BTC_INR.csv              #   Historical BTC-INR OHLCV data
│
├── models/                      # Trained model artifacts
│   ├── bitcoin_lstm.keras       #   Final LSTM model
│   ├── lstm_best.keras          #   Best LSTM checkpoint (via EarlyStopping)
│   └── bitcoin_xgb.pkl          #   Serialized XGBoost model (joblib)
│
├── src/                         # Core Python modules
│   ├── __init__.py              #   Package init
│   ├── data_loader.py           #   Yahoo Finance download + Binance live price
│   ├── indicators.py            #   9 technical indicators via the `ta` library
│   ├── preprocessing.py         #   MinMaxScaler + 60-day sliding window builder
│   ├── train_model.py           #   LSTM & XGBoost: build, train, save, load
│   ├── predict.py               #   Ensemble inference engine (70/30 blend)
│   ├── sentiment.py             #   VADER sentiment on RSS / NewsAPI headlines
│   ├── signals.py               #   Trading signal logic (BUY / SELL / HOLD)
│   ├── metrics.py               #   Comprehensive ML evaluation metrics
│   └── binance_data.py          #   Binance data utilities
│
├── dashboard/
│   └── app.py                   #   Streamlit interactive web application
│
├── main.py                      #   CLI pipeline: train → evaluate → predict → signal
├── evaluate_models.py           #   Standalone model evaluation & metrics report
├── plot_metrics.py              #   Matplotlib visualizations for model performance
├── requirements.txt             #   Python dependencies
└── README.md                    #   You are here
```

---

## 🧠 How It Works

### 1. Data Collection

- **Historical data** is downloaded from Yahoo Finance using `yfinance` for the ticker `BTC-USD` (or `BTC-INR`) starting from January 2017. Data is cached locally in `data/` as CSV to avoid redundant downloads on subsequent runs.
- **Live price** is fetched from the Binance public API endpoint (`/api/v3/ticker/price`) — no authentication or API key required.

### 2. Feature Engineering — Technical Indicators

Nine technical indicators are computed using the [`ta`](https://github.com/bukosabino/ta) library:

| Indicator | Category | Window | Description |
|---|---|---|---|
| **RSI** | Momentum | 14 | Relative Strength Index — overbought (>70) / oversold (<30) |
| **EMA-20** | Trend | 20 | Exponential Moving Average — smoothed trend line |
| **MACD** | Trend | 12/26/9 | Moving Average Convergence Divergence (line + signal + histogram) |
| **MACD Signal** | Trend | 9 | Signal line of MACD |
| **MACD Histogram** | Trend | — | Difference between MACD and signal line |
| **Bollinger Upper** | Volatility | 20 | Upper band (mean + 2σ) |
| **Bollinger Lower** | Volatility | 20 | Lower band (mean − 2σ) |
| **ATR** | Volatility | 14 | Average True Range — daily volatility measure |
| **OBV** | Volume | — | On-Balance Volume — cumulative volume flow |

### 3. Preprocessing

- **14 feature columns** are used: `Close, Open, High, Low, Volume` + the 9 indicators above.
- All features are normalized to `[0, 1]` using `MinMaxScaler`.
- **Sliding windows** of **60 consecutive days** are constructed — each sample is a `(60, 14)` matrix.
- The **target** (`y`) is the next-day scaled `Close` price.
- Data is split into **90 % training / 10 % testing** (time-ordered, no shuffling — critical for time-series integrity).

### 4. Model Training

#### LSTM (Long Short-Term Memory)

```
Input Shape: (60, 14)
    ↓
LSTM(128, return_sequences=True) → Dropout(0.2) → BatchNormalization
    ↓
LSTM(64, return_sequences=True)  → Dropout(0.2) → BatchNormalization
    ↓
LSTM(32)                         → Dropout(0.2)
    ↓
Dense(16, activation='relu')
    ↓
Dense(1)  →  Predicted Price
```

| Hyperparameter | Value |
|---|---|
| Loss Function | Huber (robust to outliers) |
| Optimizer | Adam |
| Max Epochs | 50 |
| Batch Size | 32 |
| Early Stopping | patience = 8, restores best weights |
| Checkpoint | Saves best validation model to `models/lstm_best.keras` |

#### XGBoost (Gradient Boosted Trees)

| Hyperparameter | Value |
|---|---|
| Estimators | 500 |
| Learning Rate | 0.03 |
| Max Depth | 6 |
| Subsample | 0.8 |
| Column Sample / Tree | 0.8 |
| Random State | 42 |

The input for XGBoost is the **flattened** 60 × 14 = **840-dimensional** feature vector.

### 5. Ensemble Prediction

The final predicted price is a **weighted average** of both models:

$$\hat{y}_{\text{ensemble}} = 0.70 \times \hat{y}_{\text{LSTM}} + 0.30 \times \hat{y}_{\text{XGBoost}}$$

This design captures both the **temporal sequential patterns** (LSTM's strength) and **non-linear feature interactions** (XGBoost's strength).

### 6. News Sentiment Analysis

Headlines are collected from multiple sources:

| Source | Type | API Key Needed? |
|---|---|---|
| **CoinDesk** | RSS Feed | No |
| **CryptoPanic** | RSS Feed | No |
| **Decrypt** | RSS Feed | No |
| **NewsAPI** | REST API | Yes (optional, free tier) |

Each headline is scored using **VADER** (Valence Aware Dictionary and sEntiment Reasoner):

| Score Range | Label | Interpretation |
|---|---|---|
| > +0.05 | 🟢 Positive | Bullish market mood |
| < −0.05 | 🔴 Negative | Bearish market mood |
| −0.05 to +0.05 | 🟡 Neutral | No clear sentiment |

The **overall sentiment** is the mean VADER compound score across all collected headlines.

### 7. Trading Signal Generation

The signal engine combines the **price prediction** with the **sentiment score**:

| Condition | Signal | Confidence |
|---|---|---|
| Price ↑ > 3% AND sentiment ≥ −0.05 | **BUY** | HIGH |
| Price ↑ > 0.8% AND sentiment ≥ −0.05 | **BUY** | MEDIUM |
| Price ↓ > 3% AND sentiment ≤ +0.05 | **SELL** | HIGH |
| Price ↓ > 0.8% AND sentiment ≤ +0.05 | **SELL** | MEDIUM |
| Prediction and sentiment **conflict** | **HOLD** | LOW |
| Price change within ±0.8% neutral band | **HOLD** | LOW |

Each signal includes:
- **Action** — BUY, SELL, or HOLD
- **Confidence** — HIGH, MEDIUM, or LOW
- **Stop-Loss** — 5% from current price (risk management)
- **Take-Profit** — 10% from current price (profit target)
- **Reason** — Human-readable explanation of why the signal was generated

---

## 📊 Model Evaluation Metrics

The project computes an extensive suite of regression and time-series metrics:

| Metric | What It Measures |
|---|---|
| **MAE** (Mean Absolute Error) | Average dollar error per prediction |
| **RMSE** (Root Mean Squared Error) | Penalizes large errors more heavily than MAE |
| **MSE** (Mean Squared Error) | Squared error (used internally by many loss functions) |
| **RMSLE** (Root Mean Squared Log Error) | Scale-independent log-space error |
| **MAPE** (Mean Absolute Percentage Error) | Percentage error — easy to interpret across currencies |
| **R² Score** | Proportion of variance explained (1.0 = perfect fit) |
| **Adjusted R²** | R² adjusted for number of features |
| **Directional Accuracy** | % of days where the model correctly predicted up vs. down |
| **Pearson Correlation** | Linear correlation between actual and predicted |
| **Spearman Correlation** | Rank-order correlation (robust to non-linearity) |
| **Theil's U Statistic** | Forecast quality relative to naïve prediction (< 1 = good) |
| **Residual Analysis** | Mean, Std, Min, Max of prediction errors |
| **MAPE by Price Range** | Error distribution across different price bins |

### Run Evaluation Standalone

```bash
python evaluate_models.py --currency USD
```

### Generate Metric Visualizations

```bash
python plot_metrics.py --currency USD --save
```

---

## ⚙️ Configuration & CLI Reference

### `main.py` — Full Training + Prediction Pipeline

```bash
python main.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--currency` | `USD` | Currency pair: `USD` or `INR` |
| `--newsapi` | `None` | Optional NewsAPI key for richer headline coverage |

### `evaluate_models.py` — Evaluate Trained Models

```bash
python evaluate_models.py --currency USD
```

### `plot_metrics.py` — Visualize Model Performance

```bash
python plot_metrics.py --currency USD --save
```

### `dashboard/app.py` — Interactive Web Dashboard

```bash
streamlit run dashboard/app.py
```

> All dashboard settings are configurable from the sidebar UI — no CLI flags needed.

---

## 🔑 API Keys

| Service | Required? | How to Get |
|---|---|---|
| **Binance** | No — uses public endpoint | Not needed |
| **Yahoo Finance** | No — `yfinance` is free | Not needed |
| **NewsAPI** | Optional (for richer news) | Free key at [newsapi.org](https://newsapi.org) |

> **Without a NewsAPI key**, the system automatically falls back to RSS feeds from CoinDesk, CryptoPanic, and Decrypt — so the sentiment feature works out of the box.

---

## 🛠 Tech Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow / Keras (LSTM, Dense, Dropout, BatchNormalization) |
| **Machine Learning** | XGBoost, scikit-learn (MinMaxScaler, metrics) |
| **Technical Analysis** | `ta` library (RSI, MACD, Bollinger Bands, ATR, OBV, EMA) |
| **Data Sources** | Yahoo Finance (`yfinance`), Binance REST API |
| **NLP / Sentiment** | VADER Sentiment Analyzer (`vaderSentiment`) |
| **Visualization** | Plotly (interactive charts), Matplotlib (static plots) |
| **Web Dashboard** | Streamlit |
| **Model Serialization** | Keras `.keras` format, `joblib` `.pkl` format |
| **Statistics** | SciPy (Pearson, Spearman correlation) |

---

## 🗺 Roadmap

- [ ] Add Transformer-based model (Temporal Fusion Transformer)
- [ ] Incorporate on-chain metrics (hash rate, active addresses, mempool size)
- [ ] Multi-step forecasting (3-day, 7-day, 30-day horizons)
- [ ] Backtesting engine with simulated portfolio P&L
- [ ] Docker container for one-command deployment
- [ ] Automated retraining on a schedule (cron / GitHub Actions)
- [ ] Deploy dashboard to Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m "Add amazing feature"`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

Please make sure your code follows the existing style and passes all existing tests.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

> **This project is for educational and portfolio purposes only.** It does not constitute financial advice. Cryptocurrency markets are extremely volatile and unpredictable. Past performance of any model does not guarantee future results. The trading signals generated by this system should not be used as the sole basis for any investment decision. **Always do your own research (DYOR) before making any trading or investment decisions.**

---

<div align="center">

**Built with ❤️ using Python, TensorFlow, XGBoost & Streamlit**

If you found this project useful, consider giving it a ⭐!

</div>
