"""
dashboard/app.py
----------------
Interactive Streamlit dashboard — Bitcoin AI + News Sentiment

Run:
    streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader   import load_data, get_live_price
from src.indicators    import add_indicators
from src.preprocessing import preprocess_data, inverse_close
from src.train_model   import load_lstm, load_xgb
from src.predict       import predict_ensemble, predict_next_day
from src.sentiment     import get_sentiment_score, sentiment_label
from src.signals       import generate_signal


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bitcoin AI + Sentiment",
    page_icon="₿",
    layout="wide",
)

st.title("₿  Bitcoin AI Prediction + News Sentiment")
st.caption("LSTM + XGBoost Ensemble  ·  VADER Sentiment Analysis  ·  Live Binance Price")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    currency     = st.radio("Currency", ["USD", "INR"], horizontal=True)
    start_date   = st.date_input("Data start", value=pd.Timestamp("2018-01-01"))
    newsapi_key  = st.text_input("NewsAPI key (optional)", type="password",
                                  help="Free key at newsapi.org — leave blank to use RSS feeds")
    n_forecast   = st.slider("Forecast days to show", 30, 180, 90)
    show_bb      = st.checkbox("Bollinger Bands", True)
    show_ema     = st.checkbox("EMA 20", True)
    refresh_sent = st.button("🔄  Refresh Sentiment")

sym = "₹" if currency == "INR" else "$"
currency_label = currency

# ── Load data ─────────────────────────────────────────────────
@st.cache_data(show_spinner=f"Downloading BTC data in {currency} …")
def get_data(start, curr):
    df = load_data(start=str(start), currency=curr)
    return add_indicators(df)

df = get_data(start_date, currency)

# ── Price chart ───────────────────────────────────────────────
st.subheader("📈 Bitcoin Price History")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3],
    subplot_titles=(f"Price ({currency})", "RSI"),
    vertical_spacing=0.06,
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"].squeeze(), high=df["High"].squeeze(),
    low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
    name=f"BTC-{currency}",
), row=1, col=1)

if show_ema:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_20"].squeeze(),
        line=dict(color="orange", width=1.4), name="EMA 20",
    ), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"].squeeze(),
        line=dict(color="rgba(0,120,200,0.4)", dash="dot"), name="BB Upper",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"].squeeze(),
        line=dict(color="rgba(0,120,200,0.4)", dash="dot"), name="BB Lower",
        fill="tonexty", fillcolor="rgba(0,120,200,0.05)",
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"].squeeze(),
    line=dict(color="violet"), name="RSI",
), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)


# ── Sentiment panel ───────────────────────────────────────────
st.subheader("📰 News Sentiment Analysis")

if "sentiment_data" not in st.session_state or refresh_sent:
    with st.spinner("Fetching Bitcoin news …"):
        score, headlines = get_sentiment_score(
            newsapi_key=newsapi_key or None, n=20
        )
    st.session_state["sentiment_data"] = (score, headlines)

sent_score, sent_headlines = st.session_state["sentiment_data"]

col_a, col_b, col_c = st.columns(3)
col_a.metric("Sentiment Score", f"{sent_score:+.4f}")
col_b.metric("Market Mood",     sentiment_label(sent_score))
col_c.metric("Headlines analysed", len(sent_headlines))

# Sentiment bar chart
if sent_headlines:
    titles  = [h.title[:55] + "…" if len(h.title) > 55 else h.title for h in sent_headlines]
    scores  = [h.sentiment for h in sent_headlines]
    colors  = ["#2ecc71" if s > 0.05 else ("#e74c3c" if s < -0.05 else "#f39c12") for s in scores]

    fig_sent = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker_color=colors, text=[f"{s:+.3f}" for s in scores],
        textposition="outside",
    ))
    fig_sent.update_layout(
        title="Headline Sentiment Scores",
        template="plotly_dark", height=max(300, len(titles) * 28),
        xaxis_title="VADER Compound Score", yaxis_autorange="reversed",
        margin=dict(l=10, r=60),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    with st.expander("📋 All headlines"):
        for h in sent_headlines:
            st.write(f"{h.label}  **{h.title}**  ·  *{h.source}*")


# ── Model predictions ─────────────────────────────────────────
st.subheader("🤖 Price Prediction & Trading Signal")

models_ready = (
    os.path.exists("models/bitcoin_lstm.keras") and
    os.path.exists("models/bitcoin_xgb.pkl")
)

if not models_ready:
    st.warning("⚠️ Models not found. Run `python main.py` first to train.")
else:
    @st.cache_resource(show_spinner="Loading models …")
    def load_models():
        return load_lstm(), load_xgb()

    lstm_model, xgb_model = load_models()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    n_features = X_train.shape[2]

    # Actual vs predicted chart
    preds   = predict_ensemble(lstm_model, xgb_model, X_test[-n_forecast:], scaler, n_features)
    actuals = inverse_close(scaler, y_test[-n_forecast:], n_features)
    dates   = df.index[-n_forecast:]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=actuals, name="Actual",    line=dict(color="cyan")))
    fig2.add_trace(go.Scatter(x=dates, y=preds,   name="Predicted", line=dict(color="orange", dash="dash")))
    fig2.update_layout(title=f"Actual vs Predicted — last {n_forecast} days ({currency})",
                       template="plotly_dark", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    # Signal
    last_window    = X_test[[-1]]
    next_price     = predict_next_day(lstm_model, xgb_model, last_window, scaler, n_features)
    current_price  = inverse_close(scaler, y_test[[-1]], n_features)[0]

    live = get_live_price(currency=currency)
    if live:
        current_price = live

    signal = generate_signal(current_price, next_price, sent_score)

    action_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signal",          f"{action_icons[signal.action]} {signal.action}")
    c2.metric("Confidence",      signal.confidence)
    c3.metric("Current Price",   f"{sym}{current_price:,.2f}")
    c4.metric("Predicted",       f"{sym}{next_price:,.2f}", delta=f"{signal.change_pct:+.2f}%")
    c5.metric("Sentiment Input", f"{sent_score:+.3f}")

    with st.expander("📋 Signal Details"):
        st.info(signal.reason)
        col_sl, col_tp = st.columns(2)
        col_sl.metric("Stop-Loss",   f"{sym}{signal.stop_loss:,.2f}")
        col_tp.metric("Take-Profit", f"{sym}{signal.take_profit:,.2f}")
        st.warning("⚠️ Not financial advice. Always DYOR.")

# ── Recent data ───────────────────────────────────────────────
st.subheader("📋 Recent Market Data")
st.dataframe(
    df[["Open","High","Low","Close","Volume","RSI","EMA_20","MACD"]].tail(10).round(2),
    use_container_width=True,
)
st.caption("Data: Yahoo Finance · Sentiment: VADER · Live price: Binance public API")
