import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="15m Engulfing Alert",
    page_icon="📈",
    layout="wide"
)

SYMBOLS = {
    "XAUUSD": "GC=F",
    "NASDAQ": "NQ=F",
    "BTCUSD": "BTC-USD",
    "OIL": "CL=F",
}

st.title("📈 15-Minute Engulfing Alert")
st.caption("Alert-only assistant — no automatic trading.")

with st.sidebar:
    st.header("Strategy")

    mode = st.selectbox(
        "Signal mode",
        ["Strict", "Raw"]
    )

    ratio_min = st.number_input(
        "Minimum body ratio",
        min_value=1.0,
        max_value=5.0,
        value=1.2,
        step=0.1
    )

    ema_fast = st.number_input(
        "Fast EMA",
        min_value=2,
        max_value=200,
        value=50
    )

    ema_slow = st.number_input(
        "Slow EMA",
        min_value=3,
        max_value=500,
        value=200
    )

    atr_period = st.number_input(
        "ATR period",
        min_value=2,
        max_value=100,
        value=14
    )

    sl_atr = st.number_input(
        "Stop distance (ATR)",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1
    )

    rr = st.number_input(
        "Target (R)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    auto_refresh = st.checkbox(
        "Auto refresh",
        value=False
    )

    refresh = st.number_input(
        "Refresh seconds",
        min_value=15,
        max_value=600,
        value=60,
        step=15
    )


def get_data(ticker):
    df = yf.download(
        ticker,
        period="5d",
        interval="15m",
        progress=False,
        auto_adjust=False,
        prepost=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    needed = ["Open", "High", "Low", "Close"]

    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    return df.dropna(subset=needed)


def add_indicators(df):
    d = df.copy()

    d["EMA_FAST"] = (
        d["Close"]
        .ewm(span=ema_fast, adjust=False)
        .mean()
    )

    d["EMA_SLOW"] = (
        d["Close"]
        .ewm(span=ema_slow, adjust=False)
        .mean()
    )

    previous_close = d["Close"].shift(1)

    true_range = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - previous_close).abs(),
            (d["Low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = (
        true_range
        .rolling(atr_period)
        .mean()
    )

    return d


def check_signal(d):

    minimum_bars = max(
        ema_slow,
        atr_period
    ) + 3

    if len(d) < minimum_bars:
        return None

    # Latest completed candle
    i = len(d) - 2

    previous = d.iloc[i - 1]
    current = d.iloc[i]

    previous_body = abs(
        previous["Close"] - previous["Open"]
    )

    current_body = abs(
        current["Close"] - current["Open"]
    )

    if previous_body == 0:
        ratio = 0
    else:
        ratio = current_body / previous_body

    bullish = (
        previous["Close"] < previous["Open"]
        and current["Close"] > current["Open"]
        and current["Open"] <= previous["Close"]
        and current["Close"] >= previous["Open"]
    )

    bearish = (
        previous["Close"] > previous["Open"]
        and current["Close"] < current["Open"]
        and current["Open"] >= previous["Close"]
        and current["Close"] <= previous["Open"]
    )

    if not (bullish or bearish):
        return None

    bullish_trend = (
        current["Close"]
        > current["EMA_FAST"]
        > current["EMA_SLOW"]
    )

    bearish_trend = (
        current["Close"]
        < current["EMA_FAST"]
        < current["EMA_SLOW"]
    )

    if mode == "Strict":

        if ratio < ratio_min:
            return None

        if bullish and not bullish_trend:
            return None

        if bearish and not bearish_trend:
            return None

    direction = "BUY" if bullish else "SELL"

    entry = float(current["Close"])
    atr = float(current["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    stop_distance = sl_atr * atr

    if direction == "BUY":
        sl = entry - stop_distance
        tp = entry + rr * stop_distance
    else:
        sl = entry + stop_distance
        tp = entry - rr * stop_distance

    return {
        "direction": direction,
        "bar": str(current.name),
        "ratio": ratio,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "trend": (
            "Bullish"
            if bullish_trend
            else "Bearish"
            if bearish_trend
            else "Mixed"
        ),
    }


if "alerts" not in st.session_state:
    st.session_state.alerts = []


columns = st.columns(4)

for column, market in zip(columns, SYMBOLS):

    with column:

        st.subheader(market)

        data = get_data(SYMBOLS[market])

        if data.empty:
            st.error("No market data returned.")
            continue

        data = add_indicators(data)

        last = data.iloc[-2]

        st.metric(
            "Completed close",
            f"{float(last['Close']):,.4f}"
        )

        signal = check_signal(data)

        if signal:

            st.success(
                f"🚨 {signal['direction']} SIGNAL"
            )

            st.write(
                f"Engulfing ratio: "
                f"**{signal['ratio']:.2f}x**"
            )

            st.write(
                f"Entry reference: "
                f"**{signal['entry']:.4f}**"
            )

            st.write(
                f"Suggested SL: "
                f"**{signal['sl']:.4f}**"
            )

            st.write(
                f"Suggested TP: "
                f"**{signal['tp']:.4f}**"
            )

            st.write(
                f"ATR: **{signal['atr']:.4f}**"
            )

            st.write(
                f"Trend: **{signal['trend']}**"
            )

            key = (
                f"{market}|"
                f"{signal['direction']}|"
                f"{signal['bar']}"
            )

            existing = [
                x["key"]
                for x in st.session_state.alerts
            ]

            if key not in existing:

                st.session_state.alerts.append(
                    {
                        "key": key,
                        "Alert time":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        "Market": market,
                        "Direction":
                            signal["direction"],
                        "Candle":
                            signal["bar"],
                        "Entry":
                            signal["entry"],
                        "SL":
                            signal["sl"],
                        "TP":
                            signal["tp"],
                        "Ratio":
                            signal["ratio"],
                    }
                )

        else:
            st.info(
                "No qualifying signal on the "
                "latest completed candle."
            )


st.divider()

st.subheader("Alert history")

if st.session_state.alerts:

    history = pd.DataFrame(
        st.session_state.alerts[::-1]
    )

    history = history.drop(
        columns=["key"]
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

else:

    st.caption(
        "No alerts recorded in this session."
    )


st.warning(
    "V1 uses Yahoo Finance market-data proxies. "
    "These may differ from your broker's exact "
    "prices, candles, spreads and contract specifications. "
    "Verify signals against your broker before live use. "
    "This app does not place trades."
)


if auto_refresh:

    time.sleep(refresh)
    st.rerun()
