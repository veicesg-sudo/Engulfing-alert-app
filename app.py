import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ============================================================
# APP SETTINGS
# ============================================================

st.set_page_config(
    page_title="15M Engulfing Alert",
    page_icon="📊",
    layout="wide"
)

st.title("📊 15-Minute Engulfing Alert")
st.caption("Manual trading assistant — alerts only. No automatic trades.")

# ============================================================
# TWELVE DATA
# ============================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

BASE_URL = "https://api.twelvedata.com/time_series"

MARKETS = {
    "XAUUSD": {
        "name": "Gold",
        "symbol": "XAU/USD"
    },
    "BTCUSD": {
        "name": "Bitcoin",
        "symbol": "BTC/USD"
    }
}

# ============================================================
# STRATEGY SETTINGS
# ============================================================

TIMEFRAME = "15min"
BODY_RATIO = 1.2
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
SL_ATR_MULTIPLIER = 1.5
RISK_REWARD = 2.0

# ============================================================
# SESSION STATE
# ============================================================

if "signals" not in st.session_state:
    st.session_state.signals = []

if "last_signal" not in st.session_state:
    st.session_state.last_signal = {}

# ============================================================
# FUNCTIONS
# ============================================================

def get_data(symbol):
    """Download 15-minute candles from Twelve Data."""

    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing from Streamlit Secrets."
        )

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": 300,
        "apikey": API_KEY
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(
            data.get("message", "Twelve Data returned an error.")
        )

    if "values" not in data:
        raise RuntimeError(
            "No candle data was returned."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            "No market data was returned."
        )

    # Convert prices to numbers
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert timestamp
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    # --------------------------------------------------------
    # ONLY USE COMPLETED 15-MINUTE CANDLES
    # --------------------------------------------------------

    now = pd.Timestamp.now(tz="UTC")

    current_bucket = now.floor("15min")

    df = df[df["datetime"] < current_bucket].copy()

    return df


def calculate_indicators(df):
    """Calculate EMA and ATR."""

    df = df.copy()

    df["ema50"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    true_range_1 = df["high"] - df["low"]
    true_range_2 = (
        df["high"] - previous_close
    ).abs()

    true_range_3 = (
        df["low"] - previous_close
    ).abs()

    df["tr"] = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(
        ATR_PERIOD
    ).mean()

    return df


def detect_signal(df):
    """Check the most recently completed candle."""

    if len(df) < EMA_SLOW + 5:
        return None

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_body = abs(
        previous["close"] - previous["open"]
    )

    current_body = abs(
        current["close"] - current["open"]
    )

    if previous_body == 0:
        return None

    body_ratio = (
        current_body / previous_body
    )

    # --------------------------------------------------------
    # BULLISH ENGULFING
    # --------------------------------------------------------

    bullish_engulfing = (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
        and body_ratio >= BODY_RATIO
    )

    # --------------------------------------------------------
    # BEARISH ENGULFING
    # --------------------------------------------------------

    bearish_engulfing = (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
        and body_ratio >= BODY_RATIO
    )

    atr = current["atr"]

    if pd.isna(atr):
        return None

    close = current["close"]
    ema50 = current["ema50"]
    ema200 = current["ema200"]

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if bullish_engulfing:

        trend_confirmed = (
            close > ema50
            and ema50 > ema200
        )

        if trend_confirmed:

            entry = close
            sl = entry - (
                SL_ATR_MULTIPLIER * atr
            )

            risk = entry - sl

            tp = entry + (
                RISK_REWARD * risk
            )

            return {
                "signal": "BUY",
                "time": current["datetime"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr,
                "body_ratio": body_ratio,
                "ema50": ema50,
                "ema200": ema200
            }

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if bearish_engulfing:

        trend_confirmed = (
            close < ema50
            and ema50 < ema200
        )

        if trend_confirmed:

            entry = close
            sl = entry + (
                SL_ATR_MULTIPLIER * atr
            )

            risk = sl - entry

            tp = entry - (
                RISK_REWARD * risk
            )

            return {
                "signal": "SELL",
                "time": current["datetime"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "atr": atr,
                "body_ratio": body_ratio,
                "ema50": ema50,
                "ema200": ema200
            }

    return None


def format_price(value):
    """Format prices cleanly."""

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 100:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:,.2f}"

    return f"{value:,.5f}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Strategy")

st.sidebar.write(
    f"Timeframe: **{TIMEFRAME}**"
)

st.sidebar.write(
    f"Minimum body ratio: **{BODY_RATIO}×**"
)

st.sidebar.write(
    f"ATR: **{ATR_PERIOD}**"
)

st.sidebar.write(
    f"SL: **{SL_ATR_MULTIPLIER}× ATR**"
)

st.sidebar.write(
    f"TP: **{RISK_REWARD}R**"
)

st.sidebar.divider()

st.sidebar.warning(
    "Manual trading only. "
    "This app does not place orders."
)

# ============================================================
# REFRESH
# ============================================================

if st.button(
    "🔄 Refresh Markets",
    use_container_width=True
):

    st.rerun()

st.caption(
    "Refresh the app after a 15-minute candle closes "
    "to check for a new signal."
)

# ============================================================
# API CHECK
# ============================================================

if not API_KEY:

    st.error(
        "⚠️ Twelve Data API key not found."
    )

    st.info(
        "Add TWELVE_DATA_API_KEY to Streamlit Secrets."
    )

    st.stop()

# ============================================================
# MARKET DASHBOARD
# ============================================================

st.header("Markets")

columns = st.columns(2)

for index, (market_code, market) in enumerate(
    MARKETS.items()
):

    with columns[index]:

        st.subheader(
            f"{'🥇' if market_code == 'XAUUSD' else '₿'} "
            f"{market['name']}"
        )

        try:

            df = get_data(
                market["symbol"]
            )

            df = calculate_indicators(df)

            signal = detect_signal(df)

            latest = df.iloc[-1]

            # ------------------------------------------------
            # CURRENT MARKET INFORMATION
            # ------------------------------------------------

            st.metric(
                "Last completed candle",
                format_price(latest["close"])
            )

            trend = "NEUTRAL"

            if (
                latest["close"] > latest["ema50"]
                and latest["ema50"] > latest["ema200"]
            ):
                trend = "BULLISH"

            elif (
                latest["close"] < latest["ema50"]
                and latest["ema50"] < latest["ema200"]
            ):
                trend = "BEARISH"

            st.write(
                f"**Trend:** {trend}"
            )

            st.write(
                f"EMA 50: `{format_price(latest['ema50'])}`"
            )

            st.write(
                f"EMA 200: `{format_price(latest['ema200'])}`"
            )

            st.write(
                f"ATR(14): `{format_price(latest['atr'])}`"
            )

            st.write(
                "Last candle: "
                f"`{latest['datetime'].strftime('%Y-%m-%d %H:%M UTC')}`"
            )

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            if signal:

                signal_time = str(
                    signal["time"]
                )

                signal_key = (
                    market_code,
                    signal_time,
                    signal["signal"]
                )

                already_seen = (
                    signal_key
                    in st.session_state.last_signal
                )

                if not already_seen:

                    st.session_state.last_signal[
                        signal_key
                    ] = True

                    st.session_state.signals.append(
                        {
                            "Market": market["name"],
                            "Symbol": market_code,
                            "Signal": signal["signal"],
                            "Time": signal["time"],
                            "Entry": signal["entry"],
                            "Stop Loss": signal["sl"],
                            "Take Profit": signal["tp"],
                            "ATR": signal["atr"],
                            "Body Ratio": signal["body_ratio"]
                        }
                    )

                if signal["signal"] == "BUY":

                    st.success(
                        "🟢 BUY SIGNAL"
                    )

                else:

                    st.error(
                        "🔴 SELL SIGNAL"
                    )

                st.write(
                    f"**Entry:** "
                    f"`{format_price(signal['entry'])}`"
                )

                st.write(
                    f"**Suggested SL:** "
                    f"`{format_price(signal['sl'])}`"
                )

                st.write(
                    f"**Suggested TP:** "
                    f"`{format_price(signal['tp'])}`"
                )

                st.write(
                    f"**Body ratio:** "
                    f"`{signal['body_ratio']:.2f}×`"
                )

                st.warning(
                    "⚠️ This is an alert only. "
                    "If you choose to trade, execute it manually "
                    "in XM MT5."
                )

            else:

                st.info(
                    "No confirmed signal on the latest "
                    "completed candle."
                )

        except Exception as e:

            st.error(
                f"Data error for {market['name']}"
            )

            with st.expander(
                "Technical details"
            ):

                st.code(str(e))

# ============================================================
# SIGNAL HISTORY
# ============================================================

st.divider()

st.header("📜 Signal History")

if st.session_state.signals:

    history = pd.DataFrame(
        st.session_state.signals
    )

    history = history.sort_values(
        "Time",
        ascending=False
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No signals detected during this app session."
    )

# ============================================================
# STRATEGY RULES
# ============================================================

st.divider()

st.header("📋 Strategy Rules")

st.markdown(
    """
**BUY**

1. A completed bullish engulfing candle appears.
2. Current candle body is at least **1.2×** the previous candle body.
3. Close > EMA 50.
4. EMA 50 > EMA 200.
5. Suggested SL = **1.5 × ATR(14)** below entry.
6. Suggested TP = **2R**.

**SELL**

1. A completed bearish engulfing candle appears.
2. Current candle body is at least **1.2×** the previous candle body.
3. Close < EMA 50.
4. EMA 50 < EMA 200.
5. Suggested SL = **1.5 × ATR(14)** above entry.
6. Suggested TP = **2R**.

**Important:** The app never sends an order to XM.
"""
)

st.caption(
    "Trading involves risk. This tool is for strategy alerts "
    "and does not guarantee profitable results."
)
