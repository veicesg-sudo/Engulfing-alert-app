import streamlit as st
import pandas as pd
import numpy as np
import requests

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="15M Engulfing Alert",
    page_icon="📊",
    layout="wide"
)

st.title("📊 15-Minute Engulfing Alert")
st.caption(
    "Gold + BTCUSD | Manual trading only | "
    "No automatic orders"
)

# ============================================================
# SETTINGS
# ============================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

BASE_URL = "https://api.twelvedata.com/time_series"

MARKETS = {
    "XAUUSD": {
        "name": "Gold",
        "symbol": "XAU/USD",
        "icon": "🥇"
    },
    "BTCUSD": {
        "name": "Bitcoin",
        "symbol": "BTC/USD",
        "icon": "₿"
    }
}

TIMEFRAME = "15min"

# Strategy
BODY_RATIO = 1.2
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
SL_ATR_MULTIPLIER = 1.5
RISK_REWARD = 2.0

# Number of recent completed candles to inspect
SCAN_CANDLES = 80

# ============================================================
# SESSION STATE
# ============================================================

if "signals" not in st.session_state:
    st.session_state.signals = []

if "seen_signals" not in st.session_state:
    st.session_state.seen_signals = set()

# ============================================================
# HELPER
# ============================================================

def format_price(value):
    if pd.isna(value):
        return "-"

    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 100:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:,.2f}"

    return f"{value:,.5f}"


# ============================================================
# GET MARKET DATA
# ============================================================

def get_data(symbol):

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
            data.get(
                "message",
                "Twelve Data returned an error."
            )
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

    # Convert prices
    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert timestamps
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

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # ========================================================
    # REMOVE CURRENT INCOMPLETE CANDLE
    # ========================================================

    now = pd.Timestamp.now(tz="UTC")

    current_candle_start = now.floor("15min")

    df = df[
        df["datetime"] < current_candle_start
    ].copy()

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA 50
    df["ema50"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    # EMA 200
    df["ema200"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # True Range
    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"] - df["low"]
    )

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # ATR
    df["atr"] = df["tr"].rolling(
        ATR_PERIOD
    ).mean()

    return df


# ============================================================
# CHECK ENGULFING CANDLE
# ============================================================

def check_engulfing(
    previous,
    current
):

    previous_body = abs(
        previous["close"] - previous["open"]
    )

    current_body = abs(
        current["close"] - current["open"]
    )

    # Avoid division by zero
    if previous_body == 0:
        return {
            "pattern": "NONE",
            "body_ratio": 0
        }

    body_ratio = (
        current_body / previous_body
    )

    # ========================================================
    # BULLISH ENGULFING
    # ========================================================

    bullish = (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )

    # ========================================================
    # BEARISH ENGULFING
    # ========================================================

    bearish = (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )

    if bullish:
        pattern = "BULLISH ENGULFING"

    elif bearish:
        pattern = "BEARISH ENGULFING"

    else:
        pattern = "NONE"

    return {
        "pattern": pattern,
        "body_ratio": body_ratio
    }


# ============================================================
# ANALYZE CANDLE
# ============================================================

def analyze_candle(
    df,
    index
):

    if index < 1:
        return None

    previous = df.iloc[index - 1]
    current = df.iloc[index]

    engulfing = check_engulfing(
        previous,
        current
    )

    pattern = engulfing["pattern"]
    body_ratio = engulfing["body_ratio"]

    if pattern == "NONE":
        return None

    reasons = []

    # ========================================================
    # BODY REQUIREMENT
    # ========================================================

    body_ok = (
        body_ratio >= BODY_RATIO
    )

    if not body_ok:

        reasons.append(
            f"Body ratio {body_ratio:.2f}x "
            f"is below {BODY_RATIO:.1f}x"
        )

    # ========================================================
    # TREND REQUIREMENT
    # ========================================================

    close = current["close"]
    ema50 = current["ema50"]
    ema200 = current["ema200"]

    if pd.isna(ema50) or pd.isna(ema200):

        reasons.append(
            "EMA data not available yet"
        )

        trend = "UNKNOWN"

    elif pattern == "BULLISH ENGULFING":

        trend_ok = (
            close > ema50
            and ema50 > ema200
        )

        if not trend_ok:

            reasons.append(
                "BUY trend condition failed: "
                "Close must be above EMA50 and "
                "EMA50 must be above EMA200"
            )

        trend = (
            "BULLISH"
            if trend_ok
            else "NOT BULLISH"
        )

    else:

        trend_ok = (
            close < ema50
            and ema50 < ema200
        )

        if not trend_ok:

            reasons.append(
                "SELL trend condition failed: "
                "Close must be below EMA50 and "
                "EMA50 must be below EMA200"
            )

        trend = (
            "BEARISH"
            if trend_ok
            else "NOT BEARISH"
        )

    # ========================================================
    # ATR
    # ========================================================

    atr = current["atr"]

    if pd.isna(atr):

        reasons.append(
            "ATR data not available"
        )

    # ========================================================
    # VALID SIGNAL
    # ========================================================

    valid_signal = (
        body_ok
        and len(reasons) == 0
    )

    signal = None
    entry = None
    sl = None
    tp = None

    if valid_signal:

        entry = close

        if pattern == "BULLISH ENGULFING":

            signal = "BUY"

            sl = (
                entry
                - SL_ATR_MULTIPLIER * atr
            )

            risk = entry - sl

            tp = (
                entry
                + RISK_REWARD * risk
            )

        elif pattern == "BEARISH ENGULFING":

            signal = "SELL"

            sl = (
                entry
                + SL_ATR_MULTIPLIER * atr
            )

            risk = sl - entry

            tp = (
                entry
                - RISK_REWARD * risk
            )

    return {
        "time": current["datetime"],
        "pattern": pattern,
        "body_ratio": body_ratio,
        "trend": trend,
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "ema50": ema50,
        "ema200": ema200,
        "reasons": reasons
    }


# ============================================================
# SCAN RECENT CANDLES
# ============================================================

def scan_market(df):

    results = []

    start_index = max(
        1,
        len(df) - SCAN_CANDLES
    )

    for i in range(
        start_index,
        len(df)
    ):

        result = analyze_candle(
            df,
            i
        )

        if result is not None:
            results.append(result)

    return results


# ============================================================
# REGISTER NEW SIGNALS
# ============================================================

def register_signals(
    market_code,
    market_name,
    results
):

    for result in results:

        if result["signal"] not in [
            "BUY",
            "SELL"
        ]:
            continue

        signal_key = (
            market_code,
            str(result["time"]),
            result["signal"]
        )

        if signal_key in st.session_state.seen_signals:
            continue

        st.session_state.seen_signals.add(
            signal_key
        )

        st.session_state.signals.append(
            {
                "Market": market_name,
                "Symbol": market_code,
                "Signal": result["signal"],
                "Time": result["time"],
                "Entry": result["entry"],
                "Stop Loss": result["sl"],
                "Take Profit": result["tp"],
                "ATR": result["atr"],
                "Body Ratio": result["body_ratio"]
            }
        )


# ============================================================
# DISPLAY SIGNAL
# ============================================================

def display_signal(result):

    if result["signal"] == "BUY":

        st.success(
            "🟢 BUY SIGNAL"
        )

    elif result["signal"] == "SELL":

        st.error(
            "🔴 SELL SIGNAL"
        )

    st.write(
        f"**Signal candle:** "
        f"{result['time'].strftime('%Y-%m-%d %H:%M UTC')}"
    )

    st.write(
        f"**Entry:** "
        f"`{format_price(result['entry'])}`"
    )

    st.write(
        f"**Suggested SL:** "
        f"`{format_price(result['sl'])}`"
    )

    st.write(
        f"**Suggested TP:** "
        f"`{format_price(result['tp'])}`"
    )

    st.write(
        f"**Body ratio:** "
        f"`{result['body_ratio']:.2f}x`"
    )

    st.warning(
        "⚠️ Alert only. Execute manually in XM MT5."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Strategy")

st.sidebar.write(
    "**Timeframe:** 15 minutes"
)

st.sidebar.write(
    "**Minimum body:** 1.2x"
)

st.sidebar.write(
    "**EMA:** 50 / 200"
)

st.sidebar.write(
    "**ATR:** 14"
)

st.sidebar.write(
    "**Stop Loss:** 1.5 × ATR"
)

st.sidebar.write(
    "**Take Profit:** 2R"
)

st.sidebar.divider()

st.sidebar.success(
    "Automatic checking is ON"
)

st.sidebar.warning(
    "The app does NOT place trades."
)

# ============================================================
# API CHECK
# ============================================================

if not API_KEY:

    st.error(
        "⚠️ Twelve Data API key is missing."
    )

    st.info(
        "Add TWELVE_DATA_API_KEY to Streamlit Secrets."
    )

    st.stop()


# ============================================================
# AUTOMATIC REFRESH
#
# Streamlit checks the markets every 30 seconds while
# this page remains open.
# ============================================================

@st.fragment(run_every="30s")
def market_monitor():

    st.header("📡 Live Market Monitor")

    st.caption(
        "The app checks for new completed 15-minute "
        "candles automatically."
    )

    columns = st.columns(2)

    for index, (
        market_code,
        market
    ) in enumerate(MARKETS.items()):

        with columns[index]:

            st.subheader(
                f"{market['icon']} "
                f"{market['name']}"
            )

            try:

                # ------------------------------------------------
                # GET DATA
                # ------------------------------------------------

                df = get_data(
                    market["symbol"]
                )

                # ------------------------------------------------
                # INDICATORS
                # ------------------------------------------------

                df = calculate_indicators(
                    df
                )

                # ------------------------------------------------
                # SCAN RECENT CANDLES
                # ------------------------------------------------

                results = scan_market(
                    df
                )

                # Register valid signals
                register_signals(
                    market_code,
                    market["name"],
                    results
                )

                # ------------------------------------------------
                # LATEST CANDLE
                # ------------------------------------------------

                latest = df.iloc[-1]

                st.metric(
                    "Last completed candle",
                    format_price(
                        latest["close"]
                    )
                )

                # ------------------------------------------------
                # TREND
                # ------------------------------------------------

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
                    f"EMA 50: "
                    f"`{format_price(latest['ema50'])}`"
                )

                st.write(
                    f"EMA 200: "
                    f"`{format_price(latest['ema200'])}`"
                )

                st.write(
                    f"ATR(14): "
                    f"`{format_price(latest['atr'])}`"
                )

                st.write(
                    "Last completed candle: "
                    f"`{latest['datetime'].strftime('%Y-%m-%d %H:%M UTC')}`"
                )

                # =================================================
                # VALID SIGNALS FOUND
                # =================================================

                valid_results = [
                    r
                    for r in results
                    if r["signal"] in [
                        "BUY",
                        "SELL"
                    ]
                ]

                if valid_results:

                    newest_signal = valid_results[-1]

                    st.divider()

                    display_signal(
                        newest_signal
                    )

                else:

                    st.info(
                        "No confirmed BUY/SELL signal "
                        "in the recent scan."
                    )

                # =================================================
                # ENGULFING CANDLES DETECTED
                # =================================================

                st.divider()

                st.write(
                    "### 🕯️ Recent Engulfing Candles"
                )

                if results:

                    display_rows = []

                    for result in reversed(
                        results[-10:]
                    ):

                        status = (
                            "✅ VALID SIGNAL"
                            if result["signal"]
                            else "❌ REJECTED"
                        )

                        reason = (
                            "Valid strategy setup"
                            if result["signal"]
                            else " | ".join(
                                result["reasons"]
                            )
                        )

                        display_rows.append(
                            {
                                "Time": result["time"].strftime(
                                    "%d %b %H:%M"
                                ),
                                "Pattern": result["pattern"],
                                "Body": f"{result['body_ratio']:.2f}x",
                                "Status": status,
                                "Reason": reason
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(
                            display_rows
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.write(
                        "No engulfing candles found "
                        "in the recent scan."
                    )

            except Exception as e:

                st.error(
                    f"Data error for {market['name']}"
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(e)
                    )


# ============================================================
# RUN MONITOR
# ============================================================

market_monitor()


# ============================================================
# SIGNAL HISTORY
# ============================================================

st.divider()

st.header("📜 Confirmed Signal History")

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
        "No confirmed signals detected yet."
    )


# ============================================================
# STRATEGY RULES
# ============================================================

st.divider()

st.header("📋 Strategy Rules")

st.markdown(
    """
### 🟢 BUY

- Completed bullish engulfing candle
- Current body ≥ **1.2×** previous candle body
- Close > EMA50
- EMA50 > EMA200
- Suggested SL = **1.5 × ATR(14)**
- Suggested TP = **2R**

### 🔴 SELL

- Completed bearish engulfing candle
- Current body ≥ **1.2×** previous candle body
- Close < EMA50
- EMA50 < EMA200
- Suggested SL = **1.5 × ATR(14)**
- Suggested TP = **2R**

### Important

The app **only generates alerts**.

It does **not** connect to XM for order execution,
does **not** place trades, and does **not** require
your XM username or password.
"""
)

st.caption(
    "Trading involves risk. Historical or simulated "
    "signals do not guarantee future results."
)
