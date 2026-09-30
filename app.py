import streamlit as st
import pandas as pd
import requests
from zoneinfo import ZoneInfo

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
    "Gold + BTCUSD | South African Time | Manual trading only"
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

BODY_RATIO = 1.2
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
SL_ATR_MULTIPLIER = 1.5
RISK_REWARD = 2.0
SCAN_CANDLES = 80

# South African timezone
SA_TZ = ZoneInfo("Africa/Johannesburg")

# ============================================================
# SESSION STATE
# ============================================================

if "signals" not in st.session_state:
    st.session_state.signals = []

if "seen_signals" not in st.session_state:
    st.session_state.seen_signals = set()

# ============================================================
# PRICE FORMAT
# ============================================================

def format_price(value):

    if pd.isna(value):
        return "-"

    value = float(value)

    if value >= 1:
        return f"{value:,.2f}"

    return f"{value:,.5f}"


# ============================================================
# SOUTH AFRICAN TIME
# ============================================================

def sa_time(timestamp):

    if pd.isna(timestamp):
        return "-"

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    return timestamp.tz_convert(
        SA_TZ
    )


def sa_time_string(timestamp):

    return sa_time(timestamp).strftime(
        "%Y-%m-%d %H:%M SAST"
    )


# ============================================================
# GET MARKET DATA
# ============================================================

def get_data(symbol):

    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing."
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
                "Twelve Data error."
            )
        )

    if "values" not in data:
        raise RuntimeError(
            "No market data returned."
        )

    df = pd.DataFrame(
        data["values"]
    )

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

    # Remove the currently forming candle
    now = pd.Timestamp.now(
        tz="UTC"
    )

    current_candle = now.floor(
        "15min"
    )

    df = df[
        df["datetime"] < current_candle
    ].copy()

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

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

    df["atr"] = df["tr"].rolling(
        ATR_PERIOD
    ).mean()

    return df


# ============================================================
# ENGULFING DETECTION
# ============================================================

def check_engulfing(
    previous,
    current
):

    previous_body = abs(
        previous["close"]
        - previous["open"]
    )

    current_body = abs(
        current["close"]
        - current["open"]
    )

    if previous_body == 0:

        return {
            "pattern": "NONE",
            "body_ratio": 0
        }

    body_ratio = (
        current_body
        / previous_body
    )

    bullish = (
        previous["close"]
        < previous["open"]
        and current["close"]
        > current["open"]
        and current["open"]
        <= previous["close"]
        and current["close"]
        >= previous["open"]
    )

    bearish = (
        previous["close"]
        > previous["open"]
        and current["close"]
        < current["open"]
        and current["open"]
        >= previous["close"]
        and current["close"]
        <= previous["open"]
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

    if pattern == "NONE":
        return None

    body_ratio = engulfing[
        "body_ratio"
    ]

    reasons = []

    # Body requirement
    body_ok = (
        body_ratio >= BODY_RATIO
    )

    if not body_ok:

        reasons.append(
            f"Body is {body_ratio:.2f}x "
            f"instead of at least "
            f"{BODY_RATIO:.1f}x"
        )

    close = current["close"]
    ema50 = current["ema50"]
    ema200 = current["ema200"]
    atr = current["atr"]

    # Trend
    trend_ok = False

    if pd.isna(ema50) or pd.isna(ema200):

        reasons.append(
            "EMA data unavailable"
        )

    elif pattern == "BULLISH ENGULFING":

        trend_ok = (
            close > ema50
            and ema50 > ema200
        )

        if not trend_ok:

            reasons.append(
                "BUY trend condition failed"
            )

    elif pattern == "BEARISH ENGULFING":

        trend_ok = (
            close < ema50
            and ema50 < ema200
        )

        if not trend_ok:

            reasons.append(
                "SELL trend condition failed"
            )

    # ATR
    if pd.isna(atr):

        reasons.append(
            "ATR unavailable"
        )

    valid = (
        body_ok
        and trend_ok
        and not pd.isna(atr)
    )

    signal = None
    entry = None
    sl = None
    tp = None

    if valid:

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

        else:

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

    start = max(
        1,
        len(df) - SCAN_CANDLES
    )

    for i in range(
        start,
        len(df)
    ):

        result = analyze_candle(
            df,
            i
        )

        if result is not None:

            results.append(
                result
            )

    return results


# ============================================================
# SAVE SIGNAL
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

        key = (
            market_code,
            str(result["time"]),
            result["signal"]
        )

        if key in st.session_state.seen_signals:
            continue

        st.session_state.seen_signals.add(
            key
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

    else:

        st.error(
            "🔴 SELL SIGNAL"
        )

    st.write(
        "**Signal candle:** "
        + sa_time_string(
            result["time"]
        )
    )

    st.write(
        "**Entry:** `"
        + format_price(
            result["entry"]
        )
        + "`"
    )

    st.write(
        "**Suggested SL:** `"
        + format_price(
            result["sl"]
        )
        + "`"
    )

    st.write(
        "**Suggested TP:** `"
        + format_price(
            result["tp"]
        )
        + "`"
    )

    st.write(
        f"**Body ratio:** "
        f"`{result['body_ratio']:.2f}x`"
    )

    st.warning(
        "⚠️ Alert only. "
        "Execute manually in XM MT5."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Strategy"
)

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
    "**SL:** 1.5 × ATR"
)

st.sidebar.write(
    "**TP:** 2R"
)

st.sidebar.divider()

st.sidebar.success(
    "Automatic checking ON"
)

st.sidebar.info(
    "All displayed times are South African time."
)

# ============================================================
# API CHECK
# ============================================================

if not API_KEY:

    st.error(
        "⚠️ Twelve Data API key is missing."
    )

    st.stop()


# ============================================================
# LIVE MONITOR
# ============================================================

@st.fragment(run_every="30s")
def market_monitor():

    st.header(
        "📡 Live Market Monitor"
    )

    st.caption(
        "Checking every 30 seconds for "
        "new completed 15-minute candles."
    )

    columns = st.columns(2)

    for index, (
        market_code,
        market
    ) in enumerate(
        MARKETS.items()
    ):

        with columns[index]:

            st.subheader(
                market["icon"]
                + " "
                + market["name"]
            )

            try:

                df = get_data(
                    market["symbol"]
                )

                df = calculate_indicators(
                    df
                )

                results = scan_market(
                    df
                )

                register_signals(
                    market_code,
                    market["name"],
                    results
                )

                latest = df.iloc[-1]

                st.metric(
                    "Last completed candle",
                    format_price(
                        latest["close"]
                    )
                )

                # Trend
                trend = "NEUTRAL"

                if (
                    latest["close"]
                    > latest["ema50"]
                    and latest["ema50"]
                    > latest["ema200"]
                ):

                    trend = "BULLISH"

                elif (
                    latest["close"]
                    < latest["ema50"]
                    and latest["ema50"]
                    < latest["ema200"]
                ):

                    trend = "BEARISH"

                st.write(
                    f"**Trend:** {trend}"
                )

                st.write(
                    "EMA 50: `"
                    + format_price(
                        latest["ema50"]
                    )
                    + "`"
                )

                st.write(
                    "EMA 200: `"
                    + format_price(
                        latest["ema200"]
                    )
                    + "`"
                )

                st.write(
                    "ATR(14): `"
                    + format_price(
                        latest["atr"]
                    )
                    + "`"
                )

                st.write(
                    "**Last candle:** "
                    + sa_time_string(
                        latest["datetime"]
                    )
                )

                # Valid signals
                valid_results = [
                    r
                    for r in results
                    if r["signal"]
                    in ["BUY", "SELL"]
                ]

                if valid_results:

                    st.divider()

                    display_signal(
                        valid_results[-1]
                    )

                else:

                    st.info(
                        "No confirmed signal "
                        "in recent candles."
                    )

                # Engulfing table
                st.divider()

                st.write(
                    "### 🕯️ Recent Engulfing Candles"
                )

                if results:

                    rows = []

                    for result in reversed(
                        results[-10:]
                    ):

                        if result["signal"]:

                            status = (
                                "✅ VALID SIGNAL"
                            )

                            reason = (
                                "Valid setup"
                            )

                        else:

                            status = (
                                "❌ REJECTED"
                            )

                            reason = (
                                " | ".join(
                                    result["reasons"]
                                )
                            )

                        rows.append(
                            {
                                "Time": sa_time(
                                    result["time"]
                                ).strftime(
                                    "%d %b %H:%M SAST"
                                ),
                                "Pattern": result[
                                    "pattern"
                                ],
                                "Body": (
                                    f"{result['body_ratio']:.2f}x"
                                ),
                                "Status": status,
                                "Reason": reason
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.write(
                        "No engulfing candles "
                        "found recently."
                    )

            except Exception as e:

                st.error(
                    f"Data error for "
                    f"{market['name']}"
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(e)
                    )


market_monitor()


# ============================================================
# SIGNAL HISTORY
# ============================================================

st.divider()

st.header(
    "📜 Confirmed Signal History"
)

if st.session_state.signals:

    history = pd.DataFrame(
        st.session_state.signals
    )

    # Convert history to South African time
    history["Time"] = history[
        "Time"
    ].apply(
        lambda x:
        sa_time(x).strftime(
            "%Y-%m-%d %H:%M SAST"
        )
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
        "No confirmed signals yet."
    )


# ============================================================
# RULES
# ============================================================

st.divider()

st.header(
    "📋 Strategy Rules"
)

st.markdown(
    """
### 🟢 BUY

- Bullish engulfing
- Body ≥ 1.2× previous candle
- Close > EMA50
- EMA50 > EMA200
- SL = 1.5 × ATR
- TP = 2R

### 🔴 SELL

- Bearish engulfing
- Body ≥ 1.2× previous candle
- Close < EMA50
- EMA50 < EMA200
- SL = 1.5 × ATR
- TP = 2R

**All times shown in the app are South African time (SAST).**

**The app does not place trades.**
"""
)

st.caption(
    "Trading involves risk. "
    "Signals do not guarantee future results."
)
