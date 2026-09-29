import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Engulfing Alert Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 15-Minute Engulfing Alert Scanner")
st.caption("Manual trading only — this app never places trades.")

# ============================================================
# TWELVE DATA
# ============================================================

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")

if not API_KEY:
    st.error("Twelve Data API key is missing from Streamlit Secrets.")
    st.stop()

BASE_URL = "https://api.twelvedata.com/time_series"


# These are the Twelve Data symbols we can use directly.
MARKETS = {
    "XAUUSD": {
        "name": "Gold",
        "symbol": "XAU/USD"
    },
    "BTCUSD": {
        "name": "Bitcoin",
        "symbol": "BTC/USD"
    },
    "OIL": {
        "name": "WTI Oil",
        "symbol": "WTI/USD"
    },

    # IMPORTANT:
    # We will verify the exact NAS100/NASDAQ symbol separately.
    "NASDAQ": {
        "name": "NASDAQ / NAS100",
        "symbol": "NDX"
    }
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Strategy Settings")

strict_mode = st.sidebar.checkbox(
    "Strict mode",
    value=True
)

body_ratio = st.sidebar.number_input(
    "Minimum engulfing body ratio",
    min_value=1.0,
    max_value=3.0,
    value=1.2,
    step=0.1
)

ema_fast_period = st.sidebar.number_input(
    "Fast EMA",
    min_value=5,
    max_value=100,
    value=50,
    step=1
)

ema_slow_period = st.sidebar.number_input(
    "Slow EMA",
    min_value=50,
    max_value=300,
    value=200,
    step=1
)

atr_period = st.sidebar.number_input(
    "ATR period",
    min_value=5,
    max_value=50,
    value=14,
    step=1
)

sl_atr_multiplier = st.sidebar.number_input(
    "SL = ATR ×",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1
)

risk_reward = st.sidebar.number_input(
    "Risk / Reward",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

refresh_seconds = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=15,
    max_value=300,
    value=60,
    step=15
)


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=45)
def get_candles(symbol, outputsize=300):

    params = {
        "symbol": symbol,
        "interval": "15min",
        "outputsize": outputsize,
        "timezone": "UTC",
        "order": "asc",
        "apikey": API_KEY
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(
            data.get("message", "Twelve Data returned an error.")
        )

    if "values" not in data:
        raise RuntimeError(
            "No candle data returned."
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            "Empty candle response."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    df["ema_fast"] = (
        df["close"]
        .ewm(
            span=ema_fast_period,
            adjust=False
        )
        .mean()
    )

    df["ema_slow"] = (
        df["close"]
        .ewm(
            span=ema_slow_period,
            adjust=False
        )
        .mean()
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(atr_period)
        .mean()
    )

    return df


# ============================================================
# ENGULFING DETECTOR
# ============================================================

def detect_signal(df):

    if len(df) < max(
        ema_slow_period + 5,
        atr_period + 5,
        10
    ):
        return None

    # IMPORTANT:
    # We ignore the newest candle because it may still be forming.
    #
    # The last COMPLETED candle is therefore -2.
    #
    # The candle before it is -3.

    previous = df.iloc[-3]
    current = df.iloc[-2]

    current_body = abs(
        current["close"] - current["open"]
    )

    previous_body = abs(
        previous["close"] - previous["open"]
    )

    if previous_body == 0:
        return None

    body_ratio_actual = (
        current_body / previous_body
    )

    bullish_engulfing = (
        previous["close"] < previous["open"]
        and
        current["close"] > current["open"]
        and
        current["open"] <= previous["close"]
        and
        current["close"] >= previous["open"]
    )

    bearish_engulfing = (
        previous["close"] > previous["open"]
        and
        current["close"] < current["open"]
        and
        current["open"] >= previous["close"]
        and
        current["close"] <= previous["open"]
    )

    if strict_mode:
        if body_ratio_actual < body_ratio:
            return None

    ema50 = current["ema_fast"]
    ema200 = current["ema_slow"]
    atr = current["atr"]

    if pd.isna(ema50) or pd.isna(ema200) or pd.isna(atr):
        return None

    entry = float(current["close"])

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if bullish_engulfing:

        if strict_mode:

            if not (
                entry > ema50 > ema200
            ):
                return None

        stop_loss = entry - (
            sl_atr_multiplier * atr
        )

        risk = entry - stop_loss

        take_profit = entry + (
            risk_reward * risk
        )

        return {
            "direction": "BUY",
            "time": current["datetime"],
            "entry": entry,
            "sl": stop_loss,
            "tp": take_profit,
            "atr": float(atr),
            "body_ratio": float(body_ratio_actual)
        }

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if bearish_engulfing:

        if strict_mode:

            if not (
                entry < ema50 < ema200
            ):
                return None

        stop_loss = entry + (
            sl_atr_multiplier * atr
        )

        risk = stop_loss - entry

        take_profit = entry - (
            risk_reward * risk
        )

        return {
            "direction": "SELL",
            "time": current["datetime"],
            "entry": entry,
            "sl": stop_loss,
            "tp": take_profit,
            "atr": float(atr),
            "body_ratio": float(body_ratio_actual)
        }

    return None


# ============================================================
# MARKET CARD
# ============================================================

def render_market(name, market):

    st.subheader(name)

    try:

        df = get_candles(
            market["symbol"],
            outputsize=300
        )

        df = calculate_indicators(df)

        # Latest completed candle
        completed = df.iloc[-2]

        signal = detect_signal(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Last price",
                f"{completed['close']:,.5f}"
            )

        with col2:
            st.metric(
                "EMA 50",
                f"{completed['ema_fast']:,.5f}"
            )

        with col3:
            st.metric(
                "EMA 200",
                f"{completed['ema_slow']:,.5f}"
            )

        st.caption(
            "Last completed candle: "
            + completed["datetime"].strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        )

        if signal:

            direction = signal["direction"]

            if direction == "BUY":
                st.success("🟢 BUY ALERT")
            else:
                st.error("🔴 SELL ALERT")

            a, b, c, d = st.columns(4)

            with a:
                st.metric(
                    "Entry",
                    f"{signal['entry']:,.5f}"
                )

            with b:
                st.metric(
                    "Suggested SL",
                    f"{signal['sl']:,.5f}"
                )

            with c:
                st.metric(
                    "Suggested TP",
                    f"{signal['tp']:,.5f}"
                )

            with d:
                st.metric(
                    "Body ratio",
                    f"{signal['body_ratio']:.2f}x"
                )

            st.info(
                "Manual execution only. "
                "Check the corresponding XM MT5 instrument "
                "before placing any trade."
            )

        else:

            st.info(
                "No qualifying engulfing signal "
                "on the latest completed 15-minute candle."
            )

        return signal

    except Exception as e:

        st.error(
            f"Data error for {name}: {e}"
        )

        return None


# ============================================================
# MAIN DASHBOARD
# ============================================================

signals = {}

for key, market in MARKETS.items():

    signals[key] = render_market(
        market["name"],
        market
    )

    st.divider()


# ============================================================
# ALERT HISTORY
# ============================================================

st.header("Alert History")

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []


for market_key, signal in signals.items():

    if signal:

        alert_id = (
            f"{market_key}_"
            f"{signal['direction']}_"
            f"{signal['time']}"
        )

        existing_ids = [
            x["id"]
            for x in st.session_state.alert_history
        ]

        if alert_id not in existing_ids:

            st.session_state.alert_history.insert(
                0,
                {
                    "id": alert_id,
                    "market": market_key,
                    "direction": signal["direction"],
                    "time": signal["time"],
                    "entry": signal["entry"],
                    "sl": signal["sl"],
                    "tp": signal["tp"]
                }
            )


if st.session_state.alert_history:

    history_df = pd.DataFrame(
        st.session_state.alert_history
    )

    history_df = history_df.drop(
        columns=["id"],
        errors="ignore"
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No alerts detected yet.")


# ============================================================
# AUTO REFRESH
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Auto-refresh target: {refresh_seconds}s"
)

st.markdown(
    """
    <script>
    setTimeout(function() {
        window.location.reload();
    }, %d);
    </script>
    """ % (refresh_seconds * 1000),
    unsafe_allow_html=True
)
