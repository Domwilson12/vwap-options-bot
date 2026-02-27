import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime, time as dt_time
from scipy.stats import norm

# ==========================
# SETTINGS
# ==========================

SYMBOL = "SPY"
ACCOUNT_SIZE = 10000
RISK_PERCENT = 0.01
TARGET_DELTA = 0.60
DEVIATION_THRESHOLD = 0.006
RSI_PERIOD = 14

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

LAST_SIGNAL = None

# ==========================
# FUNCTIONS
# ==========================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vwap(group):
    tp = (group["High"] + group["Low"] + group["Close"]) / 3
    return (tp * group["Volume"]).cumsum() / group["Volume"].cumsum()

def send_discord(message):
    payload = {
        "content": f"@everyone\n{message}",
        "allowed_mentions": {
            "parse": ["everyone"]
        }
    }
    requests.post(WEBHOOK_URL, json=payload)

def estimate_delta(S, K, T, r=0.01, sigma=0.20, call=True):
    if T <= 0:
        return 0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    if call:
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

# ==========================
# MAIN LOOP
# ==========================

def run_bot():
    global LAST_SIGNAL

    print("Checking market...")

    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False)

    if df.empty:
        print("No data.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open","High","Low","Close","Volume"]].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    df = df[(df.index.time >= dt_time(9,30)) & (df.index.time <= dt_time(16,0))]

    if df.empty:
        print("Outside market hours.")
        return

    df["date"] = df.index.date
    df["vwap"] = df.groupby("date", group_keys=False).apply(compute_vwap)
    df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
    df.dropna(inplace=True)

    latest = df.iloc[-1]
    current_time = latest.name.time()

    if not (dt_time(10,0) <= current_time <= dt_time(15,30)):
        print("Outside trade window.")
        return

    deviation = (latest["Close"] - latest["vwap"]) / latest["vwap"]

    long_signal = deviation < -DEVIATION_THRESHOLD and latest["rsi"] < 30
    short_signal = deviation > DEVIATION_THRESHOLD and latest["rsi"] > 70

    if long_signal:
        current_signal = "LONG"
    elif short_signal:
        current_signal = "SHORT"
    else:
        current_signal = "NONE"

    if current_signal == LAST_SIGNAL or current_signal == "NONE":
        print("No new signal.")
        return

    LAST_SIGNAL = current_signal

    ticker = yf.Ticker(SYMBOL)
    expirations = ticker.options

    if not expirations:
        print("No expirations found.")
        return

    expiration = expirations[0]
    chain = ticker.option_chain(expiration)
    options = chain.calls if current_signal == "LONG" else chain.puts

    S = latest["Close"]
    T = 1/365

    best_option = None
    best_delta_diff = 999

    for _, row in options.iterrows():
        K = row["strike"]
        mid_price = (row["bid"] + row["ask"]) / 2

        if np.isnan(mid_price) or mid_price <= 0:
            continue

        delta = estimate_delta(S, K, T, call=(current_signal=="LONG"))
        diff = abs(abs(delta) - TARGET_DELTA)

        if diff < best_delta_diff:
            best_delta_diff = diff
            best_option = {
                "strike": K,
                "price": mid_price,
                "delta": delta
            }

    if not best_option:
        print("No suitable option found.")
        return

    entry_price = best_option["price"]
    delta = best_option["delta"]
    strike = best_option["strike"]

    risk_amount = ACCOUNT_SIZE * RISK_PERCENT
    contracts = int(risk_amount / (entry_price * 100))

    message = (
        f"🚨 VWAP OPTIONS SIGNAL 🚨\n\n"
        f"Direction: {current_signal}\n"
        f"Expiration: {expiration}\n"
        f"Strike: {strike}\n"
        f"Est Premium: ${round(entry_price,2)}\n"
        f"Delta: {round(delta,2)}\n\n"
        f"Contracts (1% risk): {contracts}"
    )

    send_discord(message)
    print("Signal sent.")

if __name__ == "__main__":
    while True:
        try:
            run_bot()
        except Exception as e:
            print("Error:", e)

        time.sleep(300)
