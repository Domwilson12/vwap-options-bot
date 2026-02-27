import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, time
from scipy.stats import norm

# ==========================
# SETTINGS
# ==========================

WEBHOOK_URL = "https://discord.com/api/webhooks/1476365069179682858/epTmmPV2WEhkmK59kLpp01f9SvbHWbvKi7DWo9Lo0JjhD0_P0WroOZtS4R-_NfUc46zm"
SYMBOL = "SPY"
ACCOUNT_SIZE = 10000
RISK_PERCENT = 0.01
TARGET_DELTA = 0.60
DEVIATION_THRESHOLD = 0.006
RSI_PERIOD = 14
STATE_FILE = "bot_state.txt"

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
    requests.post(WEBHOOK_URL, json={"content": message})

def get_last_signal():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return None

def save_signal(signal):
    with open(STATE_FILE, "w") as f:
        f.write(signal)

def estimate_delta(S, K, T, r=0.01, sigma=0.20, call=True):
    if T <= 0:
        return 0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    if call:
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

# ==========================
# GET PRICE DATA
# ==========================

df = yf.download(SYMBOL, period="5d", interval="5m", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open","High","Low","Close","Volume"]].copy()

if df.index.tz is None:
    df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
else:
    df.index = df.index.tz_convert("US/Eastern")

df = df[(df.index.time >= time(9,30)) & (df.index.time <= time(16,0))]

df["date"] = df.index.date
df["vwap"] = df.groupby("date", group_keys=False).apply(compute_vwap)
df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
df.dropna(inplace=True)

latest = df.iloc[-1]
current_time = latest.name.time()

if not (time(10,0) <= current_time <= time(15,30)):
    print("Outside trade window.")
    exit()

deviation = (latest["Close"] - latest["vwap"]) / latest["vwap"]

long_signal = deviation < -DEVIATION_THRESHOLD and latest["rsi"] < 30
short_signal = deviation > DEVIATION_THRESHOLD and latest["rsi"] > 70

last_signal = get_last_signal()

if long_signal:
    current_signal = "LONG"
elif short_signal:
    current_signal = "SHORT"
else:
    current_signal = "NONE"

if current_signal != last_signal and current_signal in ["LONG", "SHORT"]:

    ticker = yf.Ticker(SYMBOL)
    expirations = ticker.options
    expiration = expirations[0]  # nearest expiry

    chain = ticker.option_chain(expiration)
    options = chain.calls if current_signal == "LONG" else chain.puts

    S = latest["Close"]
    T = 1/365  # assume 1 day to expiry for intraday

    best_option = None
    best_delta_diff = 999

    for _, row in options.iterrows():
        K = row["strike"]
        mid_price = (row["bid"] + row["ask"]) / 2
        delta = estimate_delta(S, K, T, call=(current_signal=="LONG"))

        if np.isnan(mid_price) or mid_price <= 0:
            continue

        diff = abs(abs(delta) - TARGET_DELTA)
        if diff < best_delta_diff:
            best_delta_diff = diff
            best_option = {
                "strike": K,
                "price": mid_price,
                "delta": delta
            }

    if best_option:

        entry_price = best_option["price"]
        delta = best_option["delta"]
        strike = best_option["strike"]

        target_move = latest["vwap"] - S
        expected_option_move = delta * target_move

        expected_profit = expected_option_move * 100
        risk_per_contract = entry_price * 100

        risk_amount = ACCOUNT_SIZE * RISK_PERCENT
        contracts = int(risk_amount / risk_per_contract)

        message = (
            f"🚨 VWAP OPTIONS SIGNAL 🚨\n\n"
            f"Direction: {current_signal}\n"
            f"Expiration: {expiration}\n"
            f"Strike: {strike}\n"
            f"Est. Option Price: ${round(entry_price,2)}\n"
            f"Delta: {round(delta,2)}\n\n"
            f"Contracts (1% risk): {contracts}\n"
            f"Expected Profit if VWAP hit: ${round(expected_profit * contracts,2)}"
        )

        send_discord(message)
        print("Advanced options signal sent.")

    save_signal(current_signal)

else:
    print("No new signal.")