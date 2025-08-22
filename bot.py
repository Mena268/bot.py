"""
YFINANCE SIGNAL BOT — Telegram (requests only, no telegram lib)
---------------------------------------------------------------
Works on Python 3.13 (no `imghdr` problem). Uses yfinance for data and
pure Telegram HTTP API via `requests` for alerts. Strategy = EMA(9/21) + MACD + RSI filter.

Quick Start
-----------
1) Save as: D:/Documents/trading_bot/bot.py
2) Install deps:
     py -m pip install --upgrade pip
     py -m pip install yfinance pandas numpy requests
3) Edit CONFIG below if needed (tickers, interval, etc.).
4) Run:
     cd /d D:\\Documents\\trading_bot && python bot.py

What you'll see
---------------
- On start: "✅ Bot connected..."
- Then signals ONLY when a bar CLOSES and rules confirm.
- BUY = UP (Call) on Quotex.  SELL = DOWN (Put) on Quotex.

Risk note
---------
No strategy is 100% win-rate. Start tiny position sizes.
"""

import time
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# =========================
# ======= CONFIG ==========
# =========================
# ✅ Your real token (NO SPACES)
TELEGRAM_BOT_TOKEN = "8430719482:AAG2jY9ssIFNlLYXGLPnxktKwc9QHV1xbTU"
# ✅ Your numeric chat id
TELEGRAM_CHAT_ID   = "7695071336"

# Symbols (you can add/remove)
TICKERS: List[str] = [
    "EURUSD=X",
    "GBPUSD=X",
    "BTC-USD",
]

# Timeframe & history
INTERVAL = "5m"            # "1m", "2m", "5m", "15m", "1h", "1d" ...
LOOKBACK_PERIOD = "2d"     # keep small for speed
SLEEP_SECONDS = 30          # loop delay

# Strategy params
RSI_LEN = 14
EMA_FAST = 9
EMA_SLOW = 21
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# =========================
# ===== UTILITIES =========
# =========================

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_print(*args):
    print(f"[{now_iso()}]", *args, flush=True)


def tg_send(text: str):
    """Send message via Telegram HTTP API (no external telegram lib)."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, data=data, timeout=15)
        if resp.status_code != 200:
            safe_print("Telegram send failed:", resp.status_code, resp.text[:200])
    except Exception as e:
        safe_print("Telegram exception:", e)


# =========================
# ===== INDICATORS ========
# =========================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# =========================
# ===== DATA FETCH =========
# =========================

def fetch_ohlc(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker} {interval} {period}")
    # standardize column names
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    # ensure timezone-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('UTC')
    return df


# =========================
# ==== SIGNAL LOGIC ========
# =========================

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, MACD, RSI and create BUY/SELL signals on the previous bar."""
    df = df.copy()
    df["EMA_F"] = ema(df["Close"], EMA_FAST)
    df["EMA_S"] = ema(df["Close"], EMA_SLOW)
    macd_line, signal_line, hist = macd(df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["MACD"] = macd_line
    df["MACD_SIG"] = signal_line
    df["MACD_HIST"] = hist
    df["RSI"] = rsi(df["Close"], RSI_LEN)

    # Cross conditions (use previous candle to confirm)
    df["EMA_CROSS_UP"] = (df["EMA_F"] > df["EMA_S"]) & (df["EMA_F"].shift(1) <= df["EMA_S"].shift(1))
    df["EMA_CROSS_DN"] = (df["EMA_F"] < df["EMA_S"]) & (df["EMA_F"].shift(1) >= df["EMA_S"].shift(1))

    df["MACD_CROSS_UP"] = (df["MACD"] > df["MACD_SIG"]) & (df["MACD"].shift(1) <= df["MACD_SIG"].shift(1))
    df["MACD_CROSS_DN"] = (df["MACD"] < df["MACD_SIG"]) & (df["MACD"].shift(1) >= df["MACD_SIG"].shift(1))

    # Final signal (confirmed bar = -2; -1 is still forming for most intervals)
    df["BUY"] = (
        df["EMA_CROSS_UP"].shift(1).fillna(False)
        & df["MACD_CROSS_UP"].shift(1).fillna(False)
        & (df["RSI"].shift(1) > 50)
        & (df["Close"].shift(1) > df["EMA_S"].shift(1))
    )

    df["SELL"] = (
        df["EMA_CROSS_DN"].shift(1).fillna(False)
        & df["MACD_CROSS_DN"].shift(1).fillna(False)
        & (df["RSI"].shift(1) < 50)
        & (df["Close"].shift(1) < df["EMA_S"].shift(1))
    )

    return df


def last_signal(df: pd.DataFrame) -> Optional[Tuple[str, pd.Timestamp, float]]:
    if df["BUY"].iloc[-1]:
        return ("BUY", df.index[-2], float(df["Close"].iloc[-2]))
    if df["SELL"].iloc[-1]:
        return ("SELL", df.index[-2], float(df["Close"].iloc[-2]))
    return None


# =========================
# ====== MAIN LOOP =========
# =========================

def run_loop():
    tg_send("✅ Bot connected. Signals will arrive on confirmed candles.")
    sent_on_bar = set()  # to avoid duplicate alerts per bar per ticker

    while True:
        for ticker in TICKERS:
            try:
                df = fetch_ohlc(ticker, INTERVAL, LOOKBACK_PERIOD)
                df = compute_signals(df)
                sig = last_signal(df)
                if sig is None:
                    continue

                side, bar_time, price = sig
                key = f"{ticker}|{bar_time.isoformat()}|{side}"
                if key in sent_on_bar:
                    continue

                msg = (
                    f"🔥 SIGNAL\n"
                    f"Ticker: {ticker}\n"
                    f"Interval: {INTERVAL}\n"
                    f"Signal: {side} (Quotex: {'UP' if side=='BUY' else 'DOWN'})\n"
                    f"Time (bar close): {bar_time} UTC\n"
                    f"Price: {price}\n"
                    f"RSI: {round(df['RSI'].iloc[-2],2)} | EMA9: {round(df['EMA_F'].iloc[-2],5)} | EMA21: {round(df['EMA_S'].iloc[-2],5)}\n"
                    f"MACD: {round(df['MACD'].iloc[-2],5)} vs Sig: {round(df['MACD_SIG'].iloc[-2],5)}\n"
                )
                tg_send(msg)
                sent_on_bar.add(key)

            except Exception as e:
                safe_print(f"Error for {ticker}: {e}")
                traceback.print_exc()

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    safe_print("Starting yfinance signal bot…")
    safe_print(f"Tickers: {TICKERS} | Interval: {INTERVAL} | Lookback: {LOOKBACK_PERIOD}")
    run_loop()
