"""
Daily market scanner + auto-trader for Robinhood Agentic account.

Checks Vol_Breakout and Trend_Momentum signals on SPY, QQQ, and BTC.
Places a $100 market buy when signal fires and buying power is available.
Logs every run to scanner_log.csv in the repo root.
"""
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Config — override via environment variables in GitHub Actions secrets
# ─────────────────────────────────────────────────────────────────────────────
RH_USERNAME   = os.environ.get("RH_USERNAME", "")
RH_PASSWORD   = os.environ.get("RH_PASSWORD", "")
RH_TOTP_KEY   = os.environ.get("RH_TOTP_KEY", "")     # base32 TOTP secret for 2FA
RH_ACCOUNT    = os.environ.get("RH_ACCOUNT", "726496482")   # Agentic account
DRY_RUN       = os.environ.get("DRY_RUN", "false").lower() == "true"

TRADE_AMOUNT  = float(os.environ.get("TRADE_AMOUNT", "100"))   # $ per signal
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "3"))       # max open positions

SCAN_TICKERS  = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

LOG_FILE = os.path.join(os.path.dirname(__file__), "scanner_log.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    v = df["Volume"]

    for p in [10, 20, 50, 200]:
        df[f"sma{p}"] = c.rolling(p).mean()
    df["ema12"] = c.ewm(span=12, adjust=False).mean()
    df["ema26"] = c.ewm(span=26, adjust=False).mean()
    df["macd"]  = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df["roc21"] = c.pct_change(21) * 100
    df["returns"] = c.pct_change()
    df["vol20"] = df["returns"].rolling(20).std() * math.sqrt(252)
    df["vol60"] = df["returns"].rolling(60).std() * math.sqrt(252)

    df["bb_mid"]   = c.rolling(20).mean()
    df["bb_std"]   = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["vol_sma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma20"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────
def signal_vol_breakout(row: pd.Series) -> bool:
    """Vol compression + price breaking 20-day high."""
    return (
        row["vol20"] < row["vol60"] * 0.75
        and row["Close"] > row.get("price_high20", float("inf"))
        and row["vol_ratio"] > 0.8
    )


def signal_trend_momentum(row: pd.Series) -> bool:
    """Price above SMA50, RSI in 45-65, MACD bullish, volume above avg."""
    return (
        row["Close"] > row["sma50"]
        and 45 <= row["rsi14"] <= 65
        and row["vol_ratio"] > 1.0
        and row["macd"] > row["macd_signal"]
    )


def signal_rsi_recovery(row: pd.Series) -> bool:
    """RSI recovering from oversold (<30) back above 40."""
    return (
        row.get("rsi_was_oversold", False)
        and row["rsi14"] > 40
        and row["Close"] > row["sma20"]
    )


def evaluate_ticker(symbol: str, period: str = "6mo") -> dict:
    """Download data, compute indicators, return signal verdict."""
    try:
        raw = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        if len(raw) < 60:
            return {"symbol": symbol, "signal": "no_data", "price": None, "reason": "insufficient data"}

        df = compute_indicators(raw)
        # Add rolling 20-day high (shifted by 1 to avoid lookahead)
        df["price_high20"] = df["Close"].rolling(20).max().shift(1)
        # RSI oversold in last 5 days
        df["rsi_was_oversold"] = df["rsi14"].rolling(5).min() < 30

        row = df.iloc[-1]
        price = float(row["Close"])

        # Evaluate all signals
        signals_fired = []
        if signal_vol_breakout(row):
            signals_fired.append("Vol_Breakout")
        if signal_trend_momentum(row):
            signals_fired.append("Trend_Momentum")
        if signal_rsi_recovery(row):
            signals_fired.append("RSI_Recovery")

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "signal": "BUY" if signals_fired else "HOLD",
            "strategies": signals_fired,
            "rsi": round(float(row["rsi14"]), 1) if pd.notna(row["rsi14"]) else None,
            "vol_ratio": round(float(row["vol_ratio"]), 2) if pd.notna(row["vol_ratio"]) else None,
            "macd_bullish": bool(row["macd"] > row["macd_signal"]) if pd.notna(row["macd"]) else None,
            "above_sma50": bool(row["Close"] > row["sma50"]) if pd.notna(row["sma50"]) else None,
            "reason": ", ".join(signals_fired) if signals_fired else "no signal",
        }
    except Exception as e:
        return {"symbol": symbol, "signal": "error", "price": None, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Robinhood auth + trading
# ─────────────────────────────────────────────────────────────────────────────
def get_totp_code(totp_key: str) -> str:
    """Generate current TOTP code from base32 secret."""
    import base64
    import hashlib
    import hmac
    import struct

    key = base64.b32decode(totp_key.upper().replace(" ", ""))
    counter = int(time.time()) // 30
    msg = struct.pack(">Q", counter)
    h = hmac.new(key, msg, hashlib.sha1).digest()
    offset = h[-1] & 0x0F
    code = struct.unpack(">I", h[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code % 1_000_000).zfill(6)


def login_robinhood():
    try:
        import robin_stocks.robinhood as rh
    except ImportError:
        print("robin_stocks not installed. Run: pip install robin_stocks")
        sys.exit(1)

    if not RH_USERNAME or not RH_PASSWORD:
        print("RH_USERNAME / RH_PASSWORD not set.")
        sys.exit(1)

    mfa = get_totp_code(RH_TOTP_KEY) if RH_TOTP_KEY else None
    login_kwargs = {"username": RH_USERNAME, "password": RH_PASSWORD,
                    "store_session": False, "by_sms": not bool(mfa)}
    if mfa:
        login_kwargs["mfa_code"] = mfa

    rh.login(**login_kwargs)
    return rh


def get_buying_power(rh) -> float:
    try:
        profile = rh.profiles.load_account_profile(account_number=RH_ACCOUNT)
        return float(profile.get("buying_power", 0))
    except Exception:
        portfolios = rh.profiles.load_portfolio_profile(account_number=RH_ACCOUNT)
        return float(portfolios.get("withdrawable_amount", 0))


def count_open_positions(rh) -> int:
    try:
        positions = rh.account.get_open_stock_positions(account_number=RH_ACCOUNT)
        return len([p for p in positions if float(p.get("quantity", 0)) > 0])
    except Exception:
        return 0


def place_order(rh, symbol: str, amount: float) -> dict:
    """Place a fractional dollar-amount market buy."""
    if DRY_RUN:
        print(f"[DRY RUN] Would buy ${amount} of {symbol}")
        return {"status": "dry_run", "symbol": symbol, "amount": amount}
    try:
        result = rh.orders.order_buy_fractional_by_price(
            symbol,
            amount,
            account_number=RH_ACCOUNT,
            timeInForce="gfd",
        )
        return {"status": result.get("state", "unknown"), "id": result.get("id"),
                "symbol": symbol, "amount": amount}
    except Exception as e:
        return {"status": "error", "symbol": symbol, "amount": amount, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def append_log(rows: list[dict]):
    file_exists = os.path.isfile(LOG_FILE)
    if not rows:
        return
    fields = ["timestamp", "symbol", "price", "signal", "strategies",
              "rsi", "vol_ratio", "action_taken", "order_status", "notes"]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*60}")
    print(f"Daily Scanner  |  {now}  |  DRY_RUN={DRY_RUN}")
    print(f"{'='*60}")

    all_tickers = SCAN_TICKERS + CRYPTO_TICKERS
    scan_results = []

    print("\n── Scanning tickers ──")
    for symbol in all_tickers:
        result = evaluate_ticker(symbol)
        result["timestamp"] = now
        scan_results.append(result)
        flag = "🟢 BUY" if result["signal"] == "BUY" else "⚪ HOLD"
        print(f"  {symbol:10s}  {flag:8s}  ${result['price'] or '---'}  "
              f"RSI={result.get('rsi')}  {result['reason']}")

    buy_signals = [r for r in scan_results if r["signal"] == "BUY"]
    print(f"\n── {len(buy_signals)} buy signal(s) fired ──")

    log_rows = []
    if not buy_signals:
        print("No signals. No trades placed.")
        for r in scan_results:
            log_rows.append({**r, "action_taken": "none", "order_status": "—", "notes": r["reason"]})
        append_log(log_rows)
        return

    # Login and check account state
    print("\nLogging in to Robinhood...")
    rh = login_robinhood()
    buying_power = get_buying_power(rh)
    open_positions = count_open_positions(rh)
    print(f"Buying power: ${buying_power:.2f}  |  Open positions: {open_positions}/{MAX_POSITIONS}")

    trades_this_run = 0
    for result in scan_results:
        action, order_status, notes = "none", "—", result["reason"]

        if result["signal"] == "BUY":
            can_trade = (
                buying_power >= TRADE_AMOUNT
                and open_positions + trades_this_run < MAX_POSITIONS
            )
            if can_trade:
                print(f"\n  Placing ${TRADE_AMOUNT} buy on {result['symbol']}...")
                order = place_order(rh, result["symbol"].replace("-USD", ""), TRADE_AMOUNT)
                action = "buy"
                order_status = order.get("status", "unknown")
                notes = f"order_id={order.get('id', 'n/a')}  strategies={result['reason']}"
                buying_power -= TRADE_AMOUNT
                trades_this_run += 1
                print(f"  → {order_status}  {notes}")
            else:
                reason = ("insufficient buying power" if buying_power < TRADE_AMOUNT
                          else f"max positions ({MAX_POSITIONS}) reached")
                action = "skipped"
                notes = f"signal fired but {reason}"
                print(f"  Skipped {result['symbol']}: {reason}")

        log_rows.append({**result, "action_taken": action,
                         "order_status": order_status, "notes": notes})

    append_log(log_rows)
    print(f"\n── Done. {trades_this_run} trade(s) placed. Log → scanner_log.csv ──\n")


if __name__ == "__main__":
    main()
