"""
Institutional-Grade Quantitative Backtest Framework
Multi-Strategy | Multi-Regime | Walk-Forward | News Sentiment | Options & Crypto Simulation
"""
import math
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE = {
    "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM"],
    "Mega Tech":      ["AAPL", "MSFT", "GOOGL", "META"],
    "EV/Auto":        ["TSLA", "F", "GM", "RIVN"],
    "Finance":        ["JPM", "BAC", "GS", "MS"],
    "ETF Benchmarks": ["SPY", "QQQ", "IWM", "XLK"],
}
CRYPTO_UNIVERSE = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD"}
ALL_TICKERS = [t for g in UNIVERSE.values() for t in g]
DATA_URL = "https://data.alpaca.markets"
RISK_FREE = 0.04
IV_PREMIUM = 1.15
TARGET_DELTA = 0.30
DTE_BARS = 21

# Ticker → search terms for news
TICKER_SEARCH = {
    "NVDA": "NVIDIA", "AMD": "AMD chip", "INTC": "Intel",   "QCOM": "Qualcomm",
    "AAPL": "Apple",  "MSFT": "Microsoft", "GOOGL": "Google Alphabet", "META": "Meta Facebook",
    "TSLA": "Tesla",  "F": "Ford Motor",  "GM": "General Motors", "RIVN": "Rivian",
    "JPM":  "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "SPY":  "S&P 500",  "QQQ": "Nasdaq",  "IWM": "Russell 2000", "XLK": "Tech ETF",
    "BTC":  "Bitcoin",  "ETH": "Ethereum","SOL": "Solana",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetch — Alpaca
# ─────────────────────────────────────────────────────────────────────────────
def fetch_bars(symbol, days, headers):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(days * 1.5) + 30)).strftime("%Y-%m-%d")
    r = requests.get(
        f"{DATA_URL}/v2/stocks/{symbol}/bars", headers=headers,
        params={"timeframe": "1Day", "start": start, "end": end,
                "limit": 10000, "adjustment": "split", "feed": "iex"},
    )
    if r.status_code != 200 or len(r.json().get("bars", [])) < 100:
        return None
    bars = r.json()["bars"]
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"]).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)


def fetch_crypto_bars(symbol, days, headers):
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    r = requests.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars", headers=headers,
        params={"symbols": symbol, "timeframe": "1Day", "start": start, "limit": 10000},
    )
    if r.status_code != 200:
        return None
    bars = r.json().get("bars", {}).get(symbol, [])
    if len(bars) < 50:
        return None
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"]).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# News Fetch — NewsAPI
# ─────────────────────────────────────────────────────────────────────────────
def fetch_news(query, news_api_key, days_back=7, page_size=10):
    """Fetch recent news headlines for a query from NewsAPI."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    r = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query, "from": from_date, "sortBy": "relevancy",
            "pageSize": page_size, "language": "en", "apiKey": news_api_key,
        },
        timeout=10,
    )
    if r.status_code != 200:
        return []
    return r.json().get("articles", [])


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Analysis — VADER (no API key required)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def score_sentiment(text, analyzer):
    """Return compound score [-1, 1] and label."""
    if analyzer is None or not text:
        return 0.0, "neutral"
    scores = analyzer.polarity_scores(str(text))
    c = scores["compound"]
    label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
    return round(c, 4), label


def aggregate_ticker_sentiment(ticker, news_api_key, analyzer, days_back=7):
    """Fetch news + score sentiment for one ticker. Returns summary dict."""
    query = TICKER_SEARCH.get(ticker, ticker)
    articles = fetch_news(query, news_api_key, days_back=days_back, page_size=15)
    if not articles:
        return {"ticker": ticker, "articles": [], "avg_score": 0.0,
                "label": "neutral", "n": 0, "bullish": 0, "bearish": 0, "neutral_n": 0}

    scored = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '')}"
        score, label = score_sentiment(text, analyzer)
        scored.append({
            "date": str(a.get("publishedAt", ""))[:10],
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "score": score,
            "label": label,
            "url": a.get("url", ""),
        })

    avg = np.mean([s["score"] for s in scored])
    label = "positive" if avg >= 0.05 else "negative" if avg <= -0.05 else "neutral"
    return {
        "ticker": ticker,
        "articles": scored,
        "avg_score": round(float(avg), 4),
        "label": label,
        "n": len(scored),
        "bullish": sum(1 for s in scored if s["label"] == "positive"),
        "bearish": sum(1 for s in scored if s["label"] == "negative"),
        "neutral_n": sum(1 for s in scored if s["label"] == "neutral"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Options Simulation driven by sentiment
# ─────────────────────────────────────────────────────────────────────────────
def _bs_price(S, K, T, sigma, kind, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)
    return S * _norm.cdf(d1) - K * math.exp(-r * T) * _norm.cdf(d2)


def _bs_delta(S, K, T, sigma, kind, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return float(_norm.cdf(d1)) if kind == "call" else float(_norm.cdf(d1) - 1)


def _strike_for_delta(S, T, sigma, delta_target, kind, r=RISK_FREE):
    d1 = _norm.ppf(1 - delta_target) if kind == "put" else _norm.ppf(delta_target)
    return S * math.exp((r + sigma ** 2 / 2) * T - sigma * math.sqrt(T) * d1)


def simulate_news_options_trades(ticker, S, realized_vol, sentiment_score, sentiment_label,
                                 capital=100_000, dte_days=21, contracts=1):
    """
    Given sentiment, propose option trades and simulate P&L across price scenarios.

    Sentiment → strategy mapping:
      strong positive (>0.3)   → Buy ATM Call
      moderate positive        → Bull Call Spread (buy 0.40Δ call, sell 0.20Δ call)
      neutral                  → Iron Condor (sell 0.20Δ straddle wings) or no trade
      moderate negative        → Bear Put Spread
      strong negative (<-0.3)  → Buy ATM Put
    """
    T = dte_days / 252
    # IV = realized vol * premium + sentiment IV bump
    sentiment_iv_bump = abs(sentiment_score) * 0.15  # strong news spikes IV
    iv = max(realized_vol * IV_PREMIUM + sentiment_iv_bump, 0.10)

    abs_s = abs(sentiment_score)
    strong = abs_s >= 0.3

    if sentiment_label == "positive" and strong:
        strategy = "Buy ATM Call"
        K = round(S, 0)
        cost = _bs_price(S, K, T, iv, "call") * 100 * contracts
        delta = _bs_delta(S, K, T, iv, "call")
        legs = [{"type": "call", "action": "buy", "K": K, "iv": iv}]

    elif sentiment_label == "positive":
        strategy = "Bull Call Spread"
        K_buy = _strike_for_delta(S, T, iv, 0.40, "call")
        K_sell = _strike_for_delta(S, T, iv, 0.20, "call")
        cost = (_bs_price(S, K_buy, T, iv, "call") - _bs_price(S, K_sell, T, iv, "call")) * 100 * contracts
        delta = _bs_delta(S, K_buy, T, iv, "call") - _bs_delta(S, K_sell, T, iv, "call")
        legs = [{"type": "call", "action": "buy", "K": round(K_buy, 1), "iv": iv},
                {"type": "call", "action": "sell", "K": round(K_sell, 1), "iv": iv}]

    elif sentiment_label == "negative" and strong:
        strategy = "Buy ATM Put"
        K = round(S, 0)
        cost = _bs_price(S, K, T, iv, "put") * 100 * contracts
        delta = _bs_delta(S, K, T, iv, "put")
        legs = [{"type": "put", "action": "buy", "K": K, "iv": iv}]

    elif sentiment_label == "negative":
        strategy = "Bear Put Spread"
        K_buy = _strike_for_delta(S, T, iv, 0.40, "put")
        K_sell = _strike_for_delta(S, T, iv, 0.20, "put")
        cost = (_bs_price(S, K_buy, T, iv, "put") - _bs_price(S, K_sell, T, iv, "put")) * 100 * contracts
        delta = _bs_delta(S, K_buy, T, iv, "put") - _bs_delta(S, K_sell, T, iv, "put")
        legs = [{"type": "put", "action": "buy", "K": round(K_buy, 1), "iv": iv},
                {"type": "put", "action": "sell", "K": round(K_sell, 1), "iv": iv}]

    else:
        strategy = "No Trade (neutral)"
        return {
            "ticker": ticker, "strategy": strategy, "cost": 0, "delta": 0,
            "iv": round(iv * 100, 1), "legs": [], "scenarios": [],
            "sentiment": sentiment_label, "sentiment_score": sentiment_score,
        }

    # Simulate P&L across ±20% price scenarios at expiry
    scenarios = []
    for pct in [-0.20, -0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15, +0.20]:
        S_exp = S * (1 + pct)
        pnl = 0.0
        for leg in legs:
            K_leg = leg["K"]
            if leg["type"] == "call":
                intrinsic = max(S_exp - K_leg, 0.0) * 100 * contracts
            else:
                intrinsic = max(K_leg - S_exp, 0.0) * 100 * contracts
            pnl += intrinsic if leg["action"] == "buy" else -intrinsic
        net_pnl = pnl - cost
        scenarios.append({
            "Price Move": f"{pct:+.0%}",
            "S at Expiry": round(S_exp, 2),
            "Gross Value": round(pnl, 2),
            "Net P&L ($)": round(net_pnl, 2),
            "Net P&L (%)": round(net_pnl / max(cost, 1) * 100, 1) if cost > 0 else 0,
        })

    return {
        "ticker": ticker, "strategy": strategy,
        "cost": round(cost, 2), "delta": round(delta, 3),
        "iv": round(iv * 100, 1), "legs": legs,
        "scenarios": scenarios,
        "sentiment": sentiment_label, "sentiment_score": sentiment_score,
        "S": S, "dte": dte_days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Crypto Simulation driven by sentiment
# ─────────────────────────────────────────────────────────────────────────────
def simulate_crypto_trade(ticker, S, realized_vol, sentiment_score, sentiment_label,
                           capital=100_000, hold_days=7):
    """
    Simulate a spot long/short crypto trade based on news sentiment.
    Position size scaled by |sentiment_score| (0→2% of capital, 1→10%).
    """
    pos_pct = 0.02 + abs(sentiment_score) * 0.08   # 2–10% of capital
    pos_size = capital * pos_pct
    direction = 1 if sentiment_label == "positive" else -1 if sentiment_label == "negative" else 0

    if direction == 0:
        return {"ticker": ticker, "direction": "HOLD", "pos_size": 0,
                "scenarios": [], "sentiment": sentiment_label, "sentiment_score": sentiment_score}

    daily_vol = realized_vol / math.sqrt(252)
    scenarios = []
    for pct in [-0.20, -0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15, +0.20]:
        pnl = pos_size * direction * pct
        scenarios.append({
            "Price Move": f"{pct:+.0%}",
            "S at Close": round(S * (1 + pct), 2),
            "P&L ($)": round(pnl, 2),
            "P&L (%)": round(pnl / pos_size * 100, 1),
        })

    # Expected value using normal distribution over hold_days
    mu = 0.0  # no drift assumption
    sigma_hold = daily_vol * math.sqrt(hold_days)
    ev = pos_size * direction * mu  # neutral drift EV
    prob_profit = float(_norm.cdf(0, loc=mu, scale=sigma_hold)) if direction == -1 else float(1 - _norm.cdf(0, loc=mu, scale=sigma_hold))

    return {
        "ticker": ticker,
        "direction": "LONG" if direction == 1 else "SHORT",
        "pos_size": round(pos_size, 2),
        "pos_pct": round(pos_pct * 100, 1),
        "ev": round(ev, 2),
        "prob_profit": round(prob_profit * 100, 1),
        "hold_days": hold_days,
        "ann_vol": round(realized_vol * 100, 1),
        "scenarios": scenarios,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "S": S,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    for p in [10, 20, 50, 200]:
        df[f"sma{p}"] = c.rolling(p).mean()
    for p in [12, 26]:
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    for p in [5, 10, 21, 63]:
        df[f"roc{p}"] = c.pct_change(p) * 100
    df["returns"] = c.pct_change()
    df["vol20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["vol60"] = df["returns"].rolling(60).std() * np.sqrt(252)
    df["bb_mid"] = c.rolling(20).mean()
    df["bb_std"] = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["vol_sma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma20"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Regime Detection
# ─────────────────────────────────────────────────────────────────────────────
def classify_regimes(spy_df):
    df = spy_df.copy()
    df["sma200_slope"] = df["sma200"].pct_change(21)
    df["regime"] = "SIDEWAYS"
    df.loc[(df["close"] > df["sma200"]) & (df["sma200_slope"] > 0.005), "regime"] = "BULL"
    df.loc[(df["close"] < df["sma200"]) & (df["sma200_slope"] < -0.005), "regime"] = "BEAR"
    return df[["date", "regime"]]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Signals
# ─────────────────────────────────────────────────────────────────────────────
def signal_trend_momentum(df):
    s = pd.Series(0, index=df.index)
    s[(df["close"] > df["sma50"]) & (df["rsi14"] >= 45) & (df["rsi14"] <= 65) &
      (df["vol_ratio"] > 1.0) & (df["macd"] > df["macd_signal"])] = 1
    return s


def signal_mean_reversion(df):
    s = pd.Series(0, index=df.index)
    s[(df["bb_pct"] < 0.1) & (df["rsi14"] < 35) & (df["vol_ratio"] > 1.5)] = 1
    return s


def signal_volatility_breakout(df):
    s = pd.Series(0, index=df.index)
    price_high20 = df["close"].rolling(20).max().shift(1)
    s[(df["vol20"] < df["vol60"] * 0.75) & (df["close"] > price_high20) & (df["vol_ratio"] > 0.8)] = 1
    return s


def signal_trend_following(df):
    s = pd.Series(0, index=df.index)
    s[(df["sma50"] > df["sma200"]) & (df["close"] > df["sma50"]) & (df["roc21"] > 0)] = 1
    return s


def signal_rsi_divergence(df):
    s = pd.Series(0, index=df.index)
    rsi_was_oversold = df["rsi14"].rolling(5).min() < 30
    rsi_recovering = (df["rsi14"] > 40) & (df["rsi14"] > df["rsi14"].shift(2))
    s[rsi_was_oversold & rsi_recovering & (df["close"] > df["sma20"])] = 1
    return s


STRATEGIES = {
    "Trend_Momentum":  signal_trend_momentum,
    "Mean_Reversion":  signal_mean_reversion,
    "Vol_Breakout":    signal_volatility_breakout,
    "Trend_Following": signal_trend_following,
    "RSI_Recovery":    signal_rsi_divergence,
}


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────
def backtest_strategy(df, signal_fn, capital=100_000, pos_pct=0.10,
                      stop_loss=0.05, cost_bps=8, periods_per_year=252):
    df = df.copy().reset_index(drop=True)
    signals = signal_fn(df)
    cash, position, trades, equity = capital, None, [], [capital]
    cost_mult = cost_bps / 10000

    for i in range(50, len(df) - 1):
        row, next_row = df.iloc[i], df.iloc[i + 1]
        price, nxt = row["close"], next_row["open"]
        if pd.isna(row.get("rsi14")) or pd.isna(row.get("sma50")):
            equity.append(cash if not position else cash + position["shares"] * price)
            continue
        if position:
            dd = (position["entry"] - price) / position["entry"]
            roc = (price - position["entry"]) / position["entry"]
            exit_r = None
            if dd >= stop_loss:
                exit_r = f"StopLoss {dd*100:.1f}%"
            elif row["rsi14"] > 75:
                exit_r = f"RSI overbought"
            elif row["sma50"] < row["sma200"] and signal_fn.__name__ == "signal_trend_following":
                exit_r = "Death cross"
            if exit_r:
                sell_px = nxt * (1 - cost_mult)
                pnl = position["shares"] * sell_px - position["cost"]
                cash += position["shares"] * sell_px
                trades.append({"date": str(next_row["date"]), "action": "SELL",
                                "price": round(sell_px, 2), "pnl": round(pnl, 2), "reason": exit_r})
                position = None
        elif signals.iloc[i] == 1:
            spend = min(capital * pos_pct, cash)
            if spend > 500:
                buy_px = nxt * (1 + cost_mult)
                cash -= spend
                position = {"entry": buy_px, "shares": spend / buy_px, "cost": spend}
                trades.append({"date": str(next_row["date"]), "action": "BUY",
                                "price": round(buy_px, 2), "pnl": None, "reason": "Signal"})
        equity.append(cash + (position["shares"] * price if position else 0))

    if position:
        lp = df.iloc[-1]["close"]
        equity.append(cash + position["shares"] * lp)

    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    sells = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    ann_ret = (eq[-1] / eq[0]) ** (periods_per_year / len(eq)) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year) if len(rets) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    down_rets = rets[rets < 0]
    down_vol = down_rets.std() * np.sqrt(periods_per_year) if len(down_rets) > 1 else 0
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    peaks = np.maximum.accumulate(eq)
    max_dd = float(((peaks - eq) / peaks).max())
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    pnls = [t["pnl"] for t in sells]
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gw / gl if gl > 0 else float("inf")

    return {
        "trades": trades, "equity": list(eq),
        "total_trades": len(sells),
        "win_rate": round(len(wins) / len(sells) * 100) if sells else 0,
        "total_return": round((eq[-1] - capital) / capital * 100, 2),
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 2),
        "final_cap": round(eq[-1], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Options Wheel (existing)
# ─────────────────────────────────────────────────────────────────────────────
def wheel_backtest(df, capital=100_000, cost_bps=8):
    df = df.dropna(subset=["vol20"]).reset_index(drop=True)
    if len(df) < DTE_BARS + 5:
        return None
    cash, shares, opt = capital, 0.0, None
    equity, events = [], []
    cm = cost_bps / 10000

    for i in range(len(df)):
        S = df["close"].iloc[i]
        sigma_now = max(df["vol20"].iloc[i] * IV_PREMIUM, 0.05)
        if opt and i >= opt["expiry_i"]:
            K, q = opt["K"], opt["qty"]
            if opt["kind"] == "put":
                if S < K:
                    cash -= K * q * (1 + cm)
                    shares += q
                    events.append(("ASSIGNED", i, K))
                else:
                    events.append(("PUT_EXPIRED", i, K))
            else:
                if S > K:
                    cash += K * shares * (1 - cm)
                    events.append(("CALLED_AWAY", i, K))
                    shares = 0.0
                else:
                    events.append(("CALL_EXPIRED", i, K))
            opt = None
        if opt is None and i + DTE_BARS < len(df):
            T = DTE_BARS / 252
            if shares == 0:
                K = _strike_for_delta(S, T, sigma_now, TARGET_DELTA, "put")
                qty = cash / K
                cash += _bs_price(S, K, T, sigma_now, "put") * qty
                opt = {"kind": "put", "K": K, "expiry_i": i + DTE_BARS, "qty": qty}
                events.append(("SELL_PUT", i, round(K, 2)))
            else:
                K = _strike_for_delta(S, T, sigma_now, TARGET_DELTA, "call")
                cash += _bs_price(S, K, T, sigma_now, "call") * shares
                opt = {"kind": "call", "K": K, "expiry_i": i + DTE_BARS, "qty": shares}
                events.append(("SELL_CALL", i, round(K, 2)))
        liability = 0.0
        if opt:
            T_rem = (opt["expiry_i"] - i) / 252
            liability = _bs_price(S, opt["K"], T_rem, sigma_now, opt["kind"]) * opt["qty"]
        equity.append(cash + shares * S - liability)

    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    n = len(eq)
    cagr = (eq[-1] / eq[0]) ** (252 / n) - 1
    vol = rets.std() * np.sqrt(252)
    peaks = np.maximum.accumulate(eq)
    maxdd = float(((peaks - eq) / peaks).max()) * 100
    bh_cagr = (df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / n) - 1
    assigns = sum(1 for e in events if e[0] == "ASSIGNED")
    called = sum(1 for e in events if e[0] == "CALLED_AWAY")
    cycles = sum(1 for e in events if e[0] in ("SELL_PUT", "SELL_CALL"))

    return {
        "cagr": cagr * 100, "ann_vol": vol * 100,
        "sharpe": (cagr / vol) if vol > 0 else 0, "max_dd": maxdd,
        "roc_monthly": cagr / 12 * 100,
        "cycles": cycles, "assignments": assigns, "called_away": called,
        "bh_cagr": bh_cagr * 100, "equity": eq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Pro — News + Options + Crypto", layout="wide")
st.title("Institutional Quant Backtester + News Intelligence")
st.caption("Multi-Strategy | Options Simulation | Crypto | News Sentiment Overlay")

with st.sidebar:
    st.header("Alpaca (Market Data)")
    api_key = st.text_input("Alpaca API Key", type="password", placeholder="PKUR...")
    api_secret = st.text_input("Alpaca API Secret", type="password", placeholder="EwddZ...")

    st.divider()
    st.header("NewsAPI (Sentiment)")
    news_api_key = st.text_input("NewsAPI Key", type="password",
                                  help="Free at newsapi.org — 100 req/day")

    st.divider()
    st.header("Backtest Config")
    lookback_days = st.number_input("Lookback (trading days)", min_value=252, value=756, step=63)
    capital = st.number_input("Capital ($)", min_value=10000, value=100000, step=10000)
    pos_pct = st.number_input("Position size (%)", min_value=1, max_value=50, value=10) / 100
    commission_bps = st.number_input("Commission (bps)", min_value=0, value=5)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0, value=3)
    cost_bps = commission_bps + slippage_bps
    include_crypto = st.checkbox("Include BTC / ETH / SOL", value=True)

    st.divider()
    st.header("News & Options Config")
    news_days_back = st.number_input("News lookback (days)", min_value=1, max_value=30, value=7)
    options_dte = st.number_input("Options DTE", min_value=7, max_value=90, value=21)
    options_contracts = st.number_input("Contracts per trade", min_value=1, value=1)
    crypto_hold_days = st.number_input("Crypto hold period (days)", min_value=1, max_value=30, value=7)

    run_backtest_btn = st.button("Run Full Backtest", type="primary", use_container_width=True)
    run_news_btn = st.button("Run News Intelligence", use_container_width=True,
                              help="Runs independently of backtest — only needs NewsAPI key")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_news, tab_lb, tab_eq, tab_regime, tab_wheel, tab_mc, tab_stats = st.tabs([
    "News Intelligence", "Leaderboard", "Equity Curves",
    "Regime Analysis", "Options Wheel", "Monte Carlo", "Assessment"
])

# ─────────────────────────────────────────────────────────────────────────────
# NEWS INTELLIGENCE TAB (independent of backtest)
# ─────────────────────────────────────────────────────────────────────────────
with tab_news:
    st.subheader("News Sentiment → Options & Crypto Trade Signals")

    if not run_news_btn and not run_backtest_btn:
        st.info("Enter a NewsAPI key and click **Run News Intelligence** (or Run Full Backtest).")
        st.markdown("""
**How it works:**
1. Fetches last N days of news for each ticker via NewsAPI
2. Scores each headline with VADER sentiment (positive / neutral / negative)
3. Maps sentiment → options strategy (e.g. strong positive → Buy ATM Call)
4. Maps sentiment → crypto spot trade (long/short sized by conviction)
5. Shows P&L scenarios across ±20% price moves at expiry
        """)

    elif not news_api_key:
        st.warning("NewsAPI key required for News Intelligence. Enter it in the sidebar.")

    else:
        analyzer = load_vader()
        if analyzer is None:
            st.error("vaderSentiment not installed. Add `vaderSentiment` to requirements.txt and redeploy.")
        else:
            # Choose tickers to scan — equity + crypto
            equity_scan = [t for sector in UNIVERSE.values() for t in sector]
            crypto_scan = list(CRYPTO_UNIVERSE.keys()) if include_crypto else []
            all_scan = equity_scan + crypto_scan

            sentiment_results = {}
            prog = st.progress(0, text="Fetching news & scoring sentiment...")
            for idx, ticker in enumerate(all_scan):
                sentiment_results[ticker] = aggregate_ticker_sentiment(
                    ticker, news_api_key, analyzer, days_back=news_days_back
                )
                prog.progress((idx + 1) / len(all_scan), text=f"Scored {ticker} ({idx+1}/{len(all_scan)})")
            prog.empty()

            # Sentiment heatmap overview
            st.subheader("Sentiment Overview")
            heat_data = {t: s["avg_score"] for t, s in sentiment_results.items() if s["n"] > 0}
            if heat_data:
                heat_df = pd.DataFrame([
                    {"Ticker": t, "Score": s["avg_score"], "Label": s["label"],
                     "Articles": s["n"], "Bullish": s["bullish"], "Bearish": s["bearish"]}
                    for t, s in sentiment_results.items()
                ]).sort_values("Score", ascending=False)

                col_bull, col_neut, col_bear = st.columns(3)
                col_bull.metric("Bullish tickers",
                                len(heat_df[heat_df["Label"] == "positive"]))
                col_neut.metric("Neutral tickers",
                                len(heat_df[heat_df["Label"] == "neutral"]))
                col_bear.metric("Bearish tickers",
                                len(heat_df[heat_df["Label"] == "negative"]))

                def color_label(val):
                    if val == "positive":
                        return "background-color: #d4edda; color: #155724"
                    if val == "negative":
                        return "background-color: #f8d7da; color: #721c24"
                    return "background-color: #fff3cd; color: #856404"

                st.dataframe(
                    heat_df.style.applymap(color_label, subset=["Label"])
                           .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-1, vmax=1),
                    use_container_width=True, hide_index=True
                )

            # Per-ticker news + trade signals
            st.subheader("News-Driven Trade Signals")

            actionable = [t for t, s in sentiment_results.items()
                          if s["label"] != "neutral" and s["n"] >= 2]
            if not actionable:
                st.info("No strong sentiment signals found in the last few days. Try increasing 'News lookback'.")
            else:
                # We need price data for trade simulation — use yfinance as fallback if Alpaca not run
                prices_available = "processed" in st.session_state

                for ticker in actionable:
                    sent = sentiment_results[ticker]
                    with st.expander(
                        f"{'🟢' if sent['label']=='positive' else '🔴'} "
                        f"{ticker}  |  Score: {sent['avg_score']:+.3f}  |  "
                        f"{sent['bullish']}↑ {sent['bearish']}↓  ({sent['n']} articles)",
                        expanded=False,
                    ):
                        # Recent headlines
                        st.markdown("**Recent Headlines**")
                        for art in sent["articles"][:5]:
                            emoji = "🟢" if art["label"] == "positive" else "🔴" if art["label"] == "negative" else "⚪"
                            st.markdown(f"{emoji} [{art['title']}]({art['url']})  \n"
                                        f"*{art['source']} · {art['date']} · score: {art['score']:+.3f}*")

                        # Trade simulation — need current price & vol
                        if prices_available and ticker in st.session_state["processed"]:
                            df_t = st.session_state["processed"][ticker]
                            S_now = float(df_t["close"].iloc[-1])
                            vol_now = float(df_t["vol20"].dropna().iloc[-1]) if df_t["vol20"].notna().any() else 0.25

                            if ticker in CRYPTO_UNIVERSE or ticker in ("BTC", "ETH", "SOL"):
                                # Crypto simulation
                                sim = simulate_crypto_trade(
                                    ticker, S_now, vol_now,
                                    sent["avg_score"], sent["label"],
                                    capital=capital, hold_days=crypto_hold_days,
                                )
                                st.markdown(f"**Crypto Signal: {sim['direction']}**  "
                                            f"| Size: ${sim['pos_size']:,.0f} ({sim['pos_pct']}% of capital)  "
                                            f"| Ann Vol: {sim['ann_vol']}%  "
                                            f"| P(profit): {sim['prob_profit']}%")
                                if sim["scenarios"]:
                                    st.dataframe(pd.DataFrame(sim["scenarios"]),
                                                 use_container_width=True, hide_index=True)
                            else:
                                # Options simulation
                                sim = simulate_news_options_trades(
                                    ticker, S_now, vol_now,
                                    sent["avg_score"], sent["label"],
                                    capital=capital, dte_days=options_dte,
                                    contracts=options_contracts,
                                )
                                if sim["strategy"] != "No Trade (neutral)":
                                    leg_str = "  |  ".join(
                                        f"{l['action'].upper()} {l['type'].upper()} K={l['K']}"
                                        for l in sim["legs"]
                                    )
                                    st.markdown(
                                        f"**Options Signal: {sim['strategy']}**  \n"
                                        f"Legs: {leg_str}  \n"
                                        f"Cost: ${sim['cost']:,.2f}  |  Delta: {sim['delta']:+.3f}  "
                                        f"|  IV proxy: {sim['iv']}%  |  DTE: {sim['dte']}"
                                    )
                                    st.dataframe(pd.DataFrame(sim["scenarios"]),
                                                 use_container_width=True, hide_index=True)
                        else:
                            st.caption("Run Full Backtest to enable trade simulation with live prices.")

            # Crypto spotlight
            if include_crypto:
                st.subheader("Crypto Sentiment Spotlight")
                crypto_rows = []
                for t in ["BTC", "ETH", "SOL"]:
                    if t in sentiment_results:
                        s = sentiment_results[t]
                        direction = "LONG 🟢" if s["label"] == "positive" else "SHORT 🔴" if s["label"] == "negative" else "HOLD ⚪"
                        crypto_rows.append({
                            "Crypto": t, "Score": s["avg_score"], "Sentiment": s["label"],
                            "Signal": direction, "Articles": s["n"],
                            "Bullish": s["bullish"], "Bearish": s["bearish"],
                        })
                if crypto_rows:
                    st.dataframe(pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST — only runs on button click
# ─────────────────────────────────────────────────────────────────────────────
if run_backtest_btn:
    if not api_key or not api_secret:
        st.error("Alpaca API credentials required for backtest.")
        st.stop()

    HEADERS = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}

    with st.spinner("Fetching market data..."):
        raw = {}
        for t in ALL_TICKERS:
            raw[t] = fetch_bars(t, lookback_days, HEADERS)
        if include_crypto:
            crypto_days = int(lookback_days * 365 / 252)
            for name, sym in CRYPTO_UNIVERSE.items():
                raw[name] = fetch_crypto_bars(sym, crypto_days, HEADERS)

    valid = {t: df for t, df in raw.items() if df is not None}
    PERIODS_PER_YEAR = {t: (365 if t in CRYPTO_UNIVERSE else 252) for t in valid}

    with st.spinner("Computing indicators..."):
        processed = {t: add_indicators(df) for t, df in valid.items()}

    st.session_state["processed"] = processed
    st.session_state["PERIODS_PER_YEAR"] = PERIODS_PER_YEAR

    spy_regimes = classify_regimes(processed["SPY"]) if "SPY" in processed else None

    with st.spinner(f"Running {len(STRATEGIES)} × {len(processed)} backtests..."):
        all_results = {}
        for sn, fn in STRATEGIES.items():
            all_results[sn] = {}
            for ticker, df in processed.items():
                all_results[sn][ticker] = backtest_strategy(
                    df, fn, capital=capital, pos_pct=pos_pct,
                    cost_bps=cost_bps, periods_per_year=PERIODS_PER_YEAR.get(ticker, 252)
                )

    st.session_state["all_results"] = all_results
    st.session_state["spy_regimes"] = spy_regimes
    st.success(f"Done — {len(valid)} symbols loaded, {len(STRATEGIES)*len(processed)} backtests run.")

# Pull from session state for display
all_results = st.session_state.get("all_results")
processed = st.session_state.get("processed")
spy_regimes = st.session_state.get("spy_regimes")
PERIODS_PER_YEAR = st.session_state.get("PERIODS_PER_YEAR", {})

if all_results is None or processed is None:
    for tab in [tab_lb, tab_eq, tab_regime, tab_wheel, tab_mc, tab_stats]:
        with tab:
            st.info("Run Full Backtest to see results here.")
else:
    # Build leaderboard
    rows = []
    for strat in STRATEGIES:
        for ticker in processed:
            if ticker not in all_results.get(strat, {}):
                continue
            r = all_results[strat][ticker]
            rows.append({
                "Strategy": strat, "Ticker": ticker,
                "Ann Ret %": r["ann_return"], "Sharpe": r["sharpe"],
                "Sortino": r["sortino"], "Calmar": r["calmar"],
                "Max DD %": r["max_dd"], "Win Rate %": r["win_rate"],
                "Trades": r["total_trades"], "Profit F": r["profit_factor"],
            })
    lb = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

    with tab_lb:
        st.subheader("Top 25 Strategy × Ticker (min 5 trades)")
        lb_f = lb[lb["Trades"] >= 5].head(25)
        st.dataframe(lb_f.style.background_gradient(
            subset=["Sharpe", "Ann Ret %"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True)

        st.subheader("Sector Best")
        sec_rows = []
        for sector, tickers in UNIVERSE.items():
            best, best_s = None, -99
            for strat in STRATEGIES:
                for t in [x for x in tickers if x in processed]:
                    r = all_results[strat][t]
                    if r["total_trades"] >= 3 and r["sharpe"] > best_s:
                        best_s, best = r["sharpe"], (strat, t, r)
            if best:
                s, t, r = best
                sec_rows.append({"Sector": sector, "Best": f"{s}/{t}",
                                 "Sharpe": r["sharpe"], "Ann Ret %": r["ann_return"],
                                 "Max DD %": r["max_dd"], "Trades": r["total_trades"]})
        st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

    with tab_eq:
        colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#BA7517"]
        if "SPY" in processed:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            ax1 = axes[0]
            for i, sn in enumerate(STRATEGIES):
                r = all_results[sn]["SPY"]
                eq = pd.Series(r["equity"])
                ax1.plot(eq / eq.iloc[0] * 100, label=f"{sn} (S:{r['sharpe']:+.2f})",
                         color=colors[i], linewidth=1.5)
            ax1.axhline(100, color="gray", linestyle="--", alpha=0.5)
            ax1.set_title("All Strategies — SPY")
            ax1.set_ylabel("Indexed Return")
            ax1.legend(fontsize=8, ncol=2)
            ax1.grid(axis="y", alpha=0.3)
            ax1.spines[["top", "right"]].set_visible(False)

            ax2 = axes[1]
            pd_data = lb[lb["Trades"] >= 5].groupby("Strategy")["Sharpe"].median().sort_values(ascending=False)
            bar_cols = ["#1D9E75" if v > 0 else "#D85A30" for v in pd_data]
            bars = ax2.bar(pd_data.index, pd_data.values, color=bar_cols, width=0.5, zorder=3)
            for bar, val in zip(bars, pd_data.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:+.2f}", ha="center", va="bottom", fontsize=9)
            ax2.axhline(0, color="gray")
            ax2.set_title("Median Sharpe per Strategy")
            ax2.set_ylabel("Median Sharpe")
            ax2.grid(axis="y", alpha=0.3, zorder=0)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Crypto equity curves
        crypto_done = [t for t in CRYPTO_UNIVERSE if t in processed]
        if crypto_done:
            st.subheader("Crypto — Best Strategy per Asset")
            fig2, ax = plt.subplots(figsize=(14, 4))
            for i, t in enumerate(crypto_done):
                best_r, best_s = None, -99
                for sn in STRATEGIES:
                    r = all_results[sn][t]
                    if r["sharpe"] > best_s:
                        best_s, best_r, best_sn = r["sharpe"], r, sn
                if best_r:
                    eq = pd.Series(best_r["equity"])
                    ax.plot(eq / eq.iloc[0] * 100, label=f"{t}/{best_sn} (S:{best_s:+.2f})",
                            color=colors[i % len(colors)], linewidth=1.5)
            ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Crypto Best Strategies")
            ax.set_ylabel("Indexed Return")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)

        # Correlation
        if "SPY" in processed:
            st.subheader("Strategy Correlation (SPY)")
            eq_s = {sn: pd.Series(all_results[sn]["SPY"]["equity"]) for sn in STRATEGIES}
            min_len = min(len(s) for s in eq_s.values())
            eq_df = pd.DataFrame({k: v.iloc[:min_len].values for k, v in eq_s.items()})
            corr = eq_df.pct_change().dropna().corr()
            fig3, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                        annot=True, fmt=".2f", cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                        ax=ax, linewidths=0.5)
            ax.set_title("Strategy Return Correlation")
            plt.tight_layout()
            st.pyplot(fig3)

    with tab_regime:
        if spy_regimes is not None and "SPY" in processed:
            spy_df = processed["SPY"].copy().merge(spy_regimes, on="date", how="left")
            spy_df["regime"] = spy_df["regime"].fillna("SIDEWAYS")
            counts = spy_df["regime"].value_counts()
            c1, c2, c3 = st.columns(3)
            for col, reg in zip([c1, c2, c3], ["BULL", "SIDEWAYS", "BEAR"]):
                cnt = counts.get(reg, 0)
                col.metric(reg, f"{cnt} days", f"{cnt/len(spy_df)*100:.1f}%")
            regime_data = []
            for sn, fn in STRATEGIES.items():
                row = {"Strategy": sn}
                for regime in ["BULL", "BEAR", "SIDEWAYS"]:
                    idx = spy_df[spy_df["regime"] == regime].index.tolist()
                    if len(idx) < 50:
                        row[regime] = "N/A"
                        continue
                    seg = spy_df.iloc[idx[0]:idx[-1] + 1].reset_index(drop=True)
                    r = backtest_strategy(seg, fn, capital=capital, pos_pct=pos_pct, cost_bps=cost_bps)
                    row[regime] = f"{r['ann_return']:+.1f}% (S:{r['sharpe']:+.2f})"
                regime_data.append(row)
            st.dataframe(pd.DataFrame(regime_data), use_container_width=True, hide_index=True)

    with tab_wheel:
        wt = [t for t in ["SPY", "QQQ", "NVDA", "TSLA"] + list(CRYPTO_UNIVERSE.keys())
              if t in processed]
        wheel_rows = []
        wheel_equities = {}
        for t in wt:
            r = wheel_backtest(processed[t], capital=capital, cost_bps=cost_bps)
            if r:
                wheel_rows.append({
                    "Ticker": t, "Wheel CAGR": f"{r['cagr']:+.1f}%",
                    "MaxDD": f"{r['max_dd']:.1f}%", "Sharpe": f"{r['sharpe']:.2f}",
                    "ROC/mo": f"{r['roc_monthly']:.2f}%",
                    "Cycles": r["cycles"], "Assigned": r["assignments"],
                    "Called": r["called_away"], "B&H CAGR": f"{r['bh_cagr']:+.1f}%",
                })
                wheel_equities[t] = r["equity"]
        if wheel_rows:
            st.dataframe(pd.DataFrame(wheel_rows), use_container_width=True, hide_index=True)
            fig4, ax = plt.subplots(figsize=(12, 4))
            colors5 = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#BA7517", "#666666", "#999999"]
            for i, (t, eq) in enumerate(wheel_equities.items()):
                s = pd.Series(eq)
                ax.plot(s / s.iloc[0] * 100, label=t, color=colors5[i % len(colors5)], linewidth=1.5)
            ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Options Wheel Equity Curves")
            ax.set_ylabel("Indexed Return")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig4)
            st.caption("BSM + realized-vol IV proxy. Fractional contracts. Upper-bound estimate.")

    with tab_mc:
        if "SPY" in processed:
            spy_lb = lb[lb["Ticker"] == "SPY"].sort_values("Sharpe", ascending=False)
            if len(spy_lb):
                best_sn = spy_lb.iloc[0]["Strategy"]
                best_r = all_results[best_sn]["SPY"]
                sell_pnls = [t["pnl"] for t in best_r["trades"]
                             if t["action"] == "SELL" and t["pnl"] is not None]
                if len(sell_pnls) >= 5:
                    mc = np.array([
                        (capital + np.random.choice(sell_pnls, size=len(sell_pnls), replace=True).sum() - capital)
                        / capital * 100
                        for _ in range(1000)
                    ])
                    actual = best_r["total_return"]
                    pct = (mc < actual).mean() * 100
                    st.subheader(f"Monte Carlo: {best_sn} on SPY")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Actual Return", f"{actual:+.2f}%")
                    c2.metric("MC Median", f"{np.median(mc):+.2f}%")
                    c3.metric("MC 95th", f"{np.percentile(mc, 95):+.2f}%")
                    c4.metric("Percentile Rank", f"{pct:.1f}th")
                    verdict = ("Statistically significant edge (p<0.05)" if pct > 95
                               else "Moderate evidence — needs more trades" if pct > 80
                               else "Within random noise — likely luck")
                    st.info(f"**Verdict:** {verdict}")
                    fig5, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(mc, bins=50, color="#378ADD", alpha=0.7, edgecolor="white")
                    ax.axvline(actual, color="#D85A30", linewidth=2, label=f"Actual {actual:+.1f}%")
                    ax.axvline(np.percentile(mc, 95), color="gray", linestyle="--", label="95th pct")
                    ax.set_title("Monte Carlo Return Distribution")
                    ax.set_xlabel("Total Return %")
                    ax.legend()
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig5)
                else:
                    st.warning("Not enough closed trades for Monte Carlo.")

    with tab_stats:
        all_sharpes = [r["sharpe"] for strat in all_results.values()
                       for r in strat.values() if r["total_trades"] >= 5]
        all_tc = [r["total_trades"] for strat in all_results.values() for r in strat.values()]
        n_tested = len(STRATEGIES) * len(processed)
        deflated = 0.5 + 0.2 * np.log(n_tested)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Combinations", n_tested)
        c2.metric("Sharpe > 1.0", sum(1 for s in all_sharpes if s > 1.0))
        c3.metric("Sharpe > 0.5", sum(1 for s in all_sharpes if s > 0.5))
        c4.metric("Median Sharpe", f"{np.median(all_sharpes):+.3f}")

        st.warning(f"**Multiple Testing Warning:** At p=0.05, expect ~{n_tested*0.05:.0f} false positives. "
                   f"Deflated Sharpe threshold: **{deflated:.2f}**")
        st.info(f"**Sample Size:** Only {sum(1 for t in all_tc if t >= 30)} combos have 30+ trades. "
                f"Everything else is statistically anecdotal.")
        st.success("**Next steps:** Identify top combos with 30+ trades → walk-forward → "
                   "paper trade 3 months → size by quarter-Kelly")

        # Portfolio
        top_c = lb[(lb["Sharpe"] > 0.5) & (lb["Trades"] >= 5)].head(10)
        if len(top_c):
            vols = [max(all_results[r["Strategy"]][r["Ticker"]]["ann_vol"], 1.0)
                    for _, r in top_c.iterrows()]
            inv_v = [1 / v for v in vols]
            wts = [iv / sum(inv_v) for iv in inv_v]
            port_ret = sum(all_results[r["Strategy"]][r["Ticker"]]["ann_return"] * w
                           for (_, r), w in zip(top_c.iterrows(), wts))
            port_rows = [{"Ticker": r["Ticker"], "Strategy": r["Strategy"][:15],
                          "Weight": f"{w*100:.1f}%", "Alloc": f"${capital*w:,.0f}",
                          "Sharpe": all_results[r["Strategy"]][r["Ticker"]]["sharpe"],
                          "Ann Ret %": all_results[r["Strategy"]][r["Ticker"]]["ann_return"]}
                         for (_, r), w in zip(top_c.iterrows(), wts)]
            st.subheader("Portfolio (Inverse-Vol Weighted)")
            st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)
            st.metric("Portfolio Weighted Ann. Return", f"{port_ret:+.2f}%")
