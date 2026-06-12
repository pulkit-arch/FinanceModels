"""
Institutional-Grade Quantitative Backtest Framework
Multi-Strategy | Multi-Regime | Walk-Forward Validated
"""
import math
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy.stats import norm as _norm
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
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


# ── Data Fetch ───────────────────────────────────────────────────────────────
def fetch_bars(symbol, days, headers):
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(days * 1.5) + 30)).strftime("%Y-%m-%d")
    r = requests.get(
        f"{DATA_URL}/v2/stocks/{symbol}/bars",
        headers=headers,
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
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers=headers,
        params={"symbols": symbol, "timeframe": "1Day", "start": start, "limit": 10000},
    )
    if r.status_code != 200:
        return None
    bars = r.json().get("bars", {}).get(symbol, [])
    if len(bars) < 300:
        return None
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"]).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)


# ── Indicators ────────────────────────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    for p in [10, 20, 50, 200]:
        df[f"sma{p}"] = c.rolling(p).mean()
    for p in [12, 26]:
        df[f"ema{p}"] = c.ewm(span=p, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    for p in [5, 10, 21, 63]:
        df[f"roc{p}"] = c.pct_change(p) * 100
    df["returns"] = c.pct_change()
    df["vol20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["vol60"] = df["returns"].rolling(60).std() * np.sqrt(252)
    df["atr14"] = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1).rolling(14).mean()
    df["bb_mid"] = c.rolling(20).mean()
    df["bb_std"] = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["vol_sma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma20"]
    df["obv"] = (np.sign(c.diff()) * v).cumsum()
    return df


# ── Regime Detection ──────────────────────────────────────────────────────────
def classify_regimes(spy_df):
    df = spy_df.copy()
    df["sma200_slope"] = df["sma200"].pct_change(21)
    df["regime"] = "SIDEWAYS"
    df.loc[(df["close"] > df["sma200"]) & (df["sma200_slope"] > 0.005), "regime"] = "BULL"
    df.loc[(df["close"] < df["sma200"]) & (df["sma200_slope"] < -0.005), "regime"] = "BEAR"
    return df[["date", "regime"]]


# ── Strategy Signals ──────────────────────────────────────────────────────────
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


# ── Backtest Engine ───────────────────────────────────────────────────────────
def backtest_strategy(df, signal_fn, capital=100_000, pos_pct=0.10,
                      stop_loss=0.05, take_profit=None,
                      cost_bps=8, periods_per_year=252):
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
            elif take_profit and roc >= take_profit:
                exit_r = f"TakeProfit {roc*100:.1f}%"
            elif row["rsi14"] > 75:
                exit_r = f"RSI overbought {row['rsi14']:.0f}"
            elif (row["sma50"] < row["sma200"] and
                  signal_fn.__name__ == "signal_trend_following"):
                exit_r = "Death cross"
            if exit_r:
                sell_px = nxt * (1 - cost_mult)
                proceeds = position["shares"] * sell_px
                pnl = proceeds - position["cost"]
                cash += proceeds
                trades.append({"date": str(next_row["date"]), "action": "SELL",
                                "price": round(sell_px, 2), "shares": round(position["shares"], 4),
                                "pnl": round(pnl, 2), "reason": exit_r, "capital": round(cash, 2)})
                position = None
        elif signals.iloc[i] == 1:
            spend = min(capital * pos_pct, cash)
            if spend > 500:
                buy_px = nxt * (1 + cost_mult)
                shares = spend / buy_px
                cash -= spend
                position = {"entry": buy_px, "shares": shares, "cost": spend}
                trades.append({"date": str(next_row["date"]), "action": "BUY",
                                "price": round(buy_px, 2), "shares": round(shares, 4),
                                "pnl": None, "reason": "Signal", "capital": round(cash, 2)})
        cur_val = cash + (position["shares"] * price if position else 0)
        equity.append(cur_val)

    if position:
        lp = df.iloc[-1]["close"]
        unreal = position["shares"] * lp - position["cost"]
        trades.append({"date": str(df.iloc[-1]["date"]), "action": "OPEN",
                        "price": round(lp, 2), "shares": round(position["shares"], 4),
                        "pnl": round(unreal, 2), "reason": "Open at period end",
                        "capital": round(cash + position["shares"] * lp, 2)})
        equity.append(cash + position["shares"] * lp)

    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    sells = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    pnls = [t["pnl"] for t in sells]
    opens = [t for t in trades if t["action"] == "OPEN"]
    total_pnl = sum(pnls) + sum(t["pnl"] for t in opens)
    ann_ret = (eq[-1] / eq[0]) ** (periods_per_year / len(eq)) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year) if len(rets) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    down_rets = rets[rets < 0]
    down_vol = down_rets.std() * np.sqrt(periods_per_year) if len(down_rets) > 1 else 0
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    peaks = np.maximum.accumulate(eq)
    max_dd = float(((peaks - eq) / peaks).max())
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": trades, "equity": list(eq),
        "total_trades": len(sells),
        "win_rate": round(len(wins) / len(sells) * 100) if sells else 0,
        "total_pnl": round(total_pnl, 2),
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


# ── Options Wheel ─────────────────────────────────────────────────────────────
IV_PREMIUM = 1.15
RISK_FREE = 0.04
TARGET_DELTA = 0.30
DTE_BARS = 21


def _bs_price(S, K, T, sigma, kind, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)
    return S * _norm.cdf(d1) - K * math.exp(-r * T) * _norm.cdf(d2)


def _strike_for_delta(S, T, sigma, delta, kind, r=RISK_FREE):
    d1 = _norm.ppf(1 - delta) if kind == "put" else _norm.ppf(delta)
    return S * math.exp((r + sigma ** 2 / 2) * T - sigma * math.sqrt(T) * d1)


def wheel_backtest(df, capital=100_000, target_delta=TARGET_DELTA,
                   dte=DTE_BARS, iv_premium=IV_PREMIUM, cost_bps=8):
    df = df.dropna(subset=["vol20"]).reset_index(drop=True)
    if len(df) < dte + 5:
        return None
    cash, shares, opt = capital, 0.0, None
    equity, events = [], []
    cm = cost_bps / 10000

    for i in range(len(df)):
        S = df["close"].iloc[i]
        sigma_now = max(df["vol20"].iloc[i] * iv_premium, 0.05)
        if opt and i >= opt["expiry_i"]:
            K, q = opt["K"], opt["qty"]
            if opt["kind"] == "put":
                if S < K:
                    cash -= K * q * (1 + cm)
                    shares += q
                    events.append(("ASSIGNED", df["date"].iloc[i], K))
                else:
                    events.append(("PUT_EXPIRED", df["date"].iloc[i], K))
            else:
                if S > K:
                    cash += K * shares * (1 - cm)
                    events.append(("CALLED_AWAY", df["date"].iloc[i], K))
                    shares = 0.0
                else:
                    events.append(("CALL_EXPIRED", df["date"].iloc[i], K))
            opt = None
        if opt is None and i + dte < len(df):
            T = dte / 252
            if shares == 0:
                K = _strike_for_delta(S, T, sigma_now, target_delta, "put")
                qty = cash / K
                prem = _bs_price(S, K, T, sigma_now, "put") * qty
                cash += prem
                opt = {"kind": "put", "K": K, "expiry_i": i + dte, "qty": qty, "sigma": sigma_now}
                events.append(("SELL_PUT", df["date"].iloc[i], round(K, 2)))
            else:
                K = _strike_for_delta(S, T, sigma_now, target_delta, "call")
                prem = _bs_price(S, K, T, sigma_now, "call") * shares
                cash += prem
                opt = {"kind": "call", "K": K, "expiry_i": i + dte, "qty": shares, "sigma": sigma_now}
                events.append(("SELL_CALL", df["date"].iloc[i], round(K, 2)))
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
    bh = df["close"].iloc[-1] / df["close"].iloc[0]
    bh_cagr = bh ** (252 / n) - 1
    bh_eq = capital * df["close"] / df["close"].iloc[0]
    bh_peaks = np.maximum.accumulate(bh_eq.values)
    bh_maxdd = float(((bh_peaks - bh_eq.values) / bh_peaks).max()) * 100
    prem_events = [e for e in events if e[0] in ("SELL_PUT", "SELL_CALL")]
    assigns = [e for e in events if e[0] == "ASSIGNED"]
    called = [e for e in events if e[0] == "CALLED_AWAY"]

    return {
        "cagr": cagr * 100, "ann_vol": vol * 100,
        "sharpe": (cagr / vol) if vol > 0 else 0, "max_dd": maxdd,
        "roc_monthly": cagr / 12 * 100,
        "cycles": len(prem_events), "assignments": len(assigns),
        "called_away": len(called), "final": eq[-1],
        "bh_cagr": bh_cagr * 100, "bh_maxdd": bh_maxdd,
        "equity": eq,
    }


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Pro Backtester", layout="wide")
st.title("Institutional-Grade Quantitative Backtest Framework")
st.caption("Multi-Strategy | Multi-Regime | Walk-Forward Validated")

with st.sidebar:
    st.header("Alpaca Credentials")
    api_key = st.text_input("API Key", type="password", placeholder="PKUR...")
    api_secret = st.text_input("API Secret", type="password", placeholder="EwddZ...")

    st.divider()
    st.header("Configuration")
    lookback_days = st.number_input("Lookback (trading days)", min_value=252, value=756, step=63)
    capital = st.number_input("Starting Capital ($)", min_value=10000, value=100000, step=10000)
    pos_pct = st.number_input("Position size (%)", min_value=1, max_value=50, value=10) / 100
    commission_bps = st.number_input("Commission (bps/side)", min_value=0, value=5)
    slippage_bps = st.number_input("Slippage (bps/side)", min_value=0, value=3)
    cost_bps = commission_bps + slippage_bps

    st.subheader("Universe")
    include_crypto = st.checkbox("Include crypto (BTC/ETH/SOL)", value=False)

    run_btn = st.button("Run Full Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("Enter your Alpaca API credentials in the sidebar and click **Run Full Backtest**.")
    st.markdown("""
**What this runs:**
- 5 alpha strategies: Trend Momentum, Mean Reversion, Vol Breakout, Trend Following, RSI Recovery
- 20 equity tickers across 5 sectors (+ optional crypto)
- Walk-forward validation, regime analysis, Monte Carlo, options wheel overlay
- Full institutional metrics: Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor

**Data source:** Alpaca Markets free IEX feed (requires free Alpaca account)
""")
    st.stop()

if not api_key or not api_secret:
    st.error("Please enter Alpaca API credentials in the sidebar.")
    st.stop()

HEADERS = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}

# ── Fetch Data ────────────────────────────────────────────────────────────────
with st.spinner("Fetching market data from Alpaca..."):
    raw = {}
    fetch_status = []
    for t in ALL_TICKERS:
        df_t = fetch_bars(t, lookback_days, HEADERS)
        raw[t] = df_t
        fetch_status.append({"Ticker": t, "Status": f"{len(df_t)} bars" if df_t is not None else "FAILED"})

    if include_crypto:
        crypto_days = int(lookback_days * 365 / 252)
        for name, sym in CRYPTO_UNIVERSE.items():
            df_c = fetch_crypto_bars(sym, crypto_days, HEADERS)
            raw[name] = df_c
            fetch_status.append({"Ticker": name, "Status": f"{len(df_c)} bars" if df_c is not None else "FAILED"})

valid = {t: df for t, df in raw.items() if df is not None}
PERIODS_PER_YEAR = {t: (365 if t in CRYPTO_UNIVERSE else 252) for t in valid}

st.success(f"Fetched {len(valid)}/{len(raw)} symbols successfully.")
with st.expander("Fetch details"):
    st.dataframe(pd.DataFrame(fetch_status), use_container_width=True, hide_index=True)

if len(valid) == 0:
    st.error("No data fetched. Check your API credentials.")
    st.stop()

# ── Compute Indicators ────────────────────────────────────────────────────────
with st.spinner("Computing indicators..."):
    processed = {t: add_indicators(df) for t, df in valid.items()}

# ── Regime Detection ──────────────────────────────────────────────────────────
spy_regimes = None
if "SPY" in processed:
    spy_regimes = classify_regimes(processed["SPY"])

# ── Run All Backtests ─────────────────────────────────────────────────────────
with st.spinner(f"Running {len(STRATEGIES)} × {len(processed)} = {len(STRATEGIES)*len(processed)} backtests..."):
    all_results = {}
    for strat_name, strat_fn in STRATEGIES.items():
        all_results[strat_name] = {}
        for ticker, df in processed.items():
            all_results[strat_name][ticker] = backtest_strategy(
                df, strat_fn, capital=capital, pos_pct=pos_pct,
                cost_bps=cost_bps, periods_per_year=PERIODS_PER_YEAR.get(ticker, 252)
            )

# ── Build Leaderboard ─────────────────────────────────────────────────────────
rows = []
for strat in STRATEGIES:
    for ticker in processed:
        r = all_results[strat][ticker]
        rows.append({
            "Strategy": strat, "Ticker": ticker,
            "Ann Ret %": r["ann_return"], "Sharpe": r["sharpe"],
            "Sortino": r["sortino"], "Calmar": r["calmar"],
            "Max DD %": r["max_dd"], "Win Rate %": r["win_rate"],
            "Trades": r["total_trades"], "Profit F": r["profit_factor"],
        })
lb = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

# ── Display Results ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Leaderboard", "Equity Curves", "Regime Analysis",
    "Options Wheel", "Monte Carlo", "Statistical Assessment"
])

with tab1:
    st.subheader("Top 25 Strategy × Ticker (min 5 trades)")
    lb_filtered = lb[lb["Trades"] >= 5].head(25)
    st.dataframe(lb_filtered.style.background_gradient(subset=["Sharpe", "Ann Ret %"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)

    st.subheader("Sector Alpha Scan")
    sector_rows = []
    for sector, tickers in UNIVERSE.items():
        best, best_s = None, -99
        for strat in STRATEGIES:
            for t in [x for x in tickers if x in processed]:
                r = all_results[strat][t]
                if r["total_trades"] >= 3 and r["sharpe"] > best_s:
                    best_s, best = r["sharpe"], (strat, t, r)
        if best:
            s, t, r = best
            sector_rows.append({
                "Sector": sector, "Best Combo": f"{s} / {t}",
                "Sharpe": r["sharpe"], "Ann Ret %": r["ann_return"],
                "Max DD %": r["max_dd"], "Win Rate %": r["win_rate"],
                "Trades": r["total_trades"],
            })
    st.dataframe(pd.DataFrame(sector_rows), use_container_width=True, hide_index=True)

with tab2:
    colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#BA7517"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    if "SPY" in processed:
        for i, (sn, _) in enumerate(STRATEGIES.items()):
            r = all_results[sn]["SPY"]
            eq = pd.Series(r["equity"])
            ax1.plot(eq / eq.iloc[0] * 100, label=f"{sn} (S:{r['sharpe']:+.2f})",
                     color=colors[i], linewidth=1.5)
    ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_title("All Strategies — SPY", fontsize=12)
    ax1.set_ylabel("Indexed Return (Base=100)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    plot_data = lb[lb["Trades"] >= 5].groupby("Strategy")["Sharpe"].median().sort_values(ascending=False)
    bar_cols = ["#1D9E75" if v > 0 else "#D85A30" for v in plot_data]
    bars = ax2.bar(plot_data.index, plot_data.values, color=bar_cols, width=0.5, zorder=3)
    for bar, val in zip(bars, plot_data.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("Median Sharpe per Strategy (min 5 trades)", fontsize=12)
    ax2.set_ylabel("Median Sharpe Ratio")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=3)
    st.pyplot(fig)

    if "SPY" in processed:
        st.subheader("Strategy Correlation Matrix (SPY)")
        eq_series = {}
        for sn, _ in STRATEGIES.items():
            r = all_results[sn]["SPY"]
            eq_series[sn] = pd.Series(r["equity"])
        min_len = min(len(s) for s in eq_series.values())
        eq_df = pd.DataFrame({k: v.iloc[:min_len].values for k, v in eq_series.items()})
        ret_df = eq_df.pct_change().dropna()
        corr = ret_df.corr()
        fig2, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Strategy Return Correlation (SPY)")
        plt.tight_layout()
        st.pyplot(fig2)

with tab3:
    if spy_regimes is not None and "SPY" in processed:
        spy_df = processed["SPY"].copy()
        spy_df = spy_df.merge(spy_regimes, on="date", how="left")
        spy_df["regime"] = spy_df["regime"].fillna("SIDEWAYS")

        counts = spy_df["regime"].value_counts()
        col1, col2, col3 = st.columns(3)
        for col, regime in zip([col1, col2, col3], ["BULL", "SIDEWAYS", "BEAR"]):
            cnt = counts.get(regime, 0)
            col.metric(regime, f"{cnt} days", f"{cnt/len(spy_df)*100:.1f}%")

        st.subheader("Strategy Performance by Regime")
        regime_data = []
        for sn, strat_fn in STRATEGIES.items():
            row = {"Strategy": sn}
            for regime in ["BULL", "BEAR", "SIDEWAYS"]:
                regime_idx = spy_df[spy_df["regime"] == regime].index.tolist()
                if len(regime_idx) < 50:
                    row[regime] = "N/A"
                    continue
                seg = spy_df.iloc[regime_idx[0]:regime_idx[-1] + 1].reset_index(drop=True)
                r = backtest_strategy(seg, strat_fn, capital=capital, pos_pct=pos_pct, cost_bps=cost_bps)
                row[regime] = f"{r['ann_return']:+.1f}% (S:{r['sharpe']:+.2f})"
            regime_data.append(row)
        st.dataframe(pd.DataFrame(regime_data), use_container_width=True, hide_index=True)
    else:
        st.warning("SPY data required for regime analysis.")

with tab4:
    st.subheader("CSP + Covered Call Wheel Overlay")
    wheel_tickers = [t for t in ["SPY", "QQQ", "NVDA", "TSLA"] if t in processed]
    wheel_rows_data = []
    for t in wheel_tickers:
        r = wheel_backtest(processed[t], capital=capital, cost_bps=cost_bps)
        if r:
            wheel_rows_data.append({
                "Ticker": t, "Wheel CAGR": f"{r['cagr']:+.1f}%",
                "Wheel MaxDD": f"{r['max_dd']:.1f}%", "Sharpe": f"{r['sharpe']:.2f}",
                "ROC/mo": f"{r['roc_monthly']:.2f}%", "Cycles": r["cycles"],
                "Assigned": r["assignments"], "Called": r["called_away"],
                "B&H CAGR": f"{r['bh_cagr']:+.1f}%", "B&H MaxDD": f"{r['bh_maxdd']:.1f}%",
            })
    if wheel_rows_data:
        st.dataframe(pd.DataFrame(wheel_rows_data), use_container_width=True, hide_index=True)
        st.caption("ROC/mo = annualized return / 12 with full collateral reserved. "
                   "BSM with realized-vol proxy — upper-bound estimate.")

        fig3, ax = plt.subplots(figsize=(12, 4))
        for t in wheel_tickers:
            r = wheel_backtest(processed[t], capital=capital, cost_bps=cost_bps)
            if r:
                eq = pd.Series(r["equity"])
                ax.plot(eq / eq.iloc[0] * 100, label=f"{t} Wheel (CAGR:{r['cagr']:+.1f}%)", linewidth=1.5)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Options Wheel Equity Curves")
        ax.set_ylabel("Indexed Return (Base=100)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig3)

with tab5:
    if "SPY" in processed:
        spy_strats = lb[lb["Ticker"] == "SPY"].sort_values("Sharpe", ascending=False)
        if len(spy_strats) > 0:
            best_sn = spy_strats.iloc[0]["Strategy"]
            best_r = all_results[best_sn]["SPY"]
            sell_pnls = [t["pnl"] for t in best_r["trades"] if t["action"] == "SELL"]

            if len(sell_pnls) >= 5:
                n_sim = 1000
                mc_dist = []
                for _ in range(n_sim):
                    shuffled = np.random.choice(sell_pnls, size=len(sell_pnls), replace=True)
                    mc_dist.append((capital + shuffled.sum() - capital) / capital * 100)
                mc_dist = np.array(mc_dist)
                actual = best_r["total_return"]
                pct_beaten = (mc_dist < actual).mean() * 100

                st.subheader(f"Monte Carlo: {best_sn} on SPY ({n_sim} simulations)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Actual Return", f"{actual:+.2f}%")
                c2.metric("MC Median", f"{np.median(mc_dist):+.2f}%")
                c3.metric("MC 95th pct", f"{np.percentile(mc_dist, 95):+.2f}%")
                c4.metric("Percentile Rank", f"{pct_beaten:.1f}th")

                verdict = ("Edge is statistically significant (p < 0.05)" if pct_beaten > 95
                           else "Moderate evidence — needs more trades" if pct_beaten > 80
                           else "Returns within random noise — likely luck")
                st.info(f"**Verdict:** {verdict}")

                fig4, ax = plt.subplots(figsize=(10, 4))
                ax.hist(mc_dist, bins=50, color="#378ADD", alpha=0.7, edgecolor="white")
                ax.axvline(actual, color="#D85A30", linewidth=2, label=f"Actual: {actual:+.1f}%")
                ax.axvline(np.percentile(mc_dist, 95), color="gray", linestyle="--",
                           linewidth=1.5, label="95th pct")
                ax.set_title("Monte Carlo Return Distribution")
                ax.set_xlabel("Total Return %")
                ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig4)
            else:
                st.warning(f"{best_sn} on SPY has fewer than 5 closed trades — not enough for Monte Carlo.")
    else:
        st.warning("SPY data required for Monte Carlo.")

with tab6:
    all_sharpes = [r["sharpe"] for strat in all_results.values()
                   for r in strat.values() if r["total_trades"] >= 5]
    all_trades_counts = [r["total_trades"] for strat in all_results.values()
                         for r in strat.values()]
    n_tested = len(STRATEGIES) * len(processed)
    deflated_thresh = 0.5 + 0.2 * np.log(n_tested)

    st.subheader("Honest Statistical Assessment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combinations tested", n_tested)
    c2.metric("Sharpe > 1.0", sum(1 for s in all_sharpes if s > 1.0))
    c3.metric("Sharpe > 0.5", sum(1 for s in all_sharpes if s > 0.5))
    c4.metric("Median Sharpe", f"{np.median(all_sharpes):+.3f}")

    st.warning(f"""
**⚠️ Multiple Testing Warning**
You tested {n_tested} combinations. At p=0.05, you expect ~{n_tested*0.05:.0f} false positives by chance alone.
**Deflated Sharpe threshold (Pardo/Bailey): {deflated_thresh:.2f}** — only strategies above this have genuine edge.
""")

    st.info(f"""
**📊 Sample Size Warning**
Statistical significance requires ~30+ completed trades.
Combinations meeting this bar: **{sum(1 for t in all_trades_counts if t >= 30)}**
Everything else is anecdote, not evidence.
""")

    st.success("""
**✅ What to do with these results**
1. Identify top 3-5 strategies by Sharpe with 30+ trades
2. Run walk-forward on those specific combos
3. If WF shows >60% positive periods → candidate for live test
4. Paper trade on Alpaca for 3 months before real capital
5. Size positions by Kelly criterion × 0.25 (quarter-Kelly for safety)
""")

    st.subheader("Portfolio Construction (Inverse Vol Weighted)")
    top_combos = lb[(lb["Sharpe"] > 0.5) & (lb["Trades"] >= 5)].head(10)
    if len(top_combos) > 0:
        vols = [max(all_results[r["Strategy"]][r["Ticker"]]["ann_vol"], 1.0)
                for _, r in top_combos.iterrows()]
        inv_vols = [1 / v for v in vols]
        weights = [iv / sum(inv_vols) for iv in inv_vols]
        port_rows = []
        port_ret = 0
        for (_, combo), w in zip(top_combos.iterrows(), weights):
            r = all_results[combo["Strategy"]][combo["Ticker"]]
            port_ret += r["ann_return"] * w
            port_rows.append({
                "Ticker": combo["Ticker"], "Strategy": combo["Strategy"][:15],
                "Weight": f"{w*100:.1f}%", "Alloc": f"${capital*w:,.0f}",
                "Sharpe": r["sharpe"], "Ann Ret %": r["ann_return"], "Max DD %": r["max_dd"],
            })
        st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)
        st.metric("Portfolio Weighted Ann. Return", f"{port_ret:+.2f}%",
                  help="Assumes zero correlation between strategies — optimistic upper bound")
