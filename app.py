import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------
# Black-Scholes utilities
# -----------------------------
def bs_price(S, K, T, r, sigma, option_type="put"):
    """
    European Black-Scholes (no dividends). T in years.
    """
    if T <= 0:
        # intrinsic at expiry
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if sigma <= 1e-8:
        # near-zero vol => approx intrinsic discounted (rough)
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def pick_strike_by_target_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_abs_delta: float,
    option_type: str,
) -> float:
    """
    Pick an approximate strike from target delta using BS inversion.
    This is a rough method to let you specify "sell 0.30 delta".
    """
    # delta formulas:
    # call delta = N(d1)
    # put delta  = N(d1) - 1  => abs(delta)=1-N(d1)
    if T <= 0:
        return S

    if sigma <= 1e-8:
        return S

    if option_type == "call":
        # target delta in (0,1)
        d1 = norm.ppf(np.clip(target_abs_delta, 1e-6, 1 - 1e-6))
    else:
        # abs(put delta)=1-N(d1) => N(d1)=1-abs(delta)
        d1 = norm.ppf(np.clip(1 - target_abs_delta, 1e-6, 1 - 1e-6))

    # d1 = [ln(S/K) + (r+0.5*sigma^2)T] / (sigma*sqrt(T))
    # => ln(S/K) = d1*sigma*sqrt(T) - (r+0.5*sigma^2)T
    ln_S_over_K = d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma * sigma) * T
    K = S / math.exp(ln_S_over_K)
    return float(K)


# -----------------------------
# Strategy definitions
# -----------------------------
@dataclass
class StrategySpec:
    name: str
    params: Dict[str, Any]


def parse_strategy_text(text: str) -> StrategySpec:
    """
    Very simple "strategy DSL" parser:
    Examples:
      CSP(delta=0.30, dte=30, exit_pct=0.5)
      CC(delta=0.20, dte=14, exit_pct=0.5)
      PCS(delta=0.30, dte=30, width=5, credit_min=0.20)
      CCS(delta=0.20, dte=30, width=5, credit_min=0.20)

    Returns StrategySpec(name, params).
    """
    text = text.strip()
    if "(" not in text or not text.endswith(")"):
        raise ValueError("Strategy must look like NAME(param=value, ...). Example: CSP(delta=0.30, dte=30)")

    name = text.split("(", 1)[0].strip().upper()
    inside = text.split("(", 1)[1][:-1].strip()

    params: Dict[str, Any] = {}
    if inside:
        parts = [p.strip() for p in inside.split(",")]
        for p in parts:
            k, v = [x.strip() for x in p.split("=", 1)]
            # basic type coercion
            if v.lower() in ["true", "false"]:
                val = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except:
                    val = v.strip('"').strip("'")
            params[k.lower()] = val

    return StrategySpec(name=name, params=params)


# -----------------------------
# Backtest engine
# -----------------------------
@dataclass
class Position:
    kind: str  # "short_put", "short_call", "put_spread", "call_spread"
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    contracts: int

    # legs:
    # for single short option:
    K: Optional[float] = None
    option_type: Optional[str] = None  # "put" or "call"

    # for spreads:
    K_short: Optional[float] = None
    K_long: Optional[float] = None
    spread_type: Optional[str] = None  # "put" or "call"

    entry_credit: float = 0.0  # per share
    # bookkeeping:
    closed: bool = False
    close_date: Optional[pd.Timestamp] = None
    close_debit: float = 0.0  # per share (what we paid to close)


def realized_vol_annualized(returns: pd.Series) -> float:
    # daily returns -> annualized vol
    vol = returns.std(ddof=0) * math.sqrt(252)
    return float(vol) if np.isfinite(vol) else 0.0


def yearfrac(days: int) -> float:
    return max(days, 0) / 365.0


def nearest_trading_day(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    if ts in index:
        return ts
    # pick next available
    pos = index.searchsorted(ts)
    if pos >= len(index):
        return index[-1]
    return index[pos]


def run_backtest(
    px: pd.DataFrame,
    strategy: StrategySpec,
    initial_cash: float,
    contracts: int,
    r: float,
    iv_lookback: int,
    iv_floor: float,
    iv_cap: float,
    fee_per_contract: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Model-based options backtest:
    - Price options daily with BS using sigma from rolling realized vol.
    - Enter a new position whenever none is open (serial selling).
    - Exit early if option value falls below (1-exit_pct)*credit (profit take).
    - At expiry: handle exercise/assignment (simplified).
    """
    df = px.copy()
    df = df.dropna()
    df["ret"] = df["Close"].pct_change()
    df["sigma"] = df["ret"].rolling(iv_lookback).apply(lambda x: realized_vol_annualized(pd.Series(x)), raw=False)
    df["sigma"] = df["sigma"].fillna(method="bfill").fillna(iv_floor).clip(lower=iv_floor, upper=iv_cap)

    # portfolio:
    cash = initial_cash
    shares = 0  # used for CC
    open_pos: Optional[Position] = None

    equity_curve = []
    trades: List[Dict[str, Any]] = []

    # strategy params with defaults
    p = strategy.params
    dte = int(p.get("dte", 30))
    delta = float(p.get("delta", 0.30))
    exit_pct = float(p.get("exit_pct", 0.50))  # close when you've captured exit_pct of credit
    width = float(p.get("width", 5.0))  # for spreads
    credit_min = float(p.get("credit_min", 0.0))  # per share, for spreads

    name = strategy.name

    for i, (dt, row) in enumerate(df.iterrows()):
        S = float(row["Close"])
        sigma = float(row["sigma"])

        # Mark-to-market value of open position
        pos_value = 0.0
        if open_pos and not open_pos.closed:
            days_to_exp = (open_pos.expiry_date - dt).days
            T = yearfrac(days_to_exp)
            if open_pos.kind in ["short_put", "short_call"]:
                opt_price = bs_price(S, open_pos.K, T, r, sigma, open_pos.option_type)
                # short option => liability
                pos_value = -opt_price * 100 * open_pos.contracts
            elif open_pos.kind in ["put_spread", "call_spread"]:
                short_type = open_pos.spread_type
                short_price = bs_price(S, open_pos.K_short, T, r, sigma, short_type)
                long_price = bs_price(S, open_pos.K_long, T, r, sigma, short_type)
                # credit spread (short - long)
                spread_price = (short_price - long_price)
                pos_value = -spread_price * 100 * open_pos.contracts

        # equity = cash + shares + open_pos mtm
        equity = cash + shares * S + pos_value
        equity_curve.append({"Date": dt, "Equity": equity, "Cash": cash, "Shares": shares, "Underlying": S})

        # If we have an open position, consider early exit or expiry actions
        if open_pos and not open_pos.closed:
            days_to_exp = (open_pos.expiry_date - dt).days
            T = yearfrac(days_to_exp)

            # compute current debit to close
            if open_pos.kind in ["short_put", "short_call"]:
                cur_price = bs_price(S, open_pos.K, T, r, sigma, open_pos.option_type)
                debit_to_close = cur_price
                # profit capture condition: current price <= (1-exit_pct)*entry_credit
                target = (1 - exit_pct) * open_pos.entry_credit
                if debit_to_close <= target and days_to_exp > 0:
                    # close early
                    cost = debit_to_close * 100 * open_pos.contracts + fee_per_contract * open_pos.contracts
                    cash -= cost
                    open_pos.closed = True
                    open_pos.close_date = dt
                    open_pos.close_debit = debit_to_close
                    trades.append({
                        "entry_date": open_pos.entry_date,
                        "exit_date": dt,
                        "kind": open_pos.kind,
                        "K": open_pos.K,
                        "expiry": open_pos.expiry_date,
                        "credit": open_pos.entry_credit,
                        "debit": debit_to_close,
                        "contracts": open_pos.contracts,
                        "pnl_$": (open_pos.entry_credit - debit_to_close) * 100 * open_pos.contracts - fee_per_contract * open_pos.contracts * 2,
                        "note": "profit_take"
                    })

            elif open_pos.kind in ["put_spread", "call_spread"]:
                cur_short = bs_price(S, open_pos.K_short, T, r, sigma, open_pos.spread_type)
                cur_long = bs_price(S, open_pos.K_long, T, r, sigma, open_pos.spread_type)
                spread_debit = (cur_short - cur_long)  # cost to buy back spread
                target = (1 - exit_pct) * open_pos.entry_credit
                if spread_debit <= target and days_to_exp > 0:
                    cost = spread_debit * 100 * open_pos.contracts + fee_per_contract * open_pos.contracts
                    cash -= cost
                    open_pos.closed = True
                    open_pos.close_date = dt
                    open_pos.close_debit = spread_debit
                    trades.append({
                        "entry_date": open_pos.entry_date,
                        "exit_date": dt,
                        "kind": open_pos.kind,
                        "K_short": open_pos.K_short,
                        "K_long": open_pos.K_long,
                        "expiry": open_pos.expiry_date,
                        "credit": open_pos.entry_credit,
                        "debit": spread_debit,
                        "contracts": open_pos.contracts,
                        "pnl_$": (open_pos.entry_credit - spread_debit) * 100 * open_pos.contracts - fee_per_contract * open_pos.contracts * 2,
                        "note": "profit_take"
                    })

            # expiry handling
            if (not open_pos.closed) and (days_to_exp <= 0):
                # settle based on intrinsic
                if open_pos.kind == "short_put":
                    intrinsic = max(open_pos.K - S, 0.0)
                    # assignment: buy 100 shares per contract at strike
                    if intrinsic > 0:
                        buy_cost = open_pos.K * 100 * open_pos.contracts
                        cash -= buy_cost
                        shares += 100 * open_pos.contracts
                    # option expires; liability intrinsic "paid" via assignment above
                    open_pos.closed = True
                    open_pos.close_date = dt
                    open_pos.close_debit = intrinsic
                    trades.append({
                        "entry_date": open_pos.entry_date,
                        "exit_date": dt,
                        "kind": open_pos.kind,
                        "K": open_pos.K,
                        "expiry": open_pos.expiry_date,
                        "credit": open_pos.entry_credit,
                        "debit": intrinsic,
                        "contracts": open_pos.contracts,
                        "pnl_$": (open_pos.entry_credit - intrinsic) * 100 * open_pos.contracts - fee_per_contract * open_pos.contracts * 2,
                        "note": "expiry"
                    })

                elif open_pos.kind == "short_call":
                    intrinsic = max(S - open_pos.K, 0.0)
                    # if covered: shares called away
                    if intrinsic > 0 and shares >= 100 * open_pos.contracts:
                        sell_proceeds = open_pos.K * 100 * open_pos.contracts
                        cash += sell_proceeds
                        shares -= 100 * open_pos.contracts
                    open_pos.closed = True
                    open_pos.close_date = dt
                    open_pos.close_debit = intrinsic
                    trades.append({
                        "entry_date": open_pos.entry_date,
                        "exit_date": dt,
                        "kind": open_pos.kind,
                        "K": open_pos.K,
                        "expiry": open_pos.expiry_date,
                        "credit": open_pos.entry_credit,
                        "debit": intrinsic,
                        "contracts": open_pos.contracts,
                        "pnl_$": (open_pos.entry_credit - intrinsic) * 100 * open_pos.contracts - fee_per_contract * open_pos.contracts * 2,
                        "note": "expiry"
                    })

                elif open_pos.kind in ["put_spread", "call_spread"]:
                    # intrinsic of spread at expiry
                    if open_pos.spread_type == "put":
                        short_intr = max(open_pos.K_short - S, 0.0)
                        long_intr = max(open_pos.K_long - S, 0.0)
                    else:
                        short_intr = max(S - open_pos.K_short, 0.0)
                        long_intr = max(S - open_pos.K_long, 0.0)

                    spread_intr = short_intr - long_intr
                    spread_intr = max(spread_intr, 0.0)  # liability (credit spread)
                    open_pos.closed = True
                    open_pos.close_date = dt
                    open_pos.close_debit = spread_intr
                    trades.append({
                        "entry_date": open_pos.entry_date,
                        "exit_date": dt,
                        "kind": open_pos.kind,
                        "K_short": open_pos.K_short,
                        "K_long": open_pos.K_long,
                        "expiry": open_pos.expiry_date,
                        "credit": open_pos.entry_credit,
                        "debit": spread_intr,
                        "contracts": open_pos.contracts,
                        "pnl_$": (open_pos.entry_credit - spread_intr) * 100 * open_pos.contracts - fee_per_contract * open_pos.contracts * 2,
                        "note": "expiry"
                    })

        # If no open position, try to open one
        if (open_pos is None) or (open_pos.closed):
            # Create expiry date = dt + dte, aligned to trading days
            expiry_guess = dt + pd.Timedelta(days=dte)
            expiry = nearest_trading_day(df.index, expiry_guess)
            days_to_exp = (expiry - dt).days
            T = yearfrac(days_to_exp)

            if name == "CSP":
                # ensure we have enough cash for assignment
                K = pick_strike_by_target_delta(S, T, r, sigma, delta, "put")
                credit = bs_price(S, K, T, r, sigma, "put")
                collateral = K * 100 * contracts
                if cash >= collateral:
                    cash += credit * 100 * contracts - fee_per_contract * contracts
                    open_pos = Position(
                        kind="short_put",
                        entry_date=dt,
                        expiry_date=expiry,
                        contracts=contracts,
                        K=K,
                        option_type="put",
                        entry_credit=credit,
                    )
                    trades.append({
                        "entry_date": dt,
                        "exit_date": None,
                        "kind": "OPEN short_put",
                        "K": K,
                        "expiry": expiry,
                        "credit": credit,
                        "debit": None,
                        "contracts": contracts,
                        "pnl_$": None,
                        "note": "opened"
                    })

            elif name == "CC":
                # need shares to cover; if none, buy them
                needed = 100 * contracts
                if shares < needed:
                    buy_cost = (needed - shares) * S
                    if cash >= buy_cost:
                        cash -= buy_cost
                        shares = needed
                if shares >= needed:
                    K = pick_strike_by_target_delta(S, T, r, sigma, delta, "call")
                    credit = bs_price(S, K, T, r, sigma, "call")
                    cash += credit * 100 * contracts - fee_per_contract * contracts
                    open_pos = Position(
                        kind="short_call",
                        entry_date=dt,
                        expiry_date=expiry,
                        contracts=contracts,
                        K=K,
                        option_type="call",
                        entry_credit=credit,
                    )
                    trades.append({
                        "entry_date": dt,
                        "exit_date": None,
                        "kind": "OPEN short_call",
                        "K": K,
                        "expiry": expiry,
                        "credit": credit,
                        "debit": None,
                        "contracts": contracts,
                        "pnl_$": None,
                        "note": "opened"
                    })

            elif name in ["PCS", "CCS"]:
                # Vertical credit spreads:
                # PCS = put credit spread (short put higher strike, long put lower strike)
                # CCS = call credit spread (short call lower strike, long call higher strike)
                opt_type = "put" if name == "PCS" else "call"
                K_short = pick_strike_by_target_delta(S, T, r, sigma, delta, opt_type)

                if opt_type == "put":
                    K_long = K_short - width
                else:
                    K_long = K_short + width

                # compute credit
                short_price = bs_price(S, K_short, T, r, sigma, opt_type)
                long_price = bs_price(S, K_long, T, r, sigma, opt_type)
                credit = short_price - long_price

                if credit >= credit_min:
                    # max loss per spread = width - credit (per share) => cash requirement
                    max_loss = max(width - credit, 0.0) * 100 * contracts
                    if cash >= max_loss:
                        cash += credit * 100 * contracts - fee_per_contract * contracts
                        open_pos = Position(
                            kind="put_spread" if opt_type == "put" else "call_spread",
                            entry_date=dt,
                            expiry_date=expiry,
                            contracts=contracts,
                            K_short=K_short,
                            K_long=K_long,
                            spread_type=opt_type,
                            entry_credit=credit,
                        )
                        trades.append({
                            "entry_date": dt,
                            "exit_date": None,
                            "kind": f"OPEN {open_pos.kind}",
                            "K_short": K_short,
                            "K_long": K_long,
                            "expiry": expiry,
                            "credit": credit,
                            "debit": None,
                            "contracts": contracts,
                            "pnl_$": None,
                            "note": "opened"
                        })
            else:
                raise ValueError(f"Unknown strategy name: {name}. Use CSP, CC, PCS, CCS.")

    eq = pd.DataFrame(equity_curve).set_index("Date")
    tr = pd.DataFrame(trades)
    return eq, tr


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Options Strategy Backtester (Model-based)", layout="wide")
st.title("Options Strategy Backtester (CSP / CC / Credit Spreads)")

with st.sidebar:
    st.header("Backtest Inputs")

    ticker = st.text_input("Ticker", value="SPY")
    start = st.date_input("Start date", value=pd.to_datetime("2019-01-01"))
    end = st.date_input("End date", value=pd.to_datetime("2024-12-31"))

    initial_cash = st.number_input("Initial cash ($)", min_value=1000.0, value=50000.0, step=1000.0)
    contracts = st.number_input("Contracts", min_value=1, value=1, step=1)

    r = st.number_input("Risk-free rate (annual)", min_value=0.0, value=0.03, step=0.005, format="%.3f")

    st.subheader("IV proxy (rolling realized vol)")
    iv_lookback = st.number_input("Lookback (trading days)", min_value=5, value=20, step=5)
    iv_floor = st.number_input("IV floor", min_value=0.01, value=0.15, step=0.01, format="%.2f")
    iv_cap = st.number_input("IV cap", min_value=0.10, value=1.00, step=0.05, format="%.2f")

    fee_per_contract = st.number_input("Fee per contract ($)", min_value=0.0, value=0.65, step=0.05)

st.markdown(
    """
Type a strategy in the box like:

- `CSP(delta=0.30, dte=30, exit_pct=0.50)`
- `CC(delta=0.20, dte=14, exit_pct=0.50)`
- `PCS(delta=0.30, dte=30, width=5, credit_min=0.20)`  *(Put credit spread)*
- `CCS(delta=0.20, dte=30, width=5, credit_min=0.20)`  *(Call credit spread)*

**Notes**
- This is a *model-based* backtest (BS price + realized-vol IV proxy).
- For real-world accuracy, swap in real options chain history (QuantConnect, ORATS, CBOE, Polygon, etc.).
"""
)

strategy_text = st.text_input("Strategy", value="CSP(delta=0.30, dte=30, exit_pct=0.50)")
run = st.button("Run backtest")

if run:
    try:
        strat = parse_strategy_text(strategy_text)
        st.write("Parsed strategy:", strat)

        with st.spinner("Downloading price history..."):
            data = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)

        if data.empty:
            st.error("No price data returned. Check ticker and date range.")
            st.stop()

        # Standardize columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        eq, tr = run_backtest(
            px=data,
            strategy=strat,
            initial_cash=float(initial_cash),
            contracts=int(contracts),
            r=float(r),
            iv_lookback=int(iv_lookback),
            iv_floor=float(iv_floor),
            iv_cap=float(iv_cap),
            fee_per_contract=float(fee_per_contract),
        )

        # Metrics
        eq["ret"] = eq["Equity"].pct_change().fillna(0.0)
        total_return = eq["Equity"].iloc[-1] / eq["Equity"].iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / max(len(eq), 1)) - 1
        ann_vol = eq["ret"].std(ddof=0) * math.sqrt(252)
        sharpe = (ann_return - float(r)) / ann_vol if ann_vol > 1e-9 else np.nan
        max_dd = (eq["Equity"] / eq["Equity"].cummax() - 1).min()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final equity", f"${eq['Equity'].iloc[-1]:,.0f}")
        c2.metric("Total return", f"{total_return*100:.1f}%")
        c3.metric("Ann. return (approx)", f"{ann_return*100:.1f}%")
        c4.metric("Sharpe (approx)", f"{sharpe:.2f}")
        c5.metric("Max drawdown", f"{max_dd*100:.1f}%")

        # Plot equity
        fig = plt.figure()
        plt.plot(eq.index, eq["Equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        st.pyplot(fig)

        # Trades table (cleaned)
        st.subheader("Trades / Events")
        tr_show = tr.copy()
        st.dataframe(tr_show, use_container_width=True)

        # Export
        st.download_button(
            "Download equity CSV",
            data=eq.to_csv().encode("utf-8"),
            file_name=f"{ticker}_equity.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download trades CSV",
            data=tr.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_trades.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")
