# ============================================================
# PORTFOLIO MONITOR — STREAMLIT APP
# ============================================================

import textwrap
from math import sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Monitor",
    page_icon="📈",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────
BENCHMARKS = {
    "SP500 (VFV.TO)":  "VFV.TO",
    "NASDAQ (XQQ.TO)": "XQQ.TO",
    "XEQT (XEQT.TO)":  "XEQT.TO",
}
RISK_FREE_RATE     = 0.03
MAX_LOOKAHEAD_DAYS = 5
_SKIP_TICKERS      = {"CASH", "USD", "CAD", "GBP", "EUR"}

EXAMPLE_CSV = textwrap.dedent("""\
    date,ticker,amount,type
    2023-01-10,VFV.TO,5000,BUY
    2023-01-10,XQQ.TO,3000,BUY
    2023-01-10,XEQT.TO,2000,BUY
    2023-06-01,VFV.TO,4000,BUY
    2023-06-01,ZSP.TO,2000,BUY
    2024-01-15,VFV.TO,5000,BUY
    2024-01-15,XQQ.TO,3000,BUY
    2024-06-03,VFV.TO,1500,SELL
    2024-06-03,XEQT.TO,2000,BUY
""")

# ── Formatting helpers ───────────────────────────────────────
def pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"

def money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def num(x, decimals=3):
    return f"{x:.{decimals}f}" if pd.notna(x) else "—"


# ── Load & validate CSV ──────────────────────────────────────
def load_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and validate the transactions CSV.

    Required columns: date, ticker, amount, type
      - date:   trade date (any common format)
      - ticker: Yahoo Finance symbol
      - amount: positive dollar value
      - type:   BUY or SELL
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept 'purchase_date' as an alias for 'date' for backwards compatibility
    if "purchase_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"purchase_date": "date"})

    required = {"date", "ticker", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"]   = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    if df["amount"].lt(0).any():
        raise ValueError(
            "All amounts must be positive. Use the 'type' column to indicate "
            "BUY or SELL rather than using negative numbers."
        )

    # Default to BUY if no type column provided (backwards compatibility)
    if "type" not in df.columns:
        df["type"] = "BUY"
    else:
        df["type"] = df["type"].str.upper().str.strip()
        invalid_types = df[~df["type"].isin({"BUY", "SELL"})]["type"].unique()
        if len(invalid_types) > 0:
            raise ValueError(
                f"Invalid values in 'type' column: {list(invalid_types)}. "
                "Only BUY and SELL are accepted."
            )

    # Filter out non-security rows
    def is_valid_ticker(t):
        if t in _SKIP_TICKERS:
            return False
        if any(c.isdigit() for c in t):
            return False
        return True

    before = len(df)
    df = df[df["ticker"].apply(is_valid_ticker)].copy()
    skipped_n = before - len(df)
    if skipped_n > 0:
        st.info(
            f"ℹ️ {skipped_n} row(s) skipped — tickers that aren't recognisable "
            "securities (e.g. CASH, internal codes)."
        )

    return df.sort_values("date").reset_index(drop=True)


# ── Price download ───────────────────────────────────────────
def download_prices(tickers: List[str], start_date: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    prices = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) \
             else data.to_frame(name=tickers[0])
    prices = prices.dropna(how="all").ffill()

    # Retry bare tickers with .TO suffix (TSX securities)
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    retry   = [t for t in missing if not t.endswith(".TO")]
    if retry:
        data2 = yf.download(
            tickers=[t + ".TO" for t in retry],
            start=start_date.strftime("%Y-%m-%d"),
            auto_adjust=True, actions=True, progress=False,
            group_by="column", threads=False,
        )
        prices2 = data2["Close"].copy() if isinstance(data2.columns, pd.MultiIndex) \
                  else data2.to_frame(name=retry[0] + ".TO") if len(retry) == 1 \
                  else pd.DataFrame()
        prices2 = prices2.dropna(how="all").ffill()
        for t in retry:
            col = t + ".TO"
            if col in prices2.columns and not prices2[col].dropna().empty:
                prices[t] = prices2[col]

    return prices


# ── Build trades (buys = positive units, sells = negative units) ─
def build_trades(txns: pd.DataFrame, prices: pd.DataFrame) -> tuple:
    rows, skipped = [], []
    for _, row in txns.iterrows():
        date    = pd.Timestamp(row["date"]).normalize()
        ticker  = row["ticker"]
        amount  = float(row["amount"])
        is_sell = row["type"] == "SELL"

        if ticker not in prices.columns:
            skipped.append({"date": date, "ticker": ticker,
                            "amount": amount, "type": row["type"],
                            "reason": "ticker not found"})
            continue

        series = prices[ticker].dropna()
        window = series.index[
            (series.index >= date) &
            (series.index <= date + pd.Timedelta(days=MAX_LOOKAHEAD_DAYS))
        ]
        if len(window) == 0:
            skipped.append({"date": date, "ticker": ticker,
                            "amount": amount, "type": row["type"],
                            "reason": "no price in window"})
            continue

        price = float(series.loc[window[0]])
        if np.isnan(price) or price <= 0:
            skipped.append({"date": date, "ticker": ticker,
                            "amount": amount, "type": row["type"],
                            "reason": "invalid price"})
            continue

        # Sells produce negative units
        signed_units = -(amount / price) if is_sell else (amount / price)

        rows.append({
            "date":           date,
            "effective_date": window[0],
            "ticker":         ticker,
            "amount":         amount,
            "type":           row["type"],
            "trade_price":    price,
            "units":          signed_units,
        })

    return pd.DataFrame(rows), pd.DataFrame(skipped)


# ── Units matrix & portfolio value ───────────────────────────
def build_units_matrix(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(trades["ticker"].unique())
    mat = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for _, row in trades.iterrows():
        d = pd.Timestamp(row["effective_date"]).normalize()
        if d in mat.index:
            mat.loc[d, row["ticker"]] += float(row["units"])
    # Clamp to zero — can't hold negative units (handles oversell edge cases)
    return mat.cumsum().clip(lower=0)


def build_portfolio_value(units_matrix: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    common = [c for c in units_matrix.columns if c in prices.columns]
    return (units_matrix[common] * prices[common].ffill()).sum(axis=1)


# ── XIRR ─────────────────────────────────────────────────────
def xirr(cashflows: list) -> float:
    """
    Money-weighted annualized return.
    Cashflows: list of (date, amount) where negatives = money out, positives = money in.
    """
    if len(cashflows) < 2:
        return np.nan
    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = dates[0]
    years   = [(d - t0).days / 365.0 for d in dates]

    def safe_npv(r):
        try:
            return sum(a / (1 + r) ** t for a, t in zip(amounts, years))
        except (OverflowError, ZeroDivisionError):
            return np.nan

    def safe_deriv(r):
        try:
            return sum(-t * a / (1 + r) ** (t + 1) for a, t in zip(amounts, years))
        except (OverflowError, ZeroDivisionError):
            return np.nan

    rate = 0.1
    for _ in range(200):
        f, d = safe_npv(rate), safe_deriv(rate)
        if pd.isna(f) or pd.isna(d) or d == 0:
            break
        nr = rate - f / d
        nr = max(-0.999, min(nr, 100.0))
        if abs(nr - rate) < 1e-8:
            return nr
        rate = nr
    return rate


# ── Return metrics (sell-aware) ──────────────────────────────
def net_invested(txns: pd.DataFrame) -> float:
    """
    Net cash deployed: sum of buys minus sum of sells.
    This is your true cost basis — what you're actually still 'in' for.
    """
    buys  = txns.loc[txns["type"] == "BUY",  "amount"].sum()
    sells = txns.loc[txns["type"] == "SELL", "amount"].sum()
    return buys - sells


def total_return_metric(txns: pd.DataFrame, current_value: float) -> float:
    """
    Total return on net invested capital.
    (Current portfolio value - net invested) / net invested
    """
    ni = net_invested(txns)
    return (current_value - ni) / ni if ni > 0 else np.nan


def annualized_return_metric(txns: pd.DataFrame, current_value: float) -> float:
    """
    XIRR using actual cashflow signs:
      - BUY  → negative cashflow (money leaving your pocket)
      - SELL → positive cashflow (money returning to your pocket)
    Terminal value is today's portfolio value (positive cashflow).
    """
    cfs = []
    for _, row in txns.iterrows():
        date   = pd.Timestamp(row["date"]).normalize()
        amount = float(row["amount"])
        # Buys are outflows (negative), sells are inflows (positive)
        signed = amount if row["type"] == "SELL" else -amount
        cfs.append((date, signed))

    cfs.append((pd.Timestamp("today").normalize(), current_value))
    cfs.sort(key=lambda x: x[0])
    return xirr(cfs)


def annualized_vol(value_series: pd.Series, trade_dates=None) -> float:
    ret = value_series.pct_change()
    if trade_dates is not None:
        dd = pd.DatetimeIndex([pd.Timestamp(d).normalize()
                               for d in pd.to_datetime(trade_dates)])
        ret.loc[ret.index.isin(dd)] = np.nan
    return ret.dropna().std() * sqrt(252)


def sharpe_ratio(ann_ret: float, ann_vol: float, rfr: float) -> float:
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 0:
        return np.nan
    return (ann_ret - rfr) / ann_vol


# ── Benchmark series ─────────────────────────────────────────
def build_benchmark_series(ticker: str, txns: pd.DataFrame,
                            prices: pd.DataFrame) -> pd.Series:
    """
    Inflow-matched benchmark: buys invest into the benchmark,
    sells are treated as cash withdrawn (reducing benchmark units proportionally).
    """
    series   = prices[ticker].ffill().dropna()
    units_ch = pd.Series(0.0, index=series.index)

    # Group net flow per date (buys positive, sells negative dollar flow)
    flows = txns.copy()
    flows["signed_amount"] = flows.apply(
        lambda r: float(r["amount"]) if r["type"] == "BUY" else -float(r["amount"]),
        axis=1
    )
    by_date = flows.groupby("date")["signed_amount"].sum().reset_index()

    cumulative_units = 0.0
    for _, row in by_date.iterrows():
        date   = pd.Timestamp(row["date"]).normalize()
        flow   = float(row["signed_amount"])
        eligible = series.index[series.index >= date]
        if len(eligible) == 0:
            continue
        price = float(series.loc[eligible[0]])
        if np.isnan(price) or price <= 0:
            continue
        if flow > 0:
            # Buy: invest flow into benchmark
            units_ch.loc[eligible[0]] += flow / price
        else:
            # Sell: withdraw proportionally — reduce units by same fraction
            # as the sell represents of current portfolio value
            current_val = cumulative_units * price
            if current_val > 0:
                fraction    = min(abs(flow) / current_val, 1.0)
                units_ch.loc[eligible[0]] -= cumulative_units * fraction
        # Update running cumulative so next sell fraction is correct
        cumulative_units = max(0.0, cumulative_units + units_ch.loc[eligible[0]])

    return (units_ch.cumsum().clip(lower=0) * series).rename(ticker)


# ── Holdings table ───────────────────────────────────────────
def build_holdings_table(units_matrix: pd.DataFrame,
                         prices: pd.DataFrame) -> pd.DataFrame:
    if units_matrix.empty:
        return pd.DataFrame()
    latest = units_matrix.iloc[-1]
    rows = []
    for ticker, units in latest.items():
        if ticker not in prices.columns or units <= 0:
            continue
        s     = prices[ticker].ffill().dropna()
        price = float(s.iloc[-1]) if not s.empty else np.nan
        rows.append({
            "Ticker":       ticker,
            "Units":        round(float(units), 4),
            "Price":        price,
            "Market Value": float(units) * price,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        total = df["Market Value"].sum()
        df["Weight"] = df["Market Value"] / total
    return df


# ── Main computation ─────────────────────────────────────────
def compute_all(txns: pd.DataFrame) -> dict:
    all_tickers = list(txns["ticker"].unique()) + list(BENCHMARKS.values())
    all_tickers = list(dict.fromkeys(all_tickers))
    start       = txns["date"].min()

    with st.spinner("Downloading price history from Yahoo Finance…"):
        prices = download_prices(all_tickers, start)

    # Identify tickers with no price data
    no_data = [t for t in txns["ticker"].unique()
               if t not in prices.columns or prices[t].dropna().empty]
    excluded_rows = txns[txns["ticker"].isin(no_data)].copy() if no_data else pd.DataFrame()
    txns = txns[~txns["ticker"].isin(no_data)].copy()

    if txns.empty:
        raise ValueError("No priceable securities found. Check your ticker symbols.")

    trades, skipped = build_trades(txns, prices)
    if trades.empty:
        raise ValueError("No valid trades could be executed. Check ticker symbols and dates.")

    units_matrix   = build_units_matrix(trades, prices)
    portfolio_vals = build_portfolio_value(units_matrix, prices)
    current_val    = float(portfolio_vals.dropna().iloc[-1])
    trade_dates    = txns["date"]

    port_metrics = {
        "total_return":          total_return_metric(txns, current_val),
        "annualized_return":     annualized_return_metric(txns, current_val),
        "annualized_volatility": annualized_vol(portfolio_vals, trade_dates),
        "net_invested":          net_invested(txns),
        "total_bought":          txns.loc[txns["type"] == "BUY",  "amount"].sum(),
        "total_sold":            txns.loc[txns["type"] == "SELL", "amount"].sum(),
    }
    port_metrics["sharpe"] = sharpe_ratio(
        port_metrics["annualized_return"],
        port_metrics["annualized_volatility"],
        RISK_FREE_RATE,
    )

    bm_series = {
        name: build_benchmark_series(ticker, txns, prices)
        for name, ticker in BENCHMARKS.items()
        if ticker in prices.columns
    }

    bm_metrics = {}
    for name, series in bm_series.items():
        bm_val = float(series.dropna().iloc[-1])
        ar     = annualized_return_metric(txns, bm_val)
        av     = annualized_vol(series, trade_dates)
        bm_metrics[name] = {
            "ticker":                BENCHMARKS[name],
            "total_return":          total_return_metric(txns, bm_val),
            "annualized_return":     ar,
            "annualized_volatility": av,
            "sharpe":                sharpe_ratio(ar, av, RISK_FREE_RATE),
        }

    holdings = build_holdings_table(units_matrix, prices)

    return {
        "txns":              txns,
        "prices":            prices,
        "trades":            trades,
        "skipped":           skipped,
        "excluded_rows":     excluded_rows,
        "no_data_tickers":   no_data,
        "portfolio_vals":    portfolio_vals,
        "port_metrics":      port_metrics,
        "bm_series":         bm_series,
        "bm_metrics":        bm_metrics,
        "holdings":          holdings,
    }


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

st.title("📈 Portfolio Monitor")
st.caption("Upload your transactions CSV to compare your portfolio against inflow-matched benchmarks.")

# ── CSV format guide ─────────────────────────────────────────
with st.expander("📋 How to format your CSV — click to expand", expanded=True):
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
**Your CSV needs four columns:**

| Column | Format | Example |
|---|---|---|
| `date` | YYYY-MM-DD | `2024-01-15` |
| `ticker` | Yahoo Finance symbol | `VFV.TO` |
| `amount` | Dollar amount (positive, no $ sign) | `5000` |
| `type` | Transaction type | `BUY` or `SELL` |

**Tips:**
- One row per transaction — multiple securities on the same date = multiple rows
- **Amount is always positive** — use `type` to indicate BUY or SELL, not a negative number
- Canadian ETFs need the `.TO` suffix (e.g. `VFV.TO`, `XEQT.TO`); US stocks don't (e.g. `AAPL`, `VXUS`)
- Dates must be trading days, or within 5 days of one — weekends and holidays are handled automatically
- The `type` column is optional — if omitted, all rows are treated as BUY
        """)

    with col_b:
        st.markdown("**Example file contents:**")
        st.code(EXAMPLE_CSV, language="text")
        st.download_button(
            label="⬇️ Download example CSV",
            data=EXAMPLE_CSV,
            file_name="example_transactions.csv",
            mime="text/csv",
        )

st.divider()

# ── File upload ───────────────────────────────────────────────
uploaded = st.file_uploader("Upload your transactions CSV", type=["csv"])

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    try:
        txns = load_transactions(raw_df)
    except Exception as e:
        st.error(f"CSV validation error: {e}")
        st.stop()

    buys  = txns[txns["type"] == "BUY"]
    sells = txns[txns["type"] == "SELL"]
    st.success(
        f"Loaded {len(txns)} transactions — "
        f"{len(buys)} buys and {len(sells)} sells "
        f"across {txns['date'].nunique()} dates."
    )

    try:
        results = compute_all(txns)
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        st.stop()

    pm       = results["port_metrics"]
    bm       = results["bm_metrics"]
    no_data  = results["no_data_tickers"]
    excluded = results["excluded_rows"]

    # ── Missing data alert ────────────────────────────────────
    if no_data:
        excl_total = excluded["amount"].sum()
        excl_summary = (
            excluded.groupby(["ticker", "type"])["amount"]
            .agg(transactions="count", total_amount="sum")
            .reset_index()
            .rename(columns={
                "ticker": "Ticker", "type": "Type",
                "transactions": "# Transactions", "total_amount": "$ Amount",
            })
        )
        excl_summary["$ Amount"] = excl_summary["$ Amount"].apply(money)

        st.error(
            f"⚠️ **Pricing data missing for {len(no_data)} ticker(s) — "
            f"${excl_total:,.2f} excluded from analysis.** "
            "Your results are understated by the amounts shown below."
        )
        with st.expander(f"See excluded tickers ({len(no_data)})", expanded=True):
            st.dataframe(excl_summary.set_index("Ticker"), use_container_width=True)
            st.markdown("""
**Why might a ticker be missing?**
- **Wrong symbol** — Canadian ETFs need `.TO` (e.g. `VFV.TO`). US stocks use plain symbols (e.g. `AAPL`).
- **Options / warrants** — no continuous price history on Yahoo Finance
- **GICs / structured notes** — not exchange-traded
- **Internal account codes** — brokerage labels that aren't real tickers
- **Delisted or renamed** — the security changed its symbol

**What to do:** Search for each ticker at [finance.yahoo.com](https://finance.yahoo.com), copy the exact symbol shown, update your CSV, and re-upload.
            """)
        st.divider()

    # ── Summary cards ─────────────────────────────────────────
    st.subheader("Portfolio Summary")

    ni       = pm["net_invested"]
    cv       = results["portfolio_vals"].dropna().iloc[-1]
    gain     = cv - ni
    tb       = pm["total_bought"]
    ts       = pm["total_sold"]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Bought",      money(tb))
    c2.metric("Total Sold",        money(ts))
    c3.metric("Net Invested",      money(ni))
    c4.metric("Current Value",     money(cv))
    c5.metric("Gain / Loss",       money(gain), delta=pct(pm["total_return"]))
    c6.metric("Annualized Return", pct(pm["annualized_return"]))
    c7.metric("Sharpe-like",       num(pm["sharpe"]))

    st.divider()

    # ── Chart ─────────────────────────────────────────────────
    st.subheader("Growth of $100 — Portfolio vs Benchmarks")
    st.caption(
        "Normalized against net invested capital (buys minus sells). "
        "Benchmarks mirror your exact buy and sell timing."
    )

    fig        = go.Figure()
    port_vals  = results["portfolio_vals"]
    norm_base  = ni  # normalize against net invested

    port_norm = port_vals / norm_base * 100
    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name="PORTFOLIO",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.07)",
    ))

    bm_colors = {
        "SP500 (VFV.TO)":  "#3b82f6",
        "NASDAQ (XQQ.TO)": "#f97316",
        "XEQT (XEQT.TO)":  "#22c55e",
    }
    for name, series in results["bm_series"].items():
        norm = series / norm_base * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=name,
            line=dict(color=bm_colors.get(name, "#94a3b8"), width=1.5),
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Break-even", annotation_position="bottom right")

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#1e293b", size=13),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
        ),
        xaxis=dict(showgrid=False, color="#1e293b"),
        yaxis=dict(title="Growth of 100", gridcolor="#f1f5f9", color="#1e293b"),
        font=dict(color="#1e293b"),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Performance table ─────────────────────────────────────
    st.subheader("Performance Comparison")

    table_rows = []
    for name, m in bm.items():
        table_rows.append({
            "":                      name,
            "Ticker":                m["ticker"],
            "Total Return":          pct(m["total_return"]),
            "Annualized Return":     pct(m["annualized_return"]),
            "Annualized Volatility": pct(m["annualized_volatility"]),
            "Sharpe-like":           num(m["sharpe"]),
        })
    table_rows.append({
        "":                      "**PORTFOLIO**",
        "Ticker":                "—",
        "Total Return":          pct(pm["total_return"]),
        "Annualized Return":     pct(pm["annualized_return"]),
        "Annualized Volatility": pct(pm["annualized_volatility"]),
        "Sharpe-like":           num(pm["sharpe"]),
    })
    st.dataframe(pd.DataFrame(table_rows).set_index(""), use_container_width=True)

    st.divider()

    # ── Am I beating the market? ──────────────────────────────
    st.subheader("Am I Beating the Market?")

    def verdict(v):
        if pd.isna(v):  return "—"
        if v > 0:       return "✅ Beating"
        if v < 0:       return "❌ Trailing"
        return "= Matching"

    beat_rows = []
    for name, m in bm.items():
        beat_rows.append({
            "Benchmark":        name,
            "Return Verdict":   verdict(pm["total_return"]        - m["total_return"]),
            "Excess Return":    pct(pm["total_return"]            - m["total_return"]),
            "Risk-Adj Verdict": verdict(pm["sharpe"]              - m["sharpe"]),
            "Excess Sharpe":    num(pm["sharpe"]                  - m["sharpe"]),
        })
    st.dataframe(pd.DataFrame(beat_rows).set_index("Benchmark"), use_container_width=True)

    st.divider()

    # ── Current holdings ──────────────────────────────────────
    st.subheader("Current Holdings")
    h = results["holdings"].copy()
    if not h.empty:
        h["Price"]        = h["Price"].apply(money)
        h["Market Value"] = h["Market Value"].apply(money)
        h["Weight"]       = h["Weight"].apply(pct)
        st.dataframe(h, use_container_width=True)
    else:
        st.info("No current holdings found.")

    # ── Skipped trades ────────────────────────────────────────
    if not results["skipped"].empty:
        with st.expander("⚠️ Some transactions were skipped", expanded=False):
            st.dataframe(results["skipped"], use_container_width=True)

else:
    st.info("👆 Upload your CSV above to get started. Expand the guide to see the required format.")
