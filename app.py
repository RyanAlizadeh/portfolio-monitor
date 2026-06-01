# ============================================================
# PORTFOLIO MONITOR — STREAMLIT APP
# ============================================================

import io
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
RISK_FREE_RATE       = 0.03
MAX_LOOKAHEAD_DAYS   = 5

EXAMPLE_CSV = textwrap.dedent("""\
    purchase_date,ticker,amount
    2023-01-10,VFV.TO,5000
    2023-01-10,XQQ.TO,3000
    2023-01-10,XEQT.TO,2000
    2023-06-01,VFV.TO,4000
    2023-06-01,ZSP.TO,2000
    2024-01-15,VFV.TO,5000
    2024-01-15,XQQ.TO,3000
    2024-06-03,VFV.TO,4000
    2024-06-03,XEQT.TO,2000
""")

# ── Core computation functions ────────────────────────────────

# Rows with these ticker values are silently skipped — not real securities
_SKIP_TICKERS = {"CASH", "USD", "CAD", "GBP", "EUR"}

def load_purchases_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")
    missing = {"purchase_date", "ticker", "amount"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["amount"].lt(0).any():
        raise ValueError("All amounts must be non-negative.")
    # Drop known non-security rows and anything that looks like an internal code
    # (contains digits or is in the skip list)
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
        import streamlit as _st
        _st.info(f"ℹ️ {skipped_n} rows skipped — tickers that aren't recognisable securities (e.g. CASH, internal codes).")
    return df.sort_values("purchase_date").reset_index(drop=True)


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
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.to_frame(name=tickers[0])
    prices = prices.dropna(how="all").ffill()

    # For any ticker that came back empty, retry with .TO suffix (TSX securities)
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    retry = [t for t in missing if not t.endswith(".TO")]
    if retry:
        retry_to = [t + ".TO" for t in retry]
        data2 = yf.download(
            tickers=retry_to,
            start=start_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            actions=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        if isinstance(data2.columns, pd.MultiIndex):
            prices2 = data2["Close"].copy()
        else:
            prices2 = data2.to_frame(name=retry_to[0]) if len(retry_to) == 1 else pd.DataFrame()
        prices2 = prices2.dropna(how="all").ffill()
        # Map .TO columns back to the original bare ticker name
        rename_map = {t + ".TO": t for t in retry}
        prices2 = prices2.rename(columns=rename_map)
        for col in prices2.columns:
            if col in rename_map.values() and not prices2[col].dropna().empty:
                prices[col] = prices2[col]

    return prices


def purchases_to_trades(purchases: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, skipped = [], []
    for _, row in purchases.iterrows():
        date   = pd.Timestamp(row["purchase_date"]).normalize()
        ticker = row["ticker"]
        amount = float(row["amount"])
        if ticker not in prices.columns:
            skipped.append({"date": date, "ticker": ticker, "amount": amount, "reason": "ticker not found"})
            continue
        series = prices[ticker].dropna()
        window = series.index[(series.index >= date) & (series.index <= date + pd.Timedelta(days=MAX_LOOKAHEAD_DAYS))]
        if len(window) == 0:
            skipped.append({"date": date, "ticker": ticker, "amount": amount, "reason": "no price in window"})
            continue
        price = float(series.loc[window[0]])
        if np.isnan(price) or price <= 0:
            skipped.append({"date": date, "ticker": ticker, "amount": amount, "reason": "invalid price"})
            continue
        rows.append({"purchase_date": date, "effective_date": window[0], "ticker": ticker,
                     "amount": amount, "trade_price": price, "units": amount / price})
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def build_units_matrix(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(trades["ticker"].unique())
    mat = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for _, row in trades.iterrows():
        d = pd.Timestamp(row["effective_date"]).normalize()
        if d in mat.index:
            mat.loc[d, row["ticker"]] += float(row["units"])
    return mat.cumsum()


def build_portfolio_value(units_matrix: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    common = [c for c in units_matrix.columns if c in prices.columns]
    # Forward-fill each ticker's price independently before multiplying
    # so a single missing price day doesn't zero out the whole portfolio
    filled_prices = prices[common].ffill()
    daily_values = units_matrix[common] * filled_prices
    return daily_values.sum(axis=1)


def xirr(cashflows: list) -> float:
    if len(cashflows) < 2:
        return np.nan
    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = dates[0]
    years   = [(d - t0).days / 365.0 for d in dates]

    def safe_npv(rate):
        try:
            return sum(a / (1 + rate) ** t for a, t in zip(amounts, years))
        except (OverflowError, ZeroDivisionError):
            return np.nan

    def safe_npv_deriv(rate):
        try:
            return sum(-t * a / (1 + rate) ** (t + 1) for a, t in zip(amounts, years))
        except (OverflowError, ZeroDivisionError):
            return np.nan

    rate = 0.1
    for _ in range(200):
        f  = safe_npv(rate)
        df = safe_npv_deriv(rate)
        if pd.isna(f) or pd.isna(df) or df == 0:
            break
        nr = rate - f / df
        # Clamp to prevent runaway values on short histories
        nr = max(-0.999, min(nr, 100.0))
        if abs(nr - rate) < 1e-8:
            return nr
        rate = nr
    return rate


def total_return(purchases: pd.DataFrame, current_value: float) -> float:
    invested = purchases["amount"].sum()
    return (current_value - invested) / invested if invested > 0 else np.nan


def annualized_return(purchases: pd.DataFrame, current_value: float) -> float:
    cfs = [(pd.Timestamp(r["purchase_date"]).normalize(), -float(r["amount"])) for _, r in purchases.iterrows()]
    cfs.append((pd.Timestamp("today").normalize(), current_value))
    cfs.sort(key=lambda x: x[0])
    return xirr(cfs)


def annualized_vol(value_series: pd.Series, deployment_dates=None) -> float:
    ret = value_series.pct_change()
    if deployment_dates is not None:
        dd = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in pd.to_datetime(deployment_dates)])
        ret.loc[ret.index.isin(dd)] = np.nan
    return ret.dropna().std() * sqrt(252)


def sharpe(ann_ret: float, ann_vol: float, rfr: float) -> float:
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol <= 0:
        return np.nan
    return (ann_ret - rfr) / ann_vol


def build_benchmark_series(ticker: str, purchases: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    series = prices[ticker].ffill().dropna()
    by_date = purchases.groupby("purchase_date")["amount"].sum().reset_index()
    units_ch = pd.Series(0.0, index=series.index)
    for _, row in by_date.iterrows():
        date   = pd.Timestamp(row["purchase_date"]).normalize()
        amount = float(row["amount"])
        eligible = series.index[series.index >= date]
        if len(eligible) == 0:
            continue
        price = float(series.loc[eligible[0]])
        if np.isnan(price) or price <= 0:
            continue
        units_ch.loc[eligible[0]] += amount / price
    return (units_ch.cumsum() * series).rename(ticker)


def compute_all(purchases: pd.DataFrame) -> dict:
    all_tickers = list(purchases["ticker"].unique()) + list(BENCHMARKS.values())
    all_tickers = list(dict.fromkeys(all_tickers))
    start       = purchases["purchase_date"].min()

    with st.spinner("Downloading price history from Yahoo Finance…"):
        prices = download_prices(all_tickers, start)

    # Identify tickers with no usable price data
    no_data = [t for t in purchases["ticker"].unique()
               if t not in prices.columns or prices[t].dropna().empty]
    excluded_rows = pd.DataFrame()
    if no_data:
        excluded_rows = purchases[purchases["ticker"].isin(no_data)].copy()
        purchases = purchases[~purchases["ticker"].isin(no_data)].copy()

    if purchases.empty:
        raise ValueError("No priceable securities found. Check your ticker symbols.")

    trades, skipped = purchases_to_trades(purchases, prices)
    if trades.empty:
        raise ValueError("No valid trades could be created. Check your ticker symbols and dates.")

    units_matrix   = build_units_matrix(trades, prices)
    portfolio_vals = build_portfolio_value(units_matrix, prices)
    current_val    = float(portfolio_vals.dropna().iloc[-1])
    total_invested = float(purchases["amount"].sum())
    deploy_dates   = purchases["purchase_date"]

    port_metrics = {
        "total_return":          total_return(purchases, current_val),
        "annualized_return":     annualized_return(purchases, current_val),
        "annualized_volatility": annualized_vol(portfolio_vals, deploy_dates),
    }
    port_metrics["sharpe"] = sharpe(port_metrics["annualized_return"],
                                    port_metrics["annualized_volatility"], RISK_FREE_RATE)

    bm_series  = {name: build_benchmark_series(ticker, purchases, prices)
                  for name, ticker in BENCHMARKS.items() if ticker in prices.columns}

    bm_metrics = {}
    for name, series in bm_series.items():
        bm_val = float(series.dropna().iloc[-1])
        ar     = annualized_return(purchases, bm_val)
        av     = annualized_vol(series, deploy_dates)
        bm_metrics[name] = {
            "ticker":                BENCHMARKS[name],
            "total_return":          total_return(purchases, bm_val),
            "annualized_return":     ar,
            "annualized_volatility": av,
            "sharpe":                sharpe(ar, av, RISK_FREE_RATE),
        }

    holdings = build_current_holdings_table(units_matrix, prices)

    return {
        "purchases":       purchases,
        "prices":          prices,
        "trades":          trades,
        "skipped":         skipped,
        "excluded_rows":   excluded_rows,
        "no_data_tickers": no_data,
        "portfolio_vals":  portfolio_vals,
        "total_invested":  total_invested,
        "current_val":     current_val,
        "port_metrics":    port_metrics,
        "bm_series":       bm_series,
        "bm_metrics":      bm_metrics,
        "holdings":        holdings,
    }


def build_current_holdings_table(units_matrix, prices):
    if units_matrix.empty:
        return pd.DataFrame()
    latest = units_matrix.iloc[-1]
    rows = []
    for ticker, units in latest.items():
        if ticker not in prices.columns:
            continue
        s     = prices[ticker].dropna()
        price = float(s.iloc[-1]) if not s.empty else np.nan
        rows.append({"Ticker": ticker, "Units": round(float(units), 4),
                     "Price": price, "Market Value": float(units) * price})
    df = pd.DataFrame(rows)
    if not df.empty:
        total = df["Market Value"].sum()
        df["Weight"] = df["Market Value"] / total
    return df


# ── Formatting helpers ───────────────────────────────────────

def pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"

def money(x):
    return f"${x:,.2f}" if pd.notna(x) else "—"

def num(x, decimals=3):
    return f"{x:.{decimals}f}" if pd.notna(x) else "—"


# ── UI ───────────────────────────────────────────────────────

st.title("📈 Portfolio Monitor")
st.caption("Upload your purchases CSV to compare your portfolio against inflow-matched benchmarks.")

# ── CSV FORMAT GUIDE ─────────────────────────────────────────
with st.expander("📋 How to format your CSV — click to expand", expanded=True):
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("""
**Your CSV needs exactly three columns:**

| Column | Format | Example |
|---|---|---|
| `purchase_date` | YYYY-MM-DD | `2024-01-15` |
| `ticker` | Yahoo Finance symbol | `VFV.TO` |
| `amount` | Dollar amount (no $ sign) | `5000` |

**Tips:**
- One row per purchase — if you bought multiple ETFs on the same date, that's multiple rows
- Tickers must match Yahoo Finance exactly — Canadian ETFs need the `.TO` suffix (e.g. `VFV.TO`, `XEQT.TO`)
- Dates must be trading days or within 5 days of one (weekends/holidays are handled automatically)
- The file must be saved as a `.csv` file
        """)

    with col_b:
        st.markdown("**Example file contents:**")
        st.code(EXAMPLE_CSV, language="text")
        st.download_button(
            label="⬇️ Download example CSV",
            data=EXAMPLE_CSV,
            file_name="example_purchases.csv",
            mime="text/csv",
        )

st.divider()

# ── File upload ───────────────────────────────────────────────
uploaded = st.file_uploader("Upload your purchases.csv", type=["csv"])

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    try:
        purchases = load_purchases_from_df(raw_df)
    except Exception as e:
        st.error(f"CSV validation error: {e}")
        st.stop()

    st.success(f"Loaded {len(purchases)} purchase rows across {purchases['purchase_date'].nunique()} dates.")

    try:
        results = compute_all(purchases)
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        st.stop()

    pm          = results["port_metrics"]
    bm          = results["bm_metrics"]
    ti          = results["total_invested"]
    cv          = results["current_val"]
    gain        = cv - ti
    no_data     = results["no_data_tickers"]
    excluded    = results["excluded_rows"]

    # ── Missing data alert ────────────────────────────────────
    if no_data:
        excluded_summary = (
            excluded.groupby("ticker")["amount"]
            .agg(purchases="count", total_amount="sum")
            .reset_index()
            .rename(columns={"ticker": "Ticker", "purchases": "# Purchases", "total_amount": "$ Amount in CSV"})
        )
        excluded_summary["$ Amount in CSV"] = excluded_summary["$ Amount in CSV"].apply(money)
        excluded_total = excluded["amount"].sum()

        excl_msg = (
            f"⚠️ **Pricing data missing for {len(no_data)} ticker(s) — "
            f"${excluded_total:,.2f} excluded from analysis**\n\n"
            "These securities could not be priced and are not included in your portfolio value or returns. "
            "Your results are understated by the amounts shown below."
        )
        st.error(excl_msg)
        with st.expander(f"See excluded tickers ({len(no_data)})", expanded=True):
            st.dataframe(excluded_summary.set_index("Ticker"), use_container_width=True)
            st.markdown("""
**Why might a ticker be missing?**
- **Wrong symbol** — Canadian ETFs need `.TO` suffix (e.g. `VFV.TO` not `VFV`). US stocks use their plain symbol (e.g. `AAPL`, `VXUS`).
- **Options / warrants** — these have no continuous price history on Yahoo Finance
- **GICs / structured notes** — not exchange-traded, no ticker
- **Internal account codes** — brokerage-specific labels that aren't real tickers
- **Delisted or renamed** — the security may have changed its symbol

**What to do:** Check each ticker at [finance.yahoo.com](https://finance.yahoo.com), search for the security, and copy the exact symbol shown. Update your CSV and re-upload.
""")
        st.divider()

    # ── Summary cards ─────────────────────────────────────────
    st.subheader("Portfolio Summary")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Invested",      money(ti))
    c2.metric("Current Value",       money(cv))
    c3.metric("Gain / Loss",         money(gain), delta=pct(pm["total_return"]))
    c4.metric("Total Return",        pct(pm["total_return"]))
    c5.metric("Annualized Return",   pct(pm["annualized_return"]))
    c6.metric("Volatility",          pct(pm["annualized_volatility"]))
    c7.metric("Sharpe-like",         num(pm["sharpe"]))

    st.divider()

    # ── Chart ─────────────────────────────────────────────────
    st.subheader("Growth of $100 — Portfolio vs Benchmarks")
    st.caption("All series normalized against total invested capital. Benchmarks deploy on the same dates and amounts as your portfolio.")

    fig = go.Figure()
    port_series = results["portfolio_vals"]

    # Portfolio line
    port_norm = port_series / ti * 100
    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name="PORTFOLIO", line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.07)",
    ))

    bm_colors = {
        "SP500 (VFV.TO)":  "#3b82f6",
        "NASDAQ (XQQ.TO)": "#f97316",
        "XEQT (XEQT.TO)":  "#22c55e",
    }
    for name, series in results["bm_series"].items():
        norm = series / ti * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=name, line=dict(color=bm_colors.get(name, "#94a3b8"), width=1.5),
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Break-even", annotation_position="bottom right")

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color="#1e293b", size=13),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=False, color="#1e293b"),
        yaxis=dict(title="Growth of 100", gridcolor="#f1f5f9", color="#1e293b"),
        font=dict(color="#1e293b"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Performance table ─────────────────────────────────────
    st.subheader("Performance Comparison")

    table_rows = []
    for name, m in bm.items():
        table_rows.append({
            "":                       name,
            "Ticker":                 m["ticker"],
            "Total Return":           pct(m["total_return"]),
            "Annualized Return":      pct(m["annualized_return"]),
            "Annualized Volatility":  pct(m["annualized_volatility"]),
            "Sharpe-like":            num(m["sharpe"]),
        })
    table_rows.append({
        "":                       "**PORTFOLIO**",
        "Ticker":                 "—",
        "Total Return":           pct(pm["total_return"]),
        "Annualized Return":      pct(pm["annualized_return"]),
        "Annualized Volatility":  pct(pm["annualized_volatility"]),
        "Sharpe-like":            num(pm["sharpe"]),
    })
    st.dataframe(pd.DataFrame(table_rows).set_index(""), use_container_width=True)

    st.divider()

    # ── Am I beating the market? ──────────────────────────────
    st.subheader("Am I Beating the Market?")

    beat_rows = []
    for name, m in bm.items():
        excess_ret    = pm["total_return"]    - m["total_return"]
        excess_sharpe = pm["sharpe"]          - m["sharpe"]

        def verdict(v):
            if pd.isna(v):   return "—"
            if v > 0:        return "✅ Beating"
            if v < 0:        return "❌ Trailing"
            return "= Matching"

        beat_rows.append({
            "Benchmark":            name,
            "Return Verdict":       verdict(excess_ret),
            "Excess Return":        pct(excess_ret),
            "Risk-Adj Verdict":     verdict(excess_sharpe),
            "Excess Sharpe":        num(excess_sharpe),
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

    # ── Skipped trades warning ────────────────────────────────
    if not results["skipped"].empty:
        st.warning("Some purchases were skipped:")
        st.dataframe(results["skipped"], use_container_width=True)

else:
    st.info("👆 Upload your CSV above to get started. Expand the guide to see the required format.")
