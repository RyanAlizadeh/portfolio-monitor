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
DEFAULT_BENCHMARKS = {
    "SP500 (VFV.TO)":  "VFV.TO",
    "NASDAQ (XQQ.TO)": "XQQ.TO",
    "XEQT (XEQT.TO)":  "XEQT.TO",
}
DEFAULT_RISK_FREE_RATE = 0.03
MAX_LOOKAHEAD_DAYS     = 5
_SKIP_TICKERS          = {"CASH", "USD", "CAD", "GBP", "EUR"}

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

    # For tickers that came back empty, retry with .TO suffix — but only
    # for tickers that look like TSX candidates (no existing suffix, not
    # obviously US-format). We skip the retry for tickers that already
    # returned partial data or look like plain US symbols where .TO would
    # never exist (we still try, but log clearly if both attempts fail).
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    retry   = [t for t in missing if not t.endswith(".TO") and "." not in t]
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

    # Second retry: any ticker still missing, try individually as both
    # bare and .TO — we can't reliably tell from the symbol alone whether
    # a ticker is TSX or US-listed, and rate limits may have dropped either.
    still_missing = [t for t in tickers
                     if t not in prices.columns or prices[t].dropna().empty]
    for t in still_missing:
        candidates = [t] if t.endswith(".TO") else [t, t + ".TO"]
        for candidate in candidates:
            try:
                single = yf.download(
                    tickers=candidate,
                    start=start_date.strftime("%Y-%m-%d"),
                    auto_adjust=True, actions=True,
                    progress=False, threads=False,
                )
                if not single.empty:
                    col = "Close" if "Close" in single.columns else single.columns[0]
                    s = single[col].dropna().ffill()
                    if not s.empty:
                        # Always store under the original bare ticker name
                        prices[t] = s
                        break  # found data, no need to try the other candidate
            except Exception:
                pass

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


# ── Per-security attribution ─────────────────────────────────
def build_security_attribution(txns: pd.DataFrame,
                                units_matrix: pd.DataFrame,
                                prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each security, compute:
      - Total bought / sold
      - Current market value
      - Total return (XIRR) on that position alone
      - Absolute gain/loss in dollars

    Uses the same XIRR logic as the portfolio — buys are outflows,
    sells are inflows, terminal value is today's market value.
    """
    rows = []
    today = pd.Timestamp("today").normalize()

    for ticker in sorted(txns["ticker"].unique()):
        ticker_txns = txns[txns["ticker"] == ticker].copy()

        # Current market value of this position
        if ticker not in units_matrix.columns or ticker not in prices.columns:
            continue
        current_units = float(units_matrix[ticker].iloc[-1])
        price_series  = prices[ticker].ffill().dropna()
        if price_series.empty:
            continue
        current_price = float(price_series.iloc[-1])
        current_value = current_units * current_price

        total_bought = ticker_txns.loc[ticker_txns["type"] == "BUY",  "amount"].sum()
        total_sold   = ticker_txns.loc[ticker_txns["type"] == "SELL", "amount"].sum()
        ni           = total_bought - total_sold

        # XIRR cashflows for this security only
        cfs = []
        for _, row in ticker_txns.iterrows():
            date   = pd.Timestamp(row["date"]).normalize()
            amount = float(row["amount"])
            signed = amount if row["type"] == "SELL" else -amount
            cfs.append((date, signed))
        cfs.append((today, current_value))
        cfs.sort(key=lambda x: x[0])

        ann_ret   = xirr(cfs)
        total_ret = (current_value - ni) / ni if ni > 0 else np.nan
        abs_gain  = current_value - ni

        rows.append({
            "Ticker":            ticker,
            "Total Bought":      total_bought,
            "Total Sold":        total_sold,
            "Net Invested":      ni,
            "Current Value":     current_value,
            "Gain / Loss ($)":   abs_gain,
            "Total Return":      total_ret,
            "Annualized Return": ann_ret,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Weighted contribution ─────────────────────────────────
    # Contribution = (position net invested / total net invested) * annualized return
    # This answers: "how many percentage points of total return came from this position?"
    # It properly accounts for both position size AND return rate together,
    # so a large mediocre position outranks a tiny outstanding one where appropriate.
    total_ni = df["Net Invested"].sum()
    if total_ni > 0:
        df["Portfolio Weight"]  = df["Net Invested"] / total_ni
        df["Return Contribution"] = df["Portfolio Weight"] * df["Annualized Return"]
    else:
        df["Portfolio Weight"]    = np.nan
        df["Return Contribution"] = np.nan

    df = df.sort_values("Return Contribution", ascending=False).reset_index(drop=True)
    return df


# ── Drawdown ─────────────────────────────────────────────────
def compute_drawdown(series: pd.Series) -> pd.Series:
    s = series.dropna()
    return (s - s.cummax()) / s.cummax()


# ── Rolling period returns ────────────────────────────────────
def compute_rolling_returns(
    portfolio_vals: pd.Series,
    bm_series: dict,
) -> pd.DataFrame:
    """
    Simple time-weighted period returns for standard lookback windows.
    Uses the portfolio/benchmark value series directly (not XIRR).
    """
    today = portfolio_vals.dropna().index[-1]

    periods = {
        "YTD": pd.Timestamp(today.year, 1, 1),
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
        "3Y":  today - pd.DateOffset(years=3),
    }

    def period_return(series: pd.Series, start: pd.Timestamp) -> float:
        s = series.dropna()
        candidates = s.index[s.index >= start]
        if len(candidates) == 0 or len(s) == 0:
            return np.nan
        start_val = float(s.loc[candidates[0]])
        end_val   = float(s.iloc[-1])
        if start_val <= 0:
            return np.nan
        return end_val / start_val - 1

    rows = {}
    rows["PORTFOLIO"] = {
        label: period_return(portfolio_vals, start)
        for label, start in periods.items()
    }
    for name, series in bm_series.items():
        rows[name] = {
            label: period_return(series, start)
            for label, start in periods.items()
        }

    df = pd.DataFrame(rows).T
    df.index.name = ""
    return df


# ── Main computation ─────────────────────────────────────────
def compute_all(txns: pd.DataFrame, benchmarks: dict,
                risk_free_rate: float) -> dict:
    all_tickers = list(txns["ticker"].unique()) + list(benchmarks.values())
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
        risk_free_rate,
    )

    bm_series = {
        name: build_benchmark_series(ticker, txns, prices)
        for name, ticker in benchmarks.items()
        if ticker in prices.columns
    }

    bm_metrics = {}
    for name, series in bm_series.items():
        bm_val = float(series.dropna().iloc[-1])
        ar     = annualized_return_metric(txns, bm_val)
        av     = annualized_vol(series, trade_dates)
        bm_metrics[name] = {
            "ticker":                benchmarks[name],
            "total_return":          total_return_metric(txns, bm_val),
            "annualized_return":     ar,
            "annualized_volatility": av,
            "sharpe":                sharpe_ratio(ar, av, risk_free_rate),
        }

    holdings    = build_holdings_table(units_matrix, prices)
    attribution = build_security_attribution(txns, units_matrix, prices)
    rolling     = compute_rolling_returns(portfolio_vals, bm_series)

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
        "attribution":       attribution,
        "rolling":           rolling,
    }


# ════════════════════════════════════════════════════════════
# SIDEBAR — settings (rendered before main body)
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Settings")

    # ── Benchmark configuration ───────────────────────────────
    st.subheader("Benchmarks")
    selected_default_names = st.multiselect(
        "Built-in benchmarks",
        options=list(DEFAULT_BENCHMARKS.keys()),
        default=list(DEFAULT_BENCHMARKS.keys()),
        help="Select which default benchmarks to include in the analysis.",
    )
    custom_ticker_input = st.text_input(
        "Add custom benchmark (Yahoo Finance ticker)",
        placeholder="e.g. SPY, QQQ, IVV",
        help="Enter any Yahoo Finance ticker to add it as a benchmark.",
    ).strip().upper()

    # Build the active benchmarks dict
    ACTIVE_BENCHMARKS = {k: DEFAULT_BENCHMARKS[k] for k in selected_default_names}
    if custom_ticker_input:
        # Use the ticker itself as the display name
        ACTIVE_BENCHMARKS[custom_ticker_input] = custom_ticker_input

    st.divider()

    # ── Risk-free rate ────────────────────────────────────────
    st.subheader("Risk-free Rate")
    RISK_FREE_RATE = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=DEFAULT_RISK_FREE_RATE * 100,
        step=0.25,
        format="%.2f",
        help="Used to compute Sharpe ratio. Default is 3% (approximate T-bill rate).",
    ) / 100.0

    st.divider()
    st.caption("Date range filter appears here after you upload a CSV.")
    sidebar_date_placeholder = st.empty()


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
        results = compute_all(txns, benchmarks=ACTIVE_BENCHMARKS,
                              risk_free_rate=RISK_FREE_RATE)
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        st.stop()

    pm       = results["port_metrics"]
    bm       = results["bm_metrics"]
    no_data  = results["no_data_tickers"]
    excluded = results["excluded_rows"]

    # ── Date range filter (rendered into the sidebar placeholder) ──
    port_vals_full = results["portfolio_vals"].dropna()
    data_start = port_vals_full.index.min().date()
    data_end   = port_vals_full.index.max().date()

    with sidebar_date_placeholder.container():
        st.subheader("Chart Date Range")
        date_range = st.date_input(
            "Select range",
            value=(data_start, data_end),
            min_value=data_start,
            max_value=data_end,
            help="Filters the growth and drawdown charts. Summary metrics remain inception-to-date.",
        )

    # Unpack date range (user may have selected only start while picking)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filter_start = pd.Timestamp(date_range[0])
        filter_end   = pd.Timestamp(date_range[1])
    else:
        filter_start = pd.Timestamp(data_start)
        filter_end   = pd.Timestamp(data_end)

    def apply_date_filter(series: pd.Series) -> pd.Series:
        return series.loc[
            (series.index >= filter_start) & (series.index <= filter_end)
        ]

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
- **Transient download failure** — Yahoo Finance occasionally rate-limits requests. If the ticker is a real security (e.g. `VXUS`, `AAPL`), try re-uploading your CSV — it usually resolves on a second attempt.
- **Wrong symbol** — Canadian ETFs need the `.TO` suffix (e.g. `VFV.TO` not `VFV`). US-listed securities use their plain symbol (e.g. `VXUS`, `TSLA`).
- **Options / warrants** — symbols like `GME.WS` have no continuous price history on Yahoo Finance.
- **GICs / structured notes** — not exchange-traded, no Yahoo Finance ticker.
- **Internal account codes** — brokerage-specific labels like `WSE200P` that aren't real tickers.
- **Delisted or renamed** — the security may have changed its symbol.

**What to do:** Search for each ticker at [finance.yahoo.com](https://finance.yahoo.com) to confirm the exact symbol. If the ticker looks correct (e.g. `VXUS`), simply re-upload your CSV — transient failures usually clear on retry.
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

    # ── Growth chart ──────────────────────────────────────────
    st.subheader("Growth of $100 — Portfolio vs Benchmarks")
    st.caption(
        "Normalized against net invested capital (buys minus sells). "
        "Benchmarks mirror your exact buy and sell timing. "
        "Use the date range filter in the sidebar to zoom in."
    )

    # Colour palette — default benchmarks get fixed colours, custom get auto
    _BM_COLORS = {
        "SP500 (VFV.TO)":  "#3b82f6",
        "NASDAQ (XQQ.TO)": "#f97316",
        "XEQT (XEQT.TO)":  "#22c55e",
    }
    _AUTO_COLORS = ["#a855f7", "#06b6d4", "#f43f5e", "#eab308", "#14b8a6"]

    def bm_color(name: str, idx: int) -> str:
        return _BM_COLORS.get(name, _AUTO_COLORS[idx % len(_AUTO_COLORS)])

    fig       = go.Figure()
    port_vals = apply_date_filter(results["portfolio_vals"])
    norm_base = ni

    port_norm = port_vals / norm_base * 100
    fig.add_trace(go.Scatter(
        x=port_norm.index, y=port_norm.values,
        name="PORTFOLIO",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.07)",
    ))

    for idx, (name, series) in enumerate(results["bm_series"].items()):
        filtered = apply_date_filter(series)
        norm     = filtered / norm_base * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            name=name,
            line=dict(color=bm_color(name, idx), width=1.5),
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Break-even", annotation_position="bottom right")

    _chart_layout = dict(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#2d2d2d", size=13),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
        ),
        xaxis=dict(
            showgrid=False, linecolor="#2d2d2d",
            tickfont=dict(color="#2d2d2d", size=12),
            title_font=dict(color="#2d2d2d"),
        ),
        yaxis=dict(
            title="Growth of 100", gridcolor="#e2e8f0", linecolor="#2d2d2d",
            tickfont=dict(color="#2d2d2d", size=12),
            title_font=dict(color="#2d2d2d"),
        ),
        font=dict(color="#2d2d2d"),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_layout(**_chart_layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Drawdown chart ────────────────────────────────────────
    st.subheader("Drawdown")
    st.caption(
        "Peak-to-trough decline from each series' prior high. "
        "Shows how far each portfolio / benchmark fell at its worst point in the selected window."
    )

    dd_fig = go.Figure()
    port_dd = compute_drawdown(apply_date_filter(results["portfolio_vals"])) * 100
    dd_fig.add_trace(go.Scatter(
        x=port_dd.index, y=port_dd.values,
        name="PORTFOLIO",
        line=dict(color="#0f172a", width=2.5),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.07)",
    ))
    for idx, (name, series) in enumerate(results["bm_series"].items()):
        bm_dd = compute_drawdown(apply_date_filter(series)) * 100
        dd_fig.add_trace(go.Scatter(
            x=bm_dd.index, y=bm_dd.values,
            name=name,
            line=dict(color=bm_color(name, idx), width=1.5),
        ))

    dd_fig.update_layout(
        **{**_chart_layout,
           "yaxis": {
               **_chart_layout["yaxis"],
               "title": "Drawdown (%)",
               "ticksuffix": "%",
           }
        }
    )
    st.plotly_chart(dd_fig, use_container_width=True)

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

    # ── Rolling period returns ────────────────────────────────
    st.subheader("Rolling Period Returns")
    st.caption(
        "Simple time-weighted returns for standard lookback windows (not XIRR). "
        "Periods shorter than the portfolio's history show N/A."
    )
    rolling_df = results["rolling"].copy()
    # Format all cells as percentages
    rolling_fmt = rolling_df.apply(lambda col: col.map(lambda x: pct(x) if pd.notna(x) else "—"))
    st.dataframe(rolling_fmt, use_container_width=True)

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

    # ── Security attribution ──────────────────────────────────
    st.subheader("What's Driving Your Returns?")
    st.caption(
        "Ranked by **return contribution** — portfolio weight × annualized return. "
        "Shows which positions are actually driving your results, "
        "accounting for both size and performance."
    )

    attr = results["attribution"].copy()

    if not attr.empty:
        # Filter to securities with valid contribution scores
        has_ret = attr["Return Contribution"].notna()
        # Sort by contribution (already done in build_security_attribution,
        # but re-sort here to be explicit)
        ranked  = attr[has_ret].sort_values(
            "Return Contribution", ascending=False
        ).copy()

        n_show   = min(3, len(ranked))
        leaders  = ranked.head(n_show)
        laggards = ranked.tail(n_show).iloc[::-1]  # worst first

        def fmt_attr_row(row):
            gain     = row["Gain / Loss ($)"]
            gain_str = f"+{money(gain)}" if gain >= 0 else money(gain)
            contrib  = row["Return Contribution"]
            return {
                "Ticker":               row["Ticker"],
                "Contribution (pp)":    f"{contrib*100:+.2f}pp" if pd.notna(contrib) else "—",
                "Annualized Return":    pct(row["Annualized Return"]),
                "Portfolio Weight":     pct(row["Portfolio Weight"]),
                "Gain / Loss":          gain_str,
                "Net Invested":         money(row["Net Invested"]),
                "Current Value":        money(row["Current Value"]),
            }

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("#### 🟢 Top Contributors")
            st.caption("Positions adding the most to your portfolio return")
            leader_rows = [fmt_attr_row(r) for _, r in leaders.iterrows()]
            st.dataframe(
                pd.DataFrame(leader_rows).set_index("Ticker"),
                use_container_width=True,
            )

        with col_r:
            st.markdown("#### 🔴 Biggest Laggards")
            st.caption("Positions dragging down your portfolio return")
            laggard_rows = [fmt_attr_row(r) for _, r in laggards.iterrows()]
            st.dataframe(
                pd.DataFrame(laggard_rows).set_index("Ticker"),
                use_container_width=True,
            )

        # Contribution waterfall bar chart
        chart_data = ranked.copy()
        chart_data = chart_data[chart_data["Return Contribution"].notna()]
        chart_data["color"] = chart_data["Return Contribution"].apply(
            lambda x: "#22c55e" if x >= 0 else "#ef4444"
        )
        chart_data["label"] = chart_data["Return Contribution"].apply(
            lambda x: f"{x*100:+.2f}pp"
        )

        bar_fig = go.Figure(go.Bar(
            x=chart_data["Ticker"],
            y=chart_data["Return Contribution"] * 100,
            marker_color=chart_data["color"],
            text=chart_data["label"],
            textposition="outside",
            textfont=dict(size=11, color="#2d2d2d"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Contribution: %{y:.2f}pp<br>"
                "<extra></extra>"
            ),
        ))
        bar_fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
        bar_fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=80),
            title=dict(
                text="Return Contribution by Security (percentage points)",
                font=dict(size=13, color="#2d2d2d"),
            ),
            xaxis=dict(
                showgrid=False,
                linecolor="#2d2d2d",
                tickangle=0,
                tickfont=dict(size=12, color="#2d2d2d"),
                title_font=dict(color="#2d2d2d"),
            ),
            yaxis=dict(
                title="Contribution (pp)",
                gridcolor="#e2e8f0",
                linecolor="#2d2d2d",
                ticksuffix="pp",
                tickfont=dict(color="#2d2d2d", size=12),
                title_font=dict(color="#2d2d2d"),
            ),
            font=dict(color="#2d2d2d"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption(
            "**Contribution = Portfolio Weight × Annualized Return.** "
            "Shows how many percentage points each position adds to (or subtracts from) "
            "your total annualized return. Accounts for both position size and return rate — "
            "a large mediocre position can outrank a tiny outstanding one."
        )

        # Full ranked table in expander
        with st.expander("See all securities ranked by contribution", expanded=False):
            all_rows = [fmt_attr_row(r) for _, r in ranked.iterrows()]
            st.dataframe(
                pd.DataFrame(all_rows).set_index("Ticker"),
                use_container_width=True,
            )

        # Unranked tickers (too short a history for XIRR to converge)
        unranked = attr[~has_ret]
        if not unranked.empty:
            st.caption(
                f"ℹ️ {len(unranked)} ticker(s) excluded from ranking — "
                "insufficient price history for return calculation: "
                + ", ".join(unranked["Ticker"].tolist())
            )
    else:
        st.info("No attribution data available.")

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
