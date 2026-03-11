import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import pandas as pd
import numpy as np

# =============================================================================
# VALUATION CONFIG — edit these to tune all models globally
# =============================================================================
DISCOUNT_RATE    = 0.10   # 10% hurdle rate (S&P 500 long-run average)
GROWTH_RATE      = 0.05   # 5% perpetual growth assumption
TERMINAL_GROWTH  = 0.03   # 3% terminal growth (long-run GDP proxy)
DCF_YEARS        = 20     # projection horizon for DCF methods
HISTORY_YEARS    = 5      # years of historical multiples to average

# =============================================================================
# SOURCES
# =============================================================================
NYSE_URL   = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL  = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"
CORE_PRIORITY = [
    "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "BRK-B", "LLY", "AVGO"
]

# =============================================================================
# TICKER LOADER  (unchanged)
# =============================================================================
def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = requests.get(url)
            data = res.json()
            for item in data:
                symbol = (
                    item.get('Symbol') or
                    item.get('ACT Symbol') or
                    item.get('ticker')
                )
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers:
                        tickers.append(clean)
        except Exception as e:
            print(f"Could not read {url}: {e}")
    return tickers

# =============================================================================
# HELPER — safe division
# =============================================================================
def _safe(val, fallback=None):
    """Return fallback if val is None, NaN, 0 or non-finite."""
    if val is None:
        return fallback
    try:
        if not np.isfinite(float(val)):
            return fallback
    except (TypeError, ValueError):
        return fallback
    return val

# =============================================================================
# HELPER — discounted cash flow summation
# =============================================================================
def _dcf_sum(base_val_per_share, growth, discount, years):
    """
    Sums the present value of `years` annual cash flows.
    Each year's cash flow = base * (1+g)^t discounted at (1+r)^t.
    Returns None if base_val_per_share is None or <= 0.
    """
    if base_val_per_share is None or base_val_per_share <= 0:
        return None
    total = sum(
        base_val_per_share * (1 + growth) ** t / (1 + discount) ** t
        for t in range(1, years + 1)
    )
    return round(total, 2)

# =============================================================================
# HISTORICAL MULTIPLES  (Option B — 3–5yr average)
# =============================================================================
def get_historical_mean_multiples(stock, shares):
    """
    Pulls up to HISTORY_YEARS of annual financial statements and price history
    to compute mean P/E, P/S, and P/B ratios.

    Strategy:
      - Iterate over each available fiscal year end date in stock.financials
      - For each year: look up the year-end closing price
      - Derive per-share figures from annual totals ÷ shares outstanding
      - Compute the ratio for that year, collect valid observations
      - Return the mean across all valid years

    Returns a dict: {mean_pe, mean_ps, mean_pb} — any can be None.
    """
    result = {"mean_pe": None, "mean_ps": None, "mean_pb": None}

    try:
        # Annual income statement & balance sheet (columns = fiscal year-end dates)
        financials  = stock.financials          # income statement  (annual)
        balance     = stock.balance_sheet       # balance sheet     (annual)
        hist        = stock.history(period="10y")

        if financials is None or financials.empty:
            return result
        if hist is None or hist.empty:
            return result

        pe_list, ps_list, pb_list = [], [], []

        # financials columns are Timestamps (most-recent first)
        cols = list(financials.columns)[:HISTORY_YEARS]

        for col in cols:
            year = col.year

            # ---- year-end price ----
            year_hist = hist[hist.index.year == year]
            if year_hist.empty:
                continue
            price = float(year_hist['Close'].iloc[-1])
            if price <= 0:
                continue

            # ---- EPS  (Net Income / shares) ----
            try:
                net_inc_row = (
                    financials.loc['Net Income'] if 'Net Income' in financials.index
                    else financials.loc['NetIncome'] if 'NetIncome' in financials.index
                    else None
                )
                if net_inc_row is not None and shares > 0:
                    eps = float(net_inc_row[col]) / shares
                    if eps > 0:
                        pe_list.append(price / eps)
            except Exception:
                pass

            # ---- Revenue per share ----
            try:
                rev_row = (
                    financials.loc['Total Revenue'] if 'Total Revenue' in financials.index
                    else financials.loc['TotalRevenue'] if 'TotalRevenue' in financials.index
                    else None
                )
                if rev_row is not None and shares > 0:
                    rev_ps = float(rev_row[col]) / shares
                    if rev_ps > 0:
                        ps_list.append(price / rev_ps)
            except Exception:
                pass

            # ---- Book value per share ----
            try:
                if balance is not None and not balance.empty and col in balance.columns:
                    eq_row = (
                        balance.loc['Stockholders Equity'] if 'Stockholders Equity' in balance.index
                        else balance.loc['StockholdersEquity'] if 'StockholdersEquity' in balance.index
                        else balance.loc['Total Stockholder Equity'] if 'Total Stockholder Equity' in balance.index
                        else None
                    )
                    if eq_row is not None and shares > 0:
                        bvps = float(eq_row[col]) / shares
                        if bvps > 0:
                            pb_list.append(price / bvps)
            except Exception:
                pass

        # Mean of valid observations
        if pe_list:
            result["mean_pe"] = round(float(np.mean(pe_list)), 4)
        if ps_list:
            result["mean_ps"] = round(float(np.mean(ps_list)), 4)
        if pb_list:
            result["mean_pb"] = round(float(np.mean(pb_list)), 4)

    except Exception:
        pass

    return result

# =============================================================================
# VALUATION METHOD 1 — 20-Year DCF on Operating Cash Flow
# =============================================================================
def calc_dcf_operating_cf(info, shares):
    """
    Intrinsic value = sum of 20 years of discounted Operating Cash Flow per share.
    Growth: 5% | Discount: 10%
    """
    ocf = _safe(info.get('operatingCashflow'))
    if ocf is None or shares <= 0:
        return None
    return _dcf_sum(ocf / shares, GROWTH_RATE, DISCOUNT_RATE, DCF_YEARS)

# =============================================================================
# VALUATION METHOD 2 — 20-Year Discounted Free Cash Flow
# =============================================================================
def calc_dcf_fcf(info, shares):
    """
    Intrinsic value = sum of 20 years of discounted Free Cash Flow per share.
    Growth: 5% | Discount: 10%
    """
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0:
        return None
    return _dcf_sum(fcf / shares, GROWTH_RATE, DISCOUNT_RATE, DCF_YEARS)

# =============================================================================
# VALUATION METHOD 3 — 20-Year Discounted Net Income
# =============================================================================
def calc_dcf_net_income(info, shares):
    """
    Intrinsic value = sum of 20 years of discounted Net Income per share.
    Growth: 5% | Discount: 10%
    """
    ni = _safe(info.get('netIncomeToCommon'))
    if ni is None or shares <= 0:
        return None
    return _dcf_sum(ni / shares, GROWTH_RATE, DISCOUNT_RATE, DCF_YEARS)

# =============================================================================
# VALUATION METHOD 4 — DCF with Terminal Value (Gordon Growth Model)
# =============================================================================
def calc_dcf_terminal_value(info, shares):
    """
    Two-stage model:
      Stage 1 : PV of 20 years of FCF per share at 5% growth, 10% discount
      Stage 2 : Terminal value at year 20 using Gordon Growth Model
                TV = FCF_20 × (1 + terminal_growth) / (discount - terminal_growth)
                Discounted back to present at (1 + discount)^20

    Terminal growth: 3% (long-run GDP proxy)
    """
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0:
        return None

    fcf_ps = fcf / shares
    if fcf_ps <= 0:
        return None

    pv_cashflows = _dcf_sum(fcf_ps, GROWTH_RATE, DISCOUNT_RATE, DCF_YEARS)

    # FCF at end of projection period
    fcf_terminal = fcf_ps * (1 + GROWTH_RATE) ** DCF_YEARS

    # Gordon Growth terminal value discounted to present
    if DISCOUNT_RATE <= TERMINAL_GROWTH:
        return None  # math undefined if discount <= terminal growth
    tv = (fcf_terminal * (1 + TERMINAL_GROWTH)) / (DISCOUNT_RATE - TERMINAL_GROWTH)
    pv_tv = tv / (1 + DISCOUNT_RATE) ** DCF_YEARS

    return round(pv_cashflows + pv_tv, 2)

# =============================================================================
# VALUATION METHOD 5 — Mean P/S Valuation
# =============================================================================
def calc_mean_ps(info, shares, mean_ps):
    """
    Intrinsic value = historical mean P/S ratio × current revenue per share.
    Falls back to trailing P/S from info if historical mean unavailable.
    """
    rev = _safe(info.get('totalRevenue'))
    if rev is None or shares <= 0:
        return None

    rev_ps = rev / shares
    if rev_ps <= 0:
        return None

    # Prefer historical mean; fall back to current trailing ratio
    ratio = mean_ps if mean_ps is not None else _safe(info.get('priceToSalesTrailing12Months'))
    if ratio is None or ratio <= 0:
        return None

    return round(ratio * rev_ps, 2)

# =============================================================================
# VALUATION METHOD 6 — Mean P/E Valuation (ex-NRI, approximated)
# =============================================================================
def calc_mean_pe(info, shares, mean_pe):
    """
    Intrinsic value = historical mean P/E × trailing EPS.
    NRI exclusion: yfinance does not expose adjusted EPS directly.
    We use trailingEps as the best available proxy and flag it as approximate.
    Falls back to current trailing P/E if historical mean is unavailable.
    """
    eps = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0:
        return None

    ratio = mean_pe if mean_pe is not None else _safe(info.get('trailingPE'))
    if ratio is None or ratio <= 0:
        return None

    return round(ratio * eps, 2)

# =============================================================================
# VALUATION METHOD 7 — Mean P/B Valuation
# =============================================================================
def calc_mean_pb(info, mean_pb):
    """
    Intrinsic value = historical mean P/B ratio × current book value per share.
    Falls back to current P/B from info if historical mean is unavailable.
    """
    bvps = _safe(info.get('bookValue'))
    if bvps is None or bvps <= 0:
        return None

    ratio = mean_pb if mean_pb is not None else _safe(info.get('priceToBook'))
    if ratio is None or ratio <= 0:
        return None

    return round(ratio * bvps, 2)

# =============================================================================
# VALUATION METHOD 8 — Price-to-Sales-Growth (PSG)
# =============================================================================
def calc_psg(info, shares):
    """
    PSG fair value = (Revenue per share) / (P/S ÷ Revenue Growth Rate)
    Equivalent to: Rev/share × Revenue Growth Rate / P/S ratio

    Logic: analogous to PEG — a stock is fairly valued when P/S equals
    its revenue growth rate (expressed as a ratio). If P/S < growth rate,
    the stock is cheap on a growth-adjusted basis.

    Fair value price = rev_ps × (revenue_growth / ps_ratio)
    i.e., the price at which PSG = 1.0 (fairly valued)
    """
    rev = _safe(info.get('totalRevenue'))
    rev_growth = _safe(info.get('revenueGrowth'))
    ps = _safe(info.get('priceToSalesTrailing12Months'))

    if None in (rev, rev_growth, ps) or shares <= 0:
        return None
    if ps <= 0 or rev_growth <= 0:
        return None

    rev_ps = rev / shares
    return round(rev_ps * (rev_growth / ps), 2)

# =============================================================================
# VALUATION METHOD 9 — PEG Valuation (ex-NRI, approximated)
# =============================================================================
def calc_peg(info):
    """
    PEG fair value = EPS × Earnings Growth Rate (as a percentage)

    At PEG = 1.0, price = EPS × growth_rate_pct.
    This gives the 'fairly valued' price under the Lynch PEG framework.

    Growth rate source: earningsGrowth from yfinance (forward-looking analyst estimate).
    NRI caveat: same as Method 6 — trailingEps used as proxy.
    """
    eps = _safe(info.get('trailingEps'))
    growth = _safe(info.get('earningsGrowth'))

    if None in (eps, growth) or eps <= 0 or growth <= 0:
        return None

    # Peter Lynch: fair value = EPS × (growth rate as a whole number)
    growth_pct = growth * 100
    return round(eps * growth_pct, 2)

# =============================================================================
# VALUATION METHOD 10 — EV/EBITDA Valuation
# =============================================================================
def calc_ev_ebitda(info, shares):
    """
    Intrinsic value per share derived from EV/EBITDA.

    Step 1: Compute sector-implied EV using the company's own EV/EBITDA multiple.
            (We use the company's current multiple as a proxy for sector median
            since yfinance does not expose peer group data.)
    Step 2: Subtract net debt to arrive at equity value.
    Step 3: Divide by shares outstanding.

    Formula:
      Equity Value = (EBITDA × EV/EBITDA multiple) − Total Debt + Total Cash
      Intrinsic Value per Share = Equity Value / Shares

    This effectively asks: "What is the company worth if it traded at its
    own historical EV/EBITDA multiple applied to current EBITDA?"
    """
    ebitda = _safe(info.get('ebitda'))
    ev     = _safe(info.get('enterpriseValue'))
    debt   = _safe(info.get('totalDebt'), 0)
    cash   = _safe(info.get('totalCash'), 0)

    if None in (ebitda, ev) or ebitda <= 0 or ev <= 0 or shares <= 0:
        return None

    ev_ebitda_multiple = ev / ebitda
    if ev_ebitda_multiple <= 0:
        return None

    # Implied equity value
    implied_ev      = ebitda * ev_ebitda_multiple
    equity_value    = implied_ev - debt + cash
    if equity_value <= 0:
        return None

    return round(equity_value / shares, 2)

# =============================================================================
# INTRINSIC VALUE AGGREGATOR — mean of all valid methods
# =============================================================================
def calc_intrinsic_value(valuations: dict) -> dict:
    """
    Computes the mean intrinsic value across all 10 methods,
    excluding None values, negatives, and implausible outliers (> $1,000,000).

    Returns:
      {
        "intrinsic_value": float | None,
        "methods_used": int,
        "confidence": "High" | "Medium" | "Low" | "Insufficient Data"
      }
    """
    keys = [
        "1_DCF_Operating_CF",
        "2_DCF_FCF",
        "3_DCF_Net_Income",
        "4_DCF_Terminal_Value",
        "5_Mean_PS",
        "6_Mean_PE",
        "7_Mean_PB",
        "8_PSG",
        "9_PEG",
        "10_EV_EBITDA",
    ]

    valid = [
        v for k in keys
        if (v := valuations.get(k)) is not None
        and isinstance(v, (int, float))
        and 0 < v < 1_000_000
    ]

    count = len(valid)

    if count == 0:
        return {"intrinsic_value": None, "methods_used": 0, "confidence": "Insufficient Data"}

    intrinsic = round(float(np.mean(valid)), 2)

    # Confidence tier based on how many methods returned a valid result
    if count >= 7:
        confidence = "High"
    elif count >= 4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "intrinsic_value": intrinsic,
        "methods_used": count,
        "confidence": confidence
    }

# =============================================================================
# 10-YEAR PRICE HISTORY  (unchanged)
# =============================================================================
def get_10yr_history(stock):
    """Fetches the last 10 years of data, saving only the year-end closing price."""
    try:
        hist = stock.history(period="10y")
        if hist.empty:
            return {}
        yearly_prices = {}
        years = sorted(set(d.year for d in hist.index))
        for y in years:
            yearly_prices[str(y)] = round(
                hist[hist.index.year == y]['Close'].iloc[-1], 2
            )
        return yearly_prices
    except Exception:
        return {}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def analyze_ticker(ticker, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(0.5)  # Rate-limit protection (unchanged)

            stock = yf.Ticker(ticker)
            info  = stock.info

            if not info or 'sharesOutstanding' not in info:
                return None

            shares = info.get('sharesOutstanding', 1)
            if shares <= 0:
                return None

            # ------------------------------------------------------------------
            # Pull historical mean multiples (3–5yr average via annual filings)
            # ------------------------------------------------------------------
            hist_multiples = get_historical_mean_multiples(stock, shares)
            mean_pe = hist_multiples.get("mean_pe")
            mean_ps = hist_multiples.get("mean_ps")
            mean_pb = hist_multiples.get("mean_pb")

            # ------------------------------------------------------------------
            # Run all 10 valuation methods
            # ------------------------------------------------------------------
            valuations = {
                "1_DCF_Operating_CF":  calc_dcf_operating_cf(info, shares),
                "2_DCF_FCF":           calc_dcf_fcf(info, shares),
                "3_DCF_Net_Income":    calc_dcf_net_income(info, shares),
                "4_DCF_Terminal_Value":calc_dcf_terminal_value(info, shares),
                "5_Mean_PS":           calc_mean_ps(info, shares, mean_ps),
                "6_Mean_PE":           calc_mean_pe(info, shares, mean_pe),
                "7_Mean_PB":           calc_mean_pb(info, mean_pb),
                "8_PSG":               calc_psg(info, shares),
                "9_PEG":               calc_peg(info),
                "10_EV_EBITDA":        calc_ev_ebitda(info, shares),
            }

            # ------------------------------------------------------------------
            # Aggregate → intrinsic value
            # ------------------------------------------------------------------
            aggregate   = calc_intrinsic_value(valuations)
            valuations.update(aggregate)

            # Metadata: which multiples were historical vs current fallback
            valuations["_meta"] = {
                "pe_source":  "historical_mean" if mean_pe else "current_trailing",
                "ps_source":  "historical_mean" if mean_ps else "current_trailing",
                "pb_source":  "historical_mean" if mean_pb else "current_trailing",
                "pe_approx":  True,   # NRI not cleanly separated by yfinance
                "peg_approx": True,   # same reason
            }

            # ------------------------------------------------------------------
            # History & forecast  (unchanged logic, uses new intrinsic value)
            # ------------------------------------------------------------------
            price_history   = get_10yr_history(stock)
            intrinsic       = aggregate.get("intrinsic_value") or 0
            current_year    = datetime.now().year

            forecast = {
                str(current_year + i): round(intrinsic * (1 + GROWTH_RATE) ** i, 2)
                for i in range(1, 6)
            } if intrinsic > 0 else {}

            # ------------------------------------------------------------------
            # Final output object
            # ------------------------------------------------------------------
            result = {
                "valuations":      valuations,
                "10_Year_History": price_history,
                "5_Year_Forecast": forecast,
                "Last_Updated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            return ticker, result

        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                time.sleep(5)
            continue

    return None

# =============================================================================
# A–Z PARTITIONED JSON SAVER  (unchanged)
# =============================================================================
def save_partitioned_data(master_results):
    partitions = {}
    for ticker, data in master_results.items():
        letter = ticker[0].upper()
        if not letter.isalpha():
            letter = "0-9"
        if letter not in partitions:
            partitions[letter] = {}
        partitions[letter][ticker] = data

    os.makedirs('data', exist_ok=True)
    for letter, content in partitions.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)
    print("✅ Saved A–Z stock files.")

# =============================================================================
# ENTRY POINT  (unchanged)
# =============================================================================
def main():
    tickers = get_all_tickers()
    master_results = {}
    print(f"Starting analysis of {len(tickers)} tickers with 5 safe threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
                if len(master_results) % 50 == 0:
                    print(f"  Processed {len(master_results)} stocks...")

    save_partitioned_data(master_results)

if __name__ == "__main__":
    main()
