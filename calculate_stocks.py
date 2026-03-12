import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import pandas as pd
import numpy as np
try:
    from finvizfinance.quote import finvizfinance as fvz
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    print("Warning: finvizfinance not installed. Falling back to yfinance growth rates.")

# =============================================================================
# VALUATION CONFIG — tiered by fundamental quality (two-stage growth model)
#
# Stage 1 (Years 1–10)  : near-term growth — higher visibility, more optimistic
# Stage 2 (Years 11–20) : long-term growth — tapering as business matures
# Terminal growth        : perpetual rate beyond year 20 (Gordon Growth Model)
# Discount rate          : tier-adjusted hurdle rate
# =============================================================================
TIER_CONFIG = {
    "Strong": {
        "discount_rate":   0.075,  # 7.5% — blue chip quality warrants lower hurdle
        "growth_stage1":   0.12,   # 12% years 1–10
        "growth_stage2":   0.06,   # 6%  years 11–20
        "terminal_growth": 0.04,   # 4%  terminal
    },
    "Average": {
        "discount_rate":   0.10,   # 10% — standard S&P hurdle rate
        "growth_stage1":   0.07,   # 7%  years 1–10
        "growth_stage2":   0.04,   # 4%  years 11–20
        "terminal_growth": 0.03,   # 3%  terminal
    },
    "Weak": {
        "discount_rate":   0.12,   # 12% — elevated risk premium
        "growth_stage1":   0.03,   # 3%  years 1–10
        "growth_stage2":   0.02,   # 2%  years 11–20
        "terminal_growth": 0.02,   # 2%  terminal
    },
}

DCF_YEARS     = 20
HISTORY_YEARS = 5

# =============================================================================
# MULTIPLE CAPS — prevents distorted valuations from outlier ratios
#
# P/E above 50x on near-zero earnings (e.g. GME, turnarounds) produces
# astronomical fair values that are meaningless. Similarly P/S > 20x
# and P/B > 20x are capped to prevent single methods skewing the mean.
# These caps apply to both historical means and current trailing ratios.
# =============================================================================
MAX_PE = 50    # anything above this is speculative / earnings-distorted
MAX_PS = 20    # revenue multiples above this are growth-premium territory
MAX_PB = 20    # book multiples above this indicate intangible-heavy distortion

# =============================================================================
# GROWTH RATE CAPS — applied to ALL growth inputs including Finviz
#
# MAX_GROWTH_RATE: caps the stage 1 DCF growth rate per ticker.
#   Prevents distortion from yfinance reporting 500%+ growth for companies
#   recovering from near-zero earnings (e.g. GME, turnarounds).
#
# MAX_PEG_PSG_GROWTH: tighter cap specifically for PEG and PSG fair value
#   formulas. Lynch's PEG formula multiplies EPS by growth-as-whole-number,
#   so even 30% growth gives EPS x 30 which is already generous.
#   Above 30%, the result becomes an unreliable outlier.
#
# MIN_GROWTH_RATE: floor so DCF doesn't collapse to zero for shrinking firms.
# =============================================================================
MAX_GROWTH_RATE     = 0.30   # 30% cap on DCF stage 1 growth
MAX_PEG_PSG_GROWTH  = 0.30   # 30% cap specifically for PEG and PSG formulas
MIN_GROWTH_RATE     = 0.02   # 2% floor — even declining firms get minimal growth

# =============================================================================
# TERMINAL VALUE CAP — prevents DCF terminal value from dominating the mean.
# Terminal value is capped at this multiple of the DCF FCF result.
# e.g. if DCF FCF = $187, terminal value is capped at $187 x 3 = $561.
# This prevents perpetual growth assumptions from inflating weak companies.
# =============================================================================
TERMINAL_VALUE_CAP_MULTIPLIER = 3.0

# =============================================================================
# SECTOR MEDIAN EV/EBITDA MULTIPLES
# Used instead of the company's own multiple to prevent circular valuation.
# A struggling retailer is valued at the sector median, not its own premium.
# Source: long-run sector medians based on S&P 500 historical averages.
# =============================================================================
SECTOR_EV_EBITDA = {
    "Technology":                20,
    "Healthcare":                15,
    "Consumer Discretionary":    12,
    "Consumer Staples":          14,
    "Financials":                12,
    "Energy":                     8,
    "Industrials":               12,
    "Utilities":                 10,
    "Real Estate":               18,
    "Materials":                 10,
    "Communication Services":    14,
    "Default":                   12,
}

# =============================================================================
# TIER WEIGHTS — each method weighted by reliability for that tier.
# Strong companies: DCF and terminal value weighted more heavily.
# Weak companies:   asset/revenue multiples weighted more heavily.
# All weights per tier sum to 1.0.
# =============================================================================
TIER_WEIGHTS = {
    "Strong": {
        "1_DCF_Operating_CF":   0.15,
        "2_DCF_FCF":            0.15,
        "3_DCF_Net_Income":     0.08,
        "4_DCF_Terminal_Value": 0.15,
        "5_Mean_PS":            0.08,
        "6_Mean_PE":            0.10,
        "7_Mean_PB":            0.05,
        "8_PSG":                0.08,
        "9_PEG":                0.06,
        "10_EV_EBITDA":         0.10,
    },
    "Average": {
        "1_DCF_Operating_CF":   0.12,
        "2_DCF_FCF":            0.12,
        "3_DCF_Net_Income":     0.10,
        "4_DCF_Terminal_Value": 0.12,
        "5_Mean_PS":            0.10,
        "6_Mean_PE":            0.12,
        "7_Mean_PB":            0.08,
        "8_PSG":                0.08,
        "9_PEG":                0.08,
        "10_EV_EBITDA":         0.08,
    },
    "Weak": {
        "1_DCF_Operating_CF":   0.08,
        "2_DCF_FCF":            0.08,
        "3_DCF_Net_Income":     0.08,
        "4_DCF_Terminal_Value": 0.08,
        "5_Mean_PS":            0.12,
        "6_Mean_PE":            0.08,
        "7_Mean_PB":            0.15,
        "8_PSG":                0.08,
        "9_PEG":                0.08,
        "10_EV_EBITDA":         0.17,
    },
}

# =============================================================================
# SOURCES
# =============================================================================
NYSE_URL  = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"
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
# HELPER — safe value extraction
# =============================================================================
def _safe(val, fallback=None):
    if val is None:
        return fallback
    try:
        if not np.isfinite(float(val)):
            return fallback
    except (TypeError, ValueError):
        return fallback
    return val

# =============================================================================
# HELPER -- Two-stage DCF summation
# Stage 1 (Years 1-10)  : higher near-term growth rate
# Stage 2 (Years 11-20) : lower long-term taper rate
# Cash flow is compounded forward year-by-year using the appropriate
# stage rate, then discounted back at the hurdle rate.
# =============================================================================
def _dcf_sum(base_val_per_share, growth_s1, growth_s2, discount, years=20):
    if base_val_per_share is None or base_val_per_share <= 0:
        return None
    total = 0.0
    cf    = base_val_per_share
    for t in range(1, years + 1):
        rate  = growth_s1 if t <= 10 else growth_s2
        cf    = cf * (1 + rate)
        total += cf / (1 + discount) ** t
    return round(total, 2)

# =============================================================================
# HELPER — multi-year trend from annual financials
# =============================================================================
def _get_annual_series(financials, row_names):
    """
    Tries each name in row_names against the financials index.
    Returns a chronological list of floats (oldest first) or empty list.
    """
    for name in row_names:
        if name in financials.index:
            try:
                series = financials.loc[name].dropna()
                return [float(v) for v in reversed(series.values)]
            except Exception:
                continue
    return []

def _trend_score(series, max_pts=3):
    """
    Scores a chronological value series 0–max_pts:
      max_pts   : growing every year
      max_pts-1 : one declining year
      1         : flat / mixed
      0         : two or more declining years
    """
    if len(series) < 2:
        return 1
    declines = sum(1 for i in range(1, len(series)) if series[i] < series[i - 1])
    if declines == 0:
        return max_pts
    elif declines == 1:
        return max_pts - 1
    elif declines <= len(series) // 2:
        return 1
    return 0

# =============================================================================
# FINVIZ GROWTH FETCHER
#
# Fetches "EPS Next 5Y" (analyst consensus forward EPS growth) from Finviz.
# More accurate than yfinance trailing earningsGrowth for DCF inputs because:
#   - Forward analyst consensus rather than backward-looking actuals
#   - Smooths out one-off recovery spikes (e.g. GME 500%+ trailing growth)
#
# Fallback chain:
#   1. Finviz "EPS next 5Y"       (best  -- forward analyst consensus)
#   2. yfinance earningsGrowth    (ok    -- trailing, may be distorted)
#   3. MIN_GROWTH_RATE (2%)       (floor -- prevents DCF collapse)
#
# All values clamped between MIN_GROWTH_RATE and MAX_GROWTH_RATE.
# =============================================================================
def get_forward_growth(ticker, info):
    # Attempt 1: Finviz EPS Next 5Y
    if FINVIZ_AVAILABLE:
        try:
            fundamentals = fvz(ticker).ticker_fundament()
            raw_str = fundamentals.get("EPS next 5Y", "")
            if raw_str and raw_str not in ("-", "N/A", ""):
                raw     = float(raw_str.replace("%", "").strip()) / 100
                clamped = max(MIN_GROWTH_RATE, min(raw, MAX_GROWTH_RATE))
                return {"rate": clamped, "source": "finviz", "raw": round(raw, 4)}
        except Exception:
            pass

    # Attempt 2: yfinance trailing earningsGrowth
    yf_growth = _safe(info.get("earningsGrowth"))
    if yf_growth is not None and yf_growth > 0:
        clamped = max(MIN_GROWTH_RATE, min(yf_growth, MAX_GROWTH_RATE))
        return {"rate": clamped, "source": "yfinance", "raw": round(yf_growth, 4)}

    # Attempt 3: default floor
    return {"rate": MIN_GROWTH_RATE, "source": "default", "raw": MIN_GROWTH_RATE}


# =============================================================================
# QUALITY METRIC 1 — Profitability (0–10)
# =============================================================================
def score_profitability(info):
    score = 0

    # Net profit margin (3 pts)
    rev = _safe(info.get('totalRevenue'))
    ni  = _safe(info.get('netIncomeToCommon'))
    if rev and rev > 0 and ni is not None:
        margin = ni / rev
        if margin > 0.15:    score += 3
        elif margin > 0.05:  score += 2
        elif margin > 0:     score += 1

    # ROE (3 pts)
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.20:    score += 3
        elif roe > 0.10:  score += 2
        elif roe > 0:     score += 1

    # FCF margin (2 pts)
    fcf = _safe(info.get('freeCashflow'))
    if rev and rev > 0 and fcf is not None:
        fcf_margin = fcf / rev
        if fcf_margin > 0.12:   score += 2
        elif fcf_margin > 0.05: score += 1

    # ROA (2 pts)
    roa = _safe(info.get('returnOnAssets'))
    if roa is not None:
        if roa > 0.08:   score += 2
        elif roa > 0.03: score += 1

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 2 — Financial Strength (0–10)
# =============================================================================
def score_financial_strength(info):
    score = 0

    # Debt / EBITDA (3 pts)
    debt   = _safe(info.get('totalDebt'), 0)
    ebitda = _safe(info.get('ebitda'))
    if ebitda and ebitda > 0:
        ratio = debt / ebitda
        if ratio < 1:    score += 3
        elif ratio < 3:  score += 2
        elif ratio < 5:  score += 1

    # Current ratio (3 pts)
    cr = _safe(info.get('currentRatio'))
    if cr is not None:
        if cr > 2.0:    score += 3
        elif cr > 1.2:  score += 2
        elif cr > 0.8:  score += 1

    # Interest coverage (2 pts)
    interest = _safe(info.get('interestExpense'))
    if ebitda and ebitda > 0 and interest and interest > 0:
        coverage = ebitda / interest
        if coverage > 10:  score += 2
        elif coverage > 5: score += 1

    # Positive FCF (2 pts)
    fcf = _safe(info.get('freeCashflow'))
    if fcf is not None and fcf > 0:
        score += 2

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 3 — Growth (0–10)
# =============================================================================
def score_growth(info, financials):
    score = 0

    # Revenue growth YoY (3 pts)
    rev_growth = _safe(info.get('revenueGrowth'))
    if rev_growth is not None:
        if rev_growth > 0.15:   score += 3
        elif rev_growth > 0.05: score += 2
        elif rev_growth > 0:    score += 1

    # Earnings growth YoY (3 pts)
    earn_growth = _safe(info.get('earningsGrowth'))
    if earn_growth is not None:
        if earn_growth > 0.20:  score += 3
        elif earn_growth > 0.05: score += 2
        elif earn_growth > 0:   score += 1

    if financials is not None and not financials.empty:
        # 3yr revenue trend (2 pts)
        rev_series = _get_annual_series(financials, ['Total Revenue', 'TotalRevenue'])
        if rev_series:
            score += _trend_score(rev_series[-4:], max_pts=2)

        # EPS / net income trend (2 pts)
        ni_series = _get_annual_series(financials, ['Net Income', 'NetIncome'])
        if ni_series:
            score += _trend_score(ni_series[-4:], max_pts=2)

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 4 — Predictability (0–10)
# =============================================================================
def score_predictability(info, financials):
    score = 0

    if financials is not None and not financials.empty:
        # Earnings consistency (3 pts)
        ni_series = _get_annual_series(financials, ['Net Income', 'NetIncome'])
        if ni_series:
            score += _trend_score(ni_series[-5:], max_pts=3)

        # Revenue consistency (3 pts)
        rev_series = _get_annual_series(financials, ['Total Revenue', 'TotalRevenue'])
        if rev_series:
            score += _trend_score(rev_series[-5:], max_pts=3)

        # Profit margin stability (2 pts)
        if ni_series and rev_series:
            min_len = min(len(ni_series), len(rev_series), 5)
            margins = []
            for i in range(min_len):
                if rev_series[i] and rev_series[i] > 0:
                    margins.append(ni_series[i] / rev_series[i])
            if len(margins) >= 2:
                variance = float(np.std(margins))
                if variance < 0.05:   score += 2
                elif variance < 0.10: score += 1

    # FCF consistency (2 pts)
    fcf = _safe(info.get('freeCashflow'))
    ocf = _safe(info.get('operatingCashflow'))
    if fcf is not None and fcf > 0 and ocf is not None and ocf > 0:
        score += 2
    elif fcf is not None and fcf > 0:
        score += 1

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 5 — Valuation (0–10)
# =============================================================================
def score_valuation(info, intrinsic_value):
    score = 0

    # Primary signal — Margin of Safety (5 pts)
    current_price = (
        _safe(info.get('currentPrice')) or
        _safe(info.get('regularMarketPrice'))
    )
    if current_price and current_price > 0 and intrinsic_value and intrinsic_value > 0:
        mos = (intrinsic_value - current_price) / intrinsic_value
        if mos > 0.40:    score += 5   # > 40% undervalued
        elif mos > 0.20:  score += 4   # 20–40% undervalued
        elif mos > 0:     score += 3   # 0–20% undervalued
        elif mos > -0.20: score += 2   # 0–20% overvalued
        elif mos > -0.40: score += 1   # 20–40% overvalued
        # severely overvalued → 0

    # Supporting — P/E (2 pts)
    pe = _safe(info.get('trailingPE'))
    if pe is not None and pe > 0:
        if pe < 15:   score += 2
        elif pe < 40: score += 1

    # Supporting — P/S (1 pt)
    ps = _safe(info.get('priceToSalesTrailing12Months'))
    if ps is not None and ps < 2:
        score += 1

    # Supporting — P/B (2 pts)
    pb = _safe(info.get('priceToBook'))
    if pb is not None and pb > 0:
        if pb < 1:   score += 2
        elif pb < 5: score += 1

    return min(score, 10)

# =============================================================================
# QUALITY CLASSIFICATION — your full logic applied in strict priority order
# =============================================================================
def classify_quality(scores: dict, subtotal_pct: float) -> dict:
    """
    scores keys: profitability, financial_strength, growth,
                 predictability, valuation, moat (may be None)
    Returns: {label, penalty_pct, final_score_pct}
    """
    p   = scores.get('profitability', 0) or 0
    fs  = scores.get('financial_strength', 0) or 0
    g   = scores.get('growth', 0) or 0
    pre = scores.get('predictability', 0) or 0
    v   = scores.get('valuation', 0) or 0
    m   = scores.get('moat')  # None until AI Studio fills it

    label = None

    # Rule 1 — Automatic Safe
    if m is not None and all(x >= 7 for x in [m, pre, p, fs]):
        label = "Safe"

    # Rule 2 — Automatic Dangerous
    if label is None and m is not None and m <= 4 and fs <= 4:
        label = "Dangerous"

    # Rule 3 — Speculative overrides
    if label is None and m is not None:
        if (m >= 7 and fs >= 7) or (m + fs >= 15) or (m >= 7 and g >= 7):
            label = "Speculative"

    # Rule 4 — Score thresholds
    if label is None:
        if subtotal_pct >= 70:
            label = "Safe"
        elif subtotal_pct >= 66:
            label = "Safe"        # borderline: default Safe, AI can override
        elif subtotal_pct >= 60:
            label = "Speculative"
        elif subtotal_pct >= 51:
            label = "Speculative" # borderline: default Speculative, AI can override
        else:
            label = "Dangerous"

    # Penalty
    penalty = 0
    if label == "Speculative": penalty = 5
    elif label == "Dangerous": penalty = 10

    return {
        "label":           label,
        "penalty_pct":     penalty,
        "final_score_pct": max(0.0, round(subtotal_pct - penalty, 2)),
    }

# =============================================================================
# TIER RESOLVER
# =============================================================================
def resolve_tier(label: str) -> dict:
    mapping   = {"Safe": "Strong", "Speculative": "Average", "Dangerous": "Weak"}
    tier_name = mapping.get(label, "Average")
    config    = TIER_CONFIG[tier_name]
    return {
        "tier":            tier_name,
        "discount_rate":   config["discount_rate"],
        "growth_stage1":   config["growth_stage1"],
        "growth_stage2":   config["growth_stage2"],
        "terminal_growth": config["terminal_growth"],
    }

# =============================================================================
# ROE AND ROIC CALCULATOR
#
# ROE  — already available from yfinance info["returnOnEquity"]
#         Stored directly, no calculation needed.
#
# ROIC — not exposed by yfinance directly. Calculated from annual financials:
#
#   NOPAT           = Operating Income x (1 - Effective Tax Rate)
#   Invested Capital = Total Assets - Current Liabilities - Excess Cash
#   ROIC            = NOPAT / Invested Capital
#
#   Effective Tax Rate = Income Tax Expense / Pretax Income
#                        (clamped 0-50%, defaults to 21% if unavailable)
#
#   Excess Cash = max(0, Cash - 2% of Revenue)
#                 (2% of revenue is a standard working capital cash reserve)
#
# Uses the most recent fiscal year from stock.financials and stock.balance_sheet
# which are already fetched during quality scoring — no new API calls needed.
#
# Returns:
#   {
#     "roe":          float | None,   # e.g. 0.1714 = 17.14%
#     "roic":         float | None,   # e.g. 0.2831 = 28.31%
#     "roe_pct":      str   | None,   # e.g. "17.14%"
#     "roic_pct":     str   | None,   # e.g. "28.31%"
#     "roic_source":  str,            # "calculated" | "unavailable"
#   }
# =============================================================================
def calc_roe_roic(info, financials, balance):
    result = {
        "roe":         None,
        "roic":        None,
        "roe_pct":     None,
        "roic_pct":    None,
        "roic_source": "unavailable",
    }

    # --- ROE --- (direct from yfinance)
    roe = _safe(info.get("returnOnEquity"))
    if roe is not None:
        result["roe"]     = round(roe, 4)
        result["roe_pct"] = f"{round(roe * 100, 2)}%"

    # --- ROIC --- (calculated from annual statements)
    try:
        if financials is None or financials.empty: return result
        if balance    is None or balance.empty:    return result

        # Use most recent fiscal year column
        fin_col = financials.columns[0]
        bal_col = balance.columns[0]

        # --- Operating Income ---
        op_inc = None
        for name in ["Operating Income", "OperatingIncome",
                     "Total Operating Income As Reported"]:
            if name in financials.index:
                op_inc = _safe(float(financials.loc[name, fin_col]))
                break
        if op_inc is None or op_inc <= 0:
            return result

        # --- Effective Tax Rate ---
        tax_rate = 0.21   # US corporate default
        try:
            pretax = None
            for name in ["Pretax Income", "PretaxIncome",
                         "Income Before Tax", "IncomeBeforeTax"]:
                if name in financials.index:
                    pretax = _safe(float(financials.loc[name, fin_col]))
                    break
            tax_exp = None
            for name in ["Tax Provision", "TaxProvision",
                         "Income Tax Expense", "IncomeTaxExpense"]:
                if name in financials.index:
                    tax_exp = _safe(float(financials.loc[name, fin_col]))
                    break
            if pretax and pretax > 0 and tax_exp and tax_exp > 0:
                tax_rate = max(0.0, min(tax_exp / pretax, 0.50))
        except Exception:
            pass

        nopat = op_inc * (1 - tax_rate)

        # --- Invested Capital ---
        total_assets = None
        for name in ["Total Assets", "TotalAssets"]:
            if name in balance.index:
                total_assets = _safe(float(balance.loc[name, bal_col]))
                break

        current_liab = None
        for name in ["Current Liabilities", "CurrentLiabilities",
                     "Total Current Liabilities", "TotalCurrentLiabilities"]:
            if name in balance.index:
                current_liab = _safe(float(balance.loc[name, bal_col]))
                break

        cash = _safe(float(balance.loc["Cash And Cash Equivalents",    bal_col])) if "Cash And Cash Equivalents"    in balance.index else                _safe(float(balance.loc["CashAndCashEquivalents",       bal_col])) if "CashAndCashEquivalents"       in balance.index else                _safe(info.get("totalCash"), 0)

        rev = _safe(info.get("totalRevenue"), 0)
        excess_cash = max(0, (cash or 0) - 0.02 * (rev or 0))

        if total_assets is None or current_liab is None:
            return result

        invested_capital = total_assets - current_liab - excess_cash
        if invested_capital <= 0:
            return result

        roic = nopat / invested_capital
        result["roic"]        = round(roic, 4)
        result["roic_pct"]    = f"{round(roic * 100, 2)}%"
        result["roic_source"] = "calculated"

    except Exception:
        pass

    return result


# =============================================================================
# HISTORICAL MEAN MULTIPLES  (3–5yr average)
# =============================================================================
def get_historical_mean_multiples(stock, shares):
    result = {"mean_pe": None, "mean_ps": None, "mean_pb": None}
    try:
        financials = stock.financials
        balance    = stock.balance_sheet
        hist       = stock.history(period="10y")

        if financials is None or financials.empty: return result
        if hist is None or hist.empty:             return result

        pe_list, ps_list, pb_list = [], [], []
        cols = list(financials.columns)[:HISTORY_YEARS]

        for col in cols:
            year_hist = hist[hist.index.year == col.year]
            if year_hist.empty: continue
            price = float(year_hist['Close'].iloc[-1])
            if price <= 0: continue

            try:
                ni_row = (
                    financials.loc['Net Income'] if 'Net Income' in financials.index
                    else financials.loc['NetIncome'] if 'NetIncome' in financials.index
                    else None
                )
                if ni_row is not None and shares > 0:
                    eps = float(ni_row[col]) / shares
                    if eps > 0:
                        pe = price / eps
                        if pe <= MAX_PE:           # discard distorted years
                            pe_list.append(pe)
            except Exception: pass

            try:
                rev_row = (
                    financials.loc['Total Revenue'] if 'Total Revenue' in financials.index
                    else financials.loc['TotalRevenue'] if 'TotalRevenue' in financials.index
                    else None
                )
                if rev_row is not None and shares > 0:
                    rps = float(rev_row[col]) / shares
                    if rps > 0:
                        ps = price / rps
                        if ps <= MAX_PS:           # discard hyper-growth outliers
                            ps_list.append(ps)
            except Exception: pass

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
                            pb = price / bvps
                            if pb <= MAX_PB:       # discard intangible-heavy outliers
                                pb_list.append(pb)
            except Exception: pass

        if pe_list: result["mean_pe"] = round(float(np.mean(pe_list)), 4)
        if ps_list: result["mean_ps"] = round(float(np.mean(ps_list)), 4)
        if pb_list: result["mean_pb"] = round(float(np.mean(pb_list)), 4)

    except Exception: pass
    return result

# =============================================================================
# VALUATION METHODS 1–10
# =============================================================================
def calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount):
    ocf = _safe(info.get('operatingCashflow'))
    if ocf is None or shares <= 0: return None
    return _dcf_sum(ocf / shares, growth_s1, growth_s2, discount)

def calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount):
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0: return None
    return _dcf_sum(fcf / shares, growth_s1, growth_s2, discount)

def calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount):
    ni = _safe(info.get('netIncomeToCommon'))
    if ni is None or shares <= 0: return None
    return _dcf_sum(ni / shares, growth_s1, growth_s2, discount)

def calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal_growth, dcf_fcf_value=None):
    """
    Two-stage DCF with Gordon Growth terminal value.
    Terminal value is capped at TERMINAL_VALUE_CAP_MULTIPLIER x DCF FCF
    to prevent perpetual growth assumptions from inflating weak companies
    or creating unrealistic outliers that skew the weighted mean.
    """
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0: return None
    fcf_ps = fcf / shares
    if fcf_ps <= 0: return None

    pv_cf  = _dcf_sum(fcf_ps, growth_s1, growth_s2, discount)
    cf_end = fcf_ps
    for t in range(1, DCF_YEARS + 1):
        cf_end = cf_end * (1 + (growth_s1 if t <= 10 else growth_s2))
    if discount <= terminal_growth: return None
    pv_tv  = ((cf_end * (1 + terminal_growth)) / (discount - terminal_growth)) / (1 + discount) ** DCF_YEARS

    result = round(pv_cf + pv_tv, 2)

    # Apply terminal value cap relative to plain DCF FCF
    if dcf_fcf_value and dcf_fcf_value > 0:
        cap = dcf_fcf_value * TERMINAL_VALUE_CAP_MULTIPLIER
        if result > cap:
            result = round(cap, 2)

    return result

def calc_mean_ps(info, shares, mean_ps):
    rev = _safe(info.get('totalRevenue'))
    if rev is None or shares <= 0: return None
    rev_ps = rev / shares
    if rev_ps <= 0: return None
    ratio = mean_ps if mean_ps is not None else _safe(info.get('priceToSalesTrailing12Months'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PS: return None    # reject — revenue multiple too elevated to be meaningful
    return round(ratio * rev_ps, 2)

def calc_mean_pe(info, mean_pe):
    eps   = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0: return None
    ratio = mean_pe if mean_pe is not None else _safe(info.get('trailingPE'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PE: return None    # reject — earnings too distorted to value on P/E
    return round(ratio * eps, 2)

def calc_mean_pb(info, mean_pb):
    bvps  = _safe(info.get('bookValue'))
    if bvps is None or bvps <= 0: return None
    ratio = mean_pb if mean_pb is not None else _safe(info.get('priceToBook'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PB: return None    # reject — book multiple too distorted to be meaningful
    return round(ratio * bvps, 2)

def calc_psg(info, shares, forward_growth=None):
    """
    PSG fair value = Revenue per share x (growth rate as a whole number)
    Fair value is the price at which PSG = 1.0.
    Uses forward_growth (from Finviz) if provided, else falls back to
    yfinance revenueGrowth. Capped at MAX_PEG_PSG_GROWTH to prevent
    turnaround/recovery spikes from inflating the result.
    """
    rev = _safe(info.get('totalRevenue'))
    if rev is None or rev <= 0 or shares <= 0: return None

    # Prefer forward growth; fall back to trailing revenue growth
    if forward_growth is not None and forward_growth > 0:
        growth = min(forward_growth, MAX_PEG_PSG_GROWTH)
    else:
        rev_growth = _safe(info.get('revenueGrowth'))
        if rev_growth is None or rev_growth <= 0: return None
        growth = min(rev_growth, MAX_PEG_PSG_GROWTH)

    rev_ps = rev / shares
    return round(rev_ps * (growth * 100), 2)

def calc_peg(info, forward_growth=None):
    """
    PEG fair value = EPS x (growth rate as a whole number)
    Fair value is the price at which PEG = 1.0 (Lynch framework).
    Uses forward_growth (from Finviz) if provided, else yfinance earningsGrowth.
    Capped at MAX_PEG_PSG_GROWTH (30%) — above this the formula produces
    unreliable outliers (e.g. GME recovery spike giving $199 fair value).
    """
    eps = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0: return None

    # Prefer forward growth; fall back to trailing earnings growth
    if forward_growth is not None and forward_growth > 0:
        growth = min(forward_growth, MAX_PEG_PSG_GROWTH)
    else:
        yf_growth = _safe(info.get('earningsGrowth'))
        if yf_growth is None or yf_growth <= 0: return None
        growth = min(yf_growth, MAX_PEG_PSG_GROWTH)

    return round(eps * (growth * 100), 2)

def calc_ev_ebitda(info, shares):
    """
    Uses the SECTOR MEDIAN EV/EBITDA multiple rather than the company's own
    multiple. This prevents circular valuation where an overpriced stock
    (e.g. GME trading at 200x EV/EBITDA) simply validates its own premium.

    Sector is read from info['sector']. Falls back to 'Default' (12x) if
    the sector is unrecognised or missing.
    """
    ebitda = _safe(info.get('ebitda'))
    debt   = _safe(info.get('totalDebt'), 0)
    cash   = _safe(info.get('totalCash'), 0)
    if ebitda is None or ebitda <= 0 or shares <= 0: return None

    sector   = info.get('sector', 'Default') or 'Default'
    multiple = SECTOR_EV_EBITDA.get(sector, SECTOR_EV_EBITDA['Default'])

    equity_value = (ebitda * multiple) - debt + cash
    if equity_value <= 0: return None
    return round(equity_value / shares, 2)

# =============================================================================
# INTRINSIC VALUE AGGREGATOR — tier-weighted mean
#
# Each method is weighted by its reliability for the company's tier.
# Valid methods only (non-null, > 0, < $1M). Weights are renormalised
# across valid methods so the result is always a proper weighted mean
# even when some methods return null.
# =============================================================================
def calc_intrinsic_value(valuations: dict, tier: str = "Average") -> dict:
    keys = [
        "1_DCF_Operating_CF", "2_DCF_FCF", "3_DCF_Net_Income",
        "4_DCF_Terminal_Value", "5_Mean_PS", "6_Mean_PE",
        "7_Mean_PB", "8_PSG", "9_PEG", "10_EV_EBITDA",
    ]
    weights = TIER_WEIGHTS.get(tier, TIER_WEIGHTS["Average"])

    valid_vals, valid_weights = [], []
    for k in keys:
        v = valuations.get(k)
        if v is not None and isinstance(v, (int, float)) and 0 < v < 1_000_000:
            valid_vals.append(v)
            valid_weights.append(weights.get(k, 0.10))

    count = len(valid_vals)
    if count == 0:
        return {"intrinsic_value": None, "methods_used": 0, "confidence": "Insufficient Data"}

    # Renormalise weights across valid methods only
    total_w   = sum(valid_weights)
    norm_w    = [w / total_w for w in valid_weights]
    intrinsic = round(float(np.dot(valid_vals, norm_w)), 2)
    confidence = "High" if count >= 7 else "Medium" if count >= 4 else "Low"

    return {
        "intrinsic_value": intrinsic,
        "methods_used":    count,
        "confidence":      confidence,
    }

# =============================================================================
# 10-YEAR PRICE HISTORY  (unchanged)
# =============================================================================
def get_10yr_history(stock):
    try:
        hist = stock.history(period="10y")
        if hist.empty: return {}
        return {
            str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2)
            for y in sorted(set(d.year for d in hist.index))
        }
    except Exception:
        return {}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def analyze_ticker(ticker, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(0.5)

            stock = yf.Ticker(ticker)
            info  = stock.info

            if not info:
                print(f"  [SKIP] {ticker}: yfinance returned empty info")
                return None

            # Robust share count — yfinance uses different keys depending on
            # the security type. 'sharesOutstanding' is absent for many tickers
            # (meme stocks, ETFs, ADRs). Try all known alternatives in order.
            shares = (
                info.get('sharesOutstanding') or
                info.get('impliedSharesOutstanding') or
                info.get('floatShares') or
                0
            )
            if not shares or shares <= 0:
                print(f"  [SKIP] {ticker}: no share count available "
                      f"(sharesOutstanding={info.get('sharesOutstanding')}, "
                      f"impliedSharesOutstanding={info.get('impliedSharesOutstanding')}, "
                      f"floatShares={info.get('floatShares')})")
                return None

            # Pull annual financials once — reused across quality metrics
            try:
                financials = stock.financials
            except Exception:
                financials = None

            # ------------------------------------------------------------------
            # STAGE 1 — Preliminary quality scores (valuation = 0 placeholder)
            #           Used to determine tier → tier drives DCF inputs
            # ------------------------------------------------------------------
            prelim_scores = {
                "profitability":      score_profitability(info),
                "financial_strength": score_financial_strength(info),
                "growth":             score_growth(info, financials),
                "predictability":     score_predictability(info, financials),
                "valuation":          0,     # computed after intrinsic value
                "moat":               None,  # filled by AI Studio
            }
            prelim_subtotal     = sum(v for k, v in prelim_scores.items() if k not in ('moat', 'valuation') and v is not None)
            prelim_pct          = round((prelim_subtotal / 50) * 100, 2)
            prelim_class        = classify_quality(prelim_scores, prelim_pct)
            tier_data           = resolve_tier(prelim_class["label"])
            discount            = tier_data["discount_rate"]
            growth_s1           = tier_data["growth_stage1"]
            growth_s2           = tier_data["growth_stage2"]
            terminal            = tier_data["terminal_growth"]

            # ------------------------------------------------------------------
            # STAGE 2 — Forward growth rate (Finviz primary, yfinance fallback)
            # ------------------------------------------------------------------
            fwd_growth      = get_forward_growth(ticker, info)
            fwd_rate        = fwd_growth["rate"]     # clamped 2%-30%
            fwd_source      = fwd_growth["source"]   # finviz | yfinance | default
            fwd_raw         = fwd_growth["raw"]

            # Override tier stage1 growth with forward rate — more accurate
            # Stage2 tapers from forward rate to preserve two-stage shape
            growth_s1 = fwd_rate
            growth_s2 = max(MIN_GROWTH_RATE, fwd_rate * 0.50)  # stage2 = 50% of stage1

            # ------------------------------------------------------------------
            # STAGE 3 — Historical multiples
            # ------------------------------------------------------------------
            hist_multiples = get_historical_mean_multiples(stock, shares)
            mean_pe = hist_multiples.get("mean_pe")
            mean_ps = hist_multiples.get("mean_ps")
            mean_pb = hist_multiples.get("mean_pb")

            # ------------------------------------------------------------------
            # STAGE 4 — All 10 valuations using tier-adjusted inputs
            # ------------------------------------------------------------------
            # Compute DCF FCF first — used as terminal value cap reference
            dcf_fcf_val = calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount)

            valuations = {
                "1_DCF_Operating_CF":   calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount),
                "2_DCF_FCF":            dcf_fcf_val,
                "3_DCF_Net_Income":     calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount),
                "4_DCF_Terminal_Value": calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal, dcf_fcf_val),
                "5_Mean_PS":            calc_mean_ps(info, shares, mean_ps),
                "6_Mean_PE":            calc_mean_pe(info, mean_pe),
                "7_Mean_PB":            calc_mean_pb(info, mean_pb),
                "8_PSG":                calc_psg(info, shares, fwd_rate),
                "9_PEG":                calc_peg(info, fwd_rate),
                "10_EV_EBITDA":         calc_ev_ebitda(info, shares),
            }
            aggregate  = calc_intrinsic_value(valuations, tier=tier_data["tier"])
            valuations.update(aggregate)
            valuations["_meta"] = {
                "pe_source":       "historical_mean" if mean_pe else "current_trailing",
                "ps_source":       "historical_mean" if mean_ps else "current_trailing",
                "pb_source":       "historical_mean" if mean_pb else "current_trailing",
                "pe_approx":       True,
                "peg_approx":      True,
                "growth_source":   fwd_source,
                "growth_raw":      fwd_raw,
                "discount_rate":   discount,
                "growth_stage1":   growth_s1,
                "growth_stage2":   growth_s2,
                "terminal_growth": terminal,
                "tier":            tier_data["tier"],
                "weights_applied": TIER_WEIGHTS.get(tier_data["tier"], TIER_WEIGHTS["Average"]),
            }

            # ------------------------------------------------------------------
            # STAGE 4 — Final quality scores (now with real valuation score)
            # ------------------------------------------------------------------
            intrinsic_value = aggregate.get("intrinsic_value")
            final_scores = {
                **prelim_scores,
                "valuation": score_valuation(info, intrinsic_value),
            }
            python_subtotal     = sum(v for k, v in final_scores.items() if k != 'moat' and v is not None)
            python_subtotal_pct = round((python_subtotal / 50) * 100, 2)
            classification      = classify_quality(final_scores, python_subtotal_pct)

            quality = {
                "scores": {
                    "profitability":      final_scores["profitability"],
                    "financial_strength": final_scores["financial_strength"],
                    "growth":             final_scores["growth"],
                    "predictability":     final_scores["predictability"],
                    "valuation":          final_scores["valuation"],
                    "moat":               None,
                },
                "python_subtotal":      python_subtotal,
                "python_subtotal_pct":  python_subtotal_pct,
                "classification":       classification["label"],
                "penalty_pct":          classification["penalty_pct"],
                "final_score_pct":      classification["final_score_pct"],
                "note": (
                    "Moat score is null. AI Studio adds moat score and "
                    "recalculates Investment Quality Score and classification "
                    "using all 6 metrics out of 60."
                ),
                "last_calculated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            # ------------------------------------------------------------------
            # STAGE 5 — ROE and ROIC
            # Calculated here using financials already fetched in Stage 1.
            # Stored in JSON for AI Studio to display — no live Finviz call needed.
            # ------------------------------------------------------------------
            roe_roic = calc_roe_roic(info, financials, stock.balance_sheet)

            fundamentals = {
                "roe":         roe_roic["roe"],
                "roic":        roe_roic["roic"],
                "roe_pct":     roe_roic["roe_pct"],
                "roic_pct":    roe_roic["roic_pct"],
                "roic_source": roe_roic["roic_source"],
            }

            # ------------------------------------------------------------------
            # STAGE 6 — History & forecast
            # ------------------------------------------------------------------
            price_history = get_10yr_history(stock)
            current_year  = datetime.now().year
            forecast = {
                str(current_year + i): round(intrinsic_value * (1 + growth_s1) ** i, 2)
                for i in range(1, 6)
            } if intrinsic_value and intrinsic_value > 0 else {}

            return ticker, {
                "valuations":      valuations,
                "quality":         quality,
                "fundamentals":    fundamentals,
                "10_Year_History": price_history,
                "5_Year_Forecast": forecast,
                "Last_Updated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                print(f"  [RATE LIMIT] {ticker} attempt {attempt+1} — waiting 5s")
                time.sleep(5)
            else:
                print(f"  [ERROR] {ticker} attempt {attempt+1}: {err[:120]}")
            continue

    return None

# =============================================================================
# A–Z PARTITIONED JSON SAVER  (unchanged)
# =============================================================================
def save_partitioned_data(master_results):
    partitions = {}
    for ticker, data in master_results.items():
        letter = ticker[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in partitions: partitions[letter] = {}
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

    skipped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            t   = future_to_ticker[future]
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
                if len(master_results) % 50 == 0:
                    print(f"  Processed {len(master_results)} stocks...")
            else:
                skipped.append(t)

    if skipped:
        print(f"\n⚠️  {len(skipped)} tickers skipped (see [SKIP]/[ERROR] lines above):")
        for chunk in [skipped[i:i+10] for i in range(0, len(skipped), 10)]:
            print(f"   {', '.join(chunk)}")

    save_partitioned_data(master_results)

if __name__ == "__main__":
    main()
