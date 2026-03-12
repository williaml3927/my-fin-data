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
                    if eps > 0: pe_list.append(price / eps)
            except Exception: pass

            try:
                rev_row = (
                    financials.loc['Total Revenue'] if 'Total Revenue' in financials.index
                    else financials.loc['TotalRevenue'] if 'TotalRevenue' in financials.index
                    else None
                )
                if rev_row is not None and shares > 0:
                    rps = float(rev_row[col]) / shares
                    if rps > 0: ps_list.append(price / rps)
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
                        if bvps > 0: pb_list.append(price / bvps)
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

def calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal_growth):
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0: return None
    fcf_ps = fcf / shares
    if fcf_ps <= 0: return None
    pv_cf   = _dcf_sum(fcf_ps, growth_s1, growth_s2, discount)
    # Terminal value anchored to end-of-stage-2 cash flow
    cf_end  = fcf_ps
    for t in range(1, DCF_YEARS + 1):
        cf_end = cf_end * (1 + (growth_s1 if t <= 10 else growth_s2))
    if discount <= terminal_growth: return None
    pv_tv   = ((cf_end * (1 + terminal_growth)) / (discount - terminal_growth)) / (1 + discount) ** DCF_YEARS
    return round(pv_cf + pv_tv, 2)

def calc_mean_ps(info, shares, mean_ps):
    rev = _safe(info.get('totalRevenue'))
    if rev is None or shares <= 0: return None
    rev_ps = rev / shares
    if rev_ps <= 0: return None
    ratio = mean_ps if mean_ps is not None else _safe(info.get('priceToSalesTrailing12Months'))
    if ratio is None or ratio <= 0: return None
    return round(ratio * rev_ps, 2)

def calc_mean_pe(info, mean_pe):
    eps   = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0: return None
    ratio = mean_pe if mean_pe is not None else _safe(info.get('trailingPE'))
    if ratio is None or ratio <= 0: return None
    return round(ratio * eps, 2)

def calc_mean_pb(info, mean_pb):
    bvps  = _safe(info.get('bookValue'))
    if bvps is None or bvps <= 0: return None
    ratio = mean_pb if mean_pb is not None else _safe(info.get('priceToBook'))
    if ratio is None or ratio <= 0: return None
    return round(ratio * bvps, 2)

def calc_psg(info, shares):
    """
    PSG fair value = Revenue per share x (Revenue growth rate as a whole number)
    Mirrors PEG logic: fair value is the price at which PSG = 1.0
    i.e. P/S ratio equals the revenue growth rate expressed as a percentage.
    Example: AAPL rev/share $25.40 x 6.1% growth = $154.94 fair value
    """
    rev        = _safe(info.get('totalRevenue'))
    rev_growth = _safe(info.get('revenueGrowth'))
    if None in (rev, rev_growth) or shares <= 0: return None
    if rev_growth <= 0 or rev <= 0: return None
    rev_ps = rev / shares
    return round(rev_ps * (rev_growth * 100), 2)

def calc_peg(info):
    eps    = _safe(info.get('trailingEps'))
    growth = _safe(info.get('earningsGrowth'))
    if None in (eps, growth) or eps <= 0 or growth <= 0: return None
    return round(eps * (growth * 100), 2)

def calc_ev_ebitda(info, shares):
    ebitda = _safe(info.get('ebitda'))
    ev     = _safe(info.get('enterpriseValue'))
    debt   = _safe(info.get('totalDebt'), 0)
    cash   = _safe(info.get('totalCash'), 0)
    if None in (ebitda, ev) or ebitda <= 0 or ev <= 0 or shares <= 0: return None
    multiple     = ev / ebitda
    if multiple <= 0: return None
    equity_value = (ebitda * multiple) - debt + cash
    if equity_value <= 0: return None
    return round(equity_value / shares, 2)

# =============================================================================
# INTRINSIC VALUE AGGREGATOR
# =============================================================================
def calc_intrinsic_value(valuations: dict) -> dict:
    keys = [
        "1_DCF_Operating_CF", "2_DCF_FCF", "3_DCF_Net_Income",
        "4_DCF_Terminal_Value", "5_Mean_PS", "6_Mean_PE",
        "7_Mean_PB", "8_PSG", "9_PEG", "10_EV_EBITDA",
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
    confidence = "High" if count >= 7 else "Medium" if count >= 4 else "Low"
    return {
        "intrinsic_value": round(float(np.mean(valid)), 2),
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

            if not info or 'sharesOutstanding' not in info:
                return None

            shares = info.get('sharesOutstanding', 1)
            if shares <= 0:
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
            # STAGE 2 — Historical multiples
            # ------------------------------------------------------------------
            hist_multiples = get_historical_mean_multiples(stock, shares)
            mean_pe = hist_multiples.get("mean_pe")
            mean_ps = hist_multiples.get("mean_ps")
            mean_pb = hist_multiples.get("mean_pb")

            # ------------------------------------------------------------------
            # STAGE 3 — All 10 valuations using tier-adjusted inputs
            # ------------------------------------------------------------------
            valuations = {
                "1_DCF_Operating_CF":   calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount),
                "2_DCF_FCF":            calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount),
                "3_DCF_Net_Income":     calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount),
                "4_DCF_Terminal_Value": calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal),
                "5_Mean_PS":            calc_mean_ps(info, shares, mean_ps),
                "6_Mean_PE":            calc_mean_pe(info, mean_pe),
                "7_Mean_PB":            calc_mean_pb(info, mean_pb),
                "8_PSG":                calc_psg(info, shares),
                "9_PEG":                calc_peg(info),
                "10_EV_EBITDA":         calc_ev_ebitda(info, shares),
            }
            aggregate  = calc_intrinsic_value(valuations)
            valuations.update(aggregate)
            valuations["_meta"] = {
                "pe_source":       "historical_mean" if mean_pe else "current_trailing",
                "ps_source":       "historical_mean" if mean_ps else "current_trailing",
                "pb_source":       "historical_mean" if mean_pb else "current_trailing",
                "pe_approx":       True,
                "peg_approx":      True,
                "discount_rate":   discount,
                "growth_stage1":   growth_s1,
                "growth_stage2":   growth_s2,
                "terminal_growth": terminal,
                "tier":            tier_data["tier"],
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
            # STAGE 5 — History & forecast
            # ------------------------------------------------------------------
            price_history = get_10yr_history(stock)
            current_year  = datetime.now().year
            forecast = {
                str(current_year + i): round(intrinsic_value * (1 + growth_s1) ** i, 2)
                for i in range(1, 6)
            } if intrinsic_value and intrinsic_value > 0 else {}  # uses stage1 growth for near-term forecast

            return ticker, {
                "valuations":      valuations,
                "quality":         quality,
                "10_Year_History": price_history,
                "5_Year_Forecast": forecast,
                "Last_Updated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

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
