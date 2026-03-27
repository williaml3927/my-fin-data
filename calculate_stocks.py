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
# =============================================================================
TIER_CONFIG = {
    "Strong": {
        "discount_rate":   0.075,
        "growth_stage1":   0.12,
        "growth_stage2":   0.06,
        "terminal_growth": 0.04,
    },
    "Average": {
        "discount_rate":   0.10,
        "growth_stage1":   0.07,
        "growth_stage2":   0.04,
        "terminal_growth": 0.03,
    },
    "Weak": {
        "discount_rate":   0.12,
        "growth_stage1":   0.03,
        "growth_stage2":   0.02,
        "terminal_growth": 0.02,
    },
}

DCF_YEARS     = 10
HISTORY_YEARS = 5

# =============================================================================
# MULTIPLE CAPS
# =============================================================================
MAX_PE       = 50
MAX_PS       = 20
MAX_PB       = 20
MAX_EV_EBITDA = 40   # company's own EV/EBITDA capped at 40x to filter distortion

# =============================================================================
# GROWTH RATE CAPS
#
# MAX_GROWTH_RATE    : 30% ceiling on DCF stage 1 growth
# MAX_PEG_PSG_GROWTH : 30% ceiling specifically for PEG and PSG formulas
# MIN_GROWTH_RATE    : 2% floor for DCF so shrinking firms still get a value
#
# IMPORTANT — Finviz "-" handling:
#   If Finviz returns "-" for EPS Next 5Y it means analysts forecast
#   no meaningful growth. This is treated as EXACTLY 0.0 — not as a
#   fallback to yfinance. Zero growth is intentional and meaningful data.
#   PEG and PSG will return None when growth = 0 (no valid result).
#   DCF methods apply the MIN_GROWTH_RATE floor so they still produce a value.
# =============================================================================
MAX_GROWTH_RATE    = 0.30
MAX_PEG_PSG_GROWTH = 0.20   # PSG/PEG capped at 20% → max effective P/S or P/E of 20x
MAX_DCF_GROWTH_RATE = 0.15  # DCF capped at 15% — higher rates compound to absurd values over 10 years
MIN_GROWTH_RATE    = 0.02

# =============================================================================
# TERMINAL VALUE CAP
# =============================================================================
TERMINAL_VALUE_CAP_MULTIPLIER = 3.0

# =============================================================================
# TIER WEIGHTS
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
# COMPANY TYPE DETECTION
# Classifies companies into types that require different valuation approaches.
# All methods still run — typing controls WEIGHTING not method execution.
#
# Types (mutually exclusive, priority-ordered):
#   REIT         — Real estate investment trusts (SIC 65xx, quoteType REIT)
#   BANK         — Banks and thrifts (SIC 60xx)
#   INSURANCE    — Insurance companies (SIC 63xx)
#   ASSET_MGR    — Asset managers, brokers (SIC 62xx)
#   UTILITY      — Regulated utilities (SIC 49xx)
#   SPECGROWTH   — Speculative growth: high EPS fwd, low profitability
#   CYCLICAL     — Highly cyclical (materials, energy, basic industries)
#   FCF_COMPOUNDER — Strong, consistent FCF generators
#   STANDARD     — Everything else
# =============================================================================
def detect_company_type(info, prelim_scores, eps_next_5y):
    """
    Returns a string type code and a short rationale string.
    Uses yfinance info fields: sector, industry, quoteType, SIC codes
    proxied from industry names since yfinance doesn't expose raw SIC.
    """
    sector   = (info.get("sector")   or "").lower()
    industry = (info.get("industry") or "").lower()
    qtype    = (info.get("quoteType") or "").upper()

    # ── REIT detection ────────────────────────────────────────────────────────
    if qtype == "REIT" or "reit" in industry or "real estate investment trust" in industry:
        return "REIT", "Real estate investment trust — P/NAV (P/B) prioritised"

    # ── Bank detection ────────────────────────────────────────────────────────
    bank_keywords = [
        "bank", "savings", "thrift", "credit union", "mortgage bank",
        "banking", "commercial bank", "regional bank", "diversified bank",
    ]
    if any(k in industry for k in bank_keywords):
        return "BANK", "Bank or thrift — P/Tangible Book prioritised; DCF excluded"

    # ── Insurance detection ───────────────────────────────────────────────────
    insurance_keywords = [
        "insurance", "reinsurance", "surety", "title insurance",
    ]
    if any(k in industry for k in insurance_keywords):
        return "INSURANCE", "Insurance company — DNI (Net Income DCF) prioritised; ROIC excluded"

    # ── Consumer finance / payments / credit services ─────────────────────────
    # AXP = "Credit Services", Visa/MA = "Diversified Financial Services",
    # Capital One/Discover = "Credit Services", Synchrony = "Financial Services"
    # These are financial companies but NOT banks — they fund themselves via
    # debt markets, have high leverage, and should use DNI + P/B for valuation.
    consumer_fin_keywords = [
        "credit services", "consumer finance", "credit card",
        "diversified financial services", "specialty finance",
        "payment processing", "payments", "transaction processing",
        "personal finance", "mortgage finance", "student loan",
        "auto loan", "consumer lending", "financial services",
    ]
    if any(k in industry for k in consumer_fin_keywords):
        return "CONSUMER_FINANCE", (
            "Consumer finance / payments — DNI and P/B prioritised; "
            "standard debt ratios unsuitable due to balance sheet structure"
        )

    # ── Asset manager detection ───────────────────────────────────────────────
    assetmgr_keywords = [
        "asset management", "investment management", "broker",
        "capital markets", "financial advisory", "wealth management",
        "fund", "investment bank", "brokerage", "securities",
    ]
    if any(k in industry for k in assetmgr_keywords):
        return "ASSET_MGR", "Asset manager/broker — DNI prioritised; standard CF metrics unreliable"

    # ── Catch-all for financial services sector ───────────────────────────────
    if sector == "financial services":
        return "CONSUMER_FINANCE", "Financial services — DNI and P/B prioritised"

    # ── Utility detection ─────────────────────────────────────────────────────
    if sector == "utilities" or "utility" in industry or "utilities" in industry:
        return "UTILITY", "Regulated utility — P/E and EV/EBITDA prioritised; high debt is structural"

    # ── Speculative growth detection ──────────────────────────────────────────
    # Trigger: EPS Next 5Y ≥ 15%, profitability ≤ 5, AND
    #   (moat ≥ 7 OR growth ≥ 7) AND (growth ≥ 7 OR financial_strength ≥ 7)
    prof_score   = prelim_scores.get("profitability", 10) or 10
    growth_score = prelim_scores.get("growth", 0) or 0
    fs_score     = prelim_scores.get("financial_strength", 0) or 0
    eps_fwd      = eps_next_5y or 0

    ni = _safe(info.get("netIncomeToCommon"))
    has_neg_income = ni is not None and ni < 0

    if (eps_fwd >= 0.15 and prof_score <= 5 and
            (growth_score >= 7 or fs_score >= 7) and
            (has_neg_income or prof_score <= 3)):
        return "SPECGROWTH", (
            f"Speculative growth — EPS fwd {round(eps_fwd*100,1)}%, "
            f"profitability {prof_score}/10; P/S and PSG prioritised"
        )

    # ── Cyclical detection ────────────────────────────────────────────────────
    cyclical_sectors = ["energy", "basic materials", "materials"]
    cyclical_industries = ["steel", "aluminum", "copper", "mining", "oil", "gas",
                           "chemical", "fertilizer", "paper", "lumber", "shipping",
                           "airline", "cruise", "hotel", "casino", "semiconductor"]
    if sector in cyclical_sectors or any(k in industry for k in cyclical_industries):
        return "CYCLICAL", f"Cyclical business ({sector}) — EV/EBITDA and P/S prioritised over P/E"

    # ── FCF Compounder detection ───────────────────────────────────────────────
    # Strong FCF: FCF margin > 15% AND positive FCF for 3+ years (proxy: FCF > 0)
    fcf  = _safe(info.get("freeCashflow"))
    rev  = _safe(info.get("totalRevenue"))
    ocf  = _safe(info.get("operatingCashflow"))
    fcf_margin = (fcf / rev) if fcf and rev and rev > 0 else None
    if (fcf and fcf > 0 and ocf and ocf > 0 and
            fcf_margin and fcf_margin > 0.15 and
            prof_score >= 7):
        return "FCF_COMPOUNDER", (
            f"FCF compounder — FCF margin {round(fcf_margin*100,1)}%; "
            f"DCF methods prioritised"
        )

    return "STANDARD", "Standard company — balanced weighting across all methods"


# =============================================================================
# DYNAMIC WEIGHTING
# Returns method weights tailored to company type AND quality tier.
# All weights within a returned dict sum to 1.0 after normalisation.
# Methods that are inappropriate for a type receive weight 0.0 and are
# excluded from the intrinsic value calculation.
# =============================================================================
def get_dynamic_weights(company_type, tier):
    """
    Returns a weights dict {method_key: float} for the given company type
    and quality tier. The tier still modulates weights within a type
    (e.g. a Strong REIT still gets some DCF weight, a Weak REIT doesn't).
    """
    base = TIER_WEIGHTS.get(tier, TIER_WEIGHTS["Average"])

    if company_type == "REIT":
        # P/NAV (P/B) is primary. DCF on operating CF is secondary.
        # P/E, PEG, PSG excluded — earnings are often negative or distorted.
        # ROIC excluded — not meaningful for asset-heavy leveraged entities.
        return {
            "1_DCF_Operating_CF":   0.15,
            "2_DCF_FCF":            0.10,
            "3_DCF_Net_Income":     0.00,   # excluded
            "4_DCF_Terminal_Value": 0.10,
            "5_Mean_PS":            0.05,
            "6_Mean_PE":            0.00,   # excluded — earnings distorted
            "7_Mean_PB":            0.45,   # PRIMARY — P/NAV
            "8_PSG":                0.00,   # excluded
            "9_PEG":                0.00,   # excluded
            "10_EV_EBITDA":         0.15,
        }

    if company_type == "BANK":
        # P/B (tangible book) is primary.
        # DCF excluded — bank cash flows are not comparable to industrial CF.
        # P/E secondary when positive. EV/EBITDA meaningless for banks.
        return {
            "1_DCF_Operating_CF":   0.00,   # excluded — bank OCF != industrial OCF
            "2_DCF_FCF":            0.00,   # excluded
            "3_DCF_Net_Income":     0.15,   # DNI — net income is valid for banks
            "4_DCF_Terminal_Value": 0.00,   # excluded
            "5_Mean_PS":            0.10,
            "6_Mean_PE":            0.25,   # meaningful when positive
            "7_Mean_PB":            0.40,   # PRIMARY — P/Tangible Book
            "8_PSG":                0.05,
            "9_PEG":                0.05,
            "10_EV_EBITDA":         0.00,   # excluded — EV/EBITDA not used for banks
        }

    if company_type == "INSURANCE":
        # DNI (Net Income DCF) is primary — insurance earnings are the true metric.
        # ROIC excluded (OCF/FCF unreliable due to float).
        # P/B meaningful — book value is tangible for insurers.
        return {
            "1_DCF_Operating_CF":   0.00,   # excluded — float distorts OCF
            "2_DCF_FCF":            0.00,   # excluded
            "3_DCF_Net_Income":     0.40,   # PRIMARY — discounted net income
            "4_DCF_Terminal_Value": 0.00,   # excluded
            "5_Mean_PS":            0.10,
            "6_Mean_PE":            0.25,   # secondary
            "7_Mean_PB":            0.20,   # meaningful for insurers
            "8_PSG":                0.05,
            "9_PEG":                0.00,   # excluded
            "10_EV_EBITDA":         0.00,   # excluded
        }

    if company_type == "ASSET_MGR":
        # DNI primary. P/E strong secondary. EV/EBITDA useful.
        # OCF/FCF unreliable — carried interest and fund distributions distort.
        return {
            "1_DCF_Operating_CF":   0.00,   # excluded
            "2_DCF_FCF":            0.00,   # excluded
            "3_DCF_Net_Income":     0.35,   # PRIMARY — fee income drives earnings
            "4_DCF_Terminal_Value": 0.00,   # excluded
            "5_Mean_PS":            0.15,
            "6_Mean_PE":            0.25,   # strong secondary
            "7_Mean_PB":            0.10,
            "8_PSG":                0.10,
            "9_PEG":                0.05,
            "10_EV_EBITDA":         0.00,   # excluded
        }

    if company_type == "UTILITY":
        # P/E and EV/EBITDA primary. DCF valid but uses regulated ROE.
        # High debt is structural — financial strength scoring adjusted.
        return {
            "1_DCF_Operating_CF":   0.12,
            "2_DCF_FCF":            0.08,   # lower — capex is always high
            "3_DCF_Net_Income":     0.12,
            "4_DCF_Terminal_Value": 0.10,
            "5_Mean_PS":            0.08,
            "6_Mean_PE":            0.25,   # PRIMARY — regulated earnings are stable
            "7_Mean_PB":            0.10,
            "8_PSG":                0.05,
            "9_PEG":                0.05,
            "10_EV_EBITDA":         0.05,
        }

    if company_type == "SPECGROWTH":
        # P/S and PSG primary — revenue more reliable than earnings.
        # DCF on NI excluded — negative/volatile earnings.
        # DCF on OCF downweighted — cash burn is expected.
        return {
            "1_DCF_Operating_CF":   0.08,
            "2_DCF_FCF":            0.05,   # downweighted — often negative
            "3_DCF_Net_Income":     0.00,   # excluded — negative earnings
            "4_DCF_Terminal_Value": 0.05,
            "5_Mean_PS":            0.35,   # PRIMARY
            "6_Mean_PE":            0.00,   # excluded — negative/unreliable
            "7_Mean_PB":            0.07,
            "8_PSG":                0.25,   # strong secondary
            "9_PEG":                0.00,   # excluded — earnings negative
            "10_EV_EBITDA":         0.15,
        }

    if company_type == "CYCLICAL":
        # EV/EBITDA primary — normalised through-cycle earnings.
        # P/E downweighted — earnings swing widely at cycle peaks/troughs.
        # P/S secondary — revenue more stable than earnings.
        return {
            "1_DCF_Operating_CF":   0.12,
            "2_DCF_FCF":            0.10,
            "3_DCF_Net_Income":     0.05,   # downweighted — cyclical distortion
            "4_DCF_Terminal_Value": 0.10,
            "5_Mean_PS":            0.18,   # strong secondary
            "6_Mean_PE":            0.05,   # downweighted — cycle peak/trough
            "7_Mean_PB":            0.10,
            "8_PSG":                0.10,
            "9_PEG":                0.05,
            "10_EV_EBITDA":         0.15,   # PRIMARY for cyclicals
        }

    if company_type == "CONSUMER_FINANCE":
        # DNI (Net Income DCF) and P/B primary.
        # Standard FCF/OCF DCF excluded — high-leverage funding models distort.
        # P/S meaningful — revenue is stable and recurring.
        return {
            "1_DCF_Operating_CF":   0.00,   # excluded — OCF distorted by receivables
            "2_DCF_FCF":            0.00,   # excluded
            "3_DCF_Net_Income":     0.40,   # PRIMARY — net income is the true metric
            "4_DCF_Terminal_Value": 0.00,   # excluded
            "5_Mean_PS":            0.15,   # secondary — revenue is stable
            "6_Mean_PE":            0.20,   # strong secondary
            "7_Mean_PB":            0.15,   # meaningful for financial companies
            "8_PSG":                0.10,
            "9_PEG":                0.00,   # excluded
            "10_EV_EBITDA":         0.00,   # excluded — not standard for financials
        }

    if company_type == "FCF_COMPOUNDER":
        # DCF methods primary — reliable FCF makes DCF most accurate.
        # All methods used but DCF heavily weighted.
        return {
            "1_DCF_Operating_CF":   0.20,   # PRIMARY
            "2_DCF_FCF":            0.20,   # PRIMARY
            "3_DCF_Net_Income":     0.08,
            "4_DCF_Terminal_Value": 0.18,   # PRIMARY
            "5_Mean_PS":            0.07,
            "6_Mean_PE":            0.08,
            "7_Mean_PB":            0.05,
            "8_PSG":                0.05,
            "9_PEG":                0.05,
            "10_EV_EBITDA":         0.04,
        }

    # STANDARD — use tier weights unchanged
    return base


# =============================================================================
# SOURCES
# =============================================================================
NYSE_URL  = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"
CORE_PRIORITY = [
    # Original top 10
    "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "BRK-B", "LLY", "AVGO",
    # S&P 500 large caps confirmed missing from compiled JSON
    "V", "DHR", "LOW", "SCHW", "T", "ITW",
    # Growth & Tech confirmed missing
    "PLTR", "SNOW", "MSTR", "UBER",
    # Energy confirmed missing
    "BKR",
    # Financials confirmed missing
    "C", "PRU", "KKR",
    # REITs confirmed missing
    "O", "EXR", "EQR", "VTR",
    # International ADRs confirmed missing
    "PBR",
]

# =============================================================================
# FINANCIAL MODELING PREP (FMP) CONFIG
#
# FMP provides up to 10 years of annual financial statement history —
# significantly more than yfinance which caps at ~4 years.
# Used exclusively for the Financials tab charts.
#
# Free API key: https://financialmodelingprep.com/developer/docs
# Paste your key below. If left empty the script falls back to yfinance.
# =============================================================================
FMP_API_KEY = "p9L0gFU6BZv1UKrKn1GxA92CX3LQghKz"

# =============================================================================
# TICKER LOADER
# =============================================================================
# Suffixes that indicate non-operating, non-investable securities.
# These cannot be valued with DCF or multiples and are excluded.
#   W  = warrant          (e.g. AACBW, NVAWW)
#   R  = rights           (e.g. AACBR)
#   U  = unit             (e.g. AACBU — SPAC unit = share + warrant)
#   WS = warrant series   (legacy suffix)
#   WW = double warrant   (e.g. BGLWW)
#   Z  = special series   (e.g. OUSTZ, HUBCZ)
EXCLUDED_SUFFIXES = ("W", "R", "U", "WS", "WW")

# Known non-DCF-compatible security types reported by yfinance quoteType.
# ETFs, closed-end funds, index funds etc. have no earnings or cash flow.
EXCLUDED_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX", "FUTURE", "OPTION",
                         "CURRENCY", "CRYPTOCURRENCY"}

def _is_junk_ticker(symbol: str) -> bool:
    """
    Returns True if the ticker should be excluded from valuation.
    Catches warrants, rights, units, and other non-operating securities
    by suffix pattern alone — no API call needed.
    """
    # Single-letter tickers are usually closed-end funds or preferred stock
    if len(symbol) == 1:
        return True
    # Warrants, rights and units by suffix
    for suffix in EXCLUDED_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix) + 1:
            return True
    # SPAC-style patterns: 4-5 uppercase letters, no numbers
    # SPACs are identified later if they have no revenue — not pre-filtered
    # here to avoid accidentally excluding real tickers
    return False


def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res  = requests.get(url)
            data = res.json()
            for item in data:
                symbol = (
                    item.get('Symbol') or
                    item.get('ACT Symbol') or
                    item.get('ticker')
                )
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers and not _is_junk_ticker(clean):
                        tickers.append(clean)
        except Exception as e:
            print(f"Could not read {url}: {e}")
    print(f"  Loaded {len(tickers)} tickers after filtering warrants/rights/units")
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
# HELPER — two-stage DCF summation
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
    Scores a chronological value series (oldest first) for CONSISTENT
    upward trend. Specifically designed to penalise companies that had
    one exceptional year followed by weakness — a pattern that looks
    good on simple decline-counting but signals poor predictability.

    Scoring method:
      1. Count YoY growth rate for each consecutive pair
      2. Calculate the percentage of years with positive growth
      3. Also check if the END value is meaningfully higher than the START
         (ensures the overall direction is up, not just volatile)

    max_pts=3:
      3 pts — ≥ 85% of years growing AND end > start × 1.2  (strong consistent)
      2 pts — ≥ 70% of years growing AND end > start        (mostly consistent)
      1 pt  — ≥ 50% of years growing OR end > start         (mixed but positive)
      0 pts — majority declining or end ≤ start              (declining trend)

    For short series (< 4 years) falls back to simple decline counting
    since there isn't enough data for percentage-based scoring.
    """
    if len(series) < 2:
        return 1

    # Short series — use simple approach
    if len(series) < 4:
        declines = sum(1 for i in range(1, len(series)) if series[i] < series[i-1])
        if declines == 0:              return max_pts
        elif declines == 1:            return max(0, max_pts - 1)
        return 0

    # Filter out zeros/negatives from growth rate calculation
    # (negative NI years are valid data — don't skip them)
    yoy_positive = 0
    yoy_total    = 0
    for i in range(1, len(series)):
        prev = series[i - 1]
        curr = series[i]
        yoy_total += 1
        if curr > prev:
            yoy_positive += 1

    pct_growing = yoy_positive / yoy_total

    # Overall direction — is the final value higher than the starting value?
    start = series[0]
    end   = series[-1]
    # Use absolute comparison for NI which can be negative
    if start != 0:
        overall_up      = end > start
        strong_overall  = end > start * 1.20   # at least 20% higher overall
    else:
        overall_up     = end > 0
        strong_overall = end > 0

    if pct_growing >= 0.85 and strong_overall:
        return max_pts          # strong consistent upward trend
    elif pct_growing >= 0.70 and overall_up:
        return max(1, max_pts - 1)   # mostly consistent
    elif pct_growing >= 0.50 or overall_up:
        return 1                # mixed but net positive direction
    return 0                    # declining or highly volatile

# =============================================================================
# FINVIZ GROWTH FETCHER
#
# Fetches "EPS Next 5Y" from Finviz — analyst consensus forward growth.
# More accurate than yfinance trailing earningsGrowth for DCF inputs.
#
# Finviz return value handling:
#   "12.50%"  → 0.125  (valid positive growth — use it)
#   "-"        → 0.0   (analysts forecast no growth — treat as EXACTLY zero,
#                        do NOT fall through to yfinance)
#   "N/A"      → fall through to yfinance
#   exception  → fall through to yfinance
#
# Fallback chain:
#   1. Finviz EPS next 5Y   (best  — forward analyst consensus)
#   2. yfinance earningsGrowth (ok — trailing, may be distorted)
#   3. MIN_GROWTH_RATE 2%   (floor — DCF still produces a value)
#
# PEG and PSG use growth_rate directly. When growth = 0.0 those methods
# return None (no valid Lynch-formula result at zero growth), which is
# correct — they are excluded from the weighted mean for that ticker.
#
# Returns:
#   {
#     "rate":   float   — rate to use in DCF (floored at MIN_GROWTH_RATE)
#     "peg_rate": float — rate to use in PEG/PSG (NOT floored, can be 0.0)
#     "source": str     — "finviz" | "finviz_no_growth" | "yfinance" | "default"
#     "raw":    float   — unclamped value
#   }
# =============================================================================
def get_forward_growth(ticker, info):
    # --- Attempt 1: Finviz EPS Next 5Y ---
    if FINVIZ_AVAILABLE:
        try:
            fundamentals = fvz(ticker).ticker_fundament()
            raw_str      = fundamentals.get("EPS next 5Y", "")

            # Explicit "-" means analysts forecast no growth — honour it
            if raw_str and raw_str.strip() == "-":
                return {
                    "rate":     MIN_GROWTH_RATE,  # floor so DCF doesn't collapse
                    "peg_rate": 0.0,              # PEG/PSG correctly get zero
                    "source":   "finviz_no_growth",
                    "raw":      0.0,
                }

            # Valid percentage string
            if raw_str and raw_str not in ("N/A", ""):
                raw     = float(raw_str.replace("%", "").strip()) / 100
                clamped = max(MIN_GROWTH_RATE, min(raw, MAX_GROWTH_RATE))
                return {
                    "rate":     clamped,
                    "peg_rate": min(raw, MAX_PEG_PSG_GROWTH),
                    "source":   "finviz",
                    "raw":      round(raw, 4),
                }
        except Exception:
            pass  # fall through to yfinance

    # --- Attempt 2: yfinance trailing earningsGrowth ---
    yf_growth = _safe(info.get("earningsGrowth"))
    if yf_growth is not None and yf_growth > 0:
        clamped = max(MIN_GROWTH_RATE, min(yf_growth, MAX_GROWTH_RATE))
        return {
            "rate":     clamped,
            "peg_rate": min(yf_growth, MAX_PEG_PSG_GROWTH),
            "source":   "yfinance",
            "raw":      round(yf_growth, 4),
        }

    # --- Attempt 3: default floor ---
    return {
        "rate":     MIN_GROWTH_RATE,
        "peg_rate": 0.0,
        "source":   "default",
        "raw":      0.0,
    }

# =============================================================================
# QUALITY METRIC 1 — Profitability (0–10)
# =============================================================================
def _score_label(score: int) -> str:
    """
    Convert a 0-10 quality score to a consistent written label.

    Used to pre-compute assessment text in the pipeline so AI Studio
    always reads the correct label rather than inferring it from the
    raw number (which caused mismatches like score 6 = "Average" when
    it should be "Good").

    Labels are calibrated against the actual scoring rubrics:
      10    → Excellent  — top-tier, near-perfect signals
      8–9   → Strong     — clearly above average, no major weaknesses
      6–7   → Good       — solid, meets or exceeds most benchmarks
      4–5   → Adequate   — meets minimum thresholds, some concerns
      2–3   → Weak       — below par, multiple weak signals
      0–1   → Poor       — significant deficiencies
    """
    if score is None:    return "Not scored"
    if score >= 10:      return "Excellent"
    if score >= 8:       return "Strong"
    if score >= 6:       return "Good"
    if score >= 4:       return "Adequate"
    if score >= 2:       return "Weak"
    return "Poor"


def score_profitability(info):
    """Routes to company-type-appropriate profitability scoring."""
    fin_type = _detect_financial_type(info)
    if fin_type == "BANK":
        return _score_prof_bank(info)
    elif fin_type in ("INSURANCE", "ASSET_MGR", "CONSUMER_FINANCE"):
        return _score_prof_financial(info)
    elif fin_type == "REIT":
        return _score_prof_reit(info)
    return _score_prof_standard(info)


def _score_prof_bank(info):
    """
    Bank profitability (0-10).
    ROE is the primary metric — it captures how efficiently the bank
    uses shareholder equity to generate profit, which is the core
    measure for banking businesses.
    ROA measures overall asset efficiency (standard bank KPI).
    NI consistency replaces FCF margin — banks don't report FCF.
    """
    score = 0
    # ROE (4 pts) — primary profitability metric for banks
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.15:   score += 4
        elif roe > 0.10: score += 3
        elif roe > 0.07: score += 2
        elif roe > 0.03: score += 1
    # ROA (3 pts) — asset efficiency, standard bank benchmark
    roa = _safe(info.get('returnOnAssets'))
    if roa is not None:
        if roa > 0.012:  score += 3   # top-tier bank (JPM ~1.2%)
        elif roa > 0.008: score += 2  # solid bank
        elif roa > 0.004: score += 1  # marginal
    # Net income positive and margin on revenue (3 pts)
    ni  = _safe(info.get('netIncomeToCommon'))
    rev = _safe(info.get('totalRevenue'))
    if ni and ni > 0 and rev and rev > 0:
        margin = ni / rev
        if margin > 0.20:   score += 3
        elif margin > 0.12: score += 2
        elif margin > 0:    score += 1
    return min(score, 10)


def _score_prof_financial(info):
    """
    Insurance / asset manager / consumer finance profitability (0-10).
    ROE primary, profit margin secondary, NI consistency tertiary.
    """
    score = 0
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.15:   score += 4
        elif roe > 0.10: score += 3
        elif roe > 0.06: score += 2
        elif roe > 0:    score += 1
    pm = _safe(info.get('profitMargins'))
    if pm is not None:
        if pm > 0.20:   score += 3
        elif pm > 0.10: score += 2
        elif pm > 0:    score += 1
    ni = _safe(info.get('netIncomeToCommon'))
    if ni and ni > 0:   score += 3
    elif ni == 0:       score += 1
    return min(score, 10)


def _score_prof_reit(info):
    """
    REIT profitability (0-10).
    REITs must distribute 90%+ of income so retained earnings and
    traditional margins are not meaningful. OCF (proxy for FFO) and
    dividend sustainability are the appropriate metrics.
    """
    score = 0
    # Operating cash flow as FFO proxy (4 pts)
    ocf = _safe(info.get('operatingCashflow'))
    rev = _safe(info.get('totalRevenue'))
    if ocf and ocf > 0 and rev and rev > 0:
        ocf_margin = ocf / rev
        if ocf_margin > 0.35:   score += 4
        elif ocf_margin > 0.20: score += 3
        elif ocf_margin > 0.10: score += 2
        elif ocf_margin > 0:    score += 1
    # ROE (3 pts)
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.08:   score += 3
        elif roe > 0.04: score += 2
        elif roe > 0:    score += 1
    # Dividend yield as distribution proxy (3 pts)
    div = _safe(info.get('dividendYield'))
    if div and div > 0.04:   score += 3
    elif div and div > 0.02: score += 2
    elif div and div > 0:    score += 1
    return min(score, 10)


def _score_prof_standard(info):
    """Original industrial/technology profitability scoring."""
    score = 0
    rev = _safe(info.get('totalRevenue'))
    ni  = _safe(info.get('netIncomeToCommon'))
    if rev and rev > 0 and ni is not None:
        margin = ni / rev
        if margin > 0.15:    score += 3
        elif margin > 0.05:  score += 2
        elif margin > 0:     score += 1
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.20:    score += 3
        elif roe > 0.10:  score += 2
        elif roe > 0:     score += 1
    fcf = _safe(info.get('freeCashflow'))
    if rev and rev > 0 and fcf is not None:
        fcf_margin = fcf / rev
        if fcf_margin > 0.12:   score += 2
        elif fcf_margin > 0.05: score += 1
    roa = _safe(info.get('returnOnAssets'))
    if roa is not None:
        if roa > 0.08:   score += 2
        elif roa > 0.03: score += 1
    return min(score, 10)

# =============================================================================
# EARLY FINANCIAL TYPE DETECTION
# Lightweight version using only info dict fields — no prelim_scores needed.
# Called before Stage 1a so score_financial_strength can use correct signals.
# The full detect_company_type() with prelim_scores still runs in Stage 4
# for valuation weighting decisions.
# =============================================================================
def _detect_financial_type(info):
    """
    Returns one of: "BANK", "INSURANCE", "REIT", "ASSET_MGR", or None.
    Used exclusively by score_financial_strength to select the right signals.
    """
    sector   = (info.get("sector")   or "").lower()
    industry = (info.get("industry") or "").lower()
    qtype    = (info.get("quoteType") or "").upper()

    if qtype == "REIT" or "reit" in industry or "real estate investment trust" in industry:
        return "REIT"

    bank_kw = [
        "bank", "savings", "thrift", "banking", "commercial bank",
        "regional bank", "mortgage bank", "diversified bank",
    ]
    if any(k in industry for k in bank_kw):
        return "BANK"

    ins_kw = [
        "insurance", "reinsurance", "surety", "title insurance",
    ]
    if any(k in industry for k in ins_kw):
        return "INSURANCE"

    # Consumer finance, credit services, payment networks, specialty finance
    # AXP = "Credit Services", Visa/MA = "Diversified Financial Services"
    # Capital One = "Credit Services", SLM = "Financial Services"
    consumer_fin_kw = [
        "credit services", "consumer finance", "credit card",
        "diversified financial services", "specialty finance",
        "financial services", "payment processing", "payments",
        "transaction processing", "personal finance", "mortgage finance",
        "student loan", "auto loan", "consumer lending",
    ]
    if any(k in industry for k in consumer_fin_kw):
        return "CONSUMER_FINANCE"

    mgr_kw = [
        "asset management", "investment management", "capital markets",
        "wealth management", "investment bank", "financial advisory",
        "brokerage", "securities",
    ]
    if any(k in industry for k in mgr_kw):
        return "ASSET_MGR"

    # Catch-all: anything in financial services sector not already matched
    if sector == "financial services":
        return "CONSUMER_FINANCE"

    return None


# =============================================================================
# QUALITY METRIC 2 — Financial Strength (0–10)
#
# Uses the same three pre-computed ratios displayed in the Ratios tab so
# the quality score and the displayed metrics are always consistent.
#
# Current Ratio       (3 pts) — liquidity
#   > 2.0  → 3   excellent short-term coverage
#   > 1.2  → 2   adequate coverage
#   > 0.8  → 1   tight but manageable
#   ≤ 0.8  → 0   potential liquidity risk
#
# Debt-to-EBITDA      (3 pts) — leverage
#   < 1.0  → 3   minimal leverage / debt-free
#   < 3.0  → 2   moderate leverage
#   < 5.0  → 1   elevated leverage
#   ≥ 5.0  → 0   high leverage risk
#
# Debt Servicing Ratio (2 pts) — ability to service debt from operations
#   < 0.10 → 2   interest is trivial vs operating cash flow
#   < 0.25 → 1   manageable interest burden
#   ≥ 0.25 → 0   interest is a significant drain
#   = 0.0  → 2   no interest expense (debt-free)
#
# Positive FCF        (2 pts) — free cash flow generation
#   FCF > 0 and OCF > 0 → 2
#   FCF > 0 only        → 1
# =============================================================================
def score_financial_strength(info,
                              current_ratio=None,
                              debt_to_ebitda=None,
                              debt_service_ratio=None,
                              cash_debt_history=None):
    """
    Routes to a company-type-appropriate scoring path.
    Financial companies (banks, insurance, REITs, asset managers) use
    completely different signals than industrial companies — applying
    standard debt/cash metrics to them produces meaninglessly low scores.
    """
    fin_type = _detect_financial_type(info)

    if fin_type == "BANK":
        return _score_fs_bank(info)
    elif fin_type == "INSURANCE":
        return _score_fs_insurance(info)
    elif fin_type == "REIT":
        return _score_fs_reit(info, cash_debt_history)
    elif fin_type == "ASSET_MGR":
        return _score_fs_asset_mgr(info)
    elif fin_type == "CONSUMER_FINANCE":
        return _score_fs_consumer_finance(info)
    else:
        return _score_fs_standard(info, current_ratio, debt_to_ebitda,
                                   debt_service_ratio, cash_debt_history)


def _score_fs_bank(info):
    """
    Bank financial strength (0-10).
    Standard debt metrics are meaningless for banks — deposits are
    liabilities by design and interest expense is core revenue cost.

    Bank-appropriate signals:
      ROE (3 pts)          — profitability of equity capital
        > 12% → 3   strong returns on shareholder equity
        > 8%  → 2   adequate
        > 4%  → 1   marginal
      Net Interest Margin proxy (2 pts)
        Proxied from net income / total assets (ROA)
        > 1%  → 2   healthy ROA for a bank
        > 0.5%→ 1
      P/B ratio (2 pts)   — market view of book value quality
        < 1.5 → 2   trading near or below book (conservative valuation)
        < 2.5 → 1
      Positive net income (2 pts)
        NI > 0 AND growing → 2
        NI > 0 only        → 1
      Dividend coverage (1 pt)
        Paying a dividend → 1 (banks that pay divs are typically profitable)
    """
    score = 0

    # ROE
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.12:   score += 3
        elif roe > 0.08: score += 2
        elif roe > 0.04: score += 1

    # ROA proxy for net interest margin
    ni  = _safe(info.get('netIncomeToCommon'))
    ta  = _safe(info.get('totalAssets'))
    if ni and ta and ta > 0:
        roa = ni / ta
        if roa > 0.01:    score += 2
        elif roa > 0.005: score += 1

    # P/B ratio
    pb = _safe(info.get('priceToBook'))
    if pb is not None and pb > 0:
        if pb < 1.5:   score += 2
        elif pb < 2.5: score += 1

    # Positive net income
    if ni is not None:
        if ni > 0:   score += 2
        # Note: can't check NI growth here without historical data
        # AI Studio's moat score will capture franchise strength

    # Dividend (proxy for sustained profitability)
    div = _safe(info.get('dividendYield'))
    if div is not None and div > 0:
        score += 1

    return min(score, 10)


def _score_fs_insurance(info):
    """
    Insurance company financial strength (0-10).
    FCF and standard debt ratios are unreliable due to float and reserves.

    Insurance-appropriate signals:
      ROE (3 pts)          — returns on equity capital
      Net income positive (2 pts) — underwriting profitability
      P/B ratio (2 pts)   — below book suggests good reserve management
      Revenue stability (2 pts) — premium revenue is stable by nature
      Dividend coverage (1 pt)
    """
    score = 0

    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.12:   score += 3
        elif roe > 0.08: score += 2
        elif roe > 0.04: score += 1

    ni = _safe(info.get('netIncomeToCommon'))
    if ni is not None:
        if ni > 0:   score += 2
        elif ni == 0: score += 0

    pb = _safe(info.get('priceToBook'))
    if pb is not None and pb > 0:
        if pb < 1.5:   score += 2
        elif pb < 2.5: score += 1

    # Revenue — insurance premiums are stable and recurring
    rev = _safe(info.get('totalRevenue'))
    rev_growth = _safe(info.get('revenueGrowth'))
    if rev and rev > 0:
        if rev_growth is not None and rev_growth > 0:
            score += 2
        else:
            score += 1

    div = _safe(info.get('dividendYield'))
    if div is not None and div > 0:
        score += 1

    return min(score, 10)


def _score_fs_reit(info, cash_debt_history=None):
    """
    REIT financial strength (0-10).
    High debt is structural for REITs — they are required to distribute
    90%+ of taxable income, so they must use debt for growth.
    Signals focus on debt serviceability and asset quality.

      Dividend yield / payout sustainability (3 pts)
        Yield > 3% → 2, yield > 1% → 1
        FFO payout < 80% → +1 (conservative payout)
      P/B (NAV proxy) (3 pts)
        < 1.0 → 3  (trading below NAV — potential value)
        < 1.5 → 2
        < 2.0 → 1
      Positive net income or FFO proxy (2 pts)
      Revenue stability (2 pts)
    """
    score = 0

    # Dividend yield — REITs must distribute income
    div_yield = _safe(info.get('dividendYield'))
    if div_yield is not None:
        if div_yield > 0.04:   score += 2
        elif div_yield > 0.01: score += 1

    # P/B as NAV proxy
    pb = _safe(info.get('priceToBook'))
    if pb is not None and pb > 0:
        if pb < 1.0:   score += 3
        elif pb < 1.5: score += 2
        elif pb < 2.0: score += 1

    # Net income / FFO proxy
    ni = _safe(info.get('netIncomeToCommon'))
    ocf = _safe(info.get('operatingCashflow'))
    if ocf and ocf > 0:
        score += 2   # positive OCF is the REIT equivalent of FCF
    elif ni and ni > 0:
        score += 1

    # Revenue growth
    rev_growth = _safe(info.get('revenueGrowth'))
    if rev_growth is not None:
        if rev_growth > 0.05:  score += 2
        elif rev_growth > 0:   score += 1

    return min(score, 10)


def _score_fs_asset_mgr(info):
    """
    Asset manager / broker financial strength (0-10).
    Revenue is fee-based and recurring — focus on profitability and margins.
    """
    score = 0

    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.15:   score += 3
        elif roe > 0.08: score += 2
        elif roe > 0.04: score += 1

    # Profit margin
    pm = _safe(info.get('profitMargins'))
    if pm is not None:
        if pm > 0.20:   score += 2
        elif pm > 0.10: score += 1

    ni = _safe(info.get('netIncomeToCommon'))
    if ni and ni > 0:
        score += 2

    rev_growth = _safe(info.get('revenueGrowth'))
    if rev_growth and rev_growth > 0.05:
        score += 2
    elif rev_growth and rev_growth > 0:
        score += 1

    div = _safe(info.get('dividendYield'))
    if div and div > 0:
        score += 1

    return min(score, 10)


def _score_fs_consumer_finance(info):
    """
    Consumer finance / credit services / payments financial strength (0-10).
    Companies like AXP, Visa, Mastercard, Capital One, Discover.

    High debt is structural — these businesses fund loan books and card
    receivables with debt by design. Standard debt ratios produce
    misleadingly low scores. Appropriate signals:

      ROE (3 pts)          — core measure of profitability
        > 20% → 3   exceptional (AXP, Visa typically 30%+)
        > 12% → 2   strong
        > 6%  → 1   adequate
      Net income positive (2 pts)
        NI > 0 and profit margin > 15% → 2
        NI > 0 only → 1
      Revenue growth (2 pts)
        > 8%  → 2   strong top-line growth
        > 0%  → 1   growing
      P/B ratio (2 pts)
        — high P/B is expected for these asset-light models
        < 5   → 2   reasonable for payment networks/credit cos
        < 10  → 1
      Dividend (1 pt)
    """
    score = 0

    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.20:   score += 3
        elif roe > 0.12: score += 2
        elif roe > 0.06: score += 1

    ni  = _safe(info.get('netIncomeToCommon'))
    pm  = _safe(info.get('profitMargins'))
    if ni and ni > 0:
        if pm and pm > 0.15: score += 2
        else:                score += 1

    rev_growth = _safe(info.get('revenueGrowth'))
    if rev_growth is not None:
        if rev_growth > 0.08:  score += 2
        elif rev_growth > 0:   score += 1

    pb = _safe(info.get('priceToBook'))
    if pb is not None and pb > 0:
        if pb < 5:    score += 2
        elif pb < 10: score += 1

    div = _safe(info.get('dividendYield'))
    if div and div > 0:
        score += 1

    return min(score, 10)


def _score_fs_standard(info, current_ratio=None, debt_to_ebitda=None,
                        debt_service_ratio=None, cash_debt_history=None):
    """
    Standard (industrial / technology / consumer) financial strength (0-10).
    Original scoring logic — appropriate for non-financial companies.
    """
    score = 0

    # ── Current Ratio (2 pts) ─────────────────────────────────────────────────
    cr = current_ratio if current_ratio is not None else _safe(info.get('currentRatio'))
    if cr is not None:
        if cr > 2.0:    score += 2
        elif cr > 1.2:  score += 1

    # ── Debt-to-EBITDA (2 pts) ────────────────────────────────────────────────
    if debt_to_ebitda is not None:
        if debt_to_ebitda == 0:    score += 2
        elif debt_to_ebitda < 1:   score += 2
        elif debt_to_ebitda < 3:   score += 1
    else:
        debt   = _safe(info.get('totalDebt'))
        ebitda = _safe(info.get('ebitda'))
        if debt is not None and ebitda and ebitda > 0:
            ratio = debt / ebitda
            if ratio == 0:   score += 2
            elif ratio < 1:  score += 2
            elif ratio < 3:  score += 1

    # ── Debt Servicing Ratio (2 pts) ──────────────────────────────────────────
    if debt_service_ratio is not None:
        if debt_service_ratio == 0:     score += 2
        elif debt_service_ratio < 0.10: score += 2
        elif debt_service_ratio < 0.25: score += 1
    else:
        ebitda   = _safe(info.get('ebitda'))
        interest = _safe(info.get('interestExpense'))
        if ebitda and ebitda > 0 and interest and interest > 0:
            coverage = ebitda / abs(interest)
            if coverage > 10:  score += 2
            elif coverage > 5: score += 1

    # ── Net Cash Position vs Debt trend (2 pts) ───────────────────────────────
    cash_signal_scored = False
    if cash_debt_history and len(cash_debt_history) >= 3:
        try:
            sorted_yrs = sorted(cash_debt_history.keys())
            cash_vals  = [cash_debt_history[y].get("cash_balance") for y in sorted_yrs]
            debt_vals  = [cash_debt_history[y].get("total_debt")   for y in sorted_yrs]
            pairs = [(c, d) for c, d in zip(cash_vals, debt_vals)
                     if c is not None and d is not None]
            if pairs:
                latest_cash, latest_debt = pairs[-1]
                cash_exceeds_debt = latest_cash >= latest_debt
                cash_improving = (
                    pairs[-1][0] > pairs[0][0] or
                    pairs[-1][1] < pairs[0][1] or
                    (pairs[-1][0] - pairs[-1][1]) > (pairs[0][0] - pairs[0][1])
                )
                if cash_exceeds_debt and cash_improving:
                    score += 2
                elif cash_exceeds_debt or cash_improving:
                    score += 1
                cash_signal_scored = True
        except Exception:
            pass

    if not cash_signal_scored:
        # Apple and similar companies hold large short-term investment
        # portfolios alongside cash. totalCash alone (~$29B for AAPL)
        # understates liquidity — add shortTermInvestments to capture the
        # full liquid position (~$135B for AAPL = $29B + $106B).
        cash_raw = _safe(info.get('totalCash')) or 0
        mkt_sec  = _safe(info.get('shortTermInvestments')) or 0
        cash_snap = (cash_raw + mkt_sec) or None
        if not cash_snap:
            cash_snap = (_safe(info.get('cashAndCashEquivalents')) or
                         _safe(info.get('cashAndShortTermInvestments')))
        debt_snap = _safe(info.get('totalDebt'))
        if cash_snap and debt_snap is not None:
            if cash_snap > debt_snap:
                score += 2
            elif cash_snap > debt_snap * 0.5:
                score += 1

    # ── Positive FCF (2 pts) ──────────────────────────────────────────────────
    fcf = _safe(info.get('freeCashflow'))
    ocf = _safe(info.get('operatingCashflow'))
    if fcf is not None and fcf > 0 and ocf is not None and ocf > 0:
        score += 2
    elif fcf is not None and fcf > 0:
        score += 1

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 3 — Growth (0–10)
#
# Now incorporates three KPI signals alongside the trailing YoY metrics:
#
# EPS Next 5Y  (2 pts) — forward analyst consensus growth (from Finviz)
#   > 15%  → 2   strong forward growth expected
#   > 5%   → 1   moderate forward growth expected
#   ≤ 5%   → 0   low or no growth expected
#
# ROE          (2 pts) — return on equity (efficiency of retained earnings)
#   > 20%  → 2   high-quality capital allocation
#   > 10%  → 1   decent returns
#   ≤ 10%  → 0   poor capital efficiency
#
# ROIC         (2 pts) — return on invested capital (all capital, not just equity)
#   > 15%  → 2   excellent returns on total capital
#   > 8%   → 1   acceptable returns
#   ≤ 8%   → 0   poor capital deployment
#
# Revenue YoY  (2 pts — trailing) and Earnings YoY (2 pts — trailing) retained
# but capped at 2 pts each to balance forward vs trailing signals.
# Total remains capped at 10.
# =============================================================================
def score_growth(info, financials,
                 eps_next_5y=None, roe=None, roic=None):
    score = 0

    # ── EPS Next 5Y — forward analyst consensus (2 pts) ──────────────────────
    # Uses Finviz forward growth rate if available, falls back to yfinance
    if eps_next_5y is not None:
        if eps_next_5y > 0.15:   score += 2
        elif eps_next_5y > 0.05: score += 1
    else:
        # Fallback to trailing earningsGrowth if forward rate not passed
        eg = _safe(info.get('earningsGrowth'))
        if eg is not None:
            if eg > 0.15:   score += 2
            elif eg > 0.05: score += 1

    # ── ROE — Return on Equity (2 pts) ────────────────────────────────────────
    roe_val = roe if roe is not None else _safe(info.get('returnOnEquity'))
    if roe_val is not None:
        if roe_val > 0.20:   score += 2
        elif roe_val > 0.10: score += 1

    # ── ROIC — Return on Invested Capital (2 pts) ─────────────────────────────
    if roic is not None:
        if roic > 0.15:   score += 2
        elif roic > 0.08: score += 1

    # ── Revenue YoY — trailing (2 pts) ────────────────────────────────────────
    rev_growth = _safe(info.get('revenueGrowth'))
    if rev_growth is not None:
        if rev_growth > 0.15:   score += 2
        elif rev_growth > 0.05: score += 1
        elif rev_growth > 0:    score += 1

    # ── Revenue trend from financials (1 pt) ──────────────────────────────────
    if financials is not None and not financials.empty:
        rev_series = _get_annual_series(financials, ['Total Revenue', 'TotalRevenue'])
        if rev_series:
            score += _trend_score(rev_series[-4:], max_pts=1)

    return min(score, 10)

# =============================================================================
# QUALITY METRIC 4 — Predictability (0–10)
#
# Now uses the same 10-year revenue and net income series that powers the
# 10Y Revenue & Net Income chart in the Financials tab. EDGAR provides up
# to 10 years vs ~4 years from yfinance — giving a much more meaningful
# measure of long-term consistency.
#
# Long-term revenue trend   (3 pts) — is revenue consistently growing?
#   All years up            → 3
#   One declining year      → 2
#   Mixed but mostly up     → 1
#   Two+ declining years    → 0
#
# Long-term NI trend        (3 pts) — is net income consistently growing?
#   Same bands as revenue
#
# Margin stability          (2 pts) — are profit margins predictable?
#   Std dev of NI/Rev < 5%  → 2   (very stable margins)
#   Std dev < 10%           → 1
#   ≥ 10%                   → 0
#
# FCF consistency           (2 pts) — is free cash flow reliably positive?
#   FCF > 0 and OCF > 0    → 2
#   FCF > 0 only           → 1
# =============================================================================
def score_predictability(info, financials,
                          long_rev_series=None,
                          long_ni_series=None):
    """
    Routes financial companies to simpler stability-based scoring.
    Banks and insurers have structurally volatile NI due to credit cycles
    and reserve requirements — penalising them for this volatility is unfair.
    """
    fin_type = _detect_financial_type(info)
    if fin_type in ("BANK", "INSURANCE", "CONSUMER_FINANCE",
                    "ASSET_MGR", "REIT"):
        return _score_pred_financial(info)
    return _score_pred_standard(info, financials,
                                 long_rev_series, long_ni_series)


def _score_pred_financial(info):
    """
    Financial company predictability (0-10).
    Rather than penalising for NI volatility (which is structural in
    credit cycles and insurance), scores on business model stability signals:
      - Years in operation proxy (market cap rank stability)
      - Dividend consistency (only reliable predictor of stable earnings)
      - Revenue growth direction
      - ROE consistency proxy
    """
    score = 0
    # Dividend as earnings stability proxy (4 pts)
    # Companies that have paid consistent dividends for years must have
    # had stable enough earnings to support distributions
    div = _safe(info.get('dividendYield'))
    div_rate = _safe(info.get('dividendRate'))
    if div and div > 0 and div_rate and div_rate > 0:
        score += 4   # paying a dividend = evidence of earnings stability
    elif div and div > 0:
        score += 2
    # ROE positive and reasonable (3 pts)
    roe = _safe(info.get('returnOnEquity'))
    if roe is not None:
        if roe > 0.10:   score += 3
        elif roe > 0.05: score += 2
        elif roe > 0:    score += 1
    # Revenue growth direction (2 pts)
    rev_g = _safe(info.get('revenueGrowth'))
    if rev_g is not None:
        if rev_g > 0.05:  score += 2
        elif rev_g >= 0:  score += 1
    # Positive NI (1 pt) — just confirm it's profitable
    ni = _safe(info.get('netIncomeToCommon'))
    if ni and ni > 0:
        score += 1
    return min(score, 10)


def _score_pred_standard(info, financials,
                          long_rev_series=None,
                          long_ni_series=None):
    """Original EDGAR trend-based predictability for non-financial companies."""
    score = 0

    # Use long-term EDGAR series if available, fall back to stock.financials
    if long_rev_series and len(long_rev_series) >= 2:
        rev_series = long_rev_series
    elif financials is not None and not financials.empty:
        rev_series = _get_annual_series(financials, ['Total Revenue', 'TotalRevenue'])
    else:
        rev_series = []

    if long_ni_series and len(long_ni_series) >= 2:
        ni_series = long_ni_series
    elif financials is not None and not financials.empty:
        ni_series = _get_annual_series(financials, ['Net Income', 'NetIncome'])
    else:
        ni_series = []

    # ── Long-term revenue trend (3 pts) ──────────────────────────────────────
    if rev_series:
        score += _trend_score(rev_series, max_pts=3)

    # ── Long-term net income trend (3 pts) ────────────────────────────────────
    if ni_series:
        score += _trend_score(ni_series, max_pts=3)

    # ── Profit margin stability (2 pts) ───────────────────────────────────────
    if rev_series and ni_series:
        min_len = min(len(rev_series), len(ni_series))
        margins = []
        for i in range(min_len):
            if rev_series[i] and rev_series[i] > 0:
                margins.append(ni_series[i] / rev_series[i])
        if len(margins) >= 3:   # need at least 3 years for meaningful variance
            variance = float(np.std(margins))
            if variance < 0.05:   score += 2
            elif variance < 0.10: score += 1

    # ── FCF consistency (2 pts) ───────────────────────────────────────────────
    fcf = _safe(info.get('freeCashflow'))
    ocf = _safe(info.get('operatingCashflow'))
    if fcf is not None and fcf > 0 and ocf is not None and ocf > 0:
        score += 2
    elif fcf is not None and fcf > 0:
        score += 1

    return min(score, 10)

# (score_predictability routes to _score_pred_financial or _score_pred_standard above)

# =============================================================================
# QUALITY METRIC 5 — Valuation (0–10)
# =============================================================================
def score_valuation(info, intrinsic_value):
    """
    Valuation quality score 0-10.

    MOS (Margin of Safety) bands — 6 pts max:
      > 40% undervalued  → 6  (exceptional value)
      > 20% undervalued  → 5  (clear margin of safety)
      > 0%  undervalued  → 4  (fairly valued, slight discount)
      0–10% overvalued   → 3  (marginally above fair value — common for
                                quality compounders like AAPL, MSFT)
      10–25% overvalued  → 2  (moderate premium)
      25–40% overvalued  → 1  (significant premium)
      > 40% overvalued   → 0  (severely overvalued)

    The 0-10% and 10-25% bands are the key change vs the old scoring.
    Previously a stock 23% overvalued scored the same as one 39% overvalued.
    Now there is meaningful separation between "slight premium" and
    "significant premium" — a great company like Apple at a modest premium
    to our IV model still scores 2 rather than 1.

    Supporting signals — 4 pts max:
      P/E < 15  → 2 pts    P/E 15-40 → 1 pt
      P/S < 2   → 1 pt
      P/B < 1   → 1 pt     (removed P/B < 5 band — too easy to score)
    """
    score = 0
    current_price = (
        _safe(info.get('currentPrice')) or
        _safe(info.get('regularMarketPrice'))
    )
    if current_price and current_price > 0 and intrinsic_value and intrinsic_value > 0:
        mos = (intrinsic_value - current_price) / intrinsic_value
        if mos > 0.40:    score += 6   # exceptional discount  e.g. 40%+ undervalued
        elif mos > 0.20:  score += 5   # clear margin of safety e.g. 20-40% undervalued
        elif mos > 0:     score += 4   # slight discount         e.g. 0-20% undervalued
        elif mos > -0.30: score += 3   # modest premium          e.g. 0-30% overvalued
        elif mos > -0.50: score += 2   # significant premium     e.g. 30-50% overvalued
        elif mos > -0.70: score += 1   # heavily overvalued      e.g. 50-70% overvalued
        # > 70% overvalued → 0

    # P/E support (2 pts max)
    pe = _safe(info.get('trailingPE'))
    if pe is not None and pe > 0:
        if pe < 15:   score += 2
        elif pe < 40: score += 1

    # P/S support (1 pt)
    ps = _safe(info.get('priceToSalesTrailing12Months'))
    if ps is not None and ps < 2:
        score += 1

    # P/B support (1 pt — only for genuinely cheap book value)
    pb = _safe(info.get('priceToBook'))
    if pb is not None and 0 < pb < 1:
        score += 1

    return min(score, 10)

# =============================================================================
# QUALITY CLASSIFICATION
# =============================================================================
def classify_quality(scores: dict, subtotal_pct: float,
                      roe: float = None, roic: float = None,
                      eps_next_5y: float = None) -> dict:
    """
    Classify investment quality using 5 core metrics when moat is available
    (profitability, financial_strength, growth, predictability, moat) or
    4 metrics when moat is pending AI Studio scoring.

    Valuation is excluded — business quality reflects fundamental strength,
    not whether the stock is cheap today.

    Thresholds:
      ≥ 70%    → Safe
      65–69%   → Grey zone: Safe if ROE ≥ 13% AND ROIC ≥ 13%, or
                 any two of (ROE ≥ 13%, ROIC ≥ 13%, EPS ≥ 10%);
                 else Speculative
      55–64%   → Speculative
      50–54%   → Borderline: Speculative if any two of (ROE ≥ 8%,
                 ROIC ≥ 8%, EPS ≥ 8%); else Dangerous
      ≤ 49%    → Dangerous

    Moat overrides (when AI Studio has scored moat):
      Safe override:      moat ≥ 7 AND (profitability ≥ 7 OR fs ≥ 7)
                          AND subtotal_pct ≥ 55
      Dangerous override: moat ≤ 3 AND fs ≤ 4 AND subtotal_pct < 45
    """
    p   = scores.get('profitability', 0) or 0
    fs  = scores.get('financial_strength', 0) or 0
    g   = scores.get('growth', 0) or 0
    pre = scores.get('predictability', 0) or 0
    m   = scores.get('moat')

    label = None

    # Moat-based overrides
    if m is not None:
        if m >= 7 and (p >= 7 or fs >= 7) and subtotal_pct >= 55:
            label = "Safe"
        if label is None and m <= 3 and fs <= 4 and subtotal_pct < 45:
            label = "Dangerous"

    # Score-based classification
    if label is None:
        if subtotal_pct >= 70:
            label = "Safe"

        elif subtotal_pct >= 65:
            # Grey zone 65-69%: tilt Safe if any two of ROE ≥ 13%,
            # ROIC ≥ 13%, EPS Next 5Y ≥ 10%
            tb = 0
            if roe       is not None and roe       >= 0.13: tb += 1
            if roic      is not None and roic      >= 0.13: tb += 1
            if eps_next_5y is not None and eps_next_5y >= 0.10: tb += 1
            label = "Safe" if tb >= 2 else "Speculative"

        elif subtotal_pct >= 55:
            # Speculative band 55-64%
            label = "Speculative"

        elif subtotal_pct >= 50:
            # Borderline 50-54%: Speculative only if any two of
            # ROE ≥ 8%, ROIC ≥ 8%, EPS Next 5Y ≥ 8%; else Dangerous
            tb = 0
            if roe       is not None and roe       >= 0.08: tb += 1
            if roic      is not None and roic      >= 0.08: tb += 1
            if eps_next_5y is not None and eps_next_5y >= 0.08: tb += 1
            label = "Speculative" if tb >= 2 else "Dangerous"

        else:
            # ≤ 49% → Dangerous
            label = "Dangerous"

    penalty = 0
    if label == "Speculative": penalty = 3
    elif label == "Dangerous": penalty = 8

    return {
        "label":           label,
        "penalty_pct":     penalty,
        "final_score_pct": max(0.0, round(subtotal_pct - penalty, 2)),
        "tiebreaker_used": (50 <= subtotal_pct < 70 and m is None),
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
# =============================================================================
def calc_roe_roic(info, financials, balance):
    result = {
        "roe":         None,
        "roic":        None,
        "roe_pct":     None,
        "roic_pct":    None,
        "roic_source": "unavailable",
    }
    roe = _safe(info.get("returnOnEquity"))
    if roe is not None:
        result["roe"]     = round(roe, 4)
        result["roe_pct"] = f"{round(roe * 100, 2)}%"
    try:
        if financials is None or financials.empty: return result
        if balance    is None or balance.empty:    return result
        fin_col = financials.columns[0]
        bal_col = balance.columns[0]
        op_inc  = None
        for name in ["Operating Income", "OperatingIncome",
                     "Total Operating Income As Reported"]:
            if name in financials.index:
                op_inc = _safe(float(financials.loc[name, fin_col]))
                break
        if op_inc is None or op_inc <= 0:
            return result
        tax_rate = 0.21
        try:
            pretax  = None
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
        nopat        = op_inc * (1 - tax_rate)
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
        cash = (
            _safe(float(balance.loc["Cash And Cash Equivalents", bal_col]))
            if "Cash And Cash Equivalents" in balance.index else
            _safe(float(balance.loc["CashAndCashEquivalents", bal_col]))
            if "CashAndCashEquivalents" in balance.index else
            _safe(info.get("totalCash"), 0)
        )
        rev          = _safe(info.get("totalRevenue"), 0)
        excess_cash  = max(0, (cash or 0) - 0.02 * (rev or 0))
        if total_assets is None or current_liab is None:
            return result
        invested_capital = total_assets - current_liab - excess_cash
        if invested_capital <= 0:
            return result
        roic                  = nopat / invested_capital
        result["roic"]        = round(roic, 4)
        result["roic_pct"]    = f"{round(roic * 100, 2)}%"
        result["roic_source"] = "calculated"
    except Exception:
        pass
    return result

# =============================================================================
# HISTORICAL MEAN MULTIPLES  (3–5yr average P/E, P/S, P/B and EV/EBITDA)
#
# EV/EBITDA uses the company's own historical average multiple — NOT a sector
# median — so the valuation reflects what the market has historically been
# willing to pay for THIS company's earnings power specifically.
# Capped at MAX_EV_EBITDA (40x) to filter distorted years.
# =============================================================================
def get_historical_mean_multiples(stock, shares):
    result = {
        "mean_pe":       None,
        "mean_ps":       None,
        "mean_pb":       None,
        "mean_ev_ebitda": None,
    }
    try:
        financials = stock.financials
        balance    = stock.balance_sheet
        hist       = stock.history(period="10y")

        if financials is None or financials.empty: return result
        if hist is None or hist.empty:             return result

        pe_list, ps_list, pb_list, ev_ebitda_list = [], [], [], []
        cols = list(financials.columns)[:HISTORY_YEARS]

        for col in cols:
            year_hist = hist[hist.index.year == col.year]
            if year_hist.empty: continue
            price = float(year_hist['Close'].iloc[-1])
            if price <= 0: continue

            # Mean P/E
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
                        if pe <= MAX_PE:
                            pe_list.append(pe)
            except Exception: pass

            # Mean P/S
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
                        if ps <= MAX_PS:
                            ps_list.append(ps)
            except Exception: pass

            # Mean P/B
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
                            if pb <= MAX_PB:
                                pb_list.append(pb)
            except Exception: pass

            # Historical EV/EBITDA — company's own multiple
            # EV = Market Cap + Total Debt - Cash
            # We reconstruct market cap from price × shares for that year.
            try:
                if balance is not None and not balance.empty and col in balance.columns:
                    ebitda_row = None
                    for name in ["EBITDA", "Ebitda"]:
                        if name in financials.index:
                            ebitda_row = financials.loc[name]
                            break
                    # EBITDA not always in financials — derive from Operating Income + D&A
                    if ebitda_row is None:
                        op_row = (
                            financials.loc['Operating Income'] if 'Operating Income' in financials.index
                            else financials.loc['OperatingIncome'] if 'OperatingIncome' in financials.index
                            else None
                        )
                        da_row = (
                            financials.loc['Reconciled Depreciation'] if 'Reconciled Depreciation' in financials.index
                            else financials.loc['Depreciation And Amortization'] if 'Depreciation And Amortization' in financials.index
                            else None
                        )
                        if op_row is not None and da_row is not None:
                            ebitda_val = float(op_row[col]) + abs(float(da_row[col]))
                        elif op_row is not None:
                            ebitda_val = float(op_row[col])
                        else:
                            ebitda_val = None
                    else:
                        ebitda_val = float(ebitda_row[col])

                    if ebitda_val and ebitda_val > 0:
                        debt_row = (
                            balance.loc['Total Debt'] if 'Total Debt' in balance.index
                            else balance.loc['TotalDebt'] if 'TotalDebt' in balance.index
                            else balance.loc['Long Term Debt'] if 'Long Term Debt' in balance.index
                            else None
                        )
                        cash_row = (
                            balance.loc['Cash And Cash Equivalents'] if 'Cash And Cash Equivalents' in balance.index
                            else balance.loc['CashAndCashEquivalents'] if 'CashAndCashEquivalents' in balance.index
                            else None
                        )
                        debt_val = float(debt_row[col]) if debt_row is not None else 0
                        cash_val = float(cash_row[col]) if cash_row is not None else 0
                        mkt_cap  = price * shares
                        ev       = mkt_cap + debt_val - cash_val
                        if ev > 0:
                            ev_ebitda = ev / ebitda_val
                            if 0 < ev_ebitda <= MAX_EV_EBITDA:
                                ev_ebitda_list.append(ev_ebitda)
            except Exception: pass

        if pe_list:       result["mean_pe"]        = round(float(np.mean(pe_list)), 4)
        if ps_list:       result["mean_ps"]        = round(float(np.mean(ps_list)), 4)
        if pb_list:       result["mean_pb"]        = round(float(np.mean(pb_list)), 4)
        if ev_ebitda_list: result["mean_ev_ebitda"] = round(float(np.mean(ev_ebitda_list)), 4)

    except Exception: pass
    return result

# =============================================================================
# VALUATION METHODS 1–10
# =============================================================================
def calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount):
    ocf = _safe(info.get('operatingCashflow'))
    if ocf is None or shares <= 0: return None
    g1 = min(growth_s1, MAX_DCF_GROWTH_RATE)
    g2 = min(growth_s2, MAX_DCF_GROWTH_RATE * 0.5)
    return _dcf_sum(ocf / shares, g1, g2, discount)

def calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount):
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0: return None
    g1 = min(growth_s1, MAX_DCF_GROWTH_RATE)
    g2 = min(growth_s2, MAX_DCF_GROWTH_RATE * 0.5)
    return _dcf_sum(fcf / shares, g1, g2, discount)

def calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount):
    ni = _safe(info.get('netIncomeToCommon'))
    if ni is None or shares <= 0: return None
    g1 = min(growth_s1, MAX_DCF_GROWTH_RATE)
    g2 = min(growth_s2, MAX_DCF_GROWTH_RATE * 0.5)
    return _dcf_sum(ni / shares, g1, g2, discount)

def calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal_growth, dcf_fcf_value=None):
    fcf = _safe(info.get('freeCashflow'))
    if fcf is None or shares <= 0: return None
    fcf_ps = fcf / shares
    if fcf_ps <= 0: return None
    g1 = min(growth_s1, MAX_DCF_GROWTH_RATE)
    g2 = min(growth_s2, MAX_DCF_GROWTH_RATE * 0.5)
    pv_cf  = _dcf_sum(fcf_ps, g1, g2, discount)
    cf_end = fcf_ps
    for t in range(1, DCF_YEARS + 1):
        cf_end = cf_end * (1 + (g1 if t <= 10 else g2))
    if discount <= terminal_growth: return None
    pv_tv  = ((cf_end * (1 + terminal_growth)) / (discount - terminal_growth)) / (1 + discount) ** DCF_YEARS
    result = round(pv_cf + pv_tv, 2)
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
    ratio  = mean_ps if mean_ps is not None else _safe(info.get('priceToSalesTrailing12Months'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PS: return None
    return round(ratio * rev_ps, 2)

def calc_mean_pe(info, mean_pe):
    eps   = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0: return None
    ratio = mean_pe if mean_pe is not None else _safe(info.get('trailingPE'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PE: return None
    return round(ratio * eps, 2)

def calc_mean_pb(info, mean_pb):
    bvps  = _safe(info.get('bookValue'))
    if bvps is None or bvps <= 0: return None
    ratio = mean_pb if mean_pb is not None else _safe(info.get('priceToBook'))
    if ratio is None or ratio <= 0: return None
    if ratio > MAX_PB: return None
    return round(ratio * bvps, 2)

def calc_psg(info, shares, peg_rate):
    """
    PSG: Revenue per share × (growth rate as whole number).
    Uses peg_rate — the un-floored rate from get_forward_growth.
    Returns None when peg_rate is 0 (Finviz explicitly said no growth).
    """
    rev = _safe(info.get('totalRevenue'))
    if rev is None or rev <= 0 or shares <= 0: return None
    if peg_rate is None or peg_rate <= 0: return None
    growth = min(peg_rate, MAX_PEG_PSG_GROWTH)
    return round((rev / shares) * (growth * 100), 2)

def calc_peg(info, peg_rate):
    """
    PEG: EPS × (growth rate as whole number).
    Uses peg_rate — the un-floored rate from get_forward_growth.
    Returns None when peg_rate is 0 (Finviz explicitly said no growth).
    """
    eps = _safe(info.get('trailingEps'))
    if eps is None or eps <= 0: return None
    if peg_rate is None or peg_rate <= 0: return None
    growth = min(peg_rate, MAX_PEG_PSG_GROWTH)
    return round(eps * (growth * 100), 2)

def calc_ev_ebitda(info, shares, mean_ev_ebitda):
    """
    EV/EBITDA valuation using the company's own historical average multiple.
    This is fair to the company — it reflects what the market has historically
    paid for this specific business, not a generic sector benchmark.

    Falls back to the trailing EV/EBITDA from yfinance if no historical
    average is available, capped at MAX_EV_EBITDA (40x).

    Formula: Fair Value = (EBITDA × mean_EV_EBITDA - Debt + Cash) / Shares
    """
    ebitda = _safe(info.get('ebitda'))
    debt   = _safe(info.get('totalDebt'), 0)
    cash   = _safe(info.get('totalCash'), 0)
    if ebitda is None or ebitda <= 0 or shares <= 0: return None

    # Use historical average if available, else fall back to trailing
    if mean_ev_ebitda is not None and 0 < mean_ev_ebitda <= MAX_EV_EBITDA:
        multiple = mean_ev_ebitda
    else:
        trailing = _safe(info.get('enterpriseToEbitda'))
        if trailing is None or trailing <= 0 or trailing > MAX_EV_EBITDA:
            return None
        multiple = trailing

    equity_value = (ebitda * multiple) - debt + cash
    if equity_value <= 0: return None
    return round(equity_value / shares, 2)

# =============================================================================
# INTRINSIC VALUE AGGREGATOR — tier-weighted mean
# =============================================================================
def calc_intrinsic_value(valuations: dict, tier: str = "Average",
                          dynamic_weights: dict = None,
                          current_price: float = None) -> dict:
    """
    Weighted intrinsic value aggregator.

    dynamic_weights: if provided, overrides tier weights. Methods with
    weight 0.0 are excluded entirely (appropriate for REITs, banks etc.)
    even if they produced a value — this is intentional.

    current_price: if provided, methods producing values more than 5x above
    or less than 0.1x the current price are rejected as outliers. This
    prevents exploding DCF values and currency-inflated metrics from
    distorting the aggregate intrinsic value.

    Confidence penalty:
    If the spread between the highest and lowest valid method is > 100%
    of the median value, confidence is downgraded one level. A very wide
    spread signals genuine valuation uncertainty regardless of method count.
    """
    keys = [
        "1_DCF_Operating_CF", "2_DCF_FCF", "3_DCF_Net_Income",
        "4_DCF_Terminal_Value", "5_Mean_PS", "6_Mean_PE",
        "7_Mean_PB", "8_PSG", "9_PEG", "10_EV_EBITDA",
    ]
    weights = dynamic_weights if dynamic_weights is not None else               TIER_WEIGHTS.get(tier, TIER_WEIGHTS["Average"])

    # Outlier bounds — reject values implausibly far from current market price
    if current_price and current_price > 0:
        outlier_upper = current_price * 5.0   # cap at 5× current price
        outlier_lower = current_price * 0.10  # floor at 10% of current price
    else:
        outlier_upper = 1_000_000
        outlier_lower = 0

    valid_vals, valid_weights, active_methods = [], [], []
    for k in keys:
        v = valuations.get(k)
        w = weights.get(k, 0.10)
        if w == 0.0:
            continue   # method excluded for this company type
        if v is not None and isinstance(v, (int, float)) and outlier_lower < v < outlier_upper:
            valid_vals.append(v)
            valid_weights.append(w)
            active_methods.append(k)

    count = len(valid_vals)
    if count == 0:
        return {
            "intrinsic_value": None,
            "methods_used":    0,
            "confidence":      "Insufficient Data",
            "active_methods":  [],
        }

    total_w   = sum(valid_weights)
    norm_w    = [w / total_w for w in valid_weights]
    intrinsic = round(float(np.dot(valid_vals, norm_w)), 2)

    # Base confidence by method count
    if count >= 7:   confidence = "High"
    elif count >= 4: confidence = "Medium"
    else:            confidence = "Low"

    # Confidence penalty for wide spread
    if count >= 3:
        sorted_v = sorted(valid_vals)
        median_v = sorted_v[len(sorted_v) // 2]
        spread   = (sorted_v[-1] - sorted_v[0]) / median_v if median_v > 0 else 0
        if spread > 1.5:   # spread > 150% of median — very wide disagreement
            if confidence == "High":   confidence = "Medium"
            elif confidence == "Medium": confidence = "Low"

    return {
        "intrinsic_value": intrinsic,
        "methods_used":    count,
        "confidence":      confidence,
        "active_methods":  active_methods,
    }

# =============================================================================
# 10-YEAR PRICE HISTORY
# =============================================================================
def get_10yr_history(stock):
    """
    Fetch up to 10 years of annual closing prices.
    Tries progressively shorter periods and retries with a delay if
    yfinance returns empty — handles rate limits and recently listed stocks.
    """
    for period in ("10y", "5y", "3y", "2y", "1y"):
        for attempt in range(2):   # 2 attempts per period
            try:
                hist = stock.history(period=period)
                if hist is None or hist.empty:
                    if attempt == 0:
                        time.sleep(2)   # wait 2s before retry on empty
                    continue
                result = {
                    str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2)
                    for y in sorted(set(d.year for d in hist.index))
                }
                if result:
                    return result
            except Exception:
                if attempt == 0:
                    time.sleep(2)
                continue
    return {}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def analyze_ticker(ticker, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(1.0)   # 1s between requests — reduces yfinance rate limit hits

            stock = yf.Ticker(ticker)
            info  = stock.info

            if not info:
                print(f"  [SKIP] {ticker}: yfinance returned empty info")
                return None

            # Robust share count — fallback chain covers meme stocks, ETFs, ADRs
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

            # Skip pure shells, SPACs and non-operating entities.
            # These have no revenue, no earnings and no cash flow —
            # all valuation methods will return null producing useless output.
            quote_type = info.get('quoteType', '').upper()
            if quote_type in {'ETF', 'MUTUALFUND', 'INDEX', 'FUTURE',
                               'OPTION', 'CURRENCY', 'CRYPTOCURRENCY'}:
                print(f"  [SKIP] {ticker}: non-equity security ({quote_type})")
                return None

            # Skip if no revenue AND no earnings — likely a SPAC or shell
            has_revenue  = _safe(info.get('totalRevenue'))
            has_earnings = _safe(info.get('netIncomeToCommon'))
            has_assets   = _safe(info.get('totalAssets'))
            if not has_revenue and not has_earnings and not has_assets:
                print(f"  [SKIP] {ticker}: no financial data — likely SPAC or shell")
                return None

            # Pull annual financials once — reused across all stages
            try:
                financials = stock.financials
            except Exception:
                financials = None

            # ------------------------------------------------------------------
            # STAGE 1a — Pre-compute KPIs used in quality scoring
            # Calculated before quality scores so all three metrics
            # (growth, financial_strength, predictability) use the same
            # values that are displayed in the UI tabs.
            # ------------------------------------------------------------------

            # Forward growth rate (Finviz primary, yfinance fallback)
            # Moved here from Stage 2 so it feeds into score_growth
            fwd_growth = get_forward_growth(ticker, info)
            growth_s1  = fwd_growth["rate"]
            growth_s2  = max(MIN_GROWTH_RATE, fwd_growth["rate"] * 0.50)
            peg_rate   = fwd_growth["peg_rate"]
            eps_next_5y = fwd_growth["raw"]   # raw unclamped rate for growth scoring

            # ROE — direct from yfinance info
            pre_roe = _safe(info.get('returnOnEquity'))

            # ROIC — calculated from balance sheet and income statement
            try:
                pre_roic_result = calc_roe_roic(info, financials, stock.balance_sheet)
                pre_roic = pre_roic_result.get("roic")   # decimal e.g. 0.5862
            except Exception:
                pre_roic = None
            # Current Ratio
            pre_current_ratio = None
            try:
                bal_sheet_pre = stock.balance_sheet
                if bal_sheet_pre is not None and not bal_sheet_pre.empty:
                    col = bal_sheet_pre.columns[0]
                    ca, cl = None, None
                    for name in ["Current Assets", "TotalCurrentAssets",
                                 "CurrentAssets"]:
                        if name in bal_sheet_pre.index:
                            ca = _safe(float(bal_sheet_pre.loc[name, col])); break
                    for name in ["Current Liabilities", "TotalCurrentLiabilities",
                                 "CurrentLiabilities"]:
                        if name in bal_sheet_pre.index:
                            cl = _safe(float(bal_sheet_pre.loc[name, col])); break
                    if ca and cl and cl > 0:
                        pre_current_ratio = round(ca / cl, 2)
            except Exception:
                pass
            if pre_current_ratio is None:
                pre_current_ratio = _safe(info.get('currentRatio'))

            # Debt-to-EBITDA
            pre_debt_to_ebitda = None
            try:
                td = _safe(info.get('totalDebt'))
                eb = _safe(info.get('ebitda'))
                if td is not None and eb is not None and eb > 0:
                    pre_debt_to_ebitda = round(td / eb, 2)
                elif td == 0 and eb is not None and eb > 0:
                    pre_debt_to_ebitda = 0.0
            except Exception:
                pass

            # Debt Servicing Ratio = Interest Expense / Operating Cash Flow
            # Four-source fallback chain to maximise coverage:
            #   1. stock.financials income statement (most reliable)
            #   2. stock.cashflow statement (Apple, Microsoft file it here)
            #   3. EDGAR interest expense from income facts
            #   4. yfinance info.interestExpense (least reliable, last resort)
            # If a company has zero financial debt, interest expense will be
            # zero and the ratio is stored as 0.0 (not N/A).
            pre_debt_service_ratio = None
            try:
                ie = None

                # Source 1 — income statement
                if financials is not None and not financials.empty:
                    col = financials.columns[0]
                    for name in [
                        "Interest Expense",
                        "InterestExpense",
                        "Interest Expense Non Operating",
                        "InterestExpenseNonOperating",
                        "Net Interest Income",
                        "NetInterestIncome",
                        "Interest And Debt Expense",
                        "InterestAndDebtExpense",
                        "Net Non Operating Interest Income Expense",
                        "Reconciled Interest Expense",
                        "Total Other Finance Cost",
                    ]:
                        if name in financials.index:
                            raw = _safe(float(financials.loc[name, col]))
                            if raw is not None:
                                ie = abs(raw); break

                # Source 2 — cashflow statement
                if ie is None:
                    try:
                        cf_stmt = stock.cashflow
                        if cf_stmt is not None and not cf_stmt.empty:
                            col_cf = cf_stmt.columns[0]
                            for name in [
                                "Interest Paid Supplemental Data",
                                "InterestPaidSupplementalData",
                                "Interest Paid",
                                "InterestPaid",
                                "Cash Interest Paid",
                            ]:
                                if name in cf_stmt.index:
                                    raw = _safe(float(cf_stmt.loc[name, col_cf]))
                                    if raw is not None:
                                        ie = abs(raw); break
                    except Exception:
                        pass

                # Source 3 — EDGAR income facts (already fetched for EDGAR charts)
                # Now includes interestExpense directly — covers AAPL and others
                # where yfinance income statement rows don't expose it
                if ie is None:
                    try:
                        edgar_ie = get_edgar_financials(ticker)
                        if edgar_ie["source"] == "edgar":
                            inc_years = sorted(edgar_ie["income"].keys(), reverse=True)
                            if inc_years:
                                latest_inc = edgar_ie["income"][inc_years[0]]
                                ie_edgar   = latest_inc.get("interestExpense")
                                if ie_edgar is not None:
                                    ie = abs(float(ie_edgar))
                    except Exception:
                        pass

                # Source 4 — yfinance info dict (least reliable)
                if ie is None:
                    raw_info = _safe(info.get('interestExpense'))
                    if raw_info is not None:
                        ie = abs(raw_info)

                # If still None but company has no debt, set to 0.0
                if ie is None:
                    total_debt_check = _safe(info.get('totalDebt'))
                    if total_debt_check is not None and total_debt_check == 0:
                        ie = 0.0

                op_cf_pre = _safe(info.get('operatingCashflow'))
                if ie is not None and ie > 0 and op_cf_pre and op_cf_pre > 0:
                    pre_debt_service_ratio = round(ie / op_cf_pre, 4)
                elif ie == 0:
                    pre_debt_service_ratio = 0.0

            except Exception:
                pass

            # ── 10Y Cash vs Debt history from EDGAR ──────────────────────────
            # Extracts the same data that powers the 10Y Cash & Total Debt
            # chart — used by score_financial_strength for trend analysis.
            cash_debt_history = {}
            try:
                edgar_cdf = get_edgar_financials(ticker)
                if edgar_cdf["source"] == "edgar":
                    for yr, bal_row in edgar_cdf["balance"].items():
                        cash_v = bal_row.get("cashAndCashEquivalents")
                        debt_v = bal_row.get("totalDebt")
                        if cash_v is not None or debt_v is not None:
                            # Convert to millions to match chart display
                            cash_debt_history[str(yr)] = {
                                "cash_balance": round(cash_v / 1e6, 2) if cash_v else None,
                                "total_debt":   round(debt_v / 1e6, 2) if debt_v else None,
                            }
            except Exception:
                pass

            # ── Long-term revenue and NI series from EDGAR ───────────────────
            # Extract the same 10-year chronological series that powers the
            # 10Y Revenue & Net Income chart. Used by score_predictability
            # for a longer and more meaningful trend analysis than the
            # ~4 years available from stock.financials alone.
            long_rev_series = []
            long_ni_series  = []
            try:
                edgar_data = get_edgar_financials(ticker)
                if edgar_data["source"] == "edgar":
                    inc = edgar_data["income"]
                    # Build chronological lists (oldest → newest)
                    sorted_yrs = sorted(inc.keys())
                    for yr in sorted_yrs:
                        row = inc[yr]
                        rv = row.get("revenue")
                        ni = row.get("netIncome")
                        if rv is not None: long_rev_series.append(float(rv))
                        if ni is not None: long_ni_series.append(float(ni))
            except Exception:
                pass
            # Fall back to stock.financials series if EDGAR unavailable
            if not long_rev_series and financials is not None and not financials.empty:
                long_rev_series = _get_annual_series(
                    financials, ['Total Revenue', 'TotalRevenue'])
            if not long_ni_series and financials is not None and not financials.empty:
                long_ni_series = _get_annual_series(
                    financials, ['Net Income', 'NetIncome'])

            # ------------------------------------------------------------------
            # STAGE 1b — Preliminary quality scores → determines tier
            # ------------------------------------------------------------------
            prelim_scores = {
                "profitability":      score_profitability(info),
                "financial_strength": score_financial_strength(
                                          info,
                                          current_ratio=pre_current_ratio,
                                          debt_to_ebitda=pre_debt_to_ebitda,
                                          debt_service_ratio=pre_debt_service_ratio,
                                          cash_debt_history=cash_debt_history),
                "growth":             score_growth(info, financials,
                                          eps_next_5y=eps_next_5y,
                                          roe=pre_roe,
                                          roic=pre_roic),
                "predictability":     score_predictability(info, financials,
                                          long_rev_series=long_rev_series,
                                          long_ni_series=long_ni_series),
                "valuation":          0,
                "moat":               None,
            }
            # Prelim subtotal — 4-metric denominator (moat pending)
            prelim_subtotal = sum(v for k, v in prelim_scores.items()
                                  if k in ('profitability','financial_strength',
                                           'growth','predictability') and v is not None)
            prelim_pct = round((prelim_subtotal / 40) * 100, 2)
            prelim_class    = classify_quality(prelim_scores, prelim_pct,
                                               roe=pre_roe,
                                               roic=pre_roic,
                                               eps_next_5y=eps_next_5y)
            tier_data       = resolve_tier(prelim_class["label"])
            discount        = tier_data["discount_rate"]
            terminal        = tier_data["terminal_growth"]

            # ------------------------------------------------------------------
            # STAGE 2 — Forward growth rate already computed in Stage 1a
            # growth_s1, growth_s2, peg_rate, fwd_growth all set above
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # STAGE 3 — Historical multiples (P/E, P/S, P/B, EV/EBITDA)
            # ------------------------------------------------------------------
            hist_multiples  = get_historical_mean_multiples(stock, shares)
            mean_pe         = hist_multiples.get("mean_pe")
            mean_ps         = hist_multiples.get("mean_ps")
            mean_pb         = hist_multiples.get("mean_pb")
            mean_ev_ebitda  = hist_multiples.get("mean_ev_ebitda")

            # ------------------------------------------------------------------
            # STAGE 4 — Company type detection + dynamic weighting + valuations
            # ------------------------------------------------------------------

            # Detect company type BEFORE computing valuations so weighting
            # is available immediately. Uses prelim_scores which are already
            # computed by this point.
            company_type, type_rationale = detect_company_type(
                info, prelim_scores, eps_next_5y
            )
            dynamic_weights = get_dynamic_weights(company_type, tier_data["tier"])

            dcf_fcf_val = calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount)

            valuations = {
                "1_DCF_Operating_CF":   calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount),
                "2_DCF_FCF":            dcf_fcf_val,
                "3_DCF_Net_Income":     calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount),
                "4_DCF_Terminal_Value": calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal, dcf_fcf_val),
                "5_Mean_PS":            calc_mean_ps(info, shares, mean_ps),
                "6_Mean_PE":            calc_mean_pe(info, mean_pe),
                "7_Mean_PB":            calc_mean_pb(info, mean_pb),
                "8_PSG":                calc_psg(info, shares, peg_rate),
                "9_PEG":                calc_peg(info, peg_rate),
                "10_EV_EBITDA":         calc_ev_ebitda(info, shares, mean_ev_ebitda),
            }

            _cp_for_iv = (_safe(info.get('currentPrice')) or
                          _safe(info.get('regularMarketPrice')))
            aggregate = calc_intrinsic_value(
                valuations,
                tier=tier_data["tier"],
                dynamic_weights=dynamic_weights,
                current_price=_cp_for_iv,
            )
            valuations.update(aggregate)

            # Methods excluded for this company type (weight = 0.0)
            excluded_methods = [
                k for k, w in dynamic_weights.items() if w == 0.0
            ]

            _type_explanation = {
                "BANK": (
                    "Banks make money from the difference between what they charge on loans and what they pay on deposits — "
                    "not from traditional business cash flows. Because of this, standard cash flow models (DCF) do not work well for banks. "
                    "Instead, we focus on Price-to-Book (P/B), which compares the stock price to the value of the bank's assets on its books. "
                    "Banks also carry a lot of debt by design — they borrow money to lend it out — so high leverage is normal and not a red flag."
                ),
                "REIT": (
                    "Real Estate Investment Trusts (REITs) own properties and are legally required to pay out most of their income as dividends. "
                    "This means they rarely have much free cash flow left over, so DCF models give misleading results. "
                    "Instead, we use Price-to-Book (P/B) to estimate the value of the underlying properties, "
                    "and EV/EBITDA to compare the business to similar property companies."
                ),
                "INSURANCE": (
                    "Insurance companies collect premiums upfront and pay out claims later — sometimes years later. "
                    "This makes their cash flow very lumpy and hard to predict, so we rely on net income instead. "
                    "The key question for an insurer is whether they are pricing their policies correctly and managing their investment portfolio well. "
                    "Price-to-Book and net income DCF are the most reliable ways to assess this."
                ),
                "UTILITY": (
                    "Utilities like electricity and water companies operate under government regulation, which means their prices and profits are capped but also very stable. "
                    "They tend to carry a lot of debt because building infrastructure is expensive — but this is normal and expected for the industry. "
                    "Because earnings are so predictable, Price-to-Earnings (P/E) and EV/EBITDA work well. "
                    "DCF is less useful here because regulated utilities have limited ability to grow beyond what regulators allow."
                ),
                "SPECGROWTH": (
                    "High-growth companies — typically in technology or biotech — are often not yet profitable, which means earnings-based methods like P/E give no useful result. "
                    "Instead, we use Price-to-Sales (P/S), which compares the stock price to total revenue regardless of whether the company is making a profit. "
                    "The PSG ratio (Price/Sales/Growth) also adjusts for how fast revenue is growing. "
                    "These methods are imperfect but they are the best available tools when a company is still investing heavily to grow."
                ),
                "CYCLICAL": (
                    "Cyclical companies — such as semiconductors, mining, or manufacturing — have profits that rise sharply in good times and fall sharply in downturns. "
                    "This makes earnings-based methods like P/E unreliable, because the earnings figure at any single point in time can be unusually high or low. "
                    "EV/EBITDA smooths out some of this distortion by looking at earnings before interest and tax, making it a fairer comparison across the economic cycle. "
                    "Price-to-Sales is also useful as revenue tends to be more stable than profits for cyclical businesses."
                ),
                "FCF_COMPOUNDER": (
                    "Free cash flow compounders are businesses that consistently generate more cash than they need to run the business — "
                    "think of companies like consumer goods brands or software firms with subscription revenue. "
                    "Because their cash flows are reliable year after year, a Discounted Cash Flow (DCF) model works very well. "
                    "DCF asks: if I add up all the future cash flows this company will generate and adjust for the time value of money, what is it worth today? "
                    "For these companies, that calculation gives a meaningful and trustworthy answer."
                ),
                "ASSET_MGR": (
                    "Asset managers and brokers earn fees for managing other people's money. Their balance sheets hold client assets that do not belong to the firm, "
                    "which makes standard balance sheet metrics like Price-to-Book misleading. "
                    "Instead, we focus on net income and revenue-based methods, which better reflect the underlying fee-earning power of the business."
                ),
                "CONSUMER_FINANCE": (
                    "Consumer finance companies — such as credit card providers or personal lenders — earn income from the interest they charge on loans. "
                    "Like banks, they carry a lot of debt by design because lending money is their core business model. "
                    "We use net income DCF and Price-to-Book, which together capture both the earning power and the asset value of the loan book."
                ),
                "STANDARD": (
                    "This company does not fit neatly into a specialist category, so we apply a balanced mix of all ten valuation methods. "
                    "These include Discounted Cash Flow (which estimates the present value of future cash flows), "
                    "earnings multiples (P/E and PEG), revenue multiples (P/S and PSG), book value (P/B), and EV/EBITDA. "
                    "Each method approaches value from a different angle — using all of them together gives a more rounded estimate than relying on any single number."
                ),
            }.get(company_type, "A balanced mix of valuation methods has been applied to estimate the intrinsic value of this company.")

            valuations["_meta"] = {
                "pe_source":        "historical_mean" if mean_pe else "current_trailing",
                "ps_source":        "historical_mean" if mean_ps else "current_trailing",
                "pb_source":        "historical_mean" if mean_pb else "current_trailing",
                "ev_ebitda_source": "historical_mean" if mean_ev_ebitda else "trailing_or_unavailable",
                "growth_source":    fwd_growth["source"],
                "growth_raw":       fwd_growth["raw"],
                "growth_peg_rate":  peg_rate,
                "discount_rate":    discount,
                "growth_stage1":    growth_s1,
                "growth_stage2":    growth_s2,
                "terminal_growth":  terminal,
                "tier":             tier_data["tier"],
                "company_type":     company_type,
                "type_rationale":   type_rationale,
                "method_explanation": _type_explanation,
                "weights_applied":  dynamic_weights,
                "excluded_methods": excluded_methods,
                "active_methods":   aggregate.get("active_methods", []),
            }

            # ------------------------------------------------------------------
            # STAGE 5 — Final quality scores (now with real valuation score)
            # ------------------------------------------------------------------
            intrinsic_value = aggregate.get("intrinsic_value")
            final_scores = {
                **prelim_scores,
                # Recalculate with same pre-computed values for consistency
                "financial_strength": score_financial_strength(
                                          info,
                                          current_ratio=pre_current_ratio,
                                          debt_to_ebitda=pre_debt_to_ebitda,
                                          debt_service_ratio=pre_debt_service_ratio,
                                          cash_debt_history=cash_debt_history),
                "growth":             score_growth(info, financials,
                                          eps_next_5y=eps_next_5y,
                                          roe=pre_roe,
                                          roic=pre_roic),
                "predictability":     score_predictability(info, financials,
                                          long_rev_series=long_rev_series,
                                          long_ni_series=long_ni_series),
                "valuation":          score_valuation(info, intrinsic_value),
            }
            # Quality score uses 4 metrics when moat is pending (denominator 40)
            # and 5 metrics when moat is scored (denominator 50).
            # Valuation excluded — business quality ≠ current cheapness.
            QUALITY_METRICS = ('profitability', 'financial_strength',
                               'growth', 'predictability')
            python_subtotal = sum(v for k, v in final_scores.items()
                                  if k in QUALITY_METRICS and v is not None)
            # AI Studio adds moat and recalculates out of 50 at query time.
            # Python stores the 4-metric score so the chart renders before moat.
            python_subtotal_pct = round((python_subtotal / 40) * 100, 2)
            classification      = classify_quality(
                final_scores,
                python_subtotal_pct,
                roe       = pre_roe,
                roic      = pre_roic,
                eps_next_5y = eps_next_5y,
            )

            quality = {
                "scores": {
                    "profitability":      final_scores["profitability"],
                    "financial_strength": final_scores["financial_strength"],
                    "growth":             final_scores["growth"],
                    "predictability":     final_scores["predictability"],
                    "valuation":          final_scores["valuation"],
                    "moat":               None,
                },
                # Pre-computed labels — AI Studio must use these exact strings
                # for written assessments. Never derive labels from raw score
                # numbers alone (score 6 ≠ "60% = Average").
                "score_labels": {
                    "profitability":      _score_label(final_scores["profitability"]),
                    "financial_strength": _score_label(final_scores["financial_strength"]),
                    "growth":             _score_label(final_scores["growth"]),
                    "predictability":     _score_label(final_scores["predictability"]),
                    "valuation":          _score_label(final_scores["valuation"]),
                    "moat":               "Not scored",   # AI Studio adds moat at query time
                },
                # Company-type context so AI Studio knows when high debt is structural
                "company_type":        company_type,
                "company_type_note":   (
                    "Banks and REITs carry structural leverage — do not describe "
                    "financial_strength scores of 7+ as 'moderate concern' for these types."
                    if company_type in ("BANK", "REIT", "INSURANCE", "ASSET_MGR",
                                        "CONSUMER_FINANCE", "UTILITY")
                    else None
                ),
                "python_subtotal":     python_subtotal,
                "python_subtotal_pct": python_subtotal_pct,
                "classification":      classification["label"],
                "penalty_pct":         classification["penalty_pct"],
                "final_score_pct":     classification["final_score_pct"],
                "tiebreaker_used":     classification["tiebreaker_used"],
                "tiebreaker_inputs": {
                    "roe":         round(pre_roe, 4)     if pre_roe     is not None else None,
                    "roic":        round(pre_roic, 4)    if pre_roic    is not None else None,
                    "eps_next_5y": round(eps_next_5y, 4) if eps_next_5y is not None else None,
                },
                "note": (
                    "Moat score is null — AI Studio adds moat (0-10) and recalculates.\n"
                    "With moat: final_pct = (profitability + financial_strength + growth "
                    "+ predictability + moat) / 50 × 100.\n"
                    "Then re-classify using tiebreaker_inputs:\n"
                    "  ≥70%      → Safe (no penalty)\n"
                    "  65–69%    → Safe if 2+ of (roe≥0.13, roic≥0.13, eps_next_5y≥0.10) "
                    "else Speculative (−3pts)\n"
                    "  55–64%    → Speculative (−3pts)\n"
                    "  50–54%    → Speculative if 2+ of (roe≥0.08, roic≥0.08, "
                    "eps_next_5y≥0.08) else Dangerous (−8pts)\n"
                    "  <50%      → Dangerous (−8pts)\n"
                    "Moat override: if moat≥7 AND (profitability≥7 OR fs≥7) AND pct≥55 → Safe.\n"
                    "final_score_pct = max(0, final_pct − penalty)"
                ),
                "last_calculated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            # ------------------------------------------------------------------
            # STAGE 6 — ROE, ROIC and Ratios tab metrics
            # All pre-computed here so AI Studio reads from JSON at query
            # time rather than recalculating live — eliminates tab load delay.
            #
            # Current Ratio       = Current Assets / Current Liabilities
            # Debt-to-EBITDA      = Total Debt / EBITDA
            # Debt Servicing Ratio = Net Interest Expense / Operating Cash Flow
            # ------------------------------------------------------------------
            roe_roic = calc_roe_roic(info, financials, stock.balance_sheet)

            # ── Ratios — reuse values pre-computed in Stage 1a ──────────────
            # These were already calculated before Stage 1 to feed the
            # financial strength quality score. Reusing them here ensures
            # the Ratios tab and the quality score always show identical values.
            current_ratio  = pre_current_ratio
            debt_to_ebitda = pre_debt_to_ebitda

            # Debt service ratio also reused from Stage 1a
            debt_service_ratio = pre_debt_service_ratio

            fundamentals = {
                "roe":                 roe_roic["roe"],
                "roic":                roe_roic["roic"],
                "roe_pct":             roe_roic["roe_pct"],
                "roic_pct":            roe_roic["roic_pct"],
                "roic_source":         roe_roic["roic_source"],
                "current_ratio":       current_ratio,
                "debt_to_ebitda":      debt_to_ebitda,
                "debt_service_ratio":  debt_service_ratio,
            }

            # ------------------------------------------------------------------
            # STAGE 7 — Financials chart data
            # Pre-computes all four Financials tab charts so AI Studio does
            # not need to make live GuruFocus calls at query time.
            # ------------------------------------------------------------------
            financials_charts = get_financials_chart_data(stock, info, ticker)

            # ------------------------------------------------------------------
            # STAGE 8 — History, forecast and valuation band chart data
            # ------------------------------------------------------------------
            price_history = get_10yr_history(stock)
            current_year  = datetime.now().year

            # Basic 5-year forecast (IV compounded at stage1 growth rate)
            forecast = {
                str(current_year + i): round(intrinsic_value * (1 + growth_s1) ** i, 2)
                for i in range(1, 6)
            } if intrinsic_value and intrinsic_value > 0 else {}

            # ------------------------------------------------------------------
            # Valuation band chart data
            # Provides everything AI Studio needs to render the enhanced
            # 10Y History + 5Y Forecast chart with:
            #   - Historical and current market price line
            #   - Mean intrinsic value line (historical back-projection + forecast)
            #   - Upper band (optimistic scenario — highest valuation method)
            #   - Lower band (pessimistic scenario — lowest valuation method)
            #   - Green/red shading between price and IV
            #
            # Upper/lower bands use the spread of individual valuation methods:
            #   upper_iv = highest valid individual method result
            #   lower_iv = lowest valid individual method result
            # Both are then grown/shrunk by the same growth rate for forecast years
            # and back-projected for historical years using the inverse rate.
            # ------------------------------------------------------------------
            valuation_chart = {}
            forecast_meta   = {}
            try:
                if intrinsic_value and intrinsic_value > 0:

                    # ----------------------------------------------------------
                    # FORECAST ADJUSTMENT FRAMEWORK
                    #
                    # Three forecast scenarios driven by:
                    #   1. EPS Next 5Y (core quantitative input — Finviz/yfinance)
                    #   2. Financial strength score (sustainability constraint)
                    #   3. Bull/Bear multipliers (scenario range)
                    #
                    # Financial strength constrains optimism:
                    #   A high EPS growth forecast for a company with weak cash
                    #   generation or high leverage is not credible. The FS score
                    #   reduces the effective growth rate proportionally so the
                    #   base forecast reflects what the balance sheet can support.
                    #
                    # FS score → growth sustainability multiplier:
                    #   8–10 → 1.00  (no constraint — strong balance sheet)
                    #   6–7  → 0.90  (mild moderation)
                    #   4–5  → 0.75  (moderate constraint — some financial risk)
                    #   2–3  → 0.55  (significant constraint — leveraged/weak)
                    #   0–1  → 0.35  (severe constraint — distressed)
                    #
                    # Bull rate = constrained_base × 1.30 (capped at 30%)
                    # Bear rate = constrained_base × 0.55 (floored at 1%)
                    # ----------------------------------------------------------
                    fs_score = final_scores.get("financial_strength", 5) or 5
                    if fs_score >= 8:   fs_multiplier = 1.00
                    elif fs_score >= 6: fs_multiplier = 0.90
                    elif fs_score >= 4: fs_multiplier = 0.75
                    elif fs_score >= 2: fs_multiplier = 0.55
                    else:               fs_multiplier = 0.35

                    # Raw EPS forward rate — sanitize immediately to prevent
                    # NaN/Infinity from propagating into round() calls below.
                    # fwd_growth["raw"] can be 0.0 (Finviz "-") or occasionally
                    # NaN from yfinance edge cases.
                    raw_eps_raw = fwd_growth.get("raw", growth_s1)
                    try:
                        raw_eps_rate = float(raw_eps_raw)
                        if not (raw_eps_rate == raw_eps_rate) or abs(raw_eps_rate) == float('inf'):
                            raw_eps_rate = growth_s1  # NaN/Inf fallback
                    except (TypeError, ValueError):
                        raw_eps_rate = growth_s1

                    # Base rate — EPS growth constrained by financial strength
                    base_rate  = max(MIN_GROWTH_RATE,
                                     min(raw_eps_rate * fs_multiplier, MAX_GROWTH_RATE))
                    # Bull rate — optimistic: 30% above base, hard cap 30%
                    bull_rate  = max(MIN_GROWTH_RATE,
                                     min(base_rate * 1.30, MAX_GROWTH_RATE))
                    # Bear rate — pessimistic: 45% below base, hard floor 1%
                    bear_rate  = max(0.01,
                                     min(base_rate * 0.55, MAX_GROWTH_RATE))

                    # Store forecast metadata so AI Studio can explain the logic
                    was_constrained = fs_multiplier < 1.0
                    forecast_meta = {
                        "eps_next_5y_raw":       round(raw_eps_rate, 4),
                        "fs_score":              fs_score,
                        "fs_multiplier":         fs_multiplier,
                        "base_growth_rate":      round(base_rate, 4),
                        "bull_growth_rate":      round(bull_rate, 4),
                        "bear_growth_rate":      round(bear_rate, 4),
                        "growth_constrained":    was_constrained,
                        "constraint_note": (
                            f"EPS growth moderated from {round(raw_eps_rate*100,1)}% "
                            f"to {round(base_rate*100,1)}% due to financial strength "
                            f"score of {fs_score}/10"
                        ) if was_constrained else None,
                    }

                    # Current price for shading
                    current_price = (
                        _safe(info.get('currentPrice')) or
                        _safe(info.get('regularMarketPrice'))
                    )

                    # Historical back-projection using growth rates.
                    # Logic: today's IV compounded BACK in time at each scenario rate
                    # gives what the scenario IV would have been in that year.
                    #
                    # Bull case discounts LESS (higher rate → smaller past value):
                    #   bull_factor = (1 + bull_rate) ** (-years_back)
                    # Bear case discounts MORE (lower rate → larger past value):
                    #   bear_factor = (1 + bear_rate) ** (-years_back)
                    #
                    # IMPORTANT — why bull < bear historically (and this is CORRECT):
                    # The bull scenario assumes FASTER historical growth, so looking
                    # backwards the implied past IV is LOWER (the company grew more
                    # to reach today's IV). The bear scenario assumes SLOWER growth,
                    # so the implied past IV is HIGHER.
                    #
                    # However this creates a confusing chart where the "bull" band
                    # is below the "bear" band in history. To keep the chart intuitive
                    # — bull always above bear — we swap the factors for historical years:
                    #   historical bull_case  = IV * (1 + bear_rate) ** (-years_back)  ← HIGHER
                    #   historical bear_case  = IV * (1 + bull_rate) ** (-years_back)  ← LOWER
                    #
                    # This preserves the visual: bull band always on top, bear always below.
                    if price_history:
                        hist_years = sorted(price_history.keys())
                        n_hist     = len(hist_years)
                        for i, yr in enumerate(hist_years):
                            years_back = n_hist - i
                            b_factor      = (1 + base_rate) ** (-years_back)
                            # Swap: bear_rate gives higher back-projected value → assign to bull band
                            bull_h_factor = (1 + bear_rate) ** (-years_back)
                            # Swap: bull_rate gives lower back-projected value → assign to bear band
                            bear_h_factor = (1 + bull_rate) ** (-years_back)

                            def _safe_round(v):
                                try:
                                    f = float(v)
                                    return round(f, 2) if f == f and abs(f) != float('inf') else None
                                except Exception:
                                    return None

                            valuation_chart[yr] = {
                                "market_price":    price_history[yr],
                                "intrinsic_value": _safe_round(intrinsic_value * b_factor),
                                "bull_case":       _safe_round(intrinsic_value * bull_h_factor),
                                "bear_case":       _safe_round(intrinsic_value * bear_h_factor),
                            }

                    # Current year
                    valuation_chart[str(current_year)] = {
                        "market_price":    round(current_price, 2) if current_price else None,
                        "intrinsic_value": round(intrinsic_value, 2),
                        "bull_case":       round(intrinsic_value * (1 + (bull_rate - base_rate)), 2),
                        "bear_case":       round(intrinsic_value * (1 - (base_rate - bear_rate)), 2),
                    }

                    # Forecast years — three scenarios compound forward
                    for i in range(1, 6):
                        valuation_chart[str(current_year + i)] = {
                            "market_price":    None,
                            "intrinsic_value": round(intrinsic_value * (1 + base_rate) ** i, 2),
                            "bull_case":       round(intrinsic_value * (1 + bull_rate) ** i, 2),
                            "bear_case":       round(intrinsic_value * (1 + bear_rate) ** i, 2),
                        }
            except Exception:
                valuation_chart = {}
                forecast_meta   = {}

            # ------------------------------------------------------------------
            # STAGE 9 — Bull / Bear thesis
            # Combines rule-based signals with Claude Haiku API call.
            # Falls back to rule-based only if API key not set or call fails.
            # ------------------------------------------------------------------
            bull_bear = generate_bull_bear(
                ticker        = ticker,
                info          = info,
                final_scores  = final_scores,
                fundamentals  = fundamentals,
                fwd_growth    = fwd_growth,
                intrinsic_value = intrinsic_value,
                forecast_meta = forecast_meta,
            )

            # ── Quality Narrative — pre-computed so AI Studio reads from JSON ──
            # Eliminates the main source of per-query latency: AI Studio was
            # generating quality_conclusion, risk_profile, and investment_summary
            # at runtime for every search. All three are now pre-computed here
            # in a single Haiku call (or rule-based fallback if no API key).
            quality_narrative = generate_quality_narrative(
                ticker              = ticker,
                info                = info,
                final_scores        = final_scores,
                score_labels        = quality["score_labels"],
                classification      = classification["label"],
                python_subtotal_pct = python_subtotal_pct,
                fundamentals        = fundamentals,
                fwd_growth          = fwd_growth,
                forecast_meta       = forecast_meta,
                intrinsic_value     = intrinsic_value,
                company_type        = company_type,
            )

            # ── Wire moat score back into quality dict ─────────────────────
            # Haiku scored moat (0-10) inside the same API call as the
            # narrative. Now apply it: recalculate final_score_pct out of 50
            # and re-classify with the full 5-metric score.
            moat_score = quality_narrative.get("moat_score")
            if moat_score is not None:
                try:
                    moat_score = int(round(float(moat_score)))
                    moat_score = max(0, min(10, moat_score))
                except (TypeError, ValueError):
                    moat_score = None

            if moat_score is not None:
                quality["scores"]["moat"]      = moat_score
                quality["score_labels"]["moat"] = _score_label(moat_score)
                # Recalculate final score out of 50 with moat included
                FULL_METRICS = ("profitability", "financial_strength",
                                "growth", "predictability")
                full_subtotal = sum(
                    final_scores.get(m, 0) or 0 for m in FULL_METRICS
                ) + moat_score
                full_pct = round((full_subtotal / 50) * 100, 2)
                # Re-classify with full 5-metric score
                full_classification = classify_quality(
                    {**final_scores, "moat": moat_score},
                    full_pct,
                    roe       = pre_roe,
                    roic      = pre_roic,
                    eps_next_5y = eps_next_5y,
                )
                quality["final_score_pct"]  = full_classification["final_score_pct"]
                quality["classification"]   = full_classification["label"]
                quality["penalty_pct"]      = full_classification["penalty_pct"]
                quality["tiebreaker_used"]  = full_classification["tiebreaker_used"]
                quality["note"] = "Moat scored by Haiku API — final_score_pct computed out of 50."

            # ── Margin of Safety — pre-computed for hero card ──────────────
            # Positive = stock is cheaper than IV (discount) — good
            # Negative = stock is more expensive than IV (premium)
            # AI Studio reads this directly — never recomputes from scratch
            _cp  = (_safe(info.get('currentPrice')) or
                    _safe(info.get('regularMarketPrice')))
            _iv  = intrinsic_value
            if _cp and _cp > 0 and _iv and _iv > 0:
                _mos_pct = round((_iv - _cp) / _iv * 100, 1)
                # Label convention: positive = discount (below IV), negative = premium (above IV)
                if _mos_pct > 40:    _mos_label = "Strong Discount (Margin of Safety)"
                elif _mos_pct > 20:  _mos_label = "Moderate Discount"
                elif _mos_pct > 5:   _mos_label = "Slight Discount"
                elif _mos_pct > 0:   _mos_label = "Fairly Valued (Marginal Discount)"
                elif _mos_pct > -5:  _mos_label = "Fairly Valued (Marginal Premium)"
                elif _mos_pct > -20: _mos_label = "Slight Premium"
                elif _mos_pct > -40: _mos_label = "Moderate Premium"
                else:                _mos_label = "Significant Premium"
            else:
                _mos_pct  = None
                _mos_label = "Unavailable"

            # ── Company Profile — pre-computed so AI Studio never fetches live ──
            # All fields sourced from yfinance info dict (zero extra API calls).
            # AI Studio reads these directly for the header and company tab
            # instead of making a live lookup, eliminating the main source of
            # per-query latency.
            company_profile = {
                "name":              info.get("longName") or info.get("shortName") or ticker,
                "ticker":            ticker,
                "sector":            info.get("sector"),
                "industry":          info.get("industry"),
                "company_type":      company_type,
                "type_rationale":    type_rationale,
                "description":       info.get("longBusinessSummary"),
                 "description_summary": summarise_description(
                     ticker,
                     info.get("longBusinessSummary"),
                     info.get("longName") or info.get("shortName") or ticker,
                     info.get("sector"),
                     info.get("industry"),
                     company_type,
                 ),
                "website":           info.get("website"),
                "employees":         info.get("fullTimeEmployees"),
                "country":           info.get("country"),
                "city":              info.get("city"),
                "exchange":          info.get("exchange"),
                "currency":          info.get("currency"),
                "quote_type":        info.get("quoteType"),
                "market_cap":        info.get("marketCap"),
                "52w_high":          info.get("fiftyTwoWeekHigh"),
                "52w_low":           info.get("fiftyTwoWeekLow"),
                "avg_volume":        info.get("averageVolume"),
                "pe_trailing":       info.get("trailingPE"),
                "pe_forward":        info.get("forwardPE"),
                "dividend_yield":    info.get("dividendYield"),
                "beta":              info.get("beta"),
            }

            return ticker, {
                "company_profile":      company_profile,
                "current_price":        round(_cp, 2) if _cp else None,
                "intrinsic_value":      round(_iv, 2) if _iv else None,
                "margin_of_safety_pct": _mos_pct,
                "mos_label":            _mos_label,
                "valuations":         valuations,
                "quality":            quality,
                "quality_narrative":  quality_narrative,
                "fundamentals":       fundamentals,
                "financials_charts":  financials_charts,
                "valuation_chart":    valuation_chart,
                "forecast_meta":      forecast_meta,
                "bull_bear":          bull_bear,
                "10_Year_History":    price_history,
                "5_Year_Forecast":    forecast,
                "Last_Updated":       datetime.now().strftime("%Y-%m-%d %H:%M"),
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
# SEC EDGAR FINANCIAL FETCHER
#
# Fetches up to 20 years of annual (10-K) financial data from the SEC EDGAR
# XBRL API. Completely free, no API key, no rate-limit tier.
#
# SEC requires a descriptive User-Agent header identifying the application.
# Max recommended request rate: 10 requests/second — we sleep 0.1s between
# the three calls (income, balance, cashflow) per ticker.
#
# Data flow:
#   1. Fetch ticker → CIK mapping once per run (cached in EDGAR_CIK_CACHE)
#   2. Fetch company facts JSON for the ticker's CIK
#   3. Extract annual 10-K values for each GAAP concept needed
#   4. Return data keyed by fiscal year (integer)
#
# GAAP concept fallback chains handle the fact that companies report the
# same economic line item under different concept names across time.
#
# Returns:
#   {
#     "income":   { 2024: {...}, 2023: {...}, ... },
#     "balance":  { 2024: {...}, 2023: {...}, ... },
#     "cashflow": { 2024: {...}, 2023: {...}, ... },
#     "source":   "edgar" | "unavailable"
#   }
# =============================================================================
EDGAR_CIK_CACHE = {}   # ticker → zero-padded CIK string, populated once per run
EDGAR_USER_AGENT = "InvestingApp contact@example.com"   # SEC requires this header

# Annual filing form types accepted by EDGAR extractors.
# 10-K  = US domestic companies (standard annual report)
# 20-F  = Foreign private issuers (e.g. GLOB/Globant, ASML, SAP)
#          These companies file 20-F instead of 10-K but EDGAR XBRL data
#          has identical structure — the only difference is the form field.
# 40-F  = Canadian companies listed in the US (e.g. CNQ, SU)
EDGAR_ANNUAL_FORMS = {"10-K", "20-F", "40-F"}

def _edgar_get_cik(ticker):
    """Return zero-padded CIK for ticker, or None if not found."""
    global EDGAR_CIK_CACHE
    if EDGAR_CIK_CACHE:
        return EDGAR_CIK_CACHE.get(ticker.upper())
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for entry in data.values():
            t   = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            EDGAR_CIK_CACHE[t] = cik
        return EDGAR_CIK_CACHE.get(ticker.upper())
    except Exception as e:
        print(f"  [EDGAR] CIK lookup failed: {e}")
        return None


def _edgar_extract_earliest(facts, *concept_names):
    """
    Like _edgar_extract but keeps the EARLIEST filed value per period year
    rather than the most recently filed.

    Used specifically for shares outstanding so that post-split restatements
    of prior year comparatives do not corrupt the original as-filed values.
    After a stock split a company re-files prior years with adjusted share
    counts. If we use the most recently filed value, the restated post-split
    figure appears for the pre-split year, shifting the split discontinuity
    by one year and breaking the split detection algorithm.
    """
    from datetime import datetime

    def _days(start_str, end_str):
        try:
            s = datetime.strptime(start_str, "%Y-%m-%d")
            e = datetime.strptime(end_str,   "%Y-%m-%d")
            return (e - s).days
        except Exception:
            return None

    us_gaap  = facts.get("facts", {}).get("us-gaap", {})
    year_map = {}   # period_end_year → (filed_date, value)  ← earliest filed

    for name in concept_names:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        usd   = units.get("USD") or units.get("shares") or []
        for entry in usd:
            if entry.get("form") not in EDGAR_ANNUAL_FORMS:
                continue
            if entry.get("fp") not in ("FY", "CY"):
                continue
            val   = entry.get("val")
            filed = entry.get("filed", "")
            end   = entry.get("end")
            start = entry.get("start")
            if val is None or not end:
                continue
            try:
                period_year = int(end[:4])
            except (ValueError, TypeError):
                continue
            # Balance sheet snapshots have no start date — skip duration check
            if start and end:
                days = _days(start, end)
                if days is None or not (320 <= days <= 380):
                    continue
            # Keep EARLIEST filed entry per period year (opposite of _edgar_extract)
            if period_year not in year_map or filed < year_map[period_year][0]:
                year_map[period_year] = (filed, float(val))

    return {yr: v for yr, (_, v) in year_map.items()} if year_map else {}


def _edgar_extract_merge(facts, *concept_names):
    """
    Like _edgar_extract but merges results across ALL matching concepts
    rather than stopping at the first match. Older filings often use
    different concept names than newer ones (e.g. Apple used SalesRevenueNet
    before ASC 606, then switched to RevenueFromContractWithCustomer).
    Merging fills in years that any single concept would miss.
    Most-recently-filed value wins per year across all concepts.
    """
    from datetime import datetime

    def _days(start_str, end_str):
        try:
            s = datetime.strptime(start_str, "%Y-%m-%d")
            e = datetime.strptime(end_str,   "%Y-%m-%d")
            return (e - s).days
        except Exception:
            return None

    us_gaap   = facts.get("facts", {}).get("us-gaap", {})
    year_map  = {}   # period_end_year → (filed_date, value)

    for name in concept_names:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        usd   = units.get("USD") or units.get("shares") or []
        for entry in usd:
            if entry.get("form") not in EDGAR_ANNUAL_FORMS:
                continue
            if entry.get("fp") not in ("FY", "CY"):
                continue
            val   = entry.get("val")
            filed = entry.get("filed", "")
            start = entry.get("start")
            end   = entry.get("end")
            if val is None or not end:
                continue
            try:
                period_year = int(end[:4])
            except (ValueError, TypeError):
                continue
            if start and end:
                days = _days(start, end)
                if days is None or not (320 <= days <= 380):
                    continue
            if period_year not in year_map or filed > year_map[period_year][0]:
                year_map[period_year] = (filed, float(val))

    return {yr: v for yr, (_, v) in year_map.items()} if year_map else {}


def _edgar_extract(facts, *concept_names):
    """
    Search XBRL facts for the first matching concept name.
    Returns a dict of { period_year(int): value(float) } from annual 10-K filings.

    KEY FIX — keyed by END DATE YEAR, not by EDGAR fy field.

    EDGAR's fy field = the fiscal year of the FILING (e.g. Apple's FY2025 10-K
    filed Oct 2025 includes comparative FY2023 data tagged as fy=2025).
    Using fy as the key therefore labels every value 2 years too late.

    The correct key is the year from the period END date field, which directly
    represents the year the data actually covers.

    Validations:
      1. form == "10-K"         — annual filing only
      2. fp in ("FY", "CY")     — full-year period flag
      3. end date present       — required for year key
      4. Duration 320-380 days  — full year check for flow concepts (income,
                                   cashflow). Balance sheet snapshots have no
                                   start date so this check is skipped for them.
      5. Deduplication by end year — most recently filed value wins so
                                   amendments and restatements are handled.
    """
    from datetime import datetime

    def _days(start_str, end_str):
        try:
            s = datetime.strptime(start_str, "%Y-%m-%d")
            e = datetime.strptime(end_str,   "%Y-%m-%d")
            return (e - s).days
        except Exception:
            return None

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in concept_names:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        usd   = units.get("USD") or units.get("shares") or []
        year_map = {}   # period_end_year(int) → (filed_date str, value float)

        for entry in usd:
            # 1. Annual 10-K only
            if entry.get("form") not in EDGAR_ANNUAL_FORMS:
                continue
            # 2. Full-year period flag
            if entry.get("fp") not in ("FY", "CY"):
                continue

            val   = entry.get("val")
            filed = entry.get("filed", "")
            start = entry.get("start")
            end   = entry.get("end")

            if val is None or not end:
                continue

            # 3. Derive year from actual period end date
            try:
                period_year = int(end[:4])
            except (ValueError, TypeError):
                continue

            # 4. Duration check for flow concepts (start date present)
            #    Skip for balance sheet snapshots (no start date)
            if start and end:
                days = _days(start, end)
                if days is None or not (320 <= days <= 380):
                    continue

            # 5. Keep most recently filed entry per period end year
            if period_year not in year_map or filed > year_map[period_year][0]:
                year_map[period_year] = (filed, float(val))

        if year_map:
            return {yr: v for yr, (_, v) in year_map.items()}
    return {}


def get_edgar_financials(ticker):
    """Fetch 10-K annual data from SEC EDGAR for the four financials charts."""
    empty = {"income": {}, "balance": {}, "cashflow": {}, "source": "unavailable"}

    cik = _edgar_get_cik(ticker)
    if not cik:
        return empty

    try:
        url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(
            url,
            headers={"User-Agent": EDGAR_USER_AGENT},
            timeout=20,
        )
        if resp.status_code != 200:
            print(f"  [EDGAR] {ticker}: HTTP {resp.status_code}")
            return empty

        facts = resp.json()
        time.sleep(0.12)   # stay well within SEC's 10 req/s guideline

        # ── Income statement concepts ─────────────────────────────────────
        # Merge across ALL revenue concepts — Apple switched from
        # SalesRevenueNet (pre-ASC606) to RevenueFromContractWithCustomer
        # in FY2018. Merging fills in years that any single concept misses.
        revenue = _edgar_extract_merge(facts,
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense",
            "SalesRevenueServicesNet",
            "NetRevenues",
            "TotalRevenues")

        net_income = _edgar_extract(facts,
            "NetIncomeLoss", "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic")

        op_income = _edgar_extract(facts,
            "OperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest")

        pretax = _edgar_extract(facts,
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic")

        tax_exp = _edgar_extract(facts,
            "IncomeTaxExpenseBenefit")

        # Interest expense — AAPL and many others file under InterestExpense
        # or InterestPaidNet in EDGAR even when yfinance doesn't expose it
        interest_exp_edgar = _edgar_extract_merge(facts,
            "InterestExpense",
            "InterestExpenseDebt",
            "InterestCostsIncurred",
            "InterestPaidNet",
            "FinanceLeaseInterestExpense",
            "InterestAndDebtExpense")

        # Accounts Receivable — used for Revenue vs AR chart
        # AccountsReceivableNetCurrent is the standard GAAP concept.
        # Some companies use broader receivables concepts — merge to fill gaps.
        accounts_receivable = _edgar_extract_merge(facts,
            "AccountsReceivableNetCurrent",
            "AccountsReceivableNet",
            "ReceivablesNetCurrent",
            "TradeAndOtherReceivablesNetCurrent",
            "NotesAndLoansReceivableNetCurrent")

        # ── Balance sheet concepts ────────────────────────────────────────
        total_assets = _edgar_extract(facts,
            "Assets")

        current_liab = _edgar_extract(facts,
            "LiabilitiesCurrent")

        equity = _edgar_extract(facts,
            "StockholdersEquity",
            "StockholdersEquityAttributableToParent",
            "CommonStockholdersEquity",
            "RetainedEarningsAccumulatedDeficit",  # last resort proxy
            "LiabilitiesAndStockholdersEquity")    # broad fallback

        cash = _edgar_extract(facts,
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
            "Cash")

        # Financial debt — merge across concepts to fill per-company gaps.
        # Apple uses LongTermDebtNoncurrent for 2015-2022 but LongTermDebt
        # for 2013-2014 and 2023+. Merging both fills the complete history.
        total_debt = _edgar_extract_merge(facts,
            "LongTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermNotesPayable",
            "SeniorNotes",
            "UnsecuredDebt",
            "SecuredDebt")

        short_term_debt = _edgar_extract_merge(facts,
            "ShortTermBorrowings",
            "DebtCurrent",
            "LongTermDebtCurrent",
            "NotesPayableCurrent",
            "CommercialPaper",
            "ShortTermNotesPayable")

        # Shares outstanding — use EARLIEST filed value per year, not most recent.
        # Reason: after a stock split, companies re-file prior year comparatives
        # on a post-split adjusted basis. "Most recently filed wins" would then
        # pick up the restated post-split figure for pre-split years, causing
        # the discontinuity to appear one year too late and confusing the
        # split detection algorithm.
        # Using the earliest filed original value ensures split detection
        # sees the true as-reported jump between the correct pair of years.
        shares = _edgar_extract_earliest(facts,
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "WeightedAverageNumberOfSharesOutstandingDiluted",
            "CommonStockSharesOutstanding")

        # Cash & Short-term Investments — matches what GuruFocus displays.
        # Some companies (e.g. Apple) file cash and marketable securities
        # as separate EDGAR concepts rather than a single combined one.
        # Strategy: try combined concept first, then manually sum components.
        cash_combined = _edgar_extract_merge(facts,
            "CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalentsAndShortTermInvestments")

        cash_only = _edgar_extract_merge(facts,
            "CashAndCashEquivalentsAtCarryingValue",
            "CashAndDueFromBanks",
            "Cash")

        # Current marketable securities (short-term investments)
        mkt_sec = _edgar_extract_merge(facts,
            "MarketableSecuritiesCurrent",
            "AvailableForSaleSecuritiesCurrent",
            "ShortTermInvestments",
            "OtherShortTermInvestments")

        # Build cash_and_equiv: prefer combined; else cash + marketable secs
        cash_and_equiv = {}
        all_cash_years = set(cash_combined) | set(cash_only) | set(mkt_sec)
        for yr in all_cash_years:
            if yr in cash_combined:
                cash_and_equiv[yr] = cash_combined[yr]
            elif yr in cash_only:
                # Add current marketable securities if available
                c = cash_only[yr]
                m = mkt_sec.get(yr, 0) or 0
                cash_and_equiv[yr] = c + m

        # ── Cash flow concepts ────────────────────────────────────────────
        op_cf = _edgar_extract(facts,
            "NetCashProvidedByUsedInOperatingActivities")

        capex = _edgar_extract(facts,
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets")

        buybacks = _edgar_extract(facts,
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfEquity")

        # ── Organise by fiscal year ───────────────────────────────────────
        all_years = sorted(
            set(revenue) | set(net_income) | set(total_assets) | set(op_cf),
            reverse=True
        )[:10]

        income_out, balance_out, cashflow_out = {}, {}, {}

        for yr in all_years:
            income_out[yr] = {
                "revenue":           revenue.get(yr),
                "netIncome":         net_income.get(yr),
                "operatingIncome":   op_income.get(yr),
                "incomeBeforeTax":   pretax.get(yr),
                "incomeTaxExpense":  tax_exp.get(yr),
                "interestExpense":   interest_exp_edgar.get(yr),
            }
            # Combine long-term + short-term financial debt
            lt  = total_debt.get(yr)
            st  = short_term_debt.get(yr)
            combined_debt = None
            if lt is not None or st is not None:
                combined_debt = (lt or 0) + (st or 0)

            balance_out[yr] = {
                "totalAssets":                  total_assets.get(yr),
                "totalCurrentLiabilities":      current_liab.get(yr),
                "totalStockholdersEquity":       equity.get(yr),
                "cashAndCashEquivalents":        cash_and_equiv.get(yr),
                "totalDebt":                    combined_debt,
                "commonStockSharesOutstanding": shares.get(yr),
                "accountsReceivable":           accounts_receivable.get(yr),
            }
            cashflow_out[yr] = {
                "operatingCashFlow":         op_cf.get(yr),
                "capitalExpenditure":        capex.get(yr),
                "commonStockRepurchased":    buybacks.get(yr),
            }

        return {
            "income":   income_out,
            "balance":  balance_out,
            "cashflow": cashflow_out,
            "source":   "edgar",
        }

    except Exception as e:
        print(f"  [EDGAR] {ticker}: {str(e)[:100]}")
        return empty


# =============================================================================
# FMP FINANCIAL STATEMENT FETCHER
#
# Fetches 10 years of annual income statement, balance sheet and cash flow
# from Financial Modeling Prep. Used only for the financials_charts block.
#
# FMP returns lists of dicts keyed by "date" (e.g. "2024-09-28").
# We index by fiscal year (integer) for easy lookup.
#
# Returns:
#   {
#     "income":    { 2024: {...}, 2023: {...}, ... },
#     "balance":   { 2024: {...}, 2023: {...}, ... },
#     "cashflow":  { 2024: {...}, 2023: {...}, ... },
#     "source":    "fmp" | "unavailable"
#   }
#
# Returns source="unavailable" if key is empty, ticker not found, or any
# network/parse error — caller falls back to yfinance in that case.
# =============================================================================
def get_fmp_financials(ticker):
    empty = {"income": {}, "balance": {}, "cashflow": {}, "source": "unavailable"}

    if not FMP_API_KEY or FMP_API_KEY.strip() == "":
        return empty

    base = "https://financialmodelingprep.com/api/v3"
    endpoints = {
        "income":   f"{base}/income-statement/{ticker}?limit=10&apikey={FMP_API_KEY}",
        "balance":  f"{base}/balance-sheet-statement/{ticker}?limit=10&apikey={FMP_API_KEY}",
        "cashflow": f"{base}/cash-flow-statement/{ticker}?limit=10&apikey={FMP_API_KEY}",
    }

    result = {"income": {}, "balance": {}, "cashflow": {}, "source": "unavailable"}

    try:
        for key, url in endpoints.items():
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return empty
            data = resp.json()
            # FMP returns error dicts or empty lists for unknown tickers
            if not data or isinstance(data, dict):
                return empty
            for row in data:
                try:
                    year = int(row["date"][:4])
                    result[key][year] = row
                except (KeyError, ValueError):
                    continue
        result["source"] = "fmp"
        return result
    except Exception as e:
        print(f"  [FMP] {ticker}: {str(e)[:80]}")
        return empty


# =============================================================================
# FINANCIALS CHART DATA
#
# Pre-computes all data needed for the four Financials tab bar charts.
# Eliminates the need for AI Studio to make live GuruFocus API calls.
#
# Chart 1 — Revenue vs Net Income (5yr annual)
#   Source: stock.financials
#
# Chart 2 — Free Cash Flow vs Total Debt (5yr annual)
#   FCF:  stock.cashflow  → "Free Cash Flow" or derived Operating CF - CapEx
#   Debt: stock.balance_sheet → "Total Debt"
#
# Chart 3 — Shares Outstanding vs Buybacks (5yr annual)
#   Shares: stock.balance_sheet → "Ordinary Shares Number" or derived
#   Buybacks: stock.cashflow → "Repurchase Of Capital Stock" (negative = buyback)
#
# Chart 4 — 10-Year Returns Trajectory (ROE and ROIC calculated per year)
#   ROE:  Net Income / Stockholders Equity  (per year)
#   ROIC: NOPAT / Invested Capital          (per year)
#         NOPAT = Operating Income x (1 - tax rate)
#         Invested Capital = Total Assets - Current Liabilities - Excess Cash
#
# All monetary values stored in millions (USD) for readability in charts.
# All values are sanitized — None rather than NaN/Inf.
# =============================================================================
def get_financials_chart_data(stock, info, ticker=""):
    """
    Wraps all chart data fetching in a top-level try/except so that ANY
    exception (EDGAR timeout, FMP error, split logic edge case etc.) returns
    an empty result dict rather than propagating up to analyze_ticker and
    causing the entire ticker to be dropped from the JSON.
    """
    result = {
        "revenue_vs_net_income": {},
        "fcf_vs_debt":           {},
        "shares_vs_buybacks":    {},
        "returns_trajectory":    {},
        "revenue_vs_ar":         {},
        "_notes": {
            "revenue_vs_net_income": "Shows revenue vs net income over time — both should grow together.",
            "fcf_vs_debt": "Compares free cash flow to total debt — rising cash and falling debt is healthy.",
            "shares_vs_buybacks": "Tracks share count changes — buybacks reduce shares and return value to shareholders.",
            "returns_trajectory": "Shows ROE and ROIC over time — above 15% consistently suggests a competitive advantage.",
            "revenue_vs_ar": "Compares revenue growth to accounts receivable growth — AR growing faster than revenue can signal collection problems.",
        },
    }
    try:
        return _get_financials_chart_data_inner(stock, info, ticker)
    except Exception as e:
        print(f"  [CHARTS] {ticker}: chart data failed ({str(e)[:80]}), returning empty")
        return result


def _get_financials_chart_data_inner(stock, info, ticker=""):
    result = {
        "revenue_vs_net_income": {},
        "fcf_vs_debt":           {},
        "shares_vs_buybacks":    {},
        "returns_trajectory":    {},
        "revenue_vs_ar":         {},
        "_notes": {
            "revenue_vs_net_income": (
                "This chart shows how much money the company brings in (revenue) versus "
                "how much it actually keeps as profit (net income) each year. "
                "A healthy company should show both lines growing over time. "
                "If revenue grows but net income shrinks, the company is spending more "
                "than it is earning — a warning sign worth investigating."
            ),
            "fcf_vs_debt": (
                "Free cash flow is the cash left over after the company has paid for "
                "everything it needs to run and grow the business. "
                "This chart compares that cash to the company's total debt. "
                "A company with rising free cash flow and falling debt is in a strong "
                "financial position. If debt is rising faster than cash, the company "
                "may be taking on more risk than it can comfortably manage."
            ),
            "shares_vs_buybacks": (
                "When a company buys back its own shares, it reduces the total number "
                "of shares in circulation — which means each remaining share represents "
                "a larger slice of the business. "
                "This chart shows how the share count has changed over time. "
                "Consistent buybacks are generally a sign that management believes the "
                "stock is undervalued and wants to return cash to shareholders."
            ),
            "returns_trajectory": (
                "Return on Equity (ROE) measures how much profit a company generates "
                "for every dollar of shareholder money invested. "
                "Return on Invested Capital (ROIC) is similar but also includes debt. "
                "Both are expressed as a percentage — higher is better. "
                "A company consistently earning above 15% ROIC is generally considered "
                "to have a genuine competitive advantage over its peers."
            ),
            "revenue_vs_ar": (
                "Accounts receivable is money the company is owed by customers who have "
                "not yet paid their bills. "
                "This chart compares revenue growth to accounts receivable growth each year. "
                "If accounts receivable is growing much faster than revenue, it may mean "
                "the company is struggling to collect payments — sometimes a sign of "
                "aggressive or fictitious revenue reporting. "
                "A healthy company should see both lines move at a similar pace."
            ),
        },
    }

    def to_m(val):
        """Convert raw value to millions, return None if invalid."""
        try:
            v = float(val)
            if not np.isfinite(v): return None
            return round(v / 1_000_000, 2)
        except (TypeError, ValueError):
            return None

    def pct(val):
        """Convert decimal ratio to rounded percentage, return None if invalid."""
        try:
            v = float(val)
            if not np.isfinite(v): return None
            return round(v * 100, 2)
        except (TypeError, ValueError):
            return None

    def _fv(row, *keys):
        """Extract first matching key from an FMP row dict as float or None."""
        for k in keys:
            if k in row:
                try:
                    v = float(row[k])
                    if np.isfinite(v): return v
                except (TypeError, ValueError): pass
        return None

    # ------------------------------------------------------------------
    # Source priority:
    #   1. SEC EDGAR  — free, unlimited, up to 20yr history
    #   2. FMP        — paid fallback, up to 5-10yr
    #   3. yfinance   — last resort, ~4yr
    # ------------------------------------------------------------------

    # Helper — build charts from a generic data dict (same shape for EDGAR/FMP)
    def _build_charts(data, years):
        """
        Populate all four chart dicts from a normalised data source.

        Monetary values (revenue, net income, FCF, debt) are stored in
        millions of USD using to_m() so all charts display consistently.

        Shares outstanding is stored in millions of shares.
        Buybacks are stored in millions of USD.

        EDGAR stores raw full values (e.g. 391_035_000_000).
        FMP stores raw full values in the same way.
        to_m() divides by 1,000,000 and rounds to 2dp for both.
        """
        rv, fd, sb, rt, ra = {}, {}, {}, {}, {}
        for yr in years:
            inc = data["income"].get(yr, {})
            bal = data["balance"].get(yr, {})
            cf  = data["cashflow"].get(yr, {})

            # ----------------------------------------------------------
            # Chart 1 — Revenue vs Net Income (both in $M)
            # ----------------------------------------------------------
            rev = to_m(_fv(inc, "revenue"))
            ni  = to_m(_fv(inc, "netIncome"))
            if rev is not None or ni is not None:
                rv[str(yr)] = {"revenue": rev, "net_income": ni}

            # ----------------------------------------------------------
            # Chart 2 — Cash Balance vs Total Debt (both in $M)
            # Shows actual cash held on balance sheet vs financial debt.
            # FCF is also stored for reference but NOT used as cash proxy —
            # FCF can be negative even when a company holds large cash reserves.
            # ----------------------------------------------------------
            fcf_val = to_m(_fv(cf, "freeCashFlow"))
            if fcf_val is None:
                ocf_v   = _fv(cf, "operatingCashFlow")
                capex_v = _fv(cf, "capitalExpenditure")
                if ocf_v is not None and capex_v is not None:
                    fcf_val = to_m(ocf_v - abs(capex_v))
            # Actual cash balance from balance sheet
            cash_bal = to_m(_fv(bal, "cashAndCashEquivalents"))
            # Prefer totalDebt; fall back to long-term only
            debt = to_m(_fv(bal, "totalDebt"))
            if debt is None:
                debt = to_m(_fv(bal, "longTermDebt"))
            if cash_bal is not None or debt is not None or fcf_val is not None:
                fd[str(yr)] = {
                    "cash_balance":  cash_bal,   # actual cash held — use for chart
                    "free_cash_flow": fcf_val,   # FCF for reference
                    "total_debt":    debt,
                }

            # ----------------------------------------------------------
            # Chart 3 — Shares Outstanding (millions) vs Buybacks ($M)
            # Split adjustment: if latest year shares > 1.5x the next year,
            # a split likely occurred. We normalise all historical share counts
            # to the most recent share count basis so the chart is comparable.
            # This is applied post-loop below.
            # ----------------------------------------------------------
            sh_raw = _fv(bal, "commonStockSharesOutstanding", "sharesOutstanding")
            sh = round(sh_raw / 1_000_000, 2) if sh_raw else None
            bb_raw = _fv(cf, "commonStockRepurchased")
            bb = to_m(abs(bb_raw)) if bb_raw else None
            if sh is not None or bb is not None:
                sb[str(yr)] = {"shares_outstanding_m": sh, "buybacks_m": bb}

            # ----------------------------------------------------------
            # Chart 4 — ROE and ROIC (raw ratios used, not divided by 1M)
            # ----------------------------------------------------------
            roe_val = roic_val = None
            ni_val  = _fv(inc, "netIncome")
            eq_val  = _fv(bal, "totalStockholdersEquity", "stockholdersEquity")
            if ni_val is not None and eq_val and eq_val > 0:
                roe_val = pct(ni_val / eq_val)

            op_inc  = _fv(inc, "operatingIncome")
            pretax  = _fv(inc, "incomeBeforeTax")
            tax_exp = _fv(inc, "incomeTaxExpense")
            ta      = _fv(bal, "totalAssets")
            cl      = _fv(bal, "totalCurrentLiabilities")
            csh     = _fv(bal, "cashAndCashEquivalents") or 0
            rev_v   = _fv(inc, "revenue") or 0
            if op_inc and op_inc > 0 and ta and cl:
                tr = 0.21
                if pretax and pretax > 0 and tax_exp and tax_exp > 0:
                    tr = max(0.0, min(tax_exp / pretax, 0.50))
                nopat   = op_inc * (1 - tr)
                excess  = max(0, csh - 0.02 * rev_v)
                inv_cap = ta - cl - excess
                if inv_cap > 0:
                    roic_val = pct(nopat / inv_cap)
            if roe_val is not None or roic_val is not None:
                rt[str(yr)] = {"roe_pct": roe_val, "roic_pct": roic_val}

            # ----------------------------------------------------------
            # Chart 5 — Revenue vs Accounts Receivable (both in $M)
            # Compares growth rates of revenue and AR.
            # Healthy: revenue growing >= AR growth
            # Red flag: AR growing faster than revenue — may indicate
            #   aggressive revenue recognition, collection problems,
            #   or channel stuffing.
            # ----------------------------------------------------------
            rev_ar  = to_m(_fv(inc, "revenue"))
            ar_val  = to_m(_fv(bal, "accountsReceivable",
                                    "AccountsReceivableNetCurrent"))
            if rev_ar is not None or ar_val is not None:
                # YoY growth rates computed post-loop below
                ra[str(yr)] = {"revenue": rev_ar, "accounts_receivable": ar_val}

        # ── Split-adjust historical share counts ─────────────────────
        # Detect forward splits (share count jumps up) and reverse splits
        # (share count drops). Normalise all years to the most recent
        # share count basis so the chart shows a consistent series.
        # Method: walk chronologically and accumulate a split ratio.
        if sb:
            sorted_yrs = sorted(sb.keys())   # oldest first
            # Build list of (year, shares) for years that have share data
            sh_pairs = [(y, sb[y]["shares_outstanding_m"])
                        for y in sorted_yrs if sb[y].get("shares_outstanding_m")]
            if len(sh_pairs) >= 2:
                # Split detection — walk backwards from most recent year.
                # For each consecutive pair, check if the ratio is close to
                # a known split ratio (2, 3, 4, 5, 10).
                # Uses 30% tolerance to handle cases where shares don't change
                # by exactly the split ratio (e.g. GME 4:1 gives ratio 4.65
                # because shares outstanding changed during the year).
                SPLIT_RATIOS = [2, 3, 4, 5, 10]
                cumulative = 1.0
                adjusted   = {}
                for i in range(len(sh_pairs) - 1, -1, -1):
                    yr_k, sh_val = sh_pairs[i]
                    if i < len(sh_pairs) - 1 and sh_val and sh_val > 0:
                        next_sh = sh_pairs[i + 1][1]
                        ratio   = next_sh / sh_val

                        # Forward split (shares increased going forward in time)
                        matched = False
                        for sr in SPLIT_RATIOS:
                            if abs(ratio - sr) / sr < 0.30:   # within 30% of split ratio
                                cumulative *= sr
                                matched = True
                                break

                        # Reverse split (shares decreased going forward in time)
                        if not matched and 0 < ratio < 0.6:
                            for sr in SPLIT_RATIOS:
                                inv = 1.0 / sr
                                if abs(ratio - inv) / inv < 0.30:
                                    cumulative /= sr
                                    break

                    adjusted[yr_k] = round(sh_val * cumulative, 2) if sh_val else None

                # Write adjusted values back
                for yr_k in sorted_yrs:
                    if yr_k in adjusted and adjusted[yr_k] is not None:
                        sb[yr_k]["shares_outstanding_m"] = adjusted[yr_k]
                        sb[yr_k]["split_adjusted"] = True

        # ── Compute YoY growth rates for Revenue vs AR chart ─────────
        # Growth rate flags: AR growing faster than revenue = red flag
        if ra:
            sorted_ra = sorted(ra.keys())
            for i, yr_k in enumerate(sorted_ra):
                if i == 0:
                    ra[yr_k]["rev_growth_pct"] = None
                    ra[yr_k]["ar_growth_pct"]  = None
                    ra[yr_k]["ar_outpacing"]   = None
                    continue
                prev = ra[sorted_ra[i - 1]]
                curr = ra[yr_k]
                # Revenue YoY %
                prev_rev = prev.get("revenue")
                curr_rev = curr.get("revenue")
                rev_g = None
                if prev_rev and curr_rev is not None and prev_rev != 0:
                    rev_g = round((curr_rev - prev_rev) / abs(prev_rev) * 100, 1)
                # AR YoY %
                prev_ar = prev.get("accounts_receivable")
                curr_ar = curr.get("accounts_receivable")
                ar_g = None
                if prev_ar and curr_ar is not None and prev_ar != 0:
                    ar_g = round((curr_ar - prev_ar) / abs(prev_ar) * 100, 1)
                # Flag if AR outpacing revenue
                ar_outpacing = None
                if rev_g is not None and ar_g is not None:
                    ar_outpacing = ar_g > rev_g + 5   # 5pt buffer to avoid noise
                ra[yr_k]["rev_growth_pct"] = rev_g
                ra[yr_k]["ar_growth_pct"]  = ar_g
                ra[yr_k]["ar_outpacing"]   = ar_outpacing

        return rv, fd, sb, rt, ra

    # --- Attempt 1: SEC EDGAR (free, 20yr) ---
    if ticker:
        edgar = get_edgar_financials(ticker)
        if edgar["source"] == "edgar":
            print(f"  [EDGAR] {ticker}: using EDGAR data for financials charts")
            years = sorted(
                set(edgar["income"]) | set(edgar["balance"]) | set(edgar["cashflow"]),
                reverse=True
            )[:10]
            rv, fd, sb, rt, ra = _build_charts(edgar, years)
            result["revenue_vs_net_income"] = rv
            result["fcf_vs_debt"]           = fd
            result["shares_vs_buybacks"]    = sb
            result["returns_trajectory"]    = rt
            result["revenue_vs_ar"]         = ra
            return result

    # --- Attempt 2: FMP (paid, 5-10yr) ---
    fmp = get_fmp_financials(ticker) if ticker else {"source": "unavailable"}
    if fmp["source"] == "fmp":
        print(f"  [FMP] {ticker}: EDGAR unavailable, using FMP data")
        years = sorted(
            set(fmp["income"]) | set(fmp["balance"]) | set(fmp["cashflow"]),
            reverse=True
        )[:10]
        rv, fd, sb, rt, ra = _build_charts(fmp, years)
        result["revenue_vs_net_income"] = rv
        result["fcf_vs_debt"]           = fd
        result["shares_vs_buybacks"]    = sb
        result["returns_trajectory"]    = rt
        result["revenue_vs_ar"]         = ra
        return result

    # --- Attempt 3: yfinance (~4yr) ---
    print(f"  [EDGAR/FMP] {ticker}: both unavailable, falling back to yfinance")
    try:
        # yfinance annual statements — typically 4 years of history.
        # Use all available columns rather than capping at 5.
        financials = stock.financials
        cashflow   = stock.cashflow
        balance    = stock.balance_sheet

        if financials is None or financials.empty:
            return result

        cols = list(financials.columns)  # all available fiscal years (usually 4)

        # ------------------------------------------------------------------
        # Chart 1 — Revenue vs Net Income
        # ------------------------------------------------------------------
        rev_ni = {}
        for col in cols:
            year = str(col.year)
            rev, ni = None, None
            for name in ["Total Revenue", "TotalRevenue",
                         "Revenue", "Revenues",
                         "Operating Revenue", "OperatingRevenue"]:
                if name in financials.index:
                    rev = to_m(financials.loc[name, col]); break
            for name in ["Net Income", "NetIncome",
                         "Net Income Common Stockholders",
                         "Net Income Including Noncontrolling Interests",
                         "Profit Loss", "ProfitLoss"]:
                if name in financials.index:
                    ni = to_m(financials.loc[name, col]); break
            if rev is not None or ni is not None:
                rev_ni[year] = {"revenue": rev, "net_income": ni}
        result["revenue_vs_net_income"] = rev_ni

        # ------------------------------------------------------------------
        # Chart 2 — Free Cash Flow vs Total Debt
        # ------------------------------------------------------------------
        fcf_debt = {}
        for col in cols:
            year = str(col.year)
            fcf, debt = None, None

            # FCF: try direct first, then derive Operating CF - CapEx
            if cashflow is not None and not cashflow.empty and col in cashflow.columns:
                for name in ["Free Cash Flow", "FreeCashFlow"]:
                    if name in cashflow.index:
                        fcf = to_m(cashflow.loc[name, col]); break
                if fcf is None:
                    ocf, capex = None, None
                    for name in ["Operating Cash Flow", "OperatingCashFlow",
                                 "Cash Flow From Continuing Operating Activities",
                                 "Cash Flows From Used In Operating Activities",
                                 "Net Cash Provided By Operating Activities"]:
                        if name in cashflow.index:
                            ocf = to_m(cashflow.loc[name, col]); break
                    for name in ["Capital Expenditure", "CapitalExpenditure",
                                 "Purchase Of PPE",
                                 "Purchases Of Property Plant And Equipment",
                                 "Purchase Of Property Plant And Equipment"]:
                        if name in cashflow.index:
                            capex = to_m(cashflow.loc[name, col]); break
                    if ocf is not None and capex is not None:
                        fcf = round(ocf - abs(capex), 2)

            # Cash balance from balance sheet — expanded name list
            if balance is not None and not balance.empty and col in balance.columns:
                for name in ["Cash And Cash Equivalents", "CashAndCashEquivalents",
                             "Cash Cash Equivalents And Short Term Investments",
                             "CashCashEquivalentsAndShortTermInvestments",
                             "Cash And Short Term Investments", "Cash"]:
                    if name in balance.index:
                        cash_bal = to_m(balance.loc[name, col]); break

            # Debt — expanded name list, prefer total debt
            if balance is not None and not balance.empty and col in balance.columns:
                for name in ["Total Debt", "TotalDebt",
                             "Long Term Debt And Capital Lease Obligation",
                             "LongTermDebtAndCapitalLeaseObligation",
                             "Long Term Debt", "LongTermDebt",
                             "Short Long Term Debt", "ShortLongTermDebt"]:
                    if name in balance.index:
                        debt = to_m(balance.loc[name, col]); break

            if fcf is not None or debt is not None:
                fcf_debt[year] = {"free_cash_flow": fcf, "total_debt": debt}
        result["fcf_vs_debt"] = fcf_debt

        # ------------------------------------------------------------------
        # Chart 3 — Shares Outstanding vs Buybacks
        # ------------------------------------------------------------------
        shares_buybacks = {}
        for col in cols:
            year  = str(col.year)
            sh, buyback = None, None

            # Shares: try balance sheet first with expanded name list,
            # then fall back to info dict which always has sharesOutstanding
            if balance is not None and not balance.empty and col in balance.columns:
                for name in ["Ordinary Shares Number", "OrdinarySharesNumber",
                             "Share Issued", "ShareIssued",
                             "Common Stock Shares Outstanding",
                             "CommonStockSharesOutstanding",
                             "Shares Outstanding", "SharesOutstanding",
                             "Common Stock", "CommonStock"]:
                    if name in balance.index:
                        try:
                            v = float(balance.loc[name, col])
                            if np.isfinite(v) and v > 1000:  # sanity: > 1000 actual shares
                                sh = round(v / 1_000_000, 2)
                        except (TypeError, ValueError): pass
                        if sh is not None: break

            # Fallback: sharesOutstanding from info (current only — same for all years
            # but better than nothing when balance sheet lookup fails)
            if sh is None:
                raw_sh = _safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
                if raw_sh and raw_sh > 0:
                    sh = round(raw_sh / 1_000_000, 2)

            # Buybacks: repurchase of capital stock (stored as negative in cashflow)
            if cashflow is not None and not cashflow.empty and col in cashflow.columns:
                for name in ["Repurchase Of Capital Stock", "RepurchaseOfCapitalStock",
                             "Common Stock Repurchased", "CommonStockRepurchased",
                             "Purchase Of Business", "RepurchaseOfCommonStock",
                             "Repurchase Of Common Stock",
                             "Payments For Repurchase Of Common Stock"]:
                    if name in cashflow.index:
                        try:
                            v = float(cashflow.loc[name, col])
                            if np.isfinite(v):
                                buyback = round(abs(v) / 1_000_000, 2)
                        except (TypeError, ValueError): pass
                        if buyback is not None: break

            # Always write the year if we have shares (buybacks can legitimately be null)
            if sh is not None:
                shares_buybacks[year] = {
                    "shares_outstanding_m": sh,
                    "buybacks_m":           buyback,
                }
        result["shares_vs_buybacks"] = shares_buybacks

        # ------------------------------------------------------------------
        # Chart 4 — 10-Year Returns Trajectory (ROE and ROIC per year)
        #
        # yfinance stock.financials typically holds only 4 years of annual
        # statement data. To extend towards 10 years we also try:
        #   - stock.get_income_stmt() which may return more columns
        #   - stock.income_stmt (alias)
        # All sources are merged; the most recent value per year wins.
        # Years with no data are omitted cleanly rather than showing nulls.
        # ------------------------------------------------------------------
        returns = {}

        # Collect all available annual income statement sources
        fin_sources = [financials]
        try:
            extra = stock.get_income_stmt(freq="annual", pretty=False)
            if extra is not None and not extra.empty:
                fin_sources.append(extra)
        except Exception: pass
        try:
            alias = stock.income_stmt
            if alias is not None and not alias.empty:
                fin_sources.append(alias)
        except Exception: pass

        # Collect all available annual balance sheet sources
        bal_sources = [balance] if (balance is not None and not balance.empty) else []
        try:
            extra_b = stock.get_balance_sheet(freq="annual", pretty=False)
            if extra_b is not None and not extra_b.empty:
                bal_sources.append(extra_b)
        except Exception: pass

        def _get_row(sources, row_names, col):
            """Return first valid float found across sources for given row names and column."""
            for src in sources:
                if src is None or src.empty: continue
                if col not in src.columns: continue
                for name in row_names:
                    if name in src.index:
                        try:
                            v = float(src.loc[name, col])
                            if np.isfinite(v): return v
                        except (TypeError, ValueError): pass
            return None

        # Collect all fiscal year columns across all sources, up to 10 years
        all_year_cols = {}
        for src in fin_sources:
            if src is None or src.empty: continue
            for col in src.columns:
                yr = str(col.year)
                if yr not in all_year_cols:
                    all_year_cols[yr] = col
        # Keep most recent 10 years, sorted newest first
        sorted_years = sorted(all_year_cols.keys(), reverse=True)[:10]

        for yr in sorted_years:
            col      = all_year_cols[yr]
            roe_val  = None
            roic_val = None

            # ROE = Net Income / Stockholders Equity
            try:
                ni_val = _get_row(fin_sources,
                                  ["Net Income", "NetIncome",
                                   "Net Income Common Stockholders",
                                   "Net Income Including Noncontrolling Interests"], col)
                eq_val = _get_row(bal_sources,
                                  ["Stockholders Equity", "StockholdersEquity",
                                   "Total Stockholder Equity", "TotalStockholderEquity",
                                   "Common Stock Equity", "CommonStockEquity",
                                   "Total Equity Gross Minority Interest",
                                   "Stockholders Equity Attributable To Parent"], col)
                if ni_val is not None and eq_val and eq_val > 0:
                    roe_val = pct(ni_val / eq_val)
            except Exception: pass

            # ROIC = NOPAT / Invested Capital
            try:
                op_inc = _get_row(fin_sources,
                                  ["Operating Income", "OperatingIncome",
                                   "Total Operating Income As Reported",
                                   "EBIT", "Ebit",
                                   "Operating Income Loss"], col)
                if op_inc and op_inc > 0:
                    tax_rate = 0.21
                    try:
                        pretax  = _get_row(fin_sources,
                                           ["Pretax Income", "PretaxIncome",
                                            "Income Before Tax", "IncomeBeforeTax"], col)
                        tax_exp = _get_row(fin_sources,
                                           ["Tax Provision", "TaxProvision",
                                            "Income Tax Expense", "IncomeTaxExpense"], col)
                        if pretax and pretax > 0 and tax_exp and tax_exp > 0:
                            tax_rate = max(0.0, min(tax_exp / pretax, 0.50))
                    except Exception: pass

                    nopat    = op_inc * (1 - tax_rate)
                    ta       = _get_row(bal_sources,
                                        ["Total Assets", "TotalAssets"], col)
                    cl       = _get_row(bal_sources,
                                        ["Current Liabilities", "CurrentLiabilities",
                                         "Total Current Liabilities", "TotalCurrentLiabilities"], col)
                    cash_val = _get_row(bal_sources,
                                        ["Cash And Cash Equivalents", "CashAndCashEquivalents",
                                         "Cash Cash Equivalents And Short Term Investments",
                                         "Cash"], col) or 0
                    if ta and cl:
                        rev_val = _safe(info.get("totalRevenue"), 0)
                        excess  = max(0, cash_val - 0.02 * (rev_val or 0))
                        inv_cap = ta - cl - excess
                        if inv_cap > 0:
                            roic_val = pct(nopat / inv_cap)
            except Exception: pass

            if roe_val is not None or roic_val is not None:
                returns[yr] = {"roe_pct": roe_val, "roic_pct": roic_val}

        result["returns_trajectory"] = returns

        # ------------------------------------------------------------------
        # Chart 5 — Revenue vs Accounts Receivable
        # AR from balance sheet, revenue from income statement.
        # Both in $M. YoY growth rates computed to flag AR outpacing revenue.
        # ------------------------------------------------------------------
        rev_ar_yf = {}
        for col in cols:
            year  = str(col.year)
            rev_v = None
            ar_v  = None
            for name in ["Total Revenue", "TotalRevenue"]:
                if name in financials.index:
                    rev_v = to_m(financials.loc[name, col]); break
            if balance is not None and not balance.empty and col in balance.columns:
                for name in ["Accounts Receivable",
                             "AccountsReceivable",
                             "Net Receivables",
                             "NetReceivables",
                             "Receivables"]:
                    if name in balance.index:
                        ar_v = to_m(balance.loc[name, col]); break
            if rev_v is not None or ar_v is not None:
                rev_ar_yf[year] = {
                    "revenue":             rev_v,
                    "accounts_receivable": ar_v,
                }

        # Compute YoY growth rates and outpacing flag
        sorted_ra_yf = sorted(rev_ar_yf.keys())
        for i, yr_k in enumerate(sorted_ra_yf):
            if i == 0:
                rev_ar_yf[yr_k].update({
                    "rev_growth_pct": None,
                    "ar_growth_pct":  None,
                    "ar_outpacing":   None,
                })
                continue
            prev = rev_ar_yf[sorted_ra_yf[i - 1]]
            curr = rev_ar_yf[yr_k]
            prev_rev, curr_rev = prev.get("revenue"), curr.get("revenue")
            prev_ar,  curr_ar  = prev.get("accounts_receivable"), curr.get("accounts_receivable")
            rev_g = round((curr_rev - prev_rev) / abs(prev_rev) * 100, 1) if prev_rev and curr_rev is not None and prev_rev != 0 else None
            ar_g  = round((curr_ar  - prev_ar)  / abs(prev_ar)  * 100, 1) if prev_ar  and curr_ar  is not None and prev_ar  != 0 else None
            rev_ar_yf[yr_k].update({
                "rev_growth_pct": rev_g,
                "ar_growth_pct":  ar_g,
                "ar_outpacing":   (ar_g > rev_g + 5) if (rev_g is not None and ar_g is not None) else None,
            })
        result["revenue_vs_ar"] = rev_ar_yf

    except Exception:
        pass

    return result


# =============================================================================
# BULL / BEAR THESIS GENERATOR
#
# Combines rule-based signals with Anthropic API (Claude Haiku) to generate
# a pre-computed bull/bear thesis for every ticker.
#
# Step 1 — Rule-based context building:
#   Extracts the strongest bull and bear signals from already-computed data
#   (quality scores, ratios, growth metrics, valuation) into a structured
#   context dict. This ensures the API call is focused and consistent.
#
# Step 2 — Anthropic API call (Claude Haiku):
#   Sends the structured context with a tight prompt asking for exactly
#   3 bull points and 3 bear points in JSON format. Haiku is used for
#   speed and cost (~$0.001 per ticker).
#
# Step 3 — Fallback:
#   If the API call fails for any reason (rate limit, network, invalid
#   response), falls back to the rule-based signals alone. The tab will
#   still display meaningful content rather than nothing.
#
# Output structure:
#   {
#     "bull_points": ["point 1", "point 2", "point 3"],
#     "bear_points": ["point 1", "point 2", "point 3"],
#     "source": "api" | "rules",
#     "generated_at": "2026-03-17 14:30"
#   }
# =============================================================================
ANTHROPIC_API_KEY = ""   # ← paste your Anthropic API key here (optional)
                          #   Leave empty to use rule-based fallback only


def summarise_description(ticker, description, company_name, sector, industry, company_type):
    """
    Pre-compute a 1-2 paragraph company description summary (≤80 words).
    AI Studio reads description_summary directly — zero generation at query time.
    Falls back to a trimmed rule-based version when the API key is absent or the
    call fails.
    """
    if not description or not description.strip():
        return (
            f"{company_name} is a {sector or 'publicly listed'} company"
            f"{' in the ' + industry + ' industry' if industry else ''}. "
            "Detailed business description is unavailable."
        )

    # ── Rule-based fallback: keep only first 2 sentences ────────────────────
    import re
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    fallback = " ".join(sentences[:2]).strip()
    # Hard cap at 120 words
    words = fallback.split()
    if len(words) > 120:
        fallback = " ".join(words[:120]) + "…"

    if not (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip()):
        return fallback

    # ── Haiku API call ────────────────────────────────────────────────────────
    prompt = (
        f"Summarise the following company description for {company_name} ({ticker}) "
        f"into 1–2 short paragraphs of plain prose. "
        f"Maximum 80 words total. Present tense. No bullet points. "
        f"Paragraph 1: what the company does and its main revenue source. "
        f"Paragraph 2 (optional, only if materially different info exists): "
        f"what makes it distinctive. "
        f"Do not start with the company name. No founding year or employee count.\n\n"
        f"Description:\n{description[:2000]}"
    )

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if response.status_code == 200:
            result = response.json()["content"][0]["text"].strip()
            if result:
                return result
    except Exception as e:
        print(f"  [DESC SUMMARY] {ticker}: API call failed ({str(e)[:60]}), using fallback")

    return fallback


def generate_bull_bear(ticker, info, final_scores, fundamentals,
                       fwd_growth, intrinsic_value, forecast_meta):
    """
    Generate bull and bear thesis split into two time horizons:

      short_term — next 3-6 months, driven by recent news, catalysts, sentiment
      long_term  — 1-5 year view, driven by financial fundamentals and moat

    When the Anthropic API key is set, Haiku uses its web_search tool to fetch
    recent news and analyst commentary before writing the short-term points.
    Long-term points are grounded entirely in the financial data.
    Rule-based fallback fires when the API key is absent or the call fails.
    """
    company_name  = info.get("longName", ticker)
    sector        = info.get("sector", "Unknown")
    current_price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))

    mos = None
    if intrinsic_value and intrinsic_value > 0 and current_price and current_price > 0:
        mos = round((intrinsic_value - current_price) / intrinsic_value * 100, 1)

    rev_growth    = _safe(info.get("revenueGrowth"))
    profit_margin = None
    rev           = _safe(info.get("totalRevenue"))
    ni            = _safe(info.get("netIncomeToCommon"))
    if rev and rev > 0 and ni is not None:
        profit_margin = round(ni / rev * 100, 1)

    eps5y = round(fwd_growth.get("raw", 0) * 100, 1) if fwd_growth else 0

    context = {
        "ticker":              ticker,
        "company":             company_name,
        "sector":              sector,
        "current_price":       current_price,
        "intrinsic_value":     intrinsic_value,
        "margin_of_safety_pct": mos,
        "quality_scores": {
            "profitability":      final_scores.get("profitability"),
            "financial_strength": final_scores.get("financial_strength"),
            "growth":             final_scores.get("growth"),
            "predictability":     final_scores.get("predictability"),
            "valuation":          final_scores.get("valuation"),
        },
        "eps_next_5y_pct":     eps5y,
        "roe_pct":             fundamentals.get("roe_pct"),
        "roic_pct":            fundamentals.get("roic_pct"),
        "current_ratio":       fundamentals.get("current_ratio"),
        "debt_to_ebitda":      fundamentals.get("debt_to_ebitda"),
        "revenue_growth_pct":  round(rev_growth * 100, 1) if rev_growth else None,
        "profit_margin_pct":   profit_margin,
        "base_growth_rate":    round(forecast_meta.get("base_growth_rate", 0) * 100, 1),
        "growth_constrained":  forecast_meta.get("growth_constrained", False),
        "constraint_note":     forecast_meta.get("constraint_note"),
    }

    # ── Rule-based fallback points ────────────────────────────────────────────
    st_bull, st_bear = [], []   # short-term
    lt_bull, lt_bear = [], []   # long-term

    # Short-term rules
    if mos and mos > 20:
        st_bull.append(f"Trading at a {mos}% discount to estimated intrinsic value — "
                       f"technical setup favours near-term mean reversion toward ${intrinsic_value}")
    elif mos and mos < -20:
        st_bear.append(f"Trading at a {abs(mos):.0f}% premium to intrinsic value — "
                       f"elevated risk of near-term multiple compression")

    if rev_growth and rev_growth > 0.15:
        st_bull.append(f"Revenue growing at {round(rev_growth*100,1)}% year-over-year, "
                       f"indicating sustained demand momentum heading into the next quarter")
    elif rev_growth and rev_growth < -0.05:
        st_bear.append(f"Declining revenue of {round(rev_growth*100,1)}% may weigh on "
                       f"near-term earnings and investor sentiment")

    if eps5y > 10:
        st_bull.append(f"Analyst consensus of {eps5y}% EPS growth over five years "
                       f"provides a near-term catalyst if the company delivers on guidance")
    elif eps5y < 3:
        st_bear.append(f"Low consensus EPS growth forecast of {eps5y}% p.a. limits "
                       f"re-rating potential in the near term")

    # Long-term rules
    p   = final_scores.get("profitability", 0) or 0
    fs  = final_scores.get("financial_strength", 0) or 0
    pre = final_scores.get("predictability", 0) or 0

    if p >= 8:
        lt_bull.append(f"High profitability score ({p}/10) reflects durable margins and "
                       f"returns that compound value over a multi-year horizon")
    elif p <= 3:
        lt_bear.append(f"Low profitability score ({p}/10) suggests thin or negative margins "
                       f"that structurally limit long-term earnings power")

    if fs >= 8:
        lt_bull.append(f"Strong balance sheet ({fs}/10) provides the financial flexibility "
                       f"to invest through cycles and return capital to shareholders")
    elif fs <= 3:
        lt_bear.append(f"Weak balance sheet ({fs}/10) constrains long-term capital allocation "
                       f"and increases vulnerability in a downturn")

    roic_val = fundamentals.get("roic")
    if roic_val and roic_val > 0.20:
        lt_bull.append(f"ROIC of {fundamentals.get('roic_pct')} significantly exceeds cost of capital "
                       f"— the business creates substantial economic value over time")
    elif roic_val and roic_val < 0.05:
        lt_bear.append(f"ROIC of {fundamentals.get('roic_pct')} is below most cost-of-capital benchmarks, "
                       f"suggesting the business destroys value over the long run")

    if pre >= 8:
        lt_bull.append(f"Highly predictable earnings trajectory ({pre}/10) supports reliable "
                       f"long-term DCF modelling and reduces reinvestment risk")
    elif pre <= 3:
        lt_bear.append(f"Unpredictable earnings ({pre}/10) make long-term forecasting unreliable "
                       f"and compress the valuation multiple the market is willing to pay")

    if forecast_meta.get("growth_constrained"):
        lt_bear.append(f"Financial strength is constraining the assumed growth rate in "
                       f"the valuation model — {forecast_meta.get('constraint_note', '')}")

    # Pad to 3 each
    generic_st_bull = [
        f"Consensus expectations are broadly supportive for {company_name} in the near term",
        f"Current market positioning in the {sector} sector may attract tactical buyers",
        "Any positive earnings surprise could act as a near-term re-rating catalyst",
    ]
    generic_st_bear = [
        f"Broader {sector} sector headwinds could weigh on near-term sentiment",
        "Macro uncertainty and rate expectations may suppress multiple expansion",
        "Short-term earnings visibility is limited without additional guidance",
    ]
    generic_lt_bull = [
        f"{company_name} operates in a sector with long-term structural tailwinds",
        "Management has demonstrated consistent capital allocation discipline",
        "The business has the scale to compound earnings over a multi-year horizon",
    ]
    generic_lt_bear = [
        f"Competitive intensity in the {sector} sector may erode margins over time",
        "Long-term technology or regulatory disruption risk cannot be ruled out",
        "Execution risk remains a key variable over a multi-year investment horizon",
    ]

    while len(st_bull) < 3:
        st_bull.append(generic_st_bull[len(st_bull)])
    while len(st_bear) < 3:
        st_bear.append(generic_st_bear[len(st_bear)])
    while len(lt_bull) < 3:
        lt_bull.append(generic_lt_bull[len(lt_bull)])
    while len(lt_bear) < 3:
        lt_bear.append(generic_lt_bear[len(lt_bear)])

    rule_result = {
        "short_term": {"bull_points": st_bull[:3], "bear_points": st_bear[:3]},
        "long_term":  {"bull_points": lt_bull[:3], "bear_points": lt_bear[:3]},
        "source":      "rules",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ── API call with web search ──────────────────────────────────────────────
    if not (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip()):
        return rule_result

    try:
        prompt = f"""You are a senior equity analyst writing for a professional investing app.
Your task is to generate four sets of investment thesis points for {company_name} ({ticker}).

STEP 1 — Search the web for recent news on {company_name} ({ticker}):
Search for: "{ticker} {company_name} earnings outlook news 2025 2026"
Look for: recent earnings results, guidance updates, analyst upgrades/downgrades, product launches,
regulatory developments, macro tailwinds/headwinds specific to this company.

STEP 2 — Using the financial data below AND the news you found, generate:

Financial Data:
{json.dumps(context, indent=2)}

Generate exactly four arrays:

1. short_term_bull (3 points): Bull arguments for the NEXT 3-6 MONTHS.
   Ground these in recent news, upcoming catalysts, near-term earnings momentum,
   technical setup relative to intrinsic value. Be specific — reference actual events
   or data points you found.

2. short_term_bear (3 points): Bear arguments for the NEXT 3-6 MONTHS.
   Ground these in near-term headwinds, risks to the next earnings print,
   valuation concerns, or negative news. Be specific.

3. long_term_bull (3 points): Bull arguments for a 1-5 YEAR investment horizon.
   Ground these in the company's financial quality (ROIC, margins, balance sheet),
   competitive moat, and long-term growth drivers.

4. long_term_bear (3 points): Bear arguments for a 1-5 YEAR investment horizon.
   Ground these in structural risks, balance sheet constraints, competitive threats,
   or long-term execution uncertainty.

Rules:
- Each point must be 1-2 sentences, specific and factual
- Short-term points must reference something concrete (a news item, metric, or catalyst)
- Long-term points must reference financial data provided above
- Respond ONLY with valid JSON, no other text:

{{
  "short_term_bull": ["point 1", "point 2", "point 3"],
  "short_term_bear": ["point 1", "point 2", "point 3"],
  "long_term_bull":  ["point 1", "point 2", "point 3"],
  "long_term_bear":  ["point 1", "point 2", "point 3"]
}}"""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 1500,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=60,   # web search adds latency
        )

        if response.status_code == 200:
            content_blocks = response.json().get("content", [])
            # Extract the final text block (after any tool_use/tool_result blocks)
            text_blocks = [b["text"] for b in content_blocks if b.get("type") == "text"]
            if text_blocks:
                raw = text_blocks[-1].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                return {
                    "short_term": {
                        "bull_points": parsed.get("short_term_bull", st_bull[:3]),
                        "bear_points": parsed.get("short_term_bear", st_bear[:3]),
                    },
                    "long_term": {
                        "bull_points": parsed.get("long_term_bull", lt_bull[:3]),
                        "bear_points": parsed.get("long_term_bear", lt_bear[:3]),
                    },
                    "source":       "api",
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        else:
            print(f"  [BULL/BEAR] {ticker}: API returned {response.status_code}, using rules")

    except Exception as e:
        print(f"  [BULL/BEAR] {ticker}: API call failed ({str(e)[:60]}), using rules")

    return rule_result




def generate_quality_narrative(ticker, info, final_scores, score_labels,
                                classification, python_subtotal_pct,
                                fundamentals, fwd_growth, forecast_meta,
                                intrinsic_value, company_type):
    """
    Pre-compute the three Quality tab narrative blocks so AI Studio
    reads them from the JSON rather than generating them at query time.

    Blocks:
      quality_conclusion  — 2-3 sentence overall assessment of business quality
      risk_profile        — 2-3 sentence risk summary covering key vulnerabilities
      investment_summary  — 2-3 sentence summary of the investment case

    Each block has a rule-based fallback that fires when the API key is
    absent, ensuring the app always has something to display.

    All three are generated in a single Haiku API call to minimise
    pipeline runtime (one call per ticker vs three).
    """
    company_name   = info.get("longName") or info.get("shortName") or ticker
    sector         = info.get("sector", "Unknown")
    current_price  = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    mos            = None
    if intrinsic_value and intrinsic_value > 0 and current_price and current_price > 0:
        mos = round((intrinsic_value - current_price) / intrinsic_value * 100, 1)

    rev_growth    = _safe(info.get("revenueGrowth"))
    profit_margin = None
    rev_total     = _safe(info.get("totalRevenue"))
    ni_total      = _safe(info.get("netIncomeToCommon"))
    if rev_total and rev_total > 0 and ni_total is not None:
        profit_margin = round(ni_total / rev_total * 100, 1)

    p   = final_scores.get("profitability", 0)   or 0
    fs  = final_scores.get("financial_strength", 0) or 0
    g   = final_scores.get("growth", 0)           or 0
    pre = final_scores.get("predictability", 0)   or 0
    v   = final_scores.get("valuation", 0)        or 0

    p_lbl   = score_labels.get("profitability",      "")
    fs_lbl  = score_labels.get("financial_strength", "")
    g_lbl   = score_labels.get("growth",             "")
    pre_lbl = score_labels.get("predictability",     "")
    v_lbl   = score_labels.get("valuation",          "")

    roe_pct  = fundamentals.get("roe_pct")
    roic_pct = fundamentals.get("roic_pct")
    d_ebitda = fundamentals.get("debt_to_ebitda")
    cur_rat  = fundamentals.get("current_ratio")
    eps5y    = round(fwd_growth.get("raw", 0) * 100, 1)

    # ── Rule-based fallback texts ─────────────────────────────────────────────
    # Quality conclusion
    quality_words = {
        "Safe":        "strong fundamental quality",
        "Speculative": "mixed fundamental quality",
        "Dangerous":   "weak fundamental quality",
    }
    q_word = quality_words.get(classification, "moderate quality")

    if python_subtotal_pct >= 70:
        qual_strength = "scores highly across profitability, financial strength, and predictability"
    elif python_subtotal_pct >= 55:
        qual_strength = f"shows particular strength in {p_lbl.lower()} profitability" \
                        if p >= fs else f"benefits from {fs_lbl.lower()} financial strength"
    else:
        weak_dim = min(
            [("profitability", p), ("financial strength", fs),
             ("growth", g), ("predictability", pre)],
            key=lambda x: x[1]
        )[0]
        qual_strength = f"is constrained by {weak_dim}"

    roic_text = f", supported by a {roic_pct}% ROIC" if roic_pct and float(roic_pct.rstrip('%') if isinstance(roic_pct, str) else roic_pct) > 12 else ""
    rule_quality_conclusion = (
        f"{company_name} demonstrates {q_word}, scoring {round(python_subtotal_pct)}% across "
        f"four fundamental dimensions. The business {qual_strength}{roic_text}. "
        f"{'Moat assessment is pending and will adjust the final classification.' if classification == 'Speculative' else f'The current classification is {classification}.'}"
    )

    # Risk profile
    risks = []
    if d_ebitda and d_ebitda > 3.0:
        risks.append(f"elevated leverage (Debt/EBITDA: {d_ebitda:.1f}x)")
    if cur_rat and cur_rat < 1.0:
        risks.append(f"tight liquidity (current ratio: {cur_rat:.2f})")
    if g <= 3:
        risks.append("weak near-term growth outlook")
    if pre <= 3:
        risks.append("inconsistent earnings trajectory")
    if forecast_meta.get("growth_constrained"):
        risks.append("financial strength constraining growth assumptions")
    if mos and mos < -20:
        risks.append(f"valuation premium ({abs(mos):.0f}% above estimated intrinsic value)")
    if eps5y < 3:
        risks.append(f"low consensus EPS growth forecast ({eps5y}% p.a.)")

    if risks:
        risk_list = "; ".join(risks[:3])
        rule_risk_profile = (
            f"Key risks for {company_name} include {risk_list}. "
            f"{'The company operates in the ' + sector + ' sector, which carries its own cyclical and regulatory exposures.' if sector != 'Unknown' else 'Broader market and macroeconomic conditions present additional uncertainty.'} "
            f"{'A financial strength score of ' + str(fs) + '/10 suggests the balance sheet provides limited buffer against downside scenarios.' if fs <= 4 else 'The balance sheet provides a reasonable buffer against near-term downside scenarios.'}"
        )
    else:
        rule_risk_profile = (
            f"{company_name} presents a relatively contained risk profile with no major "
            f"red flags across leverage, liquidity, or earnings consistency. "
            f"{'The primary risk is valuation — the stock trades above estimated intrinsic value.' if mos and mos < 0 else 'The primary risks are macroeconomic and sector-level rather than company-specific.'}"
        )

    # Investment summary
    if mos and mos > 20:
        val_text = f"trades at a {mos:.0f}% discount to estimated intrinsic value"
    elif mos and mos < -10:
        val_text = f"trades at a {abs(mos):.0f}% premium to estimated intrinsic value"
    else:
        val_text = "trades near estimated intrinsic value"

    if classification == "Safe":
        case_text = f"represents a {q_word} business that {val_text}"
        action_text = "suitable for investors seeking quality-oriented exposure"
    elif classification == "Speculative":
        case_text = f"is a {q_word} business that {val_text}"
        action_text = "appropriate for investors with higher risk tolerance and conviction in the growth outlook"
    else:
        case_text = f"carries {q_word} with elevated risk factors, and {val_text}"
        action_text = "warrants caution — position sizing and ongoing monitoring are advisable"

    rule_investment_summary = (
        f"{company_name} {case_text}. "
        f"{'With ' + str(eps5y) + '% consensus EPS growth forecast, the growth outlook ' + ('supports the investment case' if eps5y > 8 else 'adds uncertainty to the thesis') + '.' if eps5y else ''} "
        f"Overall, the stock {action_text}."
    ).strip()

    # ── API call — all three in one request ───────────────────────────────────
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip():
        context = {
            "ticker":             ticker,
            "company":            company_name,
            "sector":             sector,
            "company_type":       company_type,
            "classification":     classification,
            "quality_score_pct":  round(python_subtotal_pct),
            "scores": {
                "profitability":      f"{p}/10 ({p_lbl})",
                "financial_strength": f"{fs}/10 ({fs_lbl})",
                "growth":             f"{g}/10 ({g_lbl})",
                "predictability":     f"{pre}/10 ({pre_lbl})",
                "valuation":          f"{v}/10 ({v_lbl})",
            },
            "roe_pct":            roe_pct,
            "roic_pct":           roic_pct,
            "debt_to_ebitda":     d_ebitda,
            "current_ratio":      cur_rat,
            "revenue_growth_pct": round(rev_growth * 100, 1) if rev_growth else None,
            "profit_margin_pct":  profit_margin,
            "eps_next_5y_pct":    eps5y,
            "current_price":      current_price,
            "intrinsic_value":    intrinsic_value,
            "margin_of_safety_pct": mos,
            "growth_constrained": forecast_meta.get("growth_constrained"),
        }

        prompt = f"""You are a senior equity analyst writing for a professional investing app.
Based on the financial data below, generate narrative content for {company_name} ({ticker}).

Data:
{json.dumps(context, indent=2)}

Generate ALL of the following in a single JSON response:

1. moat_score: An integer from 0-10 representing the strength of the company's competitive moat.
   Score on: pricing power, switching costs, network effects, cost advantages, intangible assets.
   Use the financial data provided (ROIC, margins, predictability, growth consistency) as evidence.
   10=exceptional durable moat, 7-9=strong moat, 4-6=moderate moat, 1-3=weak moat, 0=no moat.

2. moat_rationale: One sentence explaining the moat score. Be specific to this company.

3. metric_assessments: A dict with one plain-English assessment per metric (1-2 sentences each).
   Write for a beginner investor. Explain what the score means in practice — what the company
   is actually doing well or poorly, and why it matters. Do NOT use labels like "weak", "strong",
   "good", "adequate", or "poor" — describe the reality behind the number instead.
   Keys: "profitability", "financial_strength", "growth", "predictability", "valuation"

4. quality_conclusion: 2-3 sentences on overall business quality. Mention classification
   ({classification}), strongest and weakest dimensions, and what that means for the business.

5. risk_profile: 2-3 sentences on key risks. Reference actual metrics, not generic sector risks.

6. investment_summary: 2-3 sentences on the investment case covering quality, valuation,
   and growth outlook. End with a clear one-line stance.

Rules:
- For {company_type} companies, do not flag high leverage as a risk if financial_strength ≥ 7
- Be specific, factual, and grounded in the data provided
- Respond ONLY with valid JSON, no other text:

{{
  "moat_score": 0,
  "moat_rationale": "...",
  "metric_assessments": {{
    "profitability": "...",
    "financial_strength": "...",
    "growth": "...",
    "predictability": "...",
    "valuation": "..."
  }},
  "quality_conclusion": "...",
  "risk_profile": "...",
  "investment_summary": "..."
}}"""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      "claude-haiku-4-5-20251001",
                    "max_tokens": 1200,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            if response.status_code == 200:
                raw = response.json()["content"][0]["text"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                return {
                    "moat_score":          parsed.get("moat_score"),
                    "moat_rationale":      parsed.get("moat_rationale", ""),
                    "metric_assessments":  parsed.get("metric_assessments", {}),
                    "quality_conclusion":  parsed.get("quality_conclusion",  rule_quality_conclusion),
                    "risk_profile":        parsed.get("risk_profile",        rule_risk_profile),
                    "investment_summary":  parsed.get("investment_summary",  rule_investment_summary),
                    "source":              "api",
                    "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        except Exception as e:
            print(f"  [NARRATIVE] {ticker}: API call failed ({str(e)[:60]}), using rules")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    return {
        "moat_score":          None,
        "moat_rationale":      "",
        "metric_assessments":  {},
        "quality_conclusion":  rule_quality_conclusion,
        "risk_profile":        rule_risk_profile,
        "investment_summary":  rule_investment_summary,
        "source":              "rules",
        "generated_at":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _sanitize(obj):
    """
    Recursively walks the output structure and replaces any float NaN or
    Infinity with None (serialized as JSON null). Python's json.dump allows
    NaN by default but it is not valid JSON and breaks browser JSON.parse.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None   # NaN and ±Inf → null
    if isinstance(obj, str):
        # Strip control characters that break JSON parsing in the browser
        return "".join(c for c in obj if ord(c) >= 32 or c in ("\t", "\n", "\r"))
    return obj


# Max bytes per sub-partition file before starting a new chunk.
# ~1.5MB keeps files well under GitHub's "too large to display" threshold
# (~2MB) and loads fast in the browser.
MAX_PARTITION_BYTES = 1_500_000

def save_partitioned_data(master_results):
    """
    Saves ticker data into size-capped sub-partition files.

    Files are named  data/stocks_{LETTER}{N}.json
    e.g.  stocks_A1.json, stocks_A2.json, stocks_B1.json ...

    When a file would exceed MAX_PARTITION_BYTES the current chunk is
    flushed and a new one starts.  Typical result: ~30-80 tickers per
    file, 2-4 files per letter, ~80-120 files total.

    A ticker index  data/ticker_map.json  maps every symbol to its
    filename so the app can fetch the right file without guessing.
      { "AAPL": "stocks_A1", "MSFT": "stocks_M1", ... }
    """
    # Group tickers by first letter
    by_letter = {}
    for ticker, data in master_results.items():
        letter = ticker[0].upper() if ticker[0].isalpha() else "0"
        by_letter.setdefault(letter, {})[ticker] = _sanitize(data)

    os.makedirs('data', exist_ok=True)
    ticker_map = {}   # symbol → file stem (without .json)
    files_written = 0

    for letter in sorted(by_letter.keys()):
        tickers_in_letter = by_letter[letter]
        chunk = {}
        chunk_num = 1

        for ticker, data in tickers_in_letter.items():
            chunk[ticker] = data
            # Check size after adding — serialize to measure
            approx_size = len(json.dumps(chunk))
            if approx_size >= MAX_PARTITION_BYTES:
                # Flush current chunk (without the ticker that tipped it)
                chunk.pop(ticker)
                if chunk:
                    stem = f"stocks_{letter}{chunk_num}"
                    with open(f'data/{stem}.json', 'w') as f:
                        json.dump(chunk, f, indent=2)
                    for t in chunk:
                        ticker_map[t] = stem
                    files_written += 1
                    chunk_num += 1
                # Start fresh chunk with the current ticker
                chunk = {ticker: data}

        # Flush remaining tickers in this letter
        if chunk:
            stem = f"stocks_{letter}{chunk_num}"
            with open(f'data/{stem}.json', 'w') as f:
                json.dump(chunk, f, indent=2)
            for t in chunk:
                ticker_map[t] = stem
            files_written += 1

    # Save ticker → file mapping so the app knows where to fetch
    with open('data/ticker_map.json', 'w') as f:
        json.dump(ticker_map, f, indent=2)

    print(f"✅ Saved {len(master_results)} tickers across {files_written} files")
    print(f"✅ Saved ticker_map.json ({len(ticker_map)} entries)")

# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    tickers = get_all_tickers()
    master_results = {}
    print(f"Starting analysis of {len(tickers)} tickers with 3 safe threads...")
    print(f"Using 3 threads with 1s sleep to avoid yfinance rate limiting...")
    skipped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
