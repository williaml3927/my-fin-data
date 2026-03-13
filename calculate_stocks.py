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
print(“Warning: finvizfinance not installed. Falling back to yfinance growth rates.”)

# =============================================================================

# VALUATION CONFIG — tiered by fundamental quality (two-stage growth model)

# =============================================================================

TIER_CONFIG = {
“Strong”: {
“discount_rate”:   0.075,
“growth_stage1”:   0.12,
“growth_stage2”:   0.06,
“terminal_growth”: 0.04,
},
“Average”: {
“discount_rate”:   0.10,
“growth_stage1”:   0.07,
“growth_stage2”:   0.04,
“terminal_growth”: 0.03,
},
“Weak”: {
“discount_rate”:   0.12,
“growth_stage1”:   0.03,
“growth_stage2”:   0.02,
“terminal_growth”: 0.02,
},
}

DCF_YEARS     = 20
HISTORY_YEARS = 5

# =============================================================================

# MULTIPLE CAPS

# =============================================================================

MAX_PE       = 50
MAX_PS       = 20
MAX_PB       = 20
MAX_EV_EBITDA = 40   # company’s own EV/EBITDA capped at 40x to filter distortion

# =============================================================================

# GROWTH RATE CAPS

# 

# MAX_GROWTH_RATE    : 30% ceiling on DCF stage 1 growth

# MAX_PEG_PSG_GROWTH : 30% ceiling specifically for PEG and PSG formulas

# MIN_GROWTH_RATE    : 2% floor for DCF so shrinking firms still get a value

# 

# IMPORTANT — Finviz “-” handling:

# If Finviz returns “-” for EPS Next 5Y it means analysts forecast

# no meaningful growth. This is treated as EXACTLY 0.0 — not as a

# fallback to yfinance. Zero growth is intentional and meaningful data.

# PEG and PSG will return None when growth = 0 (no valid result).

# DCF methods apply the MIN_GROWTH_RATE floor so they still produce a value.

# =============================================================================

MAX_GROWTH_RATE    = 0.30
MAX_PEG_PSG_GROWTH = 0.30
MIN_GROWTH_RATE    = 0.02

# =============================================================================

# TERMINAL VALUE CAP

# =============================================================================

TERMINAL_VALUE_CAP_MULTIPLIER = 3.0

# =============================================================================

# TIER WEIGHTS

# =============================================================================

TIER_WEIGHTS = {
“Strong”: {
“1_DCF_Operating_CF”:   0.15,
“2_DCF_FCF”:            0.15,
“3_DCF_Net_Income”:     0.08,
“4_DCF_Terminal_Value”: 0.15,
“5_Mean_PS”:            0.08,
“6_Mean_PE”:            0.10,
“7_Mean_PB”:            0.05,
“8_PSG”:                0.08,
“9_PEG”:                0.06,
“10_EV_EBITDA”:         0.10,
},
“Average”: {
“1_DCF_Operating_CF”:   0.12,
“2_DCF_FCF”:            0.12,
“3_DCF_Net_Income”:     0.10,
“4_DCF_Terminal_Value”: 0.12,
“5_Mean_PS”:            0.10,
“6_Mean_PE”:            0.12,
“7_Mean_PB”:            0.08,
“8_PSG”:                0.08,
“9_PEG”:                0.08,
“10_EV_EBITDA”:         0.08,
},
“Weak”: {
“1_DCF_Operating_CF”:   0.08,
“2_DCF_FCF”:            0.08,
“3_DCF_Net_Income”:     0.08,
“4_DCF_Terminal_Value”: 0.08,
“5_Mean_PS”:            0.12,
“6_Mean_PE”:            0.08,
“7_Mean_PB”:            0.15,
“8_PSG”:                0.08,
“9_PEG”:                0.08,
“10_EV_EBITDA”:         0.17,
},
}

# =============================================================================

# SOURCES

# =============================================================================

NYSE_URL  = “https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json”
OTHER_URL = “https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json”
CORE_PRIORITY = [
“AAPL”, “TSLA”, “MSFT”, “NVDA”, “GOOGL”,
“AMZN”, “META”, “BRK-B”, “LLY”, “AVGO”
]

# =============================================================================

# TICKER LOADER

# =============================================================================

def get_all_tickers():
tickers = CORE_PRIORITY.copy()
for url in [NYSE_URL, OTHER_URL]:
try:
res  = requests.get(url)
data = res.json()
for item in data:
symbol = (
item.get(‘Symbol’) or
item.get(‘ACT Symbol’) or
item.get(‘ticker’)
)
if symbol:
clean = symbol.strip().upper().replace(’.’, ‘-’)
if clean not in tickers:
tickers.append(clean)
except Exception as e:
print(f”Could not read {url}: {e}”)
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

# Fetches “EPS Next 5Y” from Finviz — analyst consensus forward growth.

# More accurate than yfinance trailing earningsGrowth for DCF inputs.

# 

# Finviz return value handling:

# “12.50%”  → 0.125  (valid positive growth — use it)

# “-”        → 0.0   (analysts forecast no growth — treat as EXACTLY zero,

# do NOT fall through to yfinance)

# “N/A”      → fall through to yfinance

# exception  → fall through to yfinance

# 

# Fallback chain:

# 1. Finviz EPS next 5Y   (best  — forward analyst consensus)

# 2. yfinance earningsGrowth (ok — trailing, may be distorted)

# 3. MIN_GROWTH_RATE 2%   (floor — DCF still produces a value)

# 

# PEG and PSG use growth_rate directly. When growth = 0.0 those methods

# return None (no valid Lynch-formula result at zero growth), which is

# correct — they are excluded from the weighted mean for that ticker.

# 

# Returns:

# {

# “rate”:   float   — rate to use in DCF (floored at MIN_GROWTH_RATE)

# “peg_rate”: float — rate to use in PEG/PSG (NOT floored, can be 0.0)

# “source”: str     — “finviz” | “finviz_no_growth” | “yfinance” | “default”

# “raw”:    float   — unclamped value

# }

# =============================================================================

def get_forward_growth(ticker, info):
# — Attempt 1: Finviz EPS Next 5Y —
if FINVIZ_AVAILABLE:
try:
fundamentals = fvz(ticker).ticker_fundament()
raw_str      = fundamentals.get(“EPS next 5Y”, “”)

```
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
```

# =============================================================================

# QUALITY METRIC 1 — Profitability (0–10)

# =============================================================================

def score_profitability(info):
score = 0
rev = _safe(info.get(‘totalRevenue’))
ni  = _safe(info.get(‘netIncomeToCommon’))
if rev and rev > 0 and ni is not None:
margin = ni / rev
if margin > 0.15:    score += 3
elif margin > 0.05:  score += 2
elif margin > 0:     score += 1
roe = _safe(info.get(‘returnOnEquity’))
if roe is not None:
if roe > 0.20:    score += 3
elif roe > 0.10:  score += 2
elif roe > 0:     score += 1
fcf = _safe(info.get(‘freeCashflow’))
if rev and rev > 0 and fcf is not None:
fcf_margin = fcf / rev
if fcf_margin > 0.12:   score += 2
elif fcf_margin > 0.05: score += 1
roa = _safe(info.get(‘returnOnAssets’))
if roa is not None:
if roa > 0.08:   score += 2
elif roa > 0.03: score += 1
return min(score, 10)

# =============================================================================

# QUALITY METRIC 2 — Financial Strength (0–10)

# =============================================================================

def score_financial_strength(info):
score  = 0
debt   = _safe(info.get(‘totalDebt’), 0)
ebitda = _safe(info.get(‘ebitda’))
if ebitda and ebitda > 0:
ratio = debt / ebitda
if ratio < 1:    score += 3
elif ratio < 3:  score += 2
elif ratio < 5:  score += 1
cr = _safe(info.get(‘currentRatio’))
if cr is not None:
if cr > 2.0:    score += 3
elif cr > 1.2:  score += 2
elif cr > 0.8:  score += 1
interest = _safe(info.get(‘interestExpense’))
if ebitda and ebitda > 0 and interest and interest > 0:
coverage = ebitda / interest
if coverage > 10:  score += 2
elif coverage > 5: score += 1
fcf = _safe(info.get(‘freeCashflow’))
if fcf is not None and fcf > 0:
score += 2
return min(score, 10)

# =============================================================================

# QUALITY METRIC 3 — Growth (0–10)

# =============================================================================

def score_growth(info, financials):
score = 0
rev_growth  = _safe(info.get(‘revenueGrowth’))
if rev_growth is not None:
if rev_growth > 0.15:   score += 3
elif rev_growth > 0.05: score += 2
elif rev_growth > 0:    score += 1
earn_growth = _safe(info.get(‘earningsGrowth’))
if earn_growth is not None:
if earn_growth > 0.20:   score += 3
elif earn_growth > 0.05: score += 2
elif earn_growth > 0:    score += 1
if financials is not None and not financials.empty:
rev_series = _get_annual_series(financials, [‘Total Revenue’, ‘TotalRevenue’])
if rev_series:
score += _trend_score(rev_series[-4:], max_pts=2)
ni_series  = _get_annual_series(financials, [‘Net Income’, ‘NetIncome’])
if ni_series:
score += _trend_score(ni_series[-4:], max_pts=2)
return min(score, 10)

# =============================================================================

# QUALITY METRIC 4 — Predictability (0–10)

# =============================================================================

def score_predictability(info, financials):
score = 0
if financials is not None and not financials.empty:
ni_series  = _get_annual_series(financials, [‘Net Income’, ‘NetIncome’])
if ni_series:
score += _trend_score(ni_series[-5:], max_pts=3)
rev_series = _get_annual_series(financials, [‘Total Revenue’, ‘TotalRevenue’])
if rev_series:
score += _trend_score(rev_series[-5:], max_pts=3)
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
fcf = _safe(info.get(‘freeCashflow’))
ocf = _safe(info.get(‘operatingCashflow’))
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
current_price = (
_safe(info.get(‘currentPrice’)) or
_safe(info.get(‘regularMarketPrice’))
)
if current_price and current_price > 0 and intrinsic_value and intrinsic_value > 0:
mos = (intrinsic_value - current_price) / intrinsic_value
if mos > 0.40:    score += 5
elif mos > 0.20:  score += 4
elif mos > 0:     score += 3
elif mos > -0.20: score += 2
elif mos > -0.40: score += 1
pe = _safe(info.get(‘trailingPE’))
if pe is not None and pe > 0:
if pe < 15:   score += 2
elif pe < 40: score += 1
ps = _safe(info.get(‘priceToSalesTrailing12Months’))
if ps is not None and ps < 2:
score += 1
pb = _safe(info.get(‘priceToBook’))
if pb is not None and pb > 0:
if pb < 1:   score += 2
elif pb < 5: score += 1
return min(score, 10)

# =============================================================================

# QUALITY CLASSIFICATION

# =============================================================================

def classify_quality(scores: dict, subtotal_pct: float) -> dict:
p   = scores.get(‘profitability’, 0) or 0
fs  = scores.get(‘financial_strength’, 0) or 0
g   = scores.get(‘growth’, 0) or 0
pre = scores.get(‘predictability’, 0) or 0
m   = scores.get(‘moat’)

```
label = None

if m is not None and all(x >= 7 for x in [m, pre, p, fs]):
    label = "Safe"
if label is None and m is not None and m <= 4 and fs <= 4:
    label = "Dangerous"
if label is None and m is not None:
    if (m >= 7 and fs >= 7) or (m + fs >= 15) or (m >= 7 and g >= 7):
        label = "Speculative"
if label is None:
    if subtotal_pct >= 70:
        label = "Safe"
    elif subtotal_pct >= 66:
        label = "Safe"
    elif subtotal_pct >= 60:
        label = "Speculative"
    elif subtotal_pct >= 51:
        label = "Speculative"
    else:
        label = "Dangerous"

penalty = 0
if label == "Speculative": penalty = 5
elif label == "Dangerous": penalty = 10

return {
    "label":           label,
    "penalty_pct":     penalty,
    "final_score_pct": max(0.0, round(subtotal_pct - penalty, 2)),
}
```

# =============================================================================

# TIER RESOLVER

# =============================================================================

def resolve_tier(label: str) -> dict:
mapping   = {“Safe”: “Strong”, “Speculative”: “Average”, “Dangerous”: “Weak”}
tier_name = mapping.get(label, “Average”)
config    = TIER_CONFIG[tier_name]
return {
“tier”:            tier_name,
“discount_rate”:   config[“discount_rate”],
“growth_stage1”:   config[“growth_stage1”],
“growth_stage2”:   config[“growth_stage2”],
“terminal_growth”: config[“terminal_growth”],
}

# =============================================================================

# ROE AND ROIC CALCULATOR

# =============================================================================

def calc_roe_roic(info, financials, balance):
result = {
“roe”:         None,
“roic”:        None,
“roe_pct”:     None,
“roic_pct”:    None,
“roic_source”: “unavailable”,
}
roe = _safe(info.get(“returnOnEquity”))
if roe is not None:
result[“roe”]     = round(roe, 4)
result[“roe_pct”] = f”{round(roe * 100, 2)}%”
try:
if financials is None or financials.empty: return result
if balance    is None or balance.empty:    return result
fin_col = financials.columns[0]
bal_col = balance.columns[0]
op_inc  = None
for name in [“Operating Income”, “OperatingIncome”,
“Total Operating Income As Reported”]:
if name in financials.index:
op_inc = _safe(float(financials.loc[name, fin_col]))
break
if op_inc is None or op_inc <= 0:
return result
tax_rate = 0.21
try:
pretax  = None
for name in [“Pretax Income”, “PretaxIncome”,
“Income Before Tax”, “IncomeBeforeTax”]:
if name in financials.index:
pretax = _safe(float(financials.loc[name, fin_col]))
break
tax_exp = None
for name in [“Tax Provision”, “TaxProvision”,
“Income Tax Expense”, “IncomeTaxExpense”]:
if name in financials.index:
tax_exp = _safe(float(financials.loc[name, fin_col]))
break
if pretax and pretax > 0 and tax_exp and tax_exp > 0:
tax_rate = max(0.0, min(tax_exp / pretax, 0.50))
except Exception:
pass
nopat        = op_inc * (1 - tax_rate)
total_assets = None
for name in [“Total Assets”, “TotalAssets”]:
if name in balance.index:
total_assets = _safe(float(balance.loc[name, bal_col]))
break
current_liab = None
for name in [“Current Liabilities”, “CurrentLiabilities”,
“Total Current Liabilities”, “TotalCurrentLiabilities”]:
if name in balance.index:
current_liab = _safe(float(balance.loc[name, bal_col]))
break
cash = (
_safe(float(balance.loc[“Cash And Cash Equivalents”, bal_col]))
if “Cash And Cash Equivalents” in balance.index else
_safe(float(balance.loc[“CashAndCashEquivalents”, bal_col]))
if “CashAndCashEquivalents” in balance.index else
_safe(info.get(“totalCash”), 0)
)
rev          = _safe(info.get(“totalRevenue”), 0)
excess_cash  = max(0, (cash or 0) - 0.02 * (rev or 0))
if total_assets is None or current_liab is None:
return result
invested_capital = total_assets - current_liab - excess_cash
if invested_capital <= 0:
return result
roic                  = nopat / invested_capital
result[“roic”]        = round(roic, 4)
result[“roic_pct”]    = f”{round(roic * 100, 2)}%”
result[“roic_source”] = “calculated”
except Exception:
pass
return result

# =============================================================================

# HISTORICAL MEAN MULTIPLES  (3–5yr average P/E, P/S, P/B and EV/EBITDA)

# 

# EV/EBITDA uses the company’s own historical average multiple — NOT a sector

# median — so the valuation reflects what the market has historically been

# willing to pay for THIS company’s earnings power specifically.

# Capped at MAX_EV_EBITDA (40x) to filter distorted years.

# =============================================================================

def get_historical_mean_multiples(stock, shares):
result = {
“mean_pe”:       None,
“mean_ps”:       None,
“mean_pb”:       None,
“mean_ev_ebitda”: None,
}
try:
financials = stock.financials
balance    = stock.balance_sheet
hist       = stock.history(period=“10y”)

```
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
```

# =============================================================================

# VALUATION METHODS 1–10

# =============================================================================

def calc_dcf_operating_cf(info, shares, growth_s1, growth_s2, discount):
ocf = _safe(info.get(‘operatingCashflow’))
if ocf is None or shares <= 0: return None
return _dcf_sum(ocf / shares, growth_s1, growth_s2, discount)

def calc_dcf_fcf(info, shares, growth_s1, growth_s2, discount):
fcf = _safe(info.get(‘freeCashflow’))
if fcf is None or shares <= 0: return None
return _dcf_sum(fcf / shares, growth_s1, growth_s2, discount)

def calc_dcf_net_income(info, shares, growth_s1, growth_s2, discount):
ni = _safe(info.get(‘netIncomeToCommon’))
if ni is None or shares <= 0: return None
return _dcf_sum(ni / shares, growth_s1, growth_s2, discount)

def calc_dcf_terminal_value(info, shares, growth_s1, growth_s2, discount, terminal_growth, dcf_fcf_value=None):
fcf = _safe(info.get(‘freeCashflow’))
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
if dcf_fcf_value and dcf_fcf_value > 0:
cap = dcf_fcf_value * TERMINAL_VALUE_CAP_MULTIPLIER
if result > cap:
result = round(cap, 2)
return result

def calc_mean_ps(info, shares, mean_ps):
rev = _safe(info.get(‘totalRevenue’))
if rev is None or shares <= 0: return None
rev_ps = rev / shares
if rev_ps <= 0: return None
ratio  = mean_ps if mean_ps is not None else _safe(info.get(‘priceToSalesTrailing12Months’))
if ratio is None or ratio <= 0: return None
if ratio > MAX_PS: return None
return round(ratio * rev_ps, 2)

def calc_mean_pe(info, mean_pe):
eps   = _safe(info.get(‘trailingEps’))
if eps is None or eps <= 0: return None
ratio = mean_pe if mean_pe is not None else _safe(info.get(‘trailingPE’))
if ratio is None or ratio <= 0: return None
if ratio > MAX_PE: return None
return round(ratio * eps, 2)

def calc_mean_pb(info, mean_pb):
bvps  = _safe(info.get(‘bookValue’))
if bvps is None or bvps <= 0: return None
ratio = mean_pb if mean_pb is not None else _safe(info.get(‘priceToBook’))
if ratio is None or ratio <= 0: return None
if ratio > MAX_PB: return None
return round(ratio * bvps, 2)

def calc_psg(info, shares, peg_rate):
“””
PSG: Revenue per share × (growth rate as whole number).
Uses peg_rate — the un-floored rate from get_forward_growth.
Returns None when peg_rate is 0 (Finviz explicitly said no growth).
“””
rev = _safe(info.get(‘totalRevenue’))
if rev is None or rev <= 0 or shares <= 0: return None
if peg_rate is None or peg_rate <= 0: return None
growth = min(peg_rate, MAX_PEG_PSG_GROWTH)
return round((rev / shares) * (growth * 100), 2)

def calc_peg(info, peg_rate):
“””
PEG: EPS × (growth rate as whole number).
Uses peg_rate — the un-floored rate from get_forward_growth.
Returns None when peg_rate is 0 (Finviz explicitly said no growth).
“””
eps = _safe(info.get(‘trailingEps’))
if eps is None or eps <= 0: return None
if peg_rate is None or peg_rate <= 0: return None
growth = min(peg_rate, MAX_PEG_PSG_GROWTH)
return round(eps * (growth * 100), 2)

def calc_ev_ebitda(info, shares, mean_ev_ebitda):
“””
EV/EBITDA valuation using the company’s own historical average multiple.
This is fair to the company — it reflects what the market has historically
paid for this specific business, not a generic sector benchmark.

```
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
```

# =============================================================================

# INTRINSIC VALUE AGGREGATOR — tier-weighted mean

# =============================================================================

def calc_intrinsic_value(valuations: dict, tier: str = “Average”) -> dict:
keys = [
“1_DCF_Operating_CF”, “2_DCF_FCF”, “3_DCF_Net_Income”,
“4_DCF_Terminal_Value”, “5_Mean_PS”, “6_Mean_PE”,
“7_Mean_PB”, “8_PSG”, “9_PEG”, “10_EV_EBITDA”,
]
weights = TIER_WEIGHTS.get(tier, TIER_WEIGHTS[“Average”])
valid_vals, valid_weights = [], []
for k in keys:
v = valuations.get(k)
if v is not None and isinstance(v, (int, float)) and 0 < v < 1_000_000:
valid_vals.append(v)
valid_weights.append(weights.get(k, 0.10))
count = len(valid_vals)
if count == 0:
return {“intrinsic_value”: None, “methods_used”: 0, “confidence”: “Insufficient Data”}
total_w   = sum(valid_weights)
norm_w    = [w / total_w for w in valid_weights]
intrinsic  = round(float(np.dot(valid_vals, norm_w)), 2)
confidence = “High” if count >= 7 else “Medium” if count >= 4 else “Low”
return {“intrinsic_value”: intrinsic, “methods_used”: count, “confidence”: confidence}

# =============================================================================

# 10-YEAR PRICE HISTORY

# =============================================================================

def get_10yr_history(stock):
try:
hist = stock.history(period=“10y”)
if hist.empty: return {}
return {
str(y): round(hist[hist.index.year == y][‘Close’].iloc[-1], 2)
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

```
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

        # Pull annual financials once — reused across all stages
        try:
            financials = stock.financials
        except Exception:
            financials = None

        # ------------------------------------------------------------------
        # STAGE 1 — Preliminary quality scores → determines tier
        # ------------------------------------------------------------------
        prelim_scores = {
            "profitability":      score_profitability(info),
            "financial_strength": score_financial_strength(info),
            "growth":             score_growth(info, financials),
            "predictability":     score_predictability(info, financials),
            "valuation":          0,
            "moat":               None,
        }
        prelim_subtotal = sum(v for k, v in prelim_scores.items() if k not in ('moat', 'valuation') and v is not None)
        prelim_pct      = round((prelim_subtotal / 50) * 100, 2)
        prelim_class    = classify_quality(prelim_scores, prelim_pct)
        tier_data       = resolve_tier(prelim_class["label"])
        discount        = tier_data["discount_rate"]
        terminal        = tier_data["terminal_growth"]

        # ------------------------------------------------------------------
        # STAGE 2 — Forward growth rate (Finviz primary, yfinance fallback)
        # rate     : floored at MIN_GROWTH_RATE — used in DCF methods
        # peg_rate : un-floored — used in PEG and PSG (0.0 = skip method)
        # ------------------------------------------------------------------
        fwd_growth = get_forward_growth(ticker, info)
        growth_s1  = fwd_growth["rate"]
        growth_s2  = max(MIN_GROWTH_RATE, fwd_growth["rate"] * 0.50)
        peg_rate   = fwd_growth["peg_rate"]

        # ------------------------------------------------------------------
        # STAGE 3 — Historical multiples (P/E, P/S, P/B, EV/EBITDA)
        # ------------------------------------------------------------------
        hist_multiples  = get_historical_mean_multiples(stock, shares)
        mean_pe         = hist_multiples.get("mean_pe")
        mean_ps         = hist_multiples.get("mean_ps")
        mean_pb         = hist_multiples.get("mean_pb")
        mean_ev_ebitda  = hist_multiples.get("mean_ev_ebitda")

        # ------------------------------------------------------------------
        # STAGE 4 — All 10 valuations
        # ------------------------------------------------------------------
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
        aggregate  = calc_intrinsic_value(valuations, tier=tier_data["tier"])
        valuations.update(aggregate)
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
            "weights_applied":  TIER_WEIGHTS.get(tier_data["tier"], TIER_WEIGHTS["Average"]),
        }

        # ------------------------------------------------------------------
        # STAGE 5 — Final quality scores (now with real valuation score)
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
            "python_subtotal":     python_subtotal,
            "python_subtotal_pct": python_subtotal_pct,
            "classification":      classification["label"],
            "penalty_pct":         classification["penalty_pct"],
            "final_score_pct":     classification["final_score_pct"],
            "note": (
                "Moat score is null. AI Studio adds moat score and "
                "recalculates Investment Quality Score and classification "
                "using all 6 metrics out of 60."
            ),
            "last_calculated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        # ------------------------------------------------------------------
        # STAGE 6 — ROE and ROIC (pre-computed for AI Studio display)
        # ------------------------------------------------------------------
        roe_roic     = calc_roe_roic(info, financials, stock.balance_sheet)
        fundamentals = {
            "roe":         roe_roic["roe"],
            "roic":        roe_roic["roic"],
            "roe_pct":     roe_roic["roe_pct"],
            "roic_pct":    roe_roic["roic_pct"],
            "roic_source": roe_roic["roic_source"],
        }

        # ------------------------------------------------------------------
        # STAGE 7 — History & forecast
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
```

# =============================================================================

# A–Z PARTITIONED JSON SAVER

# =============================================================================

def save_partitioned_data(master_results):
partitions = {}
for ticker, data in master_results.items():
letter = ticker[0].upper()
if not letter.isalpha(): letter = “0-9”
if letter not in partitions: partitions[letter] = {}
partitions[letter][ticker] = data
os.makedirs(‘data’, exist_ok=True)
for letter, content in partitions.items():
with open(f’data/stocks_{letter}.json’, ‘w’) as f:
json.dump(content, f, indent=4)
print(“✅ Saved A–Z stock files.”)

# =============================================================================

# ENTRY POINT

# =============================================================================

def main():
tickers = get_all_tickers()
master_results = {}
print(f”Starting analysis of {len(tickers)} tickers with 5 safe threads…”)
skipped = []
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
for future in concurrent.futures.as_completed(future_to_ticker):
t   = future_to_ticker[future]
res = future.result()
if res:
master_results[res[0]] = res[1]
if len(master_results) % 50 == 0:
print(f”  Processed {len(master_results)} stocks…”)
else:
skipped.append(t)
if skipped:
print(f”\n⚠️  {len(skipped)} tickers skipped (see [SKIP]/[ERROR] lines above):”)
for chunk in [skipped[i:i+10] for i in range(0, len(skipped), 10)]:
print(f”   {’, ’.join(chunk)}”)
save_partitioned_data(master_results)

if **name** == “**main**”:
main()
