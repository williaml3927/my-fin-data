import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import math # Required for NaN detection

# --- SOURCES ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"
CORE_PRIORITY = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO"]

def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = requests.get(url)
            data = res.json()
            for item in data:
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers: tickers.append(clean)
        except: continue
    return tickers 

def discount_sum(base_value, growth, discount, years=20):
    if base_value <= 0: return 0
    return sum([base_value * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)])

# --- NEW HELPERS FOR DATA SAFETY ---

def clean_json(obj):
    """Recursively replaces NaN and Infinity with 0 to prevent JSON crashes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def safe_get(info_dict, key, default=0):
    """Safely extracts data from yfinance, preventing 'NoneType' math crashes."""
    val = info_dict.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# -----------------------------------

def analyze_ticker(ticker):
    try:
        time.sleep(0.5) 
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or safe_get(info, 'sharesOutstanding', 0) == 0: 
            return None
        
        # --- BASIC DATA (Using safe_get) ---
        shares = safe_get(info, 'sharesOutstanding', 1)
        current_price = safe_get(info, 'currentPrice', safe_get(info, 'previousClose', 1))
        fcf = safe_get(info, 'freeCashflow', 0) / shares
        ocf = safe_get(info, 'operatingCashflow', 0) / shares
        ni = safe_get(info, 'netIncomeToCommon', 0) / shares
        rev = safe_get(info, 'totalRevenue', 0) / shares
        
        raw_growth = safe_get(info, 'earningsGrowth', 0.05)
        growth = max(0.01, min(raw_growth, 0.30))
        
        # --- PRE-FETCH MULTIPLIERS ---
        pe = safe_get(info, 'trailingPE', 20)
        ps = safe_get(info, 'priceToSalesTrailing12Months', 2)
        pb = safe_get(info, 'priceToBook', 1)
        bv = safe_get(info, 'bookValue', 0)
        mcap = safe_get(info, 'marketCap', 1)

        # --- THE 10 VALUATION METHODS ---
        vals = {
            "1_DCF_OCF_20yr": round(discount_sum(ocf, growth, 0.09, 20), 2),
            "2_DCF_FCF_20yr": round(discount_sum(fcf, growth, 0.09, 20), 2),
            "3_DCF_NetInc_20yr": round(discount_sum(ni, growth, 0.09, 20), 2),
            "4_DCF_Terminal_Value": round(discount_sum(fcf, growth, 0.09, 10) + ((fcf * (1+growth)**10 * 1.02)/(0.07))/(1.09)**10, 2),
            "5_Mean_PS_Valuation": round(ps * rev, 2),
            "6_Mean_PE_Valuation_Ex_NRI": round(pe * ni, 2),
            "7_Mean_PB_Valuation": round(pb * bv, 2),
            "8_PSG_Valuation": round((ps / (growth * 100)) * (growth * 100) * rev if growth > 0 else 0, 2),
            "9_PEG_Valuation": round((pe / (growth * 100)) * (growth * 100) * ni if growth > 0 else 0, 2),
            "10_Metcalfes_Law_Proxy": round((rev**2 / mcap) if mcap > 0 else 0, 2)
        }

        # Final Fair Value Mean
        valid_prices = [v for v in vals.values() if 0 < v < 50000]
        final_fair_value = round(sum(valid_prices)/len(valid_prices), 2) if valid_prices else 0
        vals["Final_Fair_Value"] = final_fair_value
        
        # --- PRE-FETCH QUALITY METRICS ---
        beta = safe_get(info, 'beta', 1)
        profit_margins = safe_get(info, 'profitMargins', 0)
        revenue_growth = safe_get(info, 'revenueGrowth', 0)
        gross_margins = safe_get(info, 'grossMargins', 0)
        current_ratio = safe_get(info, 'currentRatio', 1)
        debt_to_equity = safe_get(info, 'debtToEquity', 100)
        payout_ratio = safe_get(info, 'payoutRatio', 0)

        # --- QUALITY SCORING (0-10) ---
        vals["Quality_Scores"] = {
            "Predictability": max(0, min(10, int(10 - (abs(beta - 1) * 5)))),
            "Profitability": max(0, min(10, int(profit_margins * 40))),
            "Growth": max(0, min(10, int(revenue_growth * 40))),
            "Moat": max(0, min(10, int(gross_margins * 20))),
            "Financial_Strength": max(0, min(10, int((current_ratio * 2) + (max(0, 150 - debt_to_equity) / 30)))),
            "Valuation": max(0, min(10, int(5 + (((final_fair_value - current_price) / final_fair_value) * 10)))) if final_fair_value > 0 else 0
        }

        # --- FINANCIAL TAB DATA (BAR CHART ARRAYS) ---
        fin = stock.financials
        bs = stock.balance_sheet
        
        financial_tab = {
            "Revenue_NetIncome": {
                "Years": [str(d.year) for d in fin.columns[:4]][::-1] if not fin.empty else [],
                "Revenue": [float(v) for v in fin.loc['Total Revenue'][:4]][::-1] if not fin.empty and 'Total Revenue' in fin.index else [],
                "Net_Income": [float(v) for v in fin.loc['Net Income'][:4]][::-1] if not fin.empty and 'Net Income' in fin.index else []
            },
            "Cash_Debt": {
                "Years": [str(d.year) for d in bs.columns[:4]][::-1] if not bs.empty else [],
                "Cash": [float(v) for v in bs.loc['Cash And Cash Equivalents'][:4]][::-1] if not bs.empty and 'Cash And Cash Equivalents' in bs.index else [],
                "Total_Debt": [float(v) for v in bs.loc['Total Debt'][:4]][::-1] if not bs.empty and 'Total Debt' in bs.index else []
            },
            "Shares_Status": {
                "Current_Shares": shares,
                "Buyback_Yield": payout_ratio * -1 
            }
        }
        vals["Financial_Tab_Data"] = financial_tab

        # --- HISTORY & FORECAST ---
        hist = stock.history(period="10y")
        vals["10_Year_History"] = {str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2) for y in sorted(list(set(hist.index.year)))} if not hist.empty else {}
        vals["5_Year_Forecast"] = {str(datetime.now().year + i): round(final_fair_value * (1 + growth)**i, 2) for i in range(1, 6)}
        
        vals["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return ticker, vals
    except Exception as e: 
        # print(f"Skipped {ticker}: {e}") # Uncomment if you want to see why specific stocks fail
        return None

def save_partitioned(master):
    os.makedirs('data', exist_ok=True)
    buckets = {}
    
    # MAGIC BULLET: Clean the entire payload of NaN values before saving
    cleaned_master = clean_json(master)
    
    for t, d in cleaned_master.items():
        l = t[0].upper() if t[0].isalpha() else "0-9"
        if l not in buckets: buckets[l] = {}
        buckets[l][t] = d
    for l, c in buckets.items():
        with open(f'data/stocks_{l}.json', 'w') as f: json.dump(c, f, indent=4)

def main():
    tickers = get_all_tickers()
    master = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
        futures = {exc.submit(analyze_ticker, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: master[res[0]] = res[1]
    save_partitioned(master)

if __name__ == "__main__": main()
