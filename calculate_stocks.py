import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time

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

def analyze_ticker(ticker):
    try:
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'sharesOutstanding' not in info: return None
        
        # Fundamentals
        shares = info.get('sharesOutstanding', 1)
        fcf = info.get('freeCashflow', 0) / shares
        ocf = info.get('operatingCashflow', 0) / shares
        ni = info.get('netIncomeToCommon', 0) / shares
        rev = info.get('totalRevenue', 0) / shares
        growth = max(0.01, min(info.get('earningsGrowth', 0.05), 0.30))
        pe = info.get('trailingPE', 20)
        ps = info.get('priceToSalesTrailing12Months', 2)
        pb = info.get('priceToBook', 1)
        
        # --- THE 10 VALUATION METHODS ---
        vals = {
            "1_DCF_OCF_20yr": round(discount_sum(ocf, growth, 0.09, 20), 2),
            "2_DCF_FCF_20yr": round(discount_sum(fcf, growth, 0.09, 20), 2),
            "3_DCF_NetInc_20yr": round(discount_sum(ni, growth, 0.09, 20), 2),
            "4_DCF_Terminal_Value": round(discount_sum(fcf, growth, 0.09, 10) + ((fcf * (1+growth)**10 * 1.02)/(0.07))/(1.09)**10, 2),
            "5_Mean_PS_Valuation": round(ps * rev, 2),
            "6_Mean_PE_Valuation_Ex_NRI": round(pe * ni, 2),
            "7_Mean_PB_Valuation": round(pb * info.get('bookValue', 0), 2),
            "8_PSG_Valuation": round((ps / (growth * 100)) * (growth * 100) * rev if growth > 0 else 0, 2),
            # FIX: Manual PEG calculation if yfinance returns None/0
            "9_PEG_Valuation": round((pe / (growth * 100)) * (growth * 100) * ni if growth > 0 else 0, 2),
            "10_Metcalfes_Law_Proxy": round((rev**2 / mc_proxy) if (mc_proxy := info.get('marketCap', 1)) else 0, 2) # Normalized Network Value
        }

        # Final Average
        valid_prices = [v for v in vals.values() if 0 < v < 50000]
        vals["Final_Fair_Value"] = round(sum(valid_prices)/len(valid_prices), 2) if valid_prices else 0
        
        # 10yr History & 5yr Forecast
        hist = stock.history(period="10y")
        vals["10_Year_History"] = {str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2) for y in sorted(list(set(hist.index.year)))} if not hist.empty else {}
        vals["5_Year_Forecast"] = {str(datetime.now().year + i): round(vals["Final_Fair_Value"] * (1 + growth)**i, 2) for i in range(1, 6)}
        vals["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        return ticker, vals
    except: return None

def save_partitioned(master):
    os.makedirs('data', exist_ok=True)
    buckets = {}
    for t, d in master.items():
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
