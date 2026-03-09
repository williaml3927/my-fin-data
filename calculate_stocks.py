import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import random
import math

# --- CONFIGURATION ---
# We use a session to mimic a browser, which is much harder for Yahoo to block
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"

# Safe fallback list in case the URLs fail
CORE_PRIORITY = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO", "V", 
    "JPM", "TSLA", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "ORCL", "MRK", 
    "COST", "ABBV", "CVX", "CRM", "BAC", "KO", "NFLX", "PEP", "TMO", "AMD"
]

def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = session.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                for item in data:
                    symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                    if symbol:
                        clean = symbol.strip().upper().replace('.', '-')
                        if clean not in tickers: 
                            tickers.append(clean)
        except Exception as e:
            print(f"⚠️ External source failed (using fallback): {url}")
            
    print(f"📊 Total tickers queued: {len(tickers)}")
    return tickers 

def clean_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return 0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def analyze_ticker(ticker_symbol):
    try:
        # Random delay to stay under the radar
        time.sleep(random.uniform(1.0, 2.5))
        
        stock = yf.Ticker(ticker_symbol, session=session)
        info = stock.info
        
        # If Yahoo returns empty info, we try one more time before giving up
        if not info or 'symbol' not in info:
            return None
        
        # Calculate basics
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if price == 0: return None
        
        # Fetch 10-year history for the charts
        hist = stock.history(period="10y")
        history_data = {}
        if not hist.empty:
            history_data = {
                str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2) 
                for y in sorted(list(set(hist.index.year)))
            }

        # Quality Scores (Calculation logic for your Quality Tab)
        # Using .get(key, 0) prevents the "missing data" crash
        quality = {
            "Predictability": round(min(10, (info.get('revenueGrowth', 0) * 50)), 1),
            "Profitability": round(min(10, (info.get('returnOnEquity', 0) * 40)), 1),
            "Growth": round(min(10, (info.get('earningsGrowth', 0) * 30)), 1),
            "Moat": 8.5 if info.get('ebitdaMargins', 0) > 0.3 else 5.0,
            "Financial_Strength": round(min(10, (info.get('quickRatio', 0) * 5)), 1),
            "Valuation": round(min(10, (20 / info.get('trailingPE', 20) * 5)), 1)
        }

        return ticker_symbol, {
            "Name": info.get('longName', ticker_symbol),
            "Price": price,
            "Market_Cap": info.get('marketCap', 0),
            "PE_Ratio": info.get('trailingPE', 0),
            "Dividend_Yield": info.get('dividendYield', 0),
            "Quality_Scores": quality,
            "10_Year_History": history_data,
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception:
        return None

def save_partitioned_data(master_results):
    os.makedirs('data', exist_ok=True)
    partitions = {}
    
    # Clean NaN values before saving
    safe_results = clean_json(master_results)
    
    for ticker, data in safe_results.items():
        letter = ticker[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in partitions: partitions[letter] = {}
        partitions[letter][ticker] = data
    
    for letter, content in partitions.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)
            
    print(f"✅ Successfully saved {len(safe_results)} stocks across {len(partitions)} files.")

def main():
    all_tickers = get_all_tickers()
    master_results = {}
    
    # Use max_workers=2. 5 is too many and will get you IP-blocked immediately.
    print(f"🚀 Starting analysis with 2 safe threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in all_tickers[:300]} # Limit to 300 for stability
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
                if len(master_results) % 10 == 0:
                    print(f"   Processed {len(master_results)} stocks...")

    save_partitioned_data(master_results)

if __name__ == "__main__":
    main()
