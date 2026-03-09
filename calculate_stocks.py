import json
import yfinance as yf
import concurrent.futures
import os
from datetime import datetime
import time
import random
import math

# Use a common browser User-Agent to avoid immediate bot detection
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

def clean_json(obj):
    """Recursively replaces NaN and Infinity with 0 for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def fetch_stock_data(ticker_symbol):
    """Fetches data for a single ticker with rate-limit protection."""
    try:
        # 1. Random delay to mimic human behavior
        time.sleep(random.uniform(1.0, 2.5))
        
        ticker = yf.Ticker(ticker_symbol)
        
        # Get Info (Key stats)
        info = ticker.info
        if not info or 'symbol' not in info:
            return None
            
        # Get 10-Year History
        hist = ticker.history(period="10y")
        yearly_prices = {}
        if not hist.empty:
            yearly_prices = {
                str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 2) 
                for y in sorted(list(set(hist.index.year)))
            }

        return ticker_symbol, {
            "Name": info.get('longName', ticker_symbol),
            "Price": info.get('currentPrice') or info.get('regularMarketPrice'),
            "Market_Cap": info.get('marketCap', 0),
            "PE_Ratio": info.get('trailingPE', 0),
            "Dividend_Yield": info.get('dividendYield', 0),
            "10_Year_History": yearly_prices,
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        print(f"Skipping {ticker_symbol} due to error: {e}")
        return None

def save_partitions(results):
    os.makedirs('data', exist_ok=True)
    buckets = {}
    
    # Scrub the entire result set of NaN/Inf before saving
    clean_results = clean_json(results)
    
    for sym, data in clean_results.items():
        letter = sym[0].upper() if sym[0].isalpha() else "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
    
    for letter, content in buckets.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    # Example list - replace with your full list of tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B"] 
    
    print(f"Updating {len(tickers)} stocks... (Slow mode enabled for rate safety)")
    
    master_results = {}
    # 2. Lower max_workers to 2 to avoid Yahoo's "Too Many Requests" wall
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_stock_data, t): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
            
            # Progress tracker
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(tickers)} processed...")

    save_partitions(master_results)
    print(f"✅ Success: {len(master_results)} stocks updated across JSON partitions.")

if __name__ == "__main__":
    main()
