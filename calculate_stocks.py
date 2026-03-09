import json
import yfinance as yf
import concurrent.futures
import os
from datetime import datetime
import time
import random
import math
import requests

# 1. NEW: Create a custom session to properly pass the User-Agent to Yahoo Finance
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

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
        # Random delay to mimic human behavior
        time.sleep(random.uniform(1.0, 2.5))
        
        # Pass the custom session to yfinance so it doesn't block us
        ticker = yf.Ticker(ticker_symbol, session=session)
        
        info = ticker.info
        if not info or 'symbol' not in info:
            return None
            
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
    
    clean_results = clean_json(results)
    
    for sym, data in clean_results.items():
        letter = sym[0].upper() if sym[0].isalpha() else "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
    
    for letter, content in buckets.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    # 2. NEW: A much larger list of the top 100 S&P 500 stocks to populate your JSON files
    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO", "V", 
        "JPM", "TSLA", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "ORCL", "MRK", 
        "COST", "ABBV", "CVX", "CRM", "BAC", "KO", "NFLX", "PEP", "TMO", "AMD", 
        "LIN", "MCD", "DIS", "ADBE", "ABT", "CSCO", "INTU", "WFC", "QCOM", "IBM", 
        "AMAT", "CAT", "CMCSA", "DHR", "VZ", "PFE", "UBER", "NOW", "BX", "GE", 
        "AMGN", "ISRG", "TXN", "PM", "BA", "HON", "COP", "UNP", "INTC", "SPGI", 
        "RTX", "LRCX", "AXP", "LOW", "PGR", "SYK", "ELV", "T", "BLK", "TJX", 
        "MDT", "C", "BSX", "VRTX", "CB", "GS", "CI", "MMC", "REGN", "ADP", 
        "SCHW", "FI", "CVS", "PANW", "GILD", "BMY", "MDLZ", "ETN", "CME", "ADI", 
        "KLAC", "SNPS", "SHW", "DE", "CDNS", "SO", "DUK", "ICE", "MO", "SLB"
    ] 
    
    print(f"Updating {len(tickers)} stocks... (Slow mode enabled for rate safety)")
    
    master_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_stock_data, t): t for t in tickers}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
            
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(tickers)} processed...")

    save_partitions(master_results)
    print(f"✅ Success: {len(master_results)} stocks updated across JSON partitions.")

if __name__ == "__main__":
    main()
