import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import random
import math 

# --- CONSTANTS & HEADERS ---
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page="
DEFILLAMA_URL = "https://api.llama.fi/protocols"

# CRITICAL: Tricks APIs into thinking this is a web browser, preventing automatic bot-blocks
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

# --- HELPERS FOR DATA SAFETY ---

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

def safe_num(val, default=0):
    """Safely converts API responses to floats, preventing 'NoneType' crashes."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# -----------------------------------

def fetch_defillama_data():
    print("Fetching on-chain data from DefiLlama...")
    try:
        res = requests.get(DEFILLAMA_URL, headers=HEADERS, timeout=15)
        if res.status_code != 200:
            print(f"⚠️ DefiLlama API Error: {res.status_code}")
            return {}
            
        protocols = res.json()
        llama_dict = {}
        for p in protocols:
            sym = p.get('symbol', '').upper()
            if sym not in llama_dict:
                llama_dict[sym] = {
                    "tvl": safe_num(p.get('tvl'), 0),
                    "fees_24h": safe_num(p.get('fees24h'), 0),
                    "revenue_24h": safe_num(p.get('revenue24h'), 0)
                }
        return llama_dict
    except Exception as e:
        print(f"⚠️ Failed to connect to DefiLlama: {e}")
        return {}

def get_crypto_history(symbol):
    try:
        crypto_stock = yf.Ticker(f"{symbol.upper()}-USD")
        hist = crypto_stock.history(period="10y")
        if hist.empty: return {}
        
        yearly_prices = {str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 4) for y in sorted(list(set(hist.index.year)))}
        return yearly_prices
    except: return {}

def analyze_coin(coin, llama_data):
    try:
        # CRITICAL: Random delay to prevent yfinance from blacklisting your IP
        time.sleep(random.uniform(0.5, 1.5)) 
        
        symbol = coin.get('symbol', '').upper()
        if not symbol: return None

        current_price = safe_num(coin.get('current_price'), 0)
        if current_price <= 0: return None # Skip invalid/dead coins

        mc = safe_num(coin.get('market_cap'), 1)
        vol = safe_num(coin.get('total_volume'), 0)
        circulating = safe_num(coin.get('circulating_supply'), 0)
        
        raw_total = coin.get('total_supply')
        raw_max = coin.get('max_supply')
        total_supply = safe_num(raw_total if raw_total is not None else raw_max, circulating)
        if total_supply <= 0: total_supply = circulating if circulating > 0 else 1.0

        dilution_ratio = (circulating / total_supply) if total_supply > 0 else 1.0
        vol_to_mc = (vol / mc) if mc > 0 else 0
        
        # --- ON-CHAIN FINANCIALS ---
        on_chain = llama_data.get(symbol, {"tvl": 0, "fees_24h": 0, "revenue_24h": 0})
        tvl = safe_num(on_chain.get("tvl"), 0)
        fees = safe_num(on_chain.get("fees_24h"), 0)
        
        locked_supply_pct = max(0, 1 - dilution_ratio) * 100
        
        financial_tab = {
            "Economic_Activity_Growth": {
                "Transaction_Volume_24h": vol,
                "TVL": tvl,
                "Fees_24h": fees,
                "Active_Addresses": "N/A (Requires Paid API)"
            },
            "Capital_Efficiency": {
                "Fees_Generated": fees,
                "Real_Yield_Proxy": round((fees / tvl * 100), 2) if tvl > 0 else 0,
                "Issuance_Risk": f"{round(locked_supply_pct, 2)}% Supply Locked"
            },
            "Sustainability": {
                "Daily_Fees": fees,
                "Emissions_Proxy": "High" if locked_supply_pct > 50 else "Low"
            },
            "Supply_Dynamics": {
                "Circulating_Supply": circulating,
                "Total_Supply": total_supply,
                "Tokens_Burned": "N/A (Requires Paid API)"
            }
        }

        return symbol, {
            "Name": coin.get('name', symbol),
            "Current_Price": current_price,
            "Financial_Tab_Data": financial_tab, 
            "Quality_Scores": {"Network_Growth": 8}, # Placeholder for your existing scores
            "10_Year_History": get_crypto_history(symbol),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e: 
        return None

def save_partitions(results):
    os.makedirs('data_crypto', exist_ok=True)
    buckets = {}
    
    cleaned_results = clean_json(results)
    
    for sym, data in cleaned_results.items():
        if not sym: continue
        letter = sym[0].upper() if sym[0].isalpha() else "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
        
    for letter, content in buckets.items():
        with open(f'data_crypto/crypto_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    print("Initializing Crypto Analysis...")
    llama_data = fetch_defillama_data() 
    
    all_coins = []
    print("Fetching market data from CoinGecko...")
    
    # CRITICAL: Fetched sequentially instead of concurrently to prevent 429 Bans
    for page in range(1, 5):
        try:
            res = requests.get(COINGECKO_URL + str(page), headers=HEADERS, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list):
                    all_coins.extend(data)
                    print(f"   -> Page {page} fetched successfully.")
                else:
                    print(f"   ⚠️ Page {page} returned unusual data format.")
            else:
                print(f"   ⚠️ CoinGecko API Error (Page {page}): {res.status_code}")
        except Exception as e:
            print(f"   ⚠️ Request failed for page {page}: {e}")
        
        # Pause between pages to keep CoinGecko happy
        time.sleep(2) 

    if not all_coins:
        print("❌ No valid coins fetched from CoinGecko. Exiting.")
        return

    master_results = {}
    print(f"\nProcessing {len(all_coins)} coins... (Throttling enabled for Yahoo Finance safety)")
    
    # Lowered max_workers to 3 so Yahoo doesn't block the connection
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as exc:
        futures = {exc.submit(analyze_coin, c, llama_data): c for c in all_coins}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: master_results[res[0]] = res[1]
            
            # Print a progress update every 50 coins
            if (i + 1) % 50 == 0:
                print(f"   -> Progress: {i + 1}/{len(all_coins)} coins analyzed...")

    save_partitions(master_results)
    print(f"\n✅ Success: {len(master_results)} crypto assets updated and saved.")

if __name__ == "__main__":
    main()
