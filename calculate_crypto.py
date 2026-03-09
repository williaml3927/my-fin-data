import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import random
import math

# --- SESSION & HEADERS SETUP ---
# This mimics a real browser session for the entire duration of the script
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page="
DEFILLAMA_URL = "https://api.llama.fi/protocols"

# --- DATA SAFETY HELPERS ---

def clean_json(obj):
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
    if val is None: return default
    try: return float(val)
    except: return default

# --- API FUNCTIONS ---

def fetch_defillama_data():
    print("Fetching on-chain data from DefiLlama...")
    try:
        res = session.get(DEFILLAMA_URL, timeout=15)
        if res.status_code != 200: return {}
        protocols = res.json()
        llama_dict = {}
        for p in protocols:
            sym = p.get('symbol', '').upper()
            if sym:
                llama_dict[sym] = {
                    "tvl": safe_num(p.get('tvl')),
                    "fees_24h": safe_num(p.get('fees24h')),
                    "revenue_24h": safe_num(p.get('revenue24h'))
                }
        return llama_dict
    except: return {}

def get_crypto_history(symbol):
    """Fetches history using the persistent session to avoid 429 blocks."""
    try:
        # Most crypto on Yahoo is formatted as SYMBOL-USD (e.g., BTC-USD)
        ticker = yf.Ticker(f"{symbol.upper()}-USD", session=session)
        hist = ticker.history(period="10y")
        if hist.empty: return {}
        
        return {
            str(y): round(hist[hist.index.year == y]['Close'].iloc[-1], 4) 
            for y in sorted(list(set(hist.index.year)))
        }
    except: return {}

def analyze_coin(coin, llama_data):
    try:
        # Crucial delay: crypto has many more assets than stocks, so we throttle more
        time.sleep(random.uniform(0.8, 2.0))
        
        symbol = coin.get('symbol', '').upper()
        if not symbol: return None

        current_price = safe_num(coin.get('current_price'))
        if current_price <= 0: return None

        mc = safe_num(coin.get('market_cap'), 1)
        vol = safe_num(coin.get('total_volume'))
        circulating = safe_num(coin.get('circulating_supply'))
        
        total_supply = safe_num(coin.get('total_supply') or coin.get('max_supply'), circulating)
        if total_supply <= 0: total_supply = circulating if circulating > 0 else 1.0

        dilution_ratio = (circulating / total_supply)
        vol_to_mc = (vol / mc)
        
        # Merge DefiLlama Data
        on_chain = llama_data.get(symbol, {"tvl": 0, "fees_24h": 0, "revenue_24h": 0})
        tvl = on_chain["tvl"]
        fees = on_chain["fees_24h"]
        
        locked_supply_pct = max(0, 1 - dilution_ratio) * 100
        
        # Map quality scores to the keys the UI expects
        quality_scores = {
            "Predictability": 7,
            "Profitability": 8 if fees > 10000 else 5,
            "Growth": 9 if vol_to_mc > 0.1 else 6,
            "Moat": 8 if tvl > 1000000 else 4,
            "Financial_Strength": 9 if dilution_ratio > 0.8 else 5,
            "Valuation": 7
        }

        return symbol, {
            "Name": coin.get('name', symbol),
            "Current_Price": current_price,
            "Market_Cap": mc,
            "Financial_Tab_Data": {
                "Economic_Activity": {"Volume": vol, "TVL": tvl, "Fees": fees},
                "Supply": {"Circulating": circulating, "Total": total_supply, "Locked_Pct": locked_supply_pct}
            },
            "Quality_Scores": quality_scores,
            "10_Year_History": get_crypto_history(symbol),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except: return None

def save_partitions(results):
    os.makedirs('data_crypto', exist_ok=True)
    buckets = {}
    cleaned_results = clean_json(results)
    
    for sym, data in cleaned_results.items():
        letter = sym[0].upper() if sym[0].isalpha() else "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
        
    for letter, content in buckets.items():
        with open(f'data_crypto/crypto_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    print("🚀 Starting Crypto Update...")
    llama_data = fetch_defillama_data() 
    
    all_coins = []
    # Fetch top 500 coins (2 pages) to keep it stable
    for page in range(1, 3):
        print(f"Fetching CoinGecko Page {page}...")
        try:
            res = session.get(COINGECKO_URL + str(page), timeout=10)
            if res.status_code == 200:
                all_coins.extend(res.json())
            time.sleep(3) # Respect CoinGecko
        except: continue

    master_results = {}
    # Use only 2 workers for Crypto. Yahoo is very sensitive to Crypto requests.
    print(f"Analyzing {len(all_coins)} assets...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exc:
        futures = {exc.submit(analyze_coin, c, llama_data): c for c in all_coins}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res: master_results[res[0]] = res[1]
            if (i+1) % 20 == 0: print(f"Progress: {i+1}/{len(all_coins)} processed")

    save_partitions(master_results)
    print(f"✅ Finished! {len(master_results)} assets saved to data_crypto/")

if __name__ == "__main__":
    main()
