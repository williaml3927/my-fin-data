import json
import requests
import concurrent.futures
from datetime import datetime
import os

# --- SETTINGS ---
# Fetching top 1000 covers 99% of user interest while staying within API limits
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page="
GOLD_MARKET_CAP = 15000000000000 # Used for Store of Value math

def fetch_page(page_num):
    """Worker to fetch a single page of 250 coins."""
    try:
        response = requests.get(COINGECKO_URL + str(page_num), timeout=10)
        return response.json()
    except:
        return []

def analyze_coin(coin):
    """Valuation logic for a single coin."""
    try:
        symbol = coin['symbol'].upper()
        mc = coin.get('market_cap') or 0
        circulating = coin.get('circulating_supply') or 0
        total_supply = coin.get('total_supply') or circulating
        
        # Determine Category
        is_sov = symbol in ["BTC", "LTC", "DOGE", "BCH", "XMR"]
        
        # 1. Scarcity/Dilution Health
        dilution = (circulating / total_supply * 100) if total_supply > 0 else 100
        
        # 2. Network Value (Metcalfe Proxy)
        # Using volume relative to market cap as a high-activity indicator
        vol_to_mc = (coin.get('total_volume', 0) / mc) if mc > 0 else 0
        
        # 3. Monetary Premium (vs Gold)
        monetary_prem = (mc / GOLD_MARKET_CAP * 100)
        
        # Final Speculative Value Logic
        if is_sov:
            # Store of Value: Weighted by gold capture and scarcity
            spec_price = coin['current_price'] * (1 + (monetary_prem/100))
        else:
            # Utility: Weighted by network activity and dilution health
            spec_price = coin['current_price'] * (dilution / 100) * (1 + vol_to_mc)

        return symbol, {
            "Name": coin['name'],
            "Current_Price": coin['current_price'],
            "Market_Cap_Rank": coin['market_cap_rank'],
            "Dilution_Health": f"{round(dilution, 2)}%",
            "Metcalfe_Activity_Score": round(vol_to_mc * 100, 2),
            "Final_Speculative_Value": round(spec_price, 4),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except:
        return None

def save_partitions(results):
    """Saves data into data_crypto/crypto_A.json, etc."""
    if not os.path.exists('data_crypto'): os.makedirs('data_crypto')
    
    buckets = {}
    for sym, data in results.items():
        letter = sym[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
        
    for letter, content in buckets.items():
        with open(f'data_crypto/crypto_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    print("Fetching Top 1000 Cryptos...")
    all_coins = []
    # Fetch 4 pages of 250 coins each
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pages = list(executor.map(fetch_page, range(1, 5)))
        for p in pages: all_coins.extend(p)
        
    print(f"Analyzing {len(all_coins)} coins...")
    master_results = {}
    for coin in all_coins:
        res = analyze_coin(coin)
        if res:
            master_results[res[0]] = res[1]
            
    save_partitions(master_results)
    print("✅ Crypto buckets updated successfully.")

if __name__ == "__main__":
    main()
