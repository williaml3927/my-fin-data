import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page="
DEFILLAMA_URL = "https://api.llama.fi/protocols"
GOLD_MARKET_CAP = 15000000000000 

# --- NEW: Fetch DefiLlama Data Once ---
def fetch_defillama_data():
    print("Fetching on-chain data from DefiLlama...")
    try:
        res = requests.get(DEFILLAMA_URL, timeout=15)
        protocols = res.json()
        # Create a lookup dictionary by symbol (e.g., 'ETH', 'SOL')
        llama_dict = {}
        for p in protocols:
            sym = p.get('symbol', '').upper()
            if sym not in llama_dict:
                llama_dict[sym] = {
                    "tvl": p.get('tvl', 0),
                    "fees_24h": p.get('fees24h', 0) or 0,
                    "revenue_24h": p.get('revenue24h', 0) or 0
                }
        return llama_dict
    except:
        return {}

def fetch_page(page_num):
    try:
        response = requests.get(COINGECKO_URL + str(page_num), timeout=10)
        return response.json()
    except Exception as e:
        return []

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
        time.sleep(0.3) 
        symbol = coin['symbol'].upper()
        mc = coin.get('market_cap') or 1
        rank = coin.get('market_cap_rank') or 1000
        vol = coin.get('total_volume') or 0
        current_price = coin['current_price']
        circulating = coin.get('circulating_supply') or 0
        total_supply = coin.get('total_supply') or coin.get('max_supply') or circulating
        ath_change = coin.get('ath_change_percentage') or -99
        
        # Base logic
        dilution_ratio = (circulating / total_supply) if total_supply > 0 else 1.0
        vol_to_mc = (vol / mc) if mc > 0 else 0
        
        # --- ON-CHAIN FINANCIALS (DefiLlama + Proxies) ---
        on_chain = llama_data.get(symbol, {"tvl": 0, "fees_24h": 0, "revenue_24h": 0})
        tvl = on_chain["tvl"]
        fees = on_chain["fees_24h"]
        
        # Proxy for Issuance/Emissions: % of supply still locked
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

        # Quality Scores & Speculative Value (from previous steps)
        # ... [Keep your existing Quality Scores and Speculative Value math here] ...
        spec_price = current_price * dilution_ratio * (1 + vol_to_mc)

        return symbol, {
            "Name": coin['name'],
            "Current_Price": current_price,
            "Financial_Tab_Data": financial_tab, # <-- New Data Block
            "Quality_Scores": {"Network_Growth": 8}, # Placeholder for your existing scores
            "10_Year_History": get_crypto_history(symbol),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except: return None

def save_partitions(results):
    os.makedirs('data_crypto', exist_ok=True)
    buckets = {}
    for sym, data in results.items():
        letter = sym[0].upper() if sym[0].isalpha() else "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
    for letter, content in buckets.items():
        with open(f'data_crypto/crypto_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)

def main():
    print("Initializing Crypto Analysis...")
    llama_data = fetch_defillama_data() # Fetch DefiLlama first
    
    all_coins = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
        pages = list(exc.map(fetch_page, range(1, 5)))
        for p in pages: all_coins.extend(p)
    
    master_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
        # Pass llama_data into the analysis function
        futures = {exc.submit(analyze_coin, c, llama_data): c for c in all_coins}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: master_results[res[0]] = res[1]

    save_partitions(master_results)
    print(f"✅ Success: {len(master_results)} crypto assets updated.")

if __name__ == "__main__":
    main()
