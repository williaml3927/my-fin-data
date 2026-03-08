import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page="
GOLD_MARKET_CAP = 15000000000000 

def fetch_page(page_num):
    try:
        response = requests.get(COINGECKO_URL + str(page_num), timeout=10)
        return response.json()
    except Exception as e:
        print(f"Error fetching page {page_num}: {e}")
        return []

def get_crypto_history(symbol):
    try:
        # yfinance uses TICKER-USD format
        crypto_stock = yf.Ticker(f"{symbol.upper()}-USD")
        hist = crypto_stock.history(period="10y")
        if hist.empty: return {}
        
        yearly_prices = {}
        years = sorted(list(set([d.year for d in hist.index])))
        for y in years:
            yearly_prices[str(y)] = round(hist[hist.index.year == y]['Close'].iloc[-1], 4)
        return yearly_prices
    except:
        return {}

def analyze_coin(coin):
    try:
        time.sleep(0.3) 
        symbol = coin['symbol'].upper()
        mc = coin.get('market_cap') or 1 # Avoid div by zero
        rank = coin.get('market_cap_rank') or 1000
        vol = coin.get('total_volume') or 0
        current_price = coin['current_price']
        circulating = coin.get('circulating_supply') or 0
        # Fallback to circulating if total_supply is null or 0
        total_supply = coin.get('total_supply') or coin.get('max_supply') or circulating 
        ath_change = coin.get('ath_change_percentage') or -99
        
        # Core Calculation Variables
        vol_to_mc = (vol / mc) # Velocity
        dilution_ratio = (circulating / total_supply) if total_supply > 0 else 1.0
        
        # --- SCORING PER UPDATED DEFINITIONS ---
        
        # 1. Network Growth (Velocity relative to size)
        # Scaled: 20% daily turnover is a 10. 
        growth_score = max(0, min(10, int(vol_to_mc * 50))) 
        
        # 2. Utility Demand (Absolute Liquidity/Participation)
        # Threshold: $100M+ daily volume = 10/10. 
        utility_score = max(0, min(10, int(vol / 10_000_000)))
        
        # 3. Network Dominance & Ecosystem Strength (Market Cap Rank)
        # Top 10 = 10, Rank 50 = 8, Rank 200 = 5, Rank 500+ = 1-2
        if rank <= 10: dominance_score = 10
        elif rank <= 50: dominance_score = 8
        elif rank <= 150: dominance_score = 6
        elif rank <= 300: dominance_score = 4
        else: dominance_score = max(1, 10 - int(rank / 100))
        
        # 4. Protocol Reliability & Credibility (Drawdown resilience)
        # 10% from ATH = 9, 50% from ATH = 5, 95% from ATH = 1
        reliability_score = max(1, min(10, int(10 + (ath_change / 10))))
        
        # 5. Survivability & Dilution Control (Circulating / Total Supply)
        # 100% unlocked = 10. 10% unlocked = 1.
        survivability_score = max(0, min(10, int(dilution_ratio * 10)))
        
        # 6. Network Adaptability (Hybrid: Activity relative to Market Cap)
        # Measures efficiency of the network in maintaining relevance/liquidity
        adaptability_score = max(0, min(10, int((vol_to_mc * 40) + (utility_score / 5))))

        quality_scores = {
            "Network_Growth": growth_score,
            "Utility_Demand": utility_score,
            "Network_Dominance_Ecosystem_Strength": dominance_score,
            "Protocol_Reliability_Credibility": reliability_score,
            "Survivability_Dilution_Control": survivability_score,
            "Network_Adaptability": adaptability_score
        }
        
        # --- SPECULATIVE VALUATION ---
        is_sov = symbol in ["BTC", "LTC", "DOGE", "BCH", "XMR"]
        if is_sov:
            monetary_prem = (mc / GOLD_MARKET_CAP * 100)
            spec_price = current_price * (1 + (monetary_prem/100))
        else:
            # Multi-factor model: Price * Unlocked Ratio * Market Activity
            spec_price = current_price * dilution_ratio * (1 + vol_to_mc)

        final_val = round(spec_price, 4)
        
        # --- HISTORY & FORECAST ---
        history_10yr = get_crypto_history(symbol)
        growth_proxy = min(0.30, max(0.05, 0.15 + (vol_to_mc / 10)))
        forecast_5yr = {str(datetime.now().year + i): round(final_val * (1 + growth_proxy)**i, 4) for i in range(1, 6)}

        return symbol, {
            "Name": coin['name'],
            "Current_Price": current_price,
            "Market_Cap_Rank": rank,
            "Final_Speculative_Value": final_val,
            "Quality_Scores": quality_scores,
            "10_Year_History": history_10yr,
            "5_Year_Forecast": forecast_5yr,
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
    all_coins = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
        pages = list(exc.map(fetch_page, range(1, 5)))
        for p in pages: all_coins.extend(p)
    
    master_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
        futures = {exc.submit(analyze_coin, c): c for c in all_coins}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: master_results[res[0]] = res[1]

    save_partitions(master_results)
    print(f"✅ Success: {len(master_results)} crypto assets updated in partitioned JSON.")

if __name__ == "__main__":
    main()
