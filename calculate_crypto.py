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
    except:
        return []

def get_crypto_history(symbol):
    """Uses yfinance (e.g., BTC-USD) to bypass CoinGecko's strict historical rate limits."""
    try:
        # yfinance uses the format TICKER-USD for crypto
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
        time.sleep(0.2) # Light delay
        symbol = coin['symbol'].upper()
        mc = coin.get('market_cap') or 0
        circulating = coin.get('circulating_supply') or 0
        total_supply = coin.get('total_supply') or circulating
        
        is_sov = symbol in ["BTC", "LTC", "DOGE", "BCH", "XMR"]
        dilution = (circulating / total_supply * 100) if total_supply > 0 else 100
        vol_to_mc = (coin.get('total_volume', 0) / mc) if mc > 0 else 0
        monetary_prem = (mc / GOLD_MARKET_CAP * 100)
        
        if is_sov:
            spec_price = coin['current_price'] * (1 + (monetary_prem/100))
        else:
            spec_price = coin['current_price'] * (dilution / 100) * (1 + vol_to_mc)

        final_val = round(spec_price, 4)
        
        # --- NEW: History & Forecast Arrays ---
        history_10yr = get_crypto_history(symbol)
        
        # Crypto proxy growth rate (15% base network growth modified by current Metcalfe activity)
        crypto_growth_proxy = min(0.30, max(0.05, 0.15 + (vol_to_mc / 10)))
        current_year = datetime.now().year
        forecast_5yr = {
            str(current_year + i): round(final_val * (1 + crypto_growth_proxy)**i, 4)
            for i in range(1, 6)
        }

        return symbol, {
            "Name": coin['name'],
            "Current_Price": coin['current_price'],
            "Market_Cap_Rank": coin['market_cap_rank'],
            "Dilution_Health": f"{round(dilution, 2)}%",
            "Metcalfe_Activity_Score": round(vol_to_mc * 100, 2),
            "Final_Speculative_Value": final_val,
            "10_Year_History": history_10yr,
            "5_Year_Forecast": forecast_5yr,
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        return None

def save_partitions(results):
    os.makedirs('data_crypto', exist_ok=True)
    buckets = {}
    for sym, data in results.items():
        letter = sym[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in buckets: buckets[letter] = {}
        buckets[letter][sym] = data
        
    for letter, content in buckets.items():
        with open(f'data_crypto/crypto_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)
    print("✅ Crypto buckets updated successfully.")

def main():
    print("Fetching Top 1000 Cryptos...")
    all_coins = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pages = list(executor.map(fetch_page, range(1, 5)))
        for p in pages: all_coins.extend(p)
        
    print(f"Analyzing {len(all_coins)} coins...")
    master_results = {}
    
    # 5 workers to safely fetch yfinance history without bans
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_coin = {executor.submit(analyze_coin, c): c for c in all_coins}
        for future in concurrent.futures.as_completed(future_to_coin):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]

    save_partitions(master_results)

if __name__ == "__main__":
    main()
