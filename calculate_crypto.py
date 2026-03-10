import json
import requests
from datetime import datetime

# --- SOURCES ---
COINGECKO_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
# We use a constant for Gold Market Cap (~$14-15 Trillion) for Method 6
GOLD_MARKET_CAP = 15000000000000 

def run_crypto_analysis():
    results = {}
    print("Fetching master coin list...")
    
    try:
        # 1. Fetch the top 500 coins by Market Cap (Smarter than fetching 15,000)
        # This ensures we get the ones people actually search for first.
        market_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
        response = requests.get(market_url)
        coins = response.json()
        
        print(f"Analyzing top {len(coins)} cryptocurrencies...")

        for coin in coins:
            symbol = coin['symbol'].upper()
            coin_id = coin['id']
            
            # --- BUCKET SEPARATION ---
            # Bucket B: Store of Value (BTC, LTC, DOGE, etc.)
            # Bucket A: Utility/Smart Contract (Everything else)
            is_store_of_value = symbol in ["BTC", "LTC", "DOGE", "BCH", "XMR"]
            
            mc = coin.get('market_cap') or 0
            fdv = coin.get('fully_diluted_valuation') or mc
            circulating = coin.get('circulating_supply') or 0
            total_supply = coin.get('total_supply') or circulating

            # --- VALUATION MATH ---
            
            # Method 6: Monetary Premium (Relative to Gold)
            monetary_premium = (mc / GOLD_MARKET_CAP) * 100 if GOLD_MARKET_CAP > 0 else 0
            
            # Method 10: Dilution Ratio (Circulating / Total)
            # High ratio = low future sell pressure from unlocks
            dilution_score = (circulating / total_supply) * 100 if total_supply > 0 else 100
            
            # Method 7: Metcalfe's Law Proxy 
            # (Since Active Addresses require a paid API, we use a Trading Volume/MC multiplier)
            metcalfe_proxy = (coin.get('total_volume', 0) ** 0.5) * 10 # Simplified proxy for network activity
            
            # Calculate a "Speculative Fair Value" 
            # For Bucket A: Based on FDV and Volume. For Bucket B: Based on Gold parity.
            if is_store_of_value:
                # Store of Value coins are valued on scarcity and gold-capture
                spec_value = mc * 1.1 # Placeholder logic: 10% premium over current
                asset_type = "Crypto - Store of Value"
            else:
                # Utility coins are valued on usage and dilution
                spec_value = mc * (dilution_score / 100)
                asset_type = "Crypto - Utility/Platform"

            results[symbol] = {
                "asset_type": asset_type,
                "Current_Price": coin.get('current_price'),
                "Method_6_Monetary_Premium": f"{round(monetary_premium, 4)}%",
                "Method_7_Metcalfe_Proxy_Value": round(metcalfe_proxy, 2),
                "Method_10_Dilution_Health": f"{round(dilution_score, 2)}%",
                "Final_Speculative_Value": round(spec_value / circulating, 2) if circulating > 0 else 0,
                "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            }

        # 2. SAVE TO JSON
        with open('crypto_valuations.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("✅ Crypto Valuations Updated!")

    except Exception as e:
        print(f"❌ Error updating crypto: {e}")

if __name__ == "__main__":
    run_crypto_analysis()
