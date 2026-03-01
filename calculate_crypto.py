import json
from datetime import datetime

def process_crypto():
    # Bucket A = Cash-Flow Networks, Bucket B = Pure Money/Store of Value
    bucket_a = ["ETH", "SOL", "UNI"] 
    bucket_b = ["BTC", "DOGE"]
    
    results = {}
    
    # Mock API Data (Replace with CoinGecko / DefiLlama API calls later)
    mock_data = {
        "ETH": {"fdv": 400000000000, "annual_fees": 2500000000, "fees_burned": 2000000000, "circulating": 120, "max_supply": 120},
        "BTC": {"market_cap": 1200000000000, "active_addresses": 1000000, "gold_mc": 15000000000000},
    }

    # --- PROCESS BUCKET A (CASH FLOW) ---
    for coin in bucket_a:
        print(f"Analyzing Cash-Flow Crypto: {coin}...")
        try:
            data = mock_data.get(coin)
            if not data: continue
            
            # Method 2: Price-to-Sales (FDV / Fees)
            p_s = data['fdv'] / data['annual_fees'] if data['annual_fees'] > 0 else 0
            
            # Method 3: Price-to-Earnings (FDV / Fees Burned or Distributed)
            p_e = data['fdv'] / data['fees_burned'] if data['fees_burned'] > 0 else 0
            
            # Method 10: Dilution Ratio (Circulating / Max Supply)
            dilution_ratio = data['circulating'] / data['max_supply'] if data['max_supply'] > 0 else 1
            
            results[coin] = {
                "asset_type": "Crypto - Cash Flow",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Method_2_Price_to_Sales": round(p_s, 2),
                "Method_3_Price_to_Earnings": round(p_e, 2),
                "Method_10_Dilution_Ratio": f"{round(dilution_ratio * 100, 2)}%"
            }
        except Exception as e:
            results[coin] = {"error": str(e)}

    # --- PROCESS BUCKET B (STORE OF VALUE) ---
    for coin in bucket_b:
        print(f"Analyzing Network Crypto: {coin}...")
        try:
            data = mock_data.get(coin)
            if not data: continue
            
            # Method 6: Monetary Premium (Coin MC / Gold MC)
            monetary_premium = data['market_cap'] / data['gold_mc']
            
            # Method 7: Metcalfe's Law (Active Addresses ^ 2 * constant)
            metcalfe_value = (data['active_addresses'] ** 2) * 0.000001
            
            results[coin] = {
                "asset_type": "Crypto - Network Value",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Method_6_Monetary_Premium_vs_Gold": f"{round(monetary_premium * 100, 2)}%",
                "Method_7_Metcalfes_Law_Value": round(metcalfe_value, 2)
            }
        except Exception as e:
            results[coin] = {"error": str(e)}

    with open('crypto_valuations.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("✅ Crypto updated!")

if __name__ == "__main__":
    process_crypto()
