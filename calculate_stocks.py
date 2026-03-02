import json
import requests
import yfinance as yf
from datetime import datetime

# --- SOURCES ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"

def get_all_tickers():
    tickers = []
    for url in [NYSE_URL, OTHER_URL]:
        try:
            response = requests.get(url)
            data = response.json()
            for item in data:
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    tickers.append(symbol.replace('.', '-'))
        except Exception: continue
    # Increased to 500 since we are using fewer methods
    return list(set(tickers))[:500] 

def calculate_dcf(base_value, growth, discount, years=20):
    if base_value <= 0: return 0
    pv_sum = sum([base_value * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)])
    return round(pv_sum, 2)

def run_analysis():
    all_tickers = get_all_tickers()
    results = {}
    
    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            # Use fast_info or limit info calls to speed up
            info = stock.info 
            
            # --- DATA POINTS ---
            shares = info.get('sharesOutstanding', 1)
            fcf = info.get('freeCashflow', 0) / shares
            net_inc = info.get('netIncomeToCommon', 0) / shares
            rev_ps = info.get('totalRevenue', 0) / shares
            
            # --- ASSUMPTIONS ---
            growth = info.get('earningsGrowth', 0.05)
            discount = 0.09 
            
            # --- THE TOP 6 METHODS ---
            vals = {
                "1_DCF_Terminal_Value": round(calculate_dcf(fcf, growth, discount, 10) + ((fcf * (1+growth)**10 * 1.02)/(discount-0.02))/(1+discount)**10, 2),
                "2_Mean_PE_Valuation": round(info.get('trailingPE', 0) * net_inc, 2) if info.get('trailingPE') else 0,
                "3_Mean_PS_Valuation": round(info.get('priceToSalesTrailing12Months', 0) * rev_ps, 2) if info.get('priceToSalesTrailing12Months') else 0,
                "4_PEG_Ratio_Valuation": info.get('pegRatio', 0),
                "5_Mean_PB_Valuation": round(info.get('priceToBook', 0) * info.get('bookValue', 0), 2) if info.get('priceToBook') else 0,
                "6_20yr_Discounted_FCF": calculate_dcf(fcf, growth, discount, 20)
            }
            
            # --- WEIGHTED FAIR VALUE ---
            # We give 40% weight to Terminal Value, 40% to PE/PEG, 20% to the others
            price_metrics = [vals["1_DCF_Terminal_Value"], vals["2_Mean_PE_Valuation"], vals["6_20yr_Discounted_FCF"]]
            valid_prices = [p for p in price_metrics if p > 0]
            
            vals["Final_Fair_Value"] = round(sum(valid_prices) / len(valid_prices), 2) if valid_prices else 0
            vals["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            results[ticker] = vals
            print(f"Synced: {ticker}")

        except Exception: continue

    with open('stocks_valuations.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_analysis()
