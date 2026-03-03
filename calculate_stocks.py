import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os

# --- SOURCES ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"

# Priority list to ensure these are always processed first
CORE_PRIORITY = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO"]

def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = requests.get(url)
            data = res.json()
            for item in data:
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers: tickers.append(clean)
        except: continue
    return tickers # No limit here; let concurrency handle the volume

def calculate_dcf(base_val, growth, discount, years=20):
    if base_val <= 0: return 0
    return round(sum([base_val * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)]), 2)

def analyze_ticker(ticker):
    """The Worker function: Analyzes a single stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'sharesOutstanding' not in info: return None
        
        shares = info.get('sharesOutstanding', 1)
        fcf = info.get('freeCashflow', 0) / shares
        net_inc = info.get('netIncomeToCommon', 0) / shares
        rev_ps = info.get('totalRevenue', 0) / shares
        growth = max(0.01, min(info.get('earningsGrowth', 0.05), 0.30))
        
        vals = {
            "1_DCF_Terminal_Value": round(calculate_dcf(fcf, growth, 0.09, 10) + ((fcf * (1+growth)**10 * 1.02)/(0.07))/(1.09)**10, 2),
            "2_Mean_PE_Valuation": round(info.get('trailingPE', 0) * net_inc, 2) if info.get('trailingPE') else 0,
            "3_Mean_PS_Valuation": round(info.get('priceToSalesTrailing12Months', 0) * rev_ps, 2) if info.get('priceToSalesTrailing12Months') else 0,
            "4_PEG_Ratio_Valuation": info.get('pegRatio', 0),
            "5_Mean_PB_Valuation": round(info.get('priceToBook', 0) * info.get('bookValue', 0), 2) if info.get('priceToBook') else 0,
            "6_20yr_Discounted_FCF": calculate_dcf(fcf, growth, 0.09, 20),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Calculate Weighted Final Fair Value
        price_metrics = [vals["1_DCF_Terminal_Value"], vals["2_Mean_PE_Valuation"], vals["6_20yr_Discounted_FCF"]]
        valid = [p for p in price_metrics if 0 < p < 50000]
        vals["Final_Fair_Value"] = round(sum(valid)/len(valid), 2) if valid else 0
        
        return ticker, vals
    except:
        return None

def save_partitioned_data(master_results):
    """Splits one big dict into A.json, B.json, etc."""
    partitions = {}
    for ticker, data in master_results.items():
        letter = ticker[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in partitions: partitions[letter] = {}
        partitions[letter][ticker] = data
    
    # Ensure a 'directory' folder exists
    if not os.path.exists('data'): os.makedirs('data')
    
    for letter, content in partitions.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)
    print(f"Successfully saved {len(partitions)} partitioned files.")

def main():
    tickers = get_all_tickers()
    print(f"Starting parallel analysis on {len(tickers)} stocks...")
    
    master_results = {}
    # 15 workers is the sweet spot for GitHub Actions speed vs API limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                ticker, data = res
                master_results[ticker] = data
                if len(master_results) % 50 == 0: print(f"Processed {len(master_results)}...")

    save_partitioned_data(master_results)

if __name__ == "__main__":
    main()
