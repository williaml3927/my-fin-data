import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os

# --- SOURCES ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"

# Core priority ensures the most searched assets never get skipped by API hiccups
CORE_PRIORITY = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO"]

def get_all_tickers():
    """Fetches every single ticker from your GitHub JSON lists."""
    tickers = CORE_PRIORITY.copy()
    
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = requests.get(url)
            data = res.json()
            for item in data:
                # Handle different key names your JSON might use
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers: 
                        tickers.append(clean)
        except Exception as e:
            print(f"Warning: Could not read from {url} - {e}")
            
    print(f"Total tickers queued for analysis: {len(tickers)}")
    return tickers # Notice there is NO [:500] limit here anymore

def calculate_dcf(base_val, growth, discount, years=20):
    """Standard Discounted Cash Flow math."""
    if base_val <= 0: return 0
    return round(sum([base_val * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)]), 2)

def analyze_ticker(ticker):
    """The Worker function: Analyzes a single stock and returns its 6 metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # If Yahoo Finance doesn't have data for this stock, skip it
        if not info or 'sharesOutstanding' not in info: 
            return None
        
        # Data Extraction
        shares = info.get('sharesOutstanding', 1)
        fcf = info.get('freeCashflow', 0) / shares
        net_inc = info.get('netIncomeToCommon', 0) / shares
        rev_ps = info.get('totalRevenue', 0) / shares
        
        # Cap growth between 1% and 30% to prevent insane valuation spikes
        growth = max(0.01, min(info.get('earningsGrowth', 0.05), 0.30))
        
        # The Top 6 Valuation Methods
        vals = {
            "1_DCF_Terminal_Value": round(calculate_dcf(fcf, growth, 0.09, 10) + ((fcf * (1+growth)**10 * 1.02)/(0.09-0.02))/(1.09)**10, 2),
            "2_Mean_PE_Valuation": round(info.get('trailingPE', 0) * net_inc, 2) if info.get('trailingPE') else 0,
            "3_Mean_PS_Valuation": round(info.get('priceToSalesTrailing12Months', 0) * rev_ps, 2) if info.get('priceToSalesTrailing12Months') else 0,
            "4_PEG_Ratio_Valuation": info.get('pegRatio', 0),
            "5_Mean_PB_Valuation": round(info.get('priceToBook', 0) * info.get('bookValue', 0), 2) if info.get('priceToBook') else 0,
            "6_20yr_Discounted_FCF": calculate_dcf(fcf, growth, 0.09, 20),
            "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Calculate Final Fair Value (Averaging only valid, reasonable prices)
        price_metrics = [vals["1_DCF_Terminal_Value"], vals["2_Mean_PE_Valuation"], vals["6_20yr_Discounted_FCF"]]
        valid = [p for p in price_metrics if 0 < p < 50000]
        vals["Final_Fair_Value"] = round(sum(valid)/len(valid), 2) if valid else 0
        
        return ticker, vals
    except Exception:
        return None # Silently fail on bad tickers to keep the script moving

def save_partitioned_data(master_results):
    """Splits the massive dictionary into A-Z JSON files."""
    partitions = {}
    
    # Sort into buckets by the first letter of the ticker
    for ticker, data in master_results.items():
        letter = ticker[0].upper()
        if not letter.isalpha(): 
            letter = "0-9" # Catch-all for tickers starting with numbers
        if letter not in partitions: 
            partitions[letter] = {}
        partitions[letter][ticker] = data
    
    # Create the 'data' folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save each bucket as its own file
    for letter, content in partitions.items():
        filepath = f'data/stocks_{letter}.json'
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=4)
            
    print(f"✅ Successfully saved {len(partitions)} A-Z partitioned files into the 'data/' folder.")

def main():
    tickers = get_all_tickers()
    master_results = {}
    
    print(f"Starting parallel analysis using 15 threads...")
    
    # Run the analysis concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                ticker, data = res
                master_results[ticker] = data
                # Print a progress update every 100 successful stocks
                if len(master_results) % 100 == 0: 
                    print(f"Processed {len(master_results)} stocks successfully...")

    save_partitioned_data(master_results)
    print(f"Done! Final count of calculated stocks: {len(master_results)}")

if __name__ == "__main__":
    main()
