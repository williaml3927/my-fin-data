import json
import requests
import yfinance as yf
import concurrent.futures
from datetime import datetime
import os
import time
import math

# --- ANTI-BOT SESSION ---
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# --- SOURCES ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"
CORE_PRIORITY = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO"]

def clean_json(obj):
    """Recursively replaces NaN and Infinity with 0 to prevent JSON crashes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return 0
        return obj
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

def get_all_tickers():
    tickers = CORE_PRIORITY.copy()
    for url in [NYSE_URL, OTHER_URL]:
        try:
            res = session.get(url, timeout=10)
            data = res.json()
            for item in data:
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    clean = symbol.strip().upper().replace('.', '-')
                    if clean not in tickers: 
                        tickers.append(clean)
        except Exception as e:
            print(f"Could not read {url}: {e}")
    return tickers 

def calculate_dcf(base_val, growth, discount, years=20):
    if base_val <= 0: return 0
    return round(sum([base_val * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)]), 2)

def get_10yr_history(stock):
    """Fetches the last 10 years of data, saving only the year-end closing price to keep JSON fast."""
    try:
        hist = stock.history(period="10y")
        if hist.empty: return {}
        
        yearly_prices = {}
        years = sorted(list(set([d.year for d in hist.index])))
        for y in years:
            yearly_prices[str(y)] = round(hist[hist.index.year == y]['Close'].iloc[-1], 2)
        return yearly_prices
    except:
        return {}

def analyze_ticker(ticker, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(0.5) # Rate-limit protection
            
            stock = yf.Ticker(ticker, session=session)
            info = stock.info
            
            if not info or 'sharesOutstanding' not in info: 
                return None
            
            # --- BASE FUNDAMENTALS ---
            shares = info.get('sharesOutstanding', 1)
            fcf = info.get('freeCashflow', 0) / shares
            ocf = info.get('operatingCashflow', 0) / shares
            net_inc = info.get('netIncomeToCommon', 0) / shares
            eps_nri = info.get('trailingEps', net_inc) # Standardized EPS usually excludes NRI
            rev_ps = info.get('totalRevenue', 0) / shares
            
            # Growth Rates (Capped between 1% and 30% for realistic forecasting)
            growth = max(0.01, min(info.get('earningsGrowth', 0.05), 0.30))
            rev_growth = max(0.01, min(info.get('revenueGrowth', 0.05), 0.30))
            
            # --- THE 10 LOCAL VALUATION MODELS ---
            
            # 1. 20-Year DCF (Operating Cash Flow)
            val_1_ocf_dcf = calculate_dcf(ocf, growth, 0.09, 20)
            
            # 2. 20-Year DCF (Free Cash Flow)
            val_2_fcf_dcf = calculate_dcf(fcf, growth, 0.09, 20)
            
            # 3. 20-Year DCF (Net Income)
            val_3_ni_dcf = calculate_dcf(eps_nri, growth, 0.09, 20)
            
            # 4. DCF with Terminal Value (10yr explicit + Terminal)
            val_4_terminal_dcf = round(calculate_dcf(fcf, growth, 0.09, 10) + ((fcf * (1+growth)**10 * 1.02)/(0.09-0.02))/(1.09)**10, 2)
            
            # 5. Mean P/S Valuation
            val_5_ps = round(info.get('priceToSalesTrailing12Months', 0) * rev_ps, 2)
            
            # 6. Mean P/E Valuation (excl. NRI)
            val_6_pe = round(info.get('trailingPE', 0) * eps_nri, 2)
            
            # 7. Mean P/B Valuation
            val_7_pb = round(info.get('priceToBook', 0) * info.get('bookValue', 0), 2)
            
            # 8. PSG Valuation (Fair P/S = Revenue Growth * 100)
            val_8_psg = round((rev_growth * 100) * rev_ps, 2)
            
            # 9. PEG Valuation (Fair P/E = Earnings Growth * 100)
            val_9_peg = round((growth * 100) * eps_nri, 2)
            
            # 10. Metcalfe's Law Valuation Proxy (V ∝ N^2)
            # Proxy network scale via Total Revenue, scaled mathematically to fit equity pricing.
            total_rev = info.get('totalRevenue', 0)
            val_10_metcalfe = round(((total_rev / 1000000) ** 2) / (shares * 1000) if shares > 0 else 0, 2)

            vals = {
                "Valuation_1_20yr_DCF_OCF": val_1_ocf_dcf,
                "Valuation_2_20yr_DCF_FCF": val_2_fcf_dcf,
                "Valuation_3_20yr_DCF_Net_Income": val_3_ni_dcf,
                "Valuation_4_DCF_Terminal_Value": val_4_terminal_dcf,
                "Valuation_5_Mean_PS": val_5_ps,
                "Valuation_6_Mean_PE_excl_NRI": val_6_pe,
                "Valuation_7_Mean_PB": val_7_pb,
                "Valuation_8_PSG": val_8_psg,
                "Valuation_9_PEG": val_9_peg,
                "Valuation_10_Metcalfes_Law": val_10_metcalfe,
            }
            
            # Final Fair Value Calculation (Averaging reasonable, non-zero metrics to avoid outliers)
            price_metrics = list(vals.values())
            valid_metrics = [p for p in price_metrics if 0 < p < 50000]
            final_fair_value = round(sum(valid_metrics)/len(valid_metrics), 2) if valid_metrics else 0
            vals["Final_Fair_Value"] = final_fair_value
            
            # --- FORECASTS & HISTORY ---
            vals["10_Year_History"] = get_10yr_history(stock)
            
            current_year = datetime.now().year
            vals["5_Year_Forecast"] = {
                str(current_year + i): round(final_fair_value * (1 + growth)**i, 2)
                for i in range(1, 6)
            }
            
            vals["Last_Updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            return ticker, vals
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e): time.sleep(5)
            continue 
    return None

def save_partitioned_data(master_results):
    partitions = {}
    
    # CRITICAL: Clean JSON of NaN/Infinity before saving!
    clean_master = clean_json(master_results)
    
    for ticker, data in clean_master.items():
        letter = ticker[0].upper()
        if not letter.isalpha(): letter = "0-9"
        if letter not in partitions: partitions[letter] = {}
        partitions[letter][ticker] = data
    
    os.makedirs('data', exist_ok=True)
    for letter, content in partitions.items():
        with open(f'data/stocks_{letter}.json', 'w') as f:
            json.dump(content, f, indent=4)
    print(f"✅ Saved A-Z Stock files. ({len(clean_master)} stocks processed)")

def main():
    tickers = get_all_tickers()
    master_results = {}
    print(f"Starting local financial modeling across {len(tickers)} stocks...")
    
    # 5 workers is the sweet spot for avoiding Yahoo Finance bans while maintaining speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            res = future.result()
            if res:
                master_results[res[0]] = res[1]
                if len(master_results) % 50 == 0: 
                    print(f"Processed {len(master_results)} stocks locally...")

    save_partitioned_data(master_results)

if __name__ == "__main__":
    main()
