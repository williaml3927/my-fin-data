import json
import requests
import yfinance as yf
from datetime import datetime

# --- YOUR SOURCE LISTS ---
NYSE_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/NYSE.json"
OTHER_URL = "https://raw.githubusercontent.com/williaml3927/NYSE-list/refs/heads/main/Other%20list.json"

def get_all_tickers():
    tickers = []
    for url in [NYSE_URL, OTHER_URL]:
        try:
            response = requests.get(url)
            data = response.json()
            # This assumes your JSON is a list of objects like [{"Symbol": "AAPL", ...}]
            # Adjust 'Symbol' if your JSON key is different (e.g., 'ticker' or 'ACT Symbol')
            for item in data:
                symbol = item.get('Symbol') or item.get('ACT Symbol') or item.get('ticker')
                if symbol:
                    # Yahoo Finance likes '-' instead of '.' (e.g., BRK.B -> BRK-B)
                    tickers.append(symbol.replace('.', '-'))
        except Exception as e:
            print(f"Could not fetch list from {url}: {e}")
    
    # Remove duplicates and limit to a manageable number for GitHub Actions (e.g., Top 500)
    # If you try to do 5,000+ at once, the script will timeout.
    return list(set(tickers))[:500] 

def calculate_dcf(base_value, growth, discount, years=20):
    if base_value <= 0: return 0
    pv_sum = sum([base_value * (1 + growth)**t / (1 + discount)**t for t in range(1, years + 1)])
    return round(pv_sum, 2)

def run_analysis():
    all_tickers = get_all_tickers()
    results = {}
    print(f"Analyzing {len(all_tickers)} stocks from your master lists...")

    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Data Extraction
            shares = info.get('sharesOutstanding', 1)
            fcf = info.get('freeCashflow', 0) / shares
            ocf = info.get('operatingCashflow', 0) / shares
            net_inc = info.get('netIncomeToCommon', 0) / shares
            rev_ps = info.get('totalRevenue', 0) / shares
            
            # Assumptions
            growth = info.get('earningsGrowth', 0.05)
            discount = 0.09
            
            # Valuation Logic
            vals = {
                "1_DCF_OCF": calculate_dcf(ocf, growth, discount),
                "2_DCF_FCF": calculate_dcf(fcf, growth, discount),
                "3_Discounted_Net_Income": calculate_dcf(net_inc, growth, discount),
                "4_Mean_PE_Valuation": round(info.get('trailingPE', 0) * net_inc, 2) if info.get('trailingPE') else 0,
                "5_Mean_PS_Valuation": round(info.get('priceToSalesTrailing12Months', 0) * rev_ps, 2) if info.get('priceToSalesTrailing12Months') else 0,
                "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            # Simple Fair Value Average
            prices = [vals["1_DCF_OCF"], vals["2_DCF_FCF"], vals["4_Mean_PE_Valuation"]]
            valid_prices = [p for p in prices if p > 0]
            vals["Final_Fair_Value"] = round(sum(valid_prices)/len(valid_prices), 2) if valid_prices else 0
            
            results[ticker] = vals
            print(f"✅ {ticker} Success")

        except Exception:
            print(f"❌ {ticker} Skipped (Data missing)")

    with open('stocks_valuations.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_analysis()
