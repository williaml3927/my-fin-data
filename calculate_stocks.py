import json
import yfinance as yf
from datetime import datetime

def dcf_20_year(base_cf, growth, discount):
    if base_cf <= 0: return 0
    return round(sum([base_cf * (1 + growth)**t / (1 + discount)**t for t in range(1, 21)]), 2)

def process_stocks():
    assets = ["AAPL", "MSFT", "TSLA"]
    results = {}
    
    for ticker in assets:
        print(f"Analyzing Stock: {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            shares = info.get('sharesOutstanding', 1)
            net_income = info.get('netIncomeToCommon', 0) / shares
            ocf = info.get('operatingCashflow', 0) / shares
            
            # Assumptions
            growth = info.get('earningsGrowth', 0.05)
            discount = 0.08
            
            results[ticker] = {
                "asset_type": "Stock",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "DCF_Operating_Cash": dcf_20_year(ocf, growth, discount),
                "Mean_PE_Valuation": round(info.get('trailingPE', 0) * net_income, 2),
                "Mean_PS_Valuation": round(info.get('priceToSalesTrailing12Months', 0) * (info.get('totalRevenue', 0) / shares), 2),
                "Mean_PB_Valuation": round(info.get('priceToBook', 0) * info.get('bookValue', 0), 2),
                "PEG_Valuation": info.get('pegRatio', 0)
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    with open('stocks_valuations.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("✅ Stocks updated!")

if __name__ == "__main__":
    process_stocks()
