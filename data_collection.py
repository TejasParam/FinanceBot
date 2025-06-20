import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataCollectionAgent:
    def __init__(self):
        self.historical_cache = {}
        
    def fetch_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        stock_data = f"{ticker}_{period}_{interval}"
        
        if stock_data in self.historical_cache:
            return self.historical_cache[stock_data]
            
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data.ffill(inplace=True)
        
        self.historical_cache[stock_data] = data
        return data

    def fetch_realtime_data(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        data = stock.fast_info
        
        return {
            'last_price': data.last_price,
            'volume': data.last_volume,
            'market_open': data._exchange_open_now,
            'previous_close': data.previous_close,
            'currency': data.currency
        }

    def fetch_fundamentals(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        return {
            'financials': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cashflow': stock.cashflow,
            'info': stock.info
        }
    

if __name__ == "__main__":
    agent = DataCollectionAgent()

    # Fetch historical data
    historical_data = agent.fetch_stock_data("MSFT", period="6mo", interval="1d")
    print(f"Historical data shape: {historical_data.shape}")

    # Fetch real-time quote
    realtime_data = agent.fetch_realtime_data("MSFT")
    print(f"Current price: {realtime_data['last_price']}")

    # Fetch fundamentals
    fundamentals = agent.fetch_fundamentals("MSFT")
    print(f"Enterprise value: {fundamentals['info'].get('enterpriseValue')}")