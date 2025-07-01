import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataCollectionAgent:
    def __init__(self):
        self.historical_cache = {}
        
    def fetch_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        stock_data = f"{ticker}_{period}_{interval}"
        try:
            if stock_data in self.historical_cache:
                return self.historical_cache[stock_data]
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty or 'Close' not in data:
                return pd.DataFrame()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Daily_Return'] = data['Close'].pct_change()
            data.ffill(inplace=True)
            self.historical_cache[stock_data] = data
            return data
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_realtime_data(self, ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            data = stock.fast_info
            return {
                'last_price': getattr(data, 'last_price', None),
                'volume': getattr(data, 'last_volume', None),
                'market_open': getattr(data, '_exchange_open_now', None),
                'previous_close': getattr(data, 'previous_close', None),
                'currency': getattr(data, 'currency', None)
            }
        except Exception as e:
            print(f"Error fetching real-time data for {ticker}: {e}")
            return {
                'last_price': None,
                'volume': None,
                'market_open': None,
                'previous_close': None,
                'currency': None
            }

    def fetch_fundamentals(self, ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            return {
                'financials': getattr(stock, 'financials', None),
                'balance_sheet': getattr(stock, 'balance_sheet', None),
                'cashflow': getattr(stock, 'cashflow', None),
                'info': getattr(stock, 'info', None)
            }
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {
                'financials': None,
                'balance_sheet': None,
                'cashflow': None,
                'info': None
            }
    

if __name__ == "__main__":
    agent = DataCollectionAgent()

    # Fetch historical data
    historical_data = agent.fetch_stock_data("MSFT", period="6mo", interval="1d")
    print(f"Historical data shape: {historical_data}")

    # Fetch real-time quote
    realtime_data = agent.fetch_realtime_data("MSFT")
    print(f"Current price: {realtime_data['last_price']}")

    # Fetch fundamentals
    fundamentals = agent.fetch_fundamentals("MSFT")
    print(f"Enterprise value: {fundamentals['info'].get('enterpriseValue')}")