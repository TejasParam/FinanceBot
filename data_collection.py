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
        data.fillna(method='ffill', inplace=True)
        
        self.historical_cache[stock_data] = data
        return data

    def fetch_realtime_data(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        data = stock.fast_info
        
        return {
            'last_price': data.last_price,
            'volume': data.last_volume,
            'market_open': data.market_open,
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