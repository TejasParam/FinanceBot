import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collection import DataCollectionAgent

class technical_analyst_agent:
    def __init__ (self):
        self.metrics = {}
    
    #use data collection agent to get historical data, realtime data, fundamental data.
    def getData(self, ticker):
        data = []
        data_collector = DataCollectionAgent()
        data.append(data_collector.fetch_stock_data(ticker=ticker))
        data.append(data_collector.fetch_realtime_data(ticker=ticker))
        data.append(data_collector.fetch_fundamentals(ticker=ticker))
        return data
    
    def calculate_Rsi_percent_change(self,ticker,period=14):
        #get last 15 closing prices
        data = self.getData(ticker)
        rsi_data = data[0].tail(period)['Close'].to_numpy()

        #calculate percent differences from each day
        differences = (np.diff(rsi_data)/rsi_data[:-1])*100

        #calculate average gaina nd average loss
        gains,losses = [],[]
        for change in differences:
            if change < 0:
                losses.append(abs(change))
                gains.append(0)
            elif change >0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(0)
        avg_gain,avg_loss = sum(gains)/period,sum(losses)/period

        #final rsi calculation
        rsi = 100 -(100 / (1 + (avg_gain/avg_loss)))
        return rsi
    

    def calculate_Rsi_price_change(self, ticker, period=14):
        data = self.getData(ticker)
        closes = data[0]['Close']
        if len(closes) < period + 1:
            return None  # Not enough data
        delta = closes.diff().tail(period)
        gain = delta.clip(lower=0).mean()
        loss = -delta.clip(upper=0).mean()
        if loss == 0:
            return 100
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi



    def calculate_simple_moving_average(self,ticker,period=14):
        data = self.getData(ticker)
        moving_avg_data = data[0].tail(period)['Close'].to_numpy()
        return np.average(moving_avg_data)
    
    def calculate_initial_exponential_moving_average(self, ticker, period = 14):
        data = self.getData(ticker)
        closing_prices = data[0].tail(period*2)['Close']
        ema = closing_prices.ewm(span=period, adjust=False).mean().iloc[-1]
        return ema
    
    def calculate_Macd(self, ticker):
        return self.calculate_initial_exponential_moving_average(ticker,period=12) - self.calculate_initial_exponential_moving_average(ticker,period=26)
    
    def get_PEratio(self, ticker):
        stock_info = yf.Ticker(ticker).info
        return stock_info["trailingPE"]



if __name__ == "__main__":
    metrics = technical_analyst_agent()
    print(metrics.calculate_Rsi('AAPL'))
    print(metrics.calculate_simple_moving_average('AAPL'))
    print(metrics.calculate_initial_exponential_moving_average('AAPL'))
    print(metrics.calculate_Macd('AAPL'))
    print(metrics.get_PEratio('AAPL'))