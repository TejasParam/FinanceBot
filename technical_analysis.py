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
    
    def calculate_Rsi(self,ticker,period=14):
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



if __name__ == "__main__":
    metrics = technical_analyst_agent()
    print(metrics.calculate_Rsi('AAPL'))