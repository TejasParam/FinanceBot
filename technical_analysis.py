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
    
    def calculate_bollinger_bands(self, ticker, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        data = self.getData(ticker)
        closing_prices = data[0].tail(period)['Close']
        sma = closing_prices.mean()
        std = closing_prices.std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return {'middle': sma, 'upper': upper_band, 'lower': lower_band}

    def calculate_stochastic_oscillator(self, ticker, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        data = self.getData(ticker)
        df = data[0].tail(k_period + 10)  # Get extra data for calculation
        
        # Calculate %K
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (3-day SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        return {'k': k.iloc[-1], 'd': d.iloc[-1]}

    def calculate_atr(self, ticker, period=14):
        """Calculate Average True Range"""
        data = self.getData(ticker)
        df = data[0].tail(period + 1)
        
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
    
    def calculate_obv(self, ticker, period=20):
        """Calculate On-Balance Volume"""
        data = self.getData(ticker)
        df = data[0].tail(period + 10)
        
        obv = 0
        obv_list = [0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv += df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv -= df['Volume'].iloc[i]
            obv_list.append(obv)
        
        return obv_list[-1]

    def calculate_vwap(self, ticker):
        """Calculate Volume-Weighted Average Price (for intraday)"""
        data = self.getData(ticker)
        df = data[0]
        
        # VWAP calculation
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TP_Volume'] = df['Typical_Price'] * df['Volume']
        df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
        
        return df['VWAP'].iloc[-1]
    
    def calculate_obv(self, ticker, period=20):
        """Calculate On-Balance Volume"""
        data = self.getData(ticker)
        df = data[0].tail(period + 10)
        
        obv = 0
        obv_list = [0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv += df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv -= df['Volume'].iloc[i]
            obv_list.append(obv)
        
        return obv_list[-1]

    def calculate_vwap(self, ticker):
        """Calculate Volume-Weighted Average Price (for intraday)"""
        data = self.getData(ticker)
        df = data[0]
        
        # VWAP calculation
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TP_Volume'] = df['Typical_Price'] * df['Volume']
        df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
        
        return df['VWAP'].iloc[-1]
    
    def get_price_to_book(self, ticker):
        """Get Price-to-Book ratio"""
        try:
            stock_info = yf.Ticker(ticker).info
            if 'priceToBook' in stock_info:
                return stock_info['priceToBook']
            return None
        except:
            return None

    def get_eps_growth(self, ticker):
        """Get EPS growth rate"""
        try:
            stock_info = yf.Ticker(ticker).info
            if 'earningsGrowth' in stock_info:
                return stock_info['earningsGrowth']
            return None
        except:
            return None

    def get_dividend_yield(self, ticker):
        """Get dividend yield"""
        try:
            stock_info = yf.Ticker(ticker).info
            if 'dividendYield' in stock_info:
                return stock_info['dividendYield']
            return 0
        except:
            return 0




if __name__ == "__main__":
    metrics = technical_analyst_agent()
    print(metrics.calculate_Rsi_percent_change('AAPL'))
    print(metrics.calculate_simple_moving_average('AAPL'))
    print(metrics.calculate_initial_exponential_moving_average('AAPL'))
    print(metrics.calculate_Macd('AAPL'))
    print(metrics.get_PEratio('AAPL'))
    print(metrics.calculate_bollinger_bands('AAPL'))
    print(metrics.calculate_stochastic_oscillator('AAPL'))
    print(metrics.calculate_atr('AAPL'))
    print(metrics.calculate_vwap('AAPL'))
    print(metrics.calculate_obv('AAPL'))
    print(metrics.get_price_to_book('AAPL'))
    print(metrics.get_eps_growth('AAPL'))
    print(metrics.get_dividend_yield('AAPL'))
