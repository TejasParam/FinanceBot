import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collection import DataCollectionAgent

class technical_analyst_agent:
    def calculate_fibonacci_retracement(self, ticker, period=30):
        """Calculate Fibonacci retracement levels for the last N periods"""
        try:
            data = self.getData(ticker)
            df = data[0].tail(period)
            if df.empty or not all(col in df for col in ['High', 'Low']):
                return None
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            levels = {
                '0.0%': high,
                '23.6%': high - 0.236 * diff,
                '38.2%': high - 0.382 * diff,
                '50.0%': high - 0.5 * diff,
                '61.8%': high - 0.618 * diff,
                '100.0%': low
            }
            return levels
        except Exception:
            return None

    def get_trailing_eps(self, ticker):
        """Get trailing EPS (Earnings Per Share)"""
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get('trailingEps', None)
        except Exception:
            return None
    def __init__(self):
        try:
            self.metrics = {}
        except Exception:
            self.metrics = {}
    
    #use data collection agent to get historical data, realtime data, fundamental data.
    def getData(self, ticker):
        try:
            data = []
            data_collector = DataCollectionAgent()
            data.append(data_collector.fetch_stock_data(ticker=ticker))
            data.append(data_collector.fetch_realtime_data(ticker=ticker))
            data.append(data_collector.fetch_fundamentals(ticker=ticker))
            return data
        except Exception:
            return [pd.DataFrame()]
    
    def calculate_Rsi_percent_change(self, ticker, period=14):
        try:
            data = self.getData(ticker)
            rsi_data = data[0].tail(period)['Close'].to_numpy()
            if len(rsi_data) < period:
                return None
            differences = (np.diff(rsi_data)/rsi_data[:-1])*100
            gains, losses = [], []
            for change in differences:
                if change < 0:
                    losses.append(abs(change))
                    gains.append(0)
                elif change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(0)
            avg_gain = sum(gains)/period if period else 0
            avg_loss = sum(losses)/period if period else 0
            if avg_loss == 0:
                return 100
            rsi = 100 -(100 / (1 + (avg_gain/avg_loss)))
            return rsi
        except Exception:
            return None
    

    def calculate_Rsi_price_change(self, ticker, period=14):
        try:
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
        except Exception:
            return None



    def calculate_simple_moving_average(self, ticker, period=14):
        try:
            data = self.getData(ticker)
            moving_avg_data = data[0].tail(period)['Close'].to_numpy()
            if len(moving_avg_data) < period:
                return None
            return np.average(moving_avg_data)
        except Exception:
            return None
    
    def calculate_initial_exponential_moving_average(self, ticker, period=14):
        try:
            data = self.getData(ticker)
            closing_prices = data[0].tail(period*2)['Close']
            if len(closing_prices) < period:
                return None
            ema = closing_prices.ewm(span=period, adjust=False).mean().iloc[-1]
            return ema
        except Exception:
            return None
    
    def calculate_Macd(self, ticker):
        try:
            ema12 = self.calculate_initial_exponential_moving_average(ticker, period=12)
            ema26 = self.calculate_initial_exponential_moving_average(ticker, period=26)
            if ema12 is None or ema26 is None:
                return None
            return ema12 - ema26
        except Exception:
            return None
    
    def get_PEratio(self, ticker):
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get("trailingPE", None)
        except Exception:
            return None
    
    def calculate_bollinger_bands(self, ticker, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            data = self.getData(ticker)
            closing_prices = data[0].tail(period)['Close']
            if len(closing_prices) < period:
                return None
            sma = closing_prices.mean()
            std = closing_prices.std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            return {'middle': sma, 'upper': upper_band, 'lower': lower_band}
        except Exception:
            return None

    def calculate_stochastic_oscillator(self, ticker, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            data = self.getData(ticker)
            df = data[0].tail(k_period + 10)
            if df.empty or not all(col in df for col in ['Low', 'High', 'Close']):
                return None
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            return {'k': k.iloc[-1], 'd': d.iloc[-1]}
        except Exception:
            return None

    def calculate_atr(self, ticker, period=14):
        """Calculate Average True Range"""
        try:
            data = self.getData(ticker)
            df = data[0].tail(period + 1)
            if df.empty or not all(col in df for col in ['High', 'Low', 'Close']):
                return None
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            return true_range.rolling(period).mean().iloc[-1]
        except Exception:
            return None
    
    def calculate_obv(self, ticker, period=20):
        """Calculate On-Balance Volume"""
        try:
            data = self.getData(ticker)
            df = data[0].tail(period + 10)
            if df.empty or 'Close' not in df or 'Volume' not in df:
                return None
            obv = 0
            obv_list = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv += df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv -= df['Volume'].iloc[i]
                obv_list.append(obv)
            return obv_list[-1]
        except Exception:
            return None

    def calculate_vwap(self, ticker):
        """Calculate Volume-Weighted Average Price (for intraday)"""
        try:
            data = self.getData(ticker)
            df = data[0]
            if df.empty or not all(col in df for col in ['High', 'Low', 'Close', 'Volume']):
                return None
            df = df.copy()
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Volume'] = df['Typical_Price'] * df['Volume']
            df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
            df['Cumulative_Volume'] = df['Volume'].cumsum()
            df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
            return df['VWAP'].iloc[-1]
        except Exception:
            return None
    
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
            return stock_info.get('priceToBook', None)
        except Exception:
            return None

    def get_eps_growth(self, ticker):
        """Get EPS growth rate"""
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get('earningsGrowth', None)
        except Exception:
            return None

    def get_dividend_yield(self, ticker):
        """Get dividend yield"""
        try:
            stock_info = yf.Ticker(ticker).info
            return stock_info.get('dividendYield', 0)
        except Exception:
            return 0
    
    def calculate_rsi(self, ticker, period=14):
        """Calculate RSI (alias for calculate_Rsi_percent_change)"""
        return self.calculate_Rsi_percent_change(ticker, period)
    
    def calculate_macd(self, ticker):
        """Calculate MACD (alias for calculate_Macd)"""
        return self.calculate_Macd(ticker)
        
    def calculate_bollinger_bands_alias(self, ticker):
        """Calculate Bollinger Bands (alias using the existing method)"""
        return self.calculate_bollinger_bands(ticker)
        
    def calculate_sma(self, ticker, period=20):
        """Calculate Simple Moving Average"""
        try:
            data = self.getData(ticker)
            df = data[0]
            if df.empty or 'Close' not in df:
                return None
            return df['Close'].rolling(window=period).mean().iloc[-1]
        except Exception:
            return None
            
    def calculate_ema(self, ticker, period=12):
        """Calculate Exponential Moving Average"""
        try:
            data = self.getData(ticker)
            df = data[0]
            if df.empty or 'Close' not in df:
                return None
            return df['Close'].ewm(span=period).mean().iloc[-1]
        except Exception:
            return None

if __name__ == "__main__":
    metrics = technical_analyst_agent()
    print("RSI (percent change):", metrics.calculate_Rsi_percent_change('AAPL'))
    print("Simple Moving Average:", metrics.calculate_simple_moving_average('AAPL'))
    print("Exponential Moving Average:", metrics.calculate_initial_exponential_moving_average('AAPL'))
    print("MACD:", metrics.calculate_Macd('AAPL'))
    print("P/E Ratio:", metrics.get_PEratio('AAPL'))
    print("Bollinger Bands (middle, upper, lower):", metrics.calculate_bollinger_bands('AAPL'))
    print("Stochastic Oscillator (k, d):", metrics.calculate_stochastic_oscillator('AAPL'))
    print("Average True Range (ATR):", metrics.calculate_atr('AAPL'))
    print("VWAP:", metrics.calculate_vwap('AAPL'))
    print("On-Balance Volume (OBV):", metrics.calculate_obv('AAPL'))
    print("Price to Book Ratio:", metrics.get_price_to_book('AAPL'))
    print("EPS Growth:", metrics.get_eps_growth('AAPL'))
    print("Dividend Yield:", metrics.get_dividend_yield('AAPL'))
    print("Fibonacci Retracement Levels:", metrics.calculate_fibonacci_retracement('AAPL'))
    print("Trailing EPS:", metrics.get_trailing_eps('AAPL'))
