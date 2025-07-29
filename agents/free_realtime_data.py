"""
Free Real-time Data Feed System
No paid APIs - uses free sources for real-time market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
import requests
import json
import websocket
from collections import deque

class FreeRealtimeDataFeed:
    """
    Free real-time data feed aggregator
    Combines multiple free sources for near real-time data
    """
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.price_cache = {}
        self.order_book_simulator = OrderBookSimulator()
        self.running = False
        
        # Free data sources
        self.sources = {
            'yahoo': YahooRealtime(),
            'finnhub': FinnhubWebsocket(),  # Free tier
            'alpaca': AlpacaPaperData(),    # Free paper trading
            'iex': IEXCloudFree(),          # Free tier
        }
        
        # Data buffers
        self.tick_buffers = {}
        self.minute_bars = {}
        
    def start_feeds(self, symbols: List[str]):
        """Start real-time feeds for given symbols"""
        self.running = True
        
        # Start each source
        for source_name, source in self.sources.items():
            try:
                source.connect(symbols, self.on_data_received)
                print(f"✓ Connected to {source_name}")
            except Exception as e:
                print(f"✗ Failed to connect to {source_name}: {e}")
        
        # Start data processor thread
        self.processor_thread = threading.Thread(target=self._process_data)
        self.processor_thread.start()
        
    def on_data_received(self, source: str, data: Dict[str, Any]):
        """Callback for receiving data from sources"""
        data['source'] = source
        data['timestamp'] = time.time()
        self.data_queue.put(data)
        
    def _process_data(self):
        """Process incoming data and create unified feed"""
        while self.running:
            try:
                # Get data with timeout
                data = self.data_queue.get(timeout=1.0)
                
                symbol = data.get('symbol')
                if not symbol:
                    continue
                
                # Update price cache
                if 'price' in data:
                    self.price_cache[symbol] = {
                        'price': data['price'],
                        'bid': data.get('bid', data['price'] - 0.01),
                        'ask': data.get('ask', data['price'] + 0.01),
                        'volume': data.get('volume', 0),
                        'timestamp': data['timestamp']
                    }
                
                # Update tick buffer
                if symbol not in self.tick_buffers:
                    self.tick_buffers[symbol] = deque(maxlen=10000)
                
                self.tick_buffers[symbol].append(data)
                
                # Create minute bars
                self._update_minute_bars(symbol, data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Data processing error: {e}")
    
    def _update_minute_bars(self, symbol: str, tick_data: Dict[str, Any]):
        """Aggregate ticks into minute bars"""
        if symbol not in self.minute_bars:
            self.minute_bars[symbol] = []
        
        current_minute = int(tick_data['timestamp'] / 60) * 60
        
        # Check if we need a new bar
        if (not self.minute_bars[symbol] or 
            self.minute_bars[symbol][-1]['timestamp'] < current_minute):
            
            # Create new bar
            new_bar = {
                'timestamp': current_minute,
                'open': tick_data.get('price', 0),
                'high': tick_data.get('price', 0),
                'low': tick_data.get('price', 0),
                'close': tick_data.get('price', 0),
                'volume': tick_data.get('volume', 0)
            }
            self.minute_bars[symbol].append(new_bar)
        else:
            # Update current bar
            bar = self.minute_bars[symbol][-1]
            price = tick_data.get('price', bar['close'])
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += tick_data.get('volume', 0)
    
    def get_current_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest data for symbol"""
        return self.price_cache.get(symbol)
    
    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get simulated Level 2 order book"""
        current_data = self.get_current_data(symbol)
        if current_data:
            return self.order_book_simulator.generate_book(
                current_data['bid'],
                current_data['ask'],
                current_data.get('volume', 100000)
            )
        return {'bids': [], 'asks': []}
    
    def stop_feeds(self):
        """Stop all feeds"""
        self.running = False
        for source in self.sources.values():
            source.disconnect()


class YahooRealtime:
    """Yahoo Finance real-time data (1-min delay)"""
    
    def connect(self, symbols: List[str], callback: Callable):
        self.symbols = symbols
        self.callback = callback
        self.running = True
        
        # Start polling thread
        self.thread = threading.Thread(target=self._poll_yahoo)
        self.thread.daemon = True
        self.thread.start()
    
    def _poll_yahoo(self):
        """Poll Yahoo Finance for updates"""
        while self.running:
            try:
                for symbol in self.symbols:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if 'regularMarketPrice' in info:
                        data = {
                            'symbol': symbol,
                            'price': info['regularMarketPrice'],
                            'bid': info.get('bid', info['regularMarketPrice']),
                            'ask': info.get('ask', info['regularMarketPrice']),
                            'volume': info.get('volume', 0),
                            'dayHigh': info.get('dayHigh'),
                            'dayLow': info.get('dayLow')
                        }
                        self.callback('yahoo', data)
                
                time.sleep(5)  # Poll every 5 seconds
                
            except Exception as e:
                print(f"Yahoo polling error: {e}")
                time.sleep(10)
    
    def disconnect(self):
        self.running = False


class FinnhubWebsocket:
    """Finnhub free websocket (real-time)"""
    
    def __init__(self):
        # Free API key - get your own at finnhub.io
        self.api_key = "YOUR_FREE_FINNHUB_KEY"  # Register for free
        self.ws = None
        
    def connect(self, symbols: List[str], callback: Callable):
        self.callback = callback
        
        # Skip if no API key
        if self.api_key == "YOUR_FREE_FINNHUB_KEY":
            return
        
        def on_message(ws, message):
            data = json.loads(message)
            if data['type'] == 'trade':
                for trade in data['data']:
                    self.callback('finnhub', {
                        'symbol': trade['s'],
                        'price': trade['p'],
                        'volume': trade['v'],
                        'timestamp': trade['t'] / 1000
                    })
        
        def on_open(ws):
            # Subscribe to symbols
            for symbol in symbols:
                ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
        
        url = f"wss://ws.finnhub.io?token={self.api_key}"
        self.ws = websocket.WebSocketApp(url,
                                        on_open=on_open,
                                        on_message=on_message)
        
        # Run in thread
        thread = threading.Thread(target=self.ws.run_forever)
        thread.daemon = True
        thread.start()
    
    def disconnect(self):
        if self.ws:
            self.ws.close()


class AlpacaPaperData:
    """Alpaca paper trading data (free)"""
    
    def __init__(self):
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        # Free paper trading API keys - register at alpaca.markets
        self.api_key = "YOUR_ALPACA_PAPER_KEY"
        self.secret_key = "YOUR_ALPACA_PAPER_SECRET"
        
    def connect(self, symbols: List[str], callback: Callable):
        self.symbols = symbols
        self.callback = callback
        
        # Skip if no API key
        if self.api_key == "YOUR_ALPACA_PAPER_KEY":
            return
            
        # Start polling thread
        self.running = True
        self.thread = threading.Thread(target=self._poll_alpaca)
        self.thread.daemon = True
        self.thread.start()
    
    def _poll_alpaca(self):
        """Poll Alpaca for latest quotes"""
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        
        while self.running:
            try:
                # Get latest quotes
                symbols_str = ','.join(self.symbols)
                url = f"{self.data_url}/v2/stocks/quotes/latest?symbols={symbols_str}"
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    
                    for symbol, quote in data.get('quotes', {}).items():
                        self.callback('alpaca', {
                            'symbol': symbol,
                            'bid': quote['bp'],
                            'ask': quote['ap'],
                            'bid_size': quote['bs'],
                            'ask_size': quote['as'],
                            'price': (quote['bp'] + quote['ap']) / 2
                        })
                
                time.sleep(1)  # Poll every second
                
            except Exception as e:
                print(f"Alpaca polling error: {e}")
                time.sleep(5)
    
    def disconnect(self):
        self.running = False


class IEXCloudFree:
    """IEX Cloud free tier data"""
    
    def __init__(self):
        # Free tier - register at iexcloud.io
        self.api_key = "YOUR_IEX_FREE_KEY"
        self.base_url = "https://cloud.iexapis.com/stable"
        
    def connect(self, symbols: List[str], callback: Callable):
        self.symbols = symbols
        self.callback = callback
        self.running = True
        
        # Skip if no API key
        if self.api_key == "YOUR_IEX_FREE_KEY":
            return
        
        # Start polling thread
        self.thread = threading.Thread(target=self._poll_iex)
        self.thread.daemon = True
        self.thread.start()
    
    def _poll_iex(self):
        """Poll IEX for quotes"""
        while self.running:
            try:
                for symbol in self.symbols:
                    url = f"{self.base_url}/stock/{symbol}/quote?token={self.api_key}"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        self.callback('iex', {
                            'symbol': symbol,
                            'price': data['latestPrice'],
                            'bid': data.get('iexBidPrice', data['latestPrice']),
                            'ask': data.get('iexAskPrice', data['latestPrice']),
                            'volume': data['volume']
                        })
                
                time.sleep(2)  # Respect rate limits
                
            except Exception as e:
                print(f"IEX polling error: {e}")
                time.sleep(10)
    
    def disconnect(self):
        self.running = False


class OrderBookSimulator:
    """Simulate Level 2 order book from Level 1 data"""
    
    def generate_book(self, bid: float, ask: float, volume: int, 
                     levels: int = 10) -> Dict[str, List[Tuple[float, int]]]:
        """Generate realistic order book"""
        
        # Calculate spread
        spread = ask - bid
        tick_size = 0.01
        
        # Generate bid side
        bids = []
        current_bid = bid
        remaining_volume = volume * 0.5  # Half volume on each side
        
        for i in range(levels):
            # Price levels get wider as we go deeper
            if i > 0:
                current_bid -= tick_size * (1 + i * 0.1)
            
            # Size decreases but with some randomness
            level_size = int(remaining_volume * np.random.uniform(0.15, 0.35))
            level_size = max(100, level_size)  # Minimum 100 shares
            
            bids.append((round(current_bid, 2), level_size))
            remaining_volume -= level_size
        
        # Generate ask side
        asks = []
        current_ask = ask
        remaining_volume = volume * 0.5
        
        for i in range(levels):
            if i > 0:
                current_ask += tick_size * (1 + i * 0.1)
            
            level_size = int(remaining_volume * np.random.uniform(0.15, 0.35))
            level_size = max(100, level_size)
            
            asks.append((round(current_ask, 2), level_size))
            remaining_volume -= level_size
        
        return {'bids': bids, 'asks': asks}


# Example usage function
def setup_free_realtime_feeds():
    """Setup and return configured real-time feed system"""
    
    feed_system = FreeRealtimeDataFeed()
    
    print("\n=== FREE REAL-TIME DATA FEEDS ===")
    print("Available sources:")
    print("1. Yahoo Finance - 1-min delayed quotes (no API key needed)")
    print("2. Finnhub - Real-time websocket (free tier with registration)")
    print("3. Alpaca - Paper trading data (free with registration)")
    print("4. IEX Cloud - Real-time quotes (free tier with registration)")
    print("\nRegister for free API keys at:")
    print("- Finnhub: https://finnhub.io")
    print("- Alpaca: https://alpaca.markets")
    print("- IEX: https://iexcloud.io")
    
    return feed_system