import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
import yfinance as yf
from collections import deque
import threading
import time
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataProcessor:
    """
    Real-time data processing with WebSocket support and streaming capabilities
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffers = {}  # Store real-time data for each ticker
        self.subscribers = {}  # Callbacks for data updates
        self.is_running = False
        self.update_queue = queue.Queue()
        self.websocket_connections = {}
        
    def subscribe(self, ticker: str, callback: Callable):
        """Subscribe to real-time updates for a ticker"""
        if ticker not in self.subscribers:
            self.subscribers[ticker] = []
        self.subscribers[ticker].append(callback)
        
        # Initialize buffer for this ticker
        if ticker not in self.data_buffers:
            self.data_buffers[ticker] = {
                'price': deque(maxlen=self.buffer_size),
                'volume': deque(maxlen=self.buffer_size),
                'timestamp': deque(maxlen=self.buffer_size),
                'bid': deque(maxlen=self.buffer_size),
                'ask': deque(maxlen=self.buffer_size)
            }
    
    def unsubscribe(self, ticker: str, callback: Callable):
        """Unsubscribe from real-time updates"""
        if ticker in self.subscribers and callback in self.subscribers[ticker]:
            self.subscribers[ticker].remove(callback)
    
    async def connect_websocket(self, url: str, ticker: str):
        """Connect to a WebSocket data feed"""
        try:
            async with websockets.connect(url) as websocket:
                self.websocket_connections[ticker] = websocket
                logger.info(f"Connected to WebSocket for {ticker}")
                
                while self.is_running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        self._process_websocket_data(ticker, data)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(f"WebSocket connection closed for {ticker}")
                        break
                    except Exception as e:
                        logger.error(f"Error processing WebSocket data: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
    
    def _process_websocket_data(self, ticker: str, data: Dict[str, Any]):
        """Process incoming WebSocket data"""
        try:
            # Extract relevant fields (adjust based on your data source)
            timestamp = datetime.now()
            price = data.get('price', data.get('last', None))
            volume = data.get('volume', 0)
            bid = data.get('bid', price)
            ask = data.get('ask', price)
            
            if price is not None:
                # Update buffers
                self.data_buffers[ticker]['price'].append(price)
                self.data_buffers[ticker]['volume'].append(volume)
                self.data_buffers[ticker]['timestamp'].append(timestamp)
                self.data_buffers[ticker]['bid'].append(bid)
                self.data_buffers[ticker]['ask'].append(ask)
                
                # Create update event
                update = {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'bid': bid,
                    'ask': ask,
                    'spread': ask - bid if ask and bid else 0
                }
                
                # Notify subscribers
                self._notify_subscribers(ticker, update)
                
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {e}")
    
    def _notify_subscribers(self, ticker: str, update: Dict[str, Any]):
        """Notify all subscribers of a data update"""
        if ticker in self.subscribers:
            for callback in self.subscribers[ticker]:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    def simulate_realtime_data(self, tickers: List[str], update_interval: float = 1.0):
        """
        Simulate real-time data updates using Yahoo Finance
        This is a fallback when WebSocket feeds are not available
        """
        def data_generator():
            while self.is_running:
                for ticker in tickers:
                    try:
                        # Get current data from Yahoo Finance
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        # Get latest price
                        current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
                        volume = info.get('regularMarketVolume', 0)
                        
                        # Simulate bid/ask spread
                        spread = current_price * 0.0001  # 0.01% spread
                        bid = current_price - spread / 2
                        ask = current_price + spread / 2
                        
                        # Add some random walk to simulate real-time changes
                        price_change = np.random.normal(0, current_price * 0.0001)
                        current_price += price_change
                        bid += price_change
                        ask += price_change
                        
                        # Process the simulated data
                        self._process_websocket_data(ticker, {
                            'price': current_price,
                            'volume': volume,
                            'bid': bid,
                            'ask': ask
                        })
                        
                    except Exception as e:
                        logger.error(f"Error simulating data for {ticker}: {e}")
                
                time.sleep(update_interval)
        
        # Start in a separate thread
        thread = threading.Thread(target=data_generator)
        thread.daemon = True
        thread.start()
    
    def get_current_data(self, ticker: str) -> Dict[str, Any]:
        """Get the most recent data for a ticker"""
        if ticker not in self.data_buffers or not self.data_buffers[ticker]['price']:
            return None
        
        buffer = self.data_buffers[ticker]
        return {
            'ticker': ticker,
            'price': buffer['price'][-1],
            'volume': buffer['volume'][-1],
            'timestamp': buffer['timestamp'][-1],
            'bid': buffer['bid'][-1],
            'ask': buffer['ask'][-1],
            'spread': buffer['ask'][-1] - buffer['bid'][-1]
        }
    
    def get_buffer_data(self, ticker: str, n_points: Optional[int] = None) -> pd.DataFrame:
        """Get buffered data as a DataFrame"""
        if ticker not in self.data_buffers:
            return pd.DataFrame()
        
        buffer = self.data_buffers[ticker]
        
        # Determine how many points to return
        if n_points is None:
            n_points = len(buffer['price'])
        else:
            n_points = min(n_points, len(buffer['price']))
        
        if n_points == 0:
            return pd.DataFrame()
        
        # Create DataFrame from buffer
        data = pd.DataFrame({
            'timestamp': list(buffer['timestamp'])[-n_points:],
            'price': list(buffer['price'])[-n_points:],
            'volume': list(buffer['volume'])[-n_points:],
            'bid': list(buffer['bid'])[-n_points:],
            'ask': list(buffer['ask'])[-n_points:]
        })
        
        data['spread'] = data['ask'] - data['bid']
        data['mid_price'] = (data['bid'] + data['ask']) / 2
        
        return data
    
    def calculate_realtime_metrics(self, ticker: str, window: int = 60) -> Dict[str, float]:
        """Calculate real-time metrics from buffered data"""
        data = self.get_buffer_data(ticker, window)
        
        if data.empty or len(data) < 2:
            return {}
        
        # Calculate metrics
        price_series = data['price']
        returns = price_series.pct_change().dropna()
        
        metrics = {
            'current_price': price_series.iloc[-1],
            'price_change': price_series.iloc[-1] - price_series.iloc[0],
            'price_change_pct': (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100,
            'volatility': returns.std() * np.sqrt(252 * 6.5 * 60),  # Annualized
            'avg_spread': data['spread'].mean(),
            'avg_volume': data['volume'].mean(),
            'vwap': (data['price'] * data['volume']).sum() / data['volume'].sum() if data['volume'].sum() > 0 else price_series.mean(),
            'high': price_series.max(),
            'low': price_series.min(),
            'momentum': returns.mean() * 100
        }
        
        return metrics
    
    def start(self):
        """Start the real-time data processor"""
        self.is_running = True
        logger.info("Real-time data processor started")
    
    def stop(self):
        """Stop the real-time data processor"""
        self.is_running = False
        
        # Close WebSocket connections
        for ticker, ws in self.websocket_connections.items():
            asyncio.create_task(ws.close())
        
        logger.info("Real-time data processor stopped")


class StreamingMLPredictor:
    """
    Streaming ML predictions on real-time data
    """
    
    def __init__(self, model=None):
        self.model = model
        self.prediction_buffer = deque(maxlen=100)
        self.feature_buffer = {}
        
    def update_features(self, ticker: str, data: Dict[str, Any]):
        """Update feature buffer with new data"""
        if ticker not in self.feature_buffer:
            self.feature_buffer[ticker] = deque(maxlen=100)
        
        # Extract features from real-time data
        features = {
            'price': data['price'],
            'volume': data['volume'],
            'spread': data['spread'],
            'timestamp': data['timestamp']
        }
        
        self.feature_buffer[ticker].append(features)
    
    def generate_streaming_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate prediction based on streaming data"""
        if ticker not in self.feature_buffer or len(self.feature_buffer[ticker]) < 10:
            return None
        
        try:
            # Convert buffer to features
            buffer_data = list(self.feature_buffer[ticker])
            
            # Calculate technical features
            prices = [d['price'] for d in buffer_data]
            volumes = [d['volume'] for d in buffer_data]
            
            # Simple features for demonstration
            features = {
                'price_mean': np.mean(prices[-10:]),
                'price_std': np.std(prices[-10:]),
                'volume_mean': np.mean(volumes[-10:]),
                'price_momentum': (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0,
                'volume_ratio': volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1
            }
            
            # Make prediction (simplified for demonstration)
            # In practice, you would use your trained model here
            price_change_prediction = np.random.normal(0, 0.001)  # Placeholder
            confidence = np.random.uniform(0.5, 0.9)  # Placeholder
            
            prediction = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'current_price': prices[-1],
                'predicted_change': price_change_prediction,
                'predicted_price': prices[-1] * (1 + price_change_prediction),
                'confidence': confidence,
                'features': features
            }
            
            self.prediction_buffer.append(prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy of recent predictions"""
        if len(self.prediction_buffer) < 2:
            return {}
        
        # Simple accuracy calculation
        # In practice, you would compare predictions with actual outcomes
        return {
            'predictions_made': len(self.prediction_buffer),
            'avg_confidence': np.mean([p['confidence'] for p in self.prediction_buffer]),
            'last_update': self.prediction_buffer[-1]['timestamp']
        }


class RealTimeAlertSystem:
    """
    Real-time alert system for price movements and patterns
    """
    
    def __init__(self):
        self.alerts = {}
        self.alert_history = deque(maxlen=1000)
        
    def add_price_alert(self, ticker: str, condition: str, threshold: float, callback: Callable):
        """Add a price-based alert"""
        if ticker not in self.alerts:
            self.alerts[ticker] = []
        
        alert = {
            'type': 'price',
            'condition': condition,  # 'above', 'below', 'change_pct'
            'threshold': threshold,
            'callback': callback,
            'triggered': False,
            'created_at': datetime.now()
        }
        
        self.alerts[ticker].append(alert)
    
    def check_alerts(self, update: Dict[str, Any]):
        """Check if any alerts should be triggered"""
        ticker = update['ticker']
        
        if ticker not in self.alerts:
            return
        
        for alert in self.alerts[ticker]:
            if alert['triggered']:
                continue
            
            triggered = False
            
            if alert['type'] == 'price':
                price = update['price']
                
                if alert['condition'] == 'above' and price > alert['threshold']:
                    triggered = True
                elif alert['condition'] == 'below' and price < alert['threshold']:
                    triggered = True
                elif alert['condition'] == 'change_pct':
                    # Need historical price for percentage change
                    # This is a simplified implementation
                    pass
            
            if triggered:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now()
                alert['trigger_price'] = update['price']
                
                # Execute callback
                try:
                    alert['callback'](ticker, alert, update)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                
                # Add to history
                self.alert_history.append({
                    'ticker': ticker,
                    'alert': alert,
                    'update': update,
                    'timestamp': datetime.now()
                })


# Example usage
if __name__ == "__main__":
    # Initialize components
    processor = RealTimeDataProcessor()
    ml_predictor = StreamingMLPredictor()
    alert_system = RealTimeAlertSystem()
    
    # Define callbacks
    def on_data_update(update):
        """Handle real-time data updates"""
        print(f"Update for {update['ticker']}: ${update['price']:.2f}")
        
        # Update ML features
        ml_predictor.update_features(update['ticker'], update)
        
        # Generate prediction
        prediction = ml_predictor.generate_streaming_prediction(update['ticker'])
        if prediction:
            print(f"Prediction: ${prediction['predicted_price']:.2f} (confidence: {prediction['confidence']:.2%})")
        
        # Check alerts
        alert_system.check_alerts(update)
    
    def on_price_alert(ticker, alert, update):
        """Handle price alerts"""
        print(f"ALERT: {ticker} price {alert['condition']} {alert['threshold']:.2f} at ${update['price']:.2f}")
    
    # Subscribe to updates
    tickers = ["AAPL", "MSFT", "GOOGL"]
    for ticker in tickers:
        processor.subscribe(ticker, on_data_update)
        
        # Add some alerts
        current_price = yf.Ticker(ticker).info.get('regularMarketPrice', 100)
        alert_system.add_price_alert(ticker, 'above', current_price * 1.01, on_price_alert)
        alert_system.add_price_alert(ticker, 'below', current_price * 0.99, on_price_alert)
    
    # Start processor
    processor.start()
    
    # Simulate real-time data (in practice, you would connect to real WebSocket feeds)
    processor.simulate_realtime_data(tickers, update_interval=5.0)
    
    # Run for demonstration
    try:
        time.sleep(30)  # Run for 30 seconds
        
        # Print some metrics
        for ticker in tickers:
            metrics = processor.calculate_realtime_metrics(ticker)
            if metrics:
                print(f"\nMetrics for {ticker}:")
                print(f"  Current Price: ${metrics.get('current_price', 0):.2f}")
                print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
                print(f"  VWAP: ${metrics.get('vwap', 0):.2f}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()