"""
Alpaca Paper Trading Integration Module
Handles all interactions with Alpaca API for automated trading
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.models import Position, Order, Account
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AlpacaTradingBot:
    """Main class for Alpaca paper trading integration"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca trading bot
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (True) or live trading (False)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        
        # Initialize clients
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Trading parameters
        self.max_position_size = 0.10  # Max 10% of portfolio per position
        self.min_position_size = 0.02  # Min 2% of portfolio per position
        self.max_positions = 20  # Maximum number of positions
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        
        # Initialize account info
        self.account = None
        self.positions = {}
        self.orders = []
        
    def get_account_info(self) -> Dict:
        """Get current account information"""
        try:
            self.account = self.trading_client.get_account()
            return {
                'buying_power': float(self.account.buying_power),
                'portfolio_value': float(self.account.portfolio_value),
                'cash': float(self.account.cash),
                'equity': float(self.account.equity),
                'pattern_day_trader': self.account.pattern_day_trader,
                'trading_blocked': self.account.trading_blocked,
                'account_blocked': self.account.account_blocked
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            self.positions = {pos.symbol: pos for pos in positions}
            return self.positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_position_value(self, symbol: str) -> float:
        """Get current value of a position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            return float(pos.market_value)
        return 0.0
    
    def calculate_position_size(self, symbol: str, confidence: float, 
                              volatility: float, portfolio_value: float) -> int:
        """
        Calculate optimal position size based on Kelly Criterion and risk management
        
        Args:
            symbol: Stock symbol
            confidence: ML model confidence (0-1)
            volatility: Stock volatility
            portfolio_value: Total portfolio value
            
        Returns:
            Number of shares to buy
        """
        # Base allocation using confidence
        base_allocation = confidence * self.max_position_size
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = 1 / (1 + volatility * 10)
        adjusted_allocation = base_allocation * volatility_factor
        
        # Ensure within bounds
        final_allocation = max(self.min_position_size, 
                             min(adjusted_allocation, self.max_position_size))
        
        # Calculate dollar amount
        position_value = portfolio_value * final_allocation
        
        # Get current price
        try:
            quote = self.get_latest_quote(symbol)
            if quote:
                current_price = float(quote['ask_price'])
                shares = int(position_value / current_price)
                return max(1, shares)  # At least 1 share
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            
        return 0
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for a symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]
            return {
                'ask_price': float(quote.ask_price),
                'bid_price': float(quote.bid_price),
                'ask_size': quote.ask_size,
                'bid_size': quote.bid_size,
                'timestamp': quote.timestamp
            }
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=days)
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def place_buy_order(self, symbol: str, quantity: int, 
                       order_type: str = 'market', 
                       limit_price: Optional[float] = None,
                       reason: str = "") -> Optional[Order]:
        """
        Place a buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            reason: Reason for the trade (for logging)
            
        Returns:
            Order object if successful
        """
        try:
            if order_type == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
            else:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            
            order = self.trading_client.submit_order(order_request)
            
            # Log the trade
            self.log_trade(symbol, 'BUY', quantity, order.id, reason)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing buy order for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol: str, quantity: int, 
                        order_type: str = 'market',
                        limit_price: Optional[float] = None,
                        reason: str = "") -> Optional[Order]:
        """Place a sell order"""
        try:
            if order_type == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
            else:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            
            order = self.trading_client.submit_order(order_request)
            
            # Log the trade
            self.log_trade(symbol, 'SELL', quantity, order.id, reason)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing sell order for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, reason: str = "") -> bool:
        """Close an entire position"""
        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                qty = int(position.qty)
                
                # Place sell order
                order = self.place_sell_order(symbol, qty, reason=reason)
                return order is not None
            else:
                self.logger.warning(f"No position found for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        try:
            request = GetOrdersRequest(status=OrderStatus.OPEN)
            orders = self.trading_client.get_orders(request)
            return orders
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.trading_client.cancel_orders()
            return True
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return False
    
    def check_stop_loss(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if position should be stopped out
        
        Returns:
            (should_stop, current_pnl_pct)
        """
        if symbol not in self.positions:
            return False, 0.0
            
        position = self.positions[symbol]
        avg_cost = float(position.avg_entry_price)
        current_price = float(position.current_price)
        
        pnl_pct = (current_price - avg_cost) / avg_cost
        
        if pnl_pct <= -self.stop_loss_pct:
            return True, pnl_pct
            
        return False, pnl_pct
    
    def check_take_profit(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if position should take profits
        
        Returns:
            (should_take_profit, current_pnl_pct)
        """
        if symbol not in self.positions:
            return False, 0.0
            
        position = self.positions[symbol]
        avg_cost = float(position.avg_entry_price)
        current_price = float(position.current_price)
        
        pnl_pct = (current_price - avg_cost) / avg_cost
        
        if pnl_pct >= self.take_profit_pct:
            return True, pnl_pct
            
        return False, pnl_pct
    
    def log_trade(self, symbol: str, action: str, quantity: int, 
                  order_id: str, reason: str):
        """Log trade details to file"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_id': order_id,
            'reason': reason,
            'account_value': float(self.account.portfolio_value) if self.account else 0
        }
        
        # Append to trades log file
        log_file = 'trades_log.json'
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            trades.append(trade_log)
            
            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            account_info = self.get_account_info()
            positions = self.get_positions()
            
            # Calculate metrics
            total_value = account_info['portfolio_value']
            cash = account_info['cash']
            positions_value = total_value - cash
            
            # Position concentration
            position_values = [float(pos.market_value) for pos in positions.values()]
            max_position = max(position_values) if position_values else 0
            concentration = max_position / total_value if total_value > 0 else 0
            
            # Calculate total P&L
            total_pnl = sum(float(pos.unrealized_pl) for pos in positions.values())
            total_pnl_pct = total_pnl / total_value if total_value > 0 else 0
            
            return {
                'total_value': total_value,
                'cash': cash,
                'positions_value': positions_value,
                'num_positions': len(positions),
                'cash_percentage': cash / total_value if total_value > 0 else 1,
                'max_position_concentration': concentration,
                'total_unrealized_pnl': total_pnl,
                'total_unrealized_pnl_pct': total_pnl_pct,
                'positions': {
                    symbol: {
                        'quantity': int(pos.qty),
                        'avg_cost': float(pos.avg_entry_price),
                        'current_price': float(pos.current_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'unrealized_pnl_pct': float(pos.unrealized_plpc) / 100
                    }
                    for symbol, pos in positions.items()
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}