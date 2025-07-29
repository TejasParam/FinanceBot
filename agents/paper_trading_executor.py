"""
Paper Trading Execution System
Simulates real execution without spending money
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

class PaperTradingExecutor:
    """
    Realistic paper trading execution system
    Simulates market microstructure effects
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}  # symbol -> Position
        self.orders = {}  # order_id -> Order
        self.order_history = []
        self.trade_history = []
        
        # Realistic execution parameters
        self.slippage_bps = 2  # 2 basis points average slippage
        self.market_impact_coef = 0.1  # 10% of square root of size
        self.commission_per_share = 0.005
        self.min_commission = 1.0
        
        # Order ID counter
        self.order_counter = 0
        
        # Performance tracking
        self.daily_pnl = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # Load saved state if exists
        self.state_file = "paper_trading_state.json"
        self.load_state()
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution"""
        
        # Generate order ID
        self.order_counter += 1
        order.order_id = f"ORD{self.order_counter:06d}"
        order.timestamp = datetime.now()
        order.status = OrderStatus.PENDING
        
        # Validate order
        validation = self._validate_order(order)
        if not validation['valid']:
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return f"Order rejected: {validation['reason']}"
        
        # Store order
        self.orders[order.order_id] = order
        
        # Execute immediately if market order
        if order.order_type == "MARKET":
            self._execute_order(order)
        
        self.save_state()
        return order.order_id
    
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order before submission"""
        
        # Check buying power
        if order.side == "BUY":
            # Get current price (simulate from last known or limit price)
            est_price = order.limit_price or self._get_market_price(order.symbol)
            required_capital = order.quantity * est_price * 1.02  # Add buffer
            
            if required_capital > self.cash:
                return {'valid': False, 'reason': 'Insufficient buying power'}
        
        # Check position for sells
        elif order.side == "SELL":
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                return {'valid': False, 'reason': 'Insufficient position'}
        
        # Check order parameters
        if order.quantity <= 0:
            return {'valid': False, 'reason': 'Invalid quantity'}
        
        if order.order_type == "LIMIT" and not order.limit_price:
            return {'valid': False, 'reason': 'Limit price required for limit orders'}
        
        return {'valid': True}
    
    def _execute_order(self, order: Order):
        """Execute order with realistic fills"""
        
        # Get market price
        market_price = self._get_market_price(order.symbol)
        
        # Calculate slippage and impact
        slippage = self._calculate_slippage(order, market_price)
        market_impact = self._calculate_market_impact(order)
        
        # Determine fill price
        if order.side == "BUY":
            fill_price = market_price * (1 + slippage + market_impact)
        else:
            fill_price = market_price * (1 - slippage - market_impact)
        
        # Check limit price
        if order.order_type == "LIMIT":
            if order.side == "BUY" and fill_price > order.limit_price:
                return  # No fill
            elif order.side == "SELL" and fill_price < order.limit_price:
                return  # No fill
        
        # Simulate partial fills for large orders
        fill_quantity = order.quantity
        if order.quantity > 1000:
            # Chance of partial fill
            if np.random.random() < 0.3:
                fill_quantity = int(order.quantity * np.random.uniform(0.5, 0.9))
                order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.FILLED
        
        # Update order
        order.filled_quantity = fill_quantity
        order.average_fill_price = fill_price
        
        # Calculate commission
        commission = max(self.min_commission, fill_quantity * self.commission_per_share)
        
        # Update positions and cash
        self._update_position(order, fill_quantity, fill_price, commission)
        
        # Record trade
        trade = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': datetime.now(),
            'slippage_bps': slippage * 10000,
            'impact_bps': market_impact * 10000
        }
        self.trade_history.append(trade)
        self.metrics['total_trades'] += 1
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order.order_id]
    
    def _calculate_slippage(self, order: Order, market_price: float) -> float:
        """Calculate realistic slippage"""
        
        # Base slippage
        base_slippage = self.slippage_bps / 10000
        
        # Add randomness
        random_factor = np.random.normal(1, 0.3)
        
        # Size adjustment (larger orders have more slippage)
        size_factor = np.log10(order.quantity) / 3
        
        return base_slippage * random_factor * (1 + size_factor)
    
    def _calculate_market_impact(self, order: Order) -> float:
        """Calculate market impact based on order size"""
        
        # Estimate average daily volume (would use real data)
        adv_estimate = 1000000
        
        # Size as percentage of ADV
        size_pct = order.quantity / adv_estimate
        
        # Square root market impact model
        impact = self.market_impact_coef * np.sqrt(size_pct)
        
        return min(impact, 0.01)  # Cap at 1%
    
    def _update_position(self, order: Order, quantity: int, price: float, commission: float):
        """Update positions and cash after fill"""
        
        if order.side == "BUY":
            # Deduct cash
            total_cost = quantity * price + commission
            self.cash -= total_cost
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                # Average cost calculation
                total_quantity = pos.quantity + quantity
                pos.average_cost = ((pos.quantity * pos.average_cost) + (quantity * price)) / total_quantity
                pos.quantity = total_quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=quantity,
                    average_cost=price,
                    current_price=price,
                    unrealized_pnl=0.0
                )
        
        else:  # SELL
            # Add cash
            total_proceeds = quantity * price - commission
            self.cash += total_proceeds
            
            # Update position
            pos = self.positions[order.symbol]
            
            # Calculate realized P&L
            realized_pnl = (price - pos.average_cost) * quantity - commission
            pos.realized_pnl += realized_pnl
            self.metrics['total_pnl'] += realized_pnl
            
            if realized_pnl > 0:
                self.metrics['winning_trades'] += 1
            
            # Update quantity
            pos.quantity -= quantity
            
            # Remove position if fully closed
            if pos.quantity == 0:
                del self.positions[order.symbol]
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price (simulated)"""
        
        # In production, this would fetch real quotes
        # For now, simulate with some randomness
        base_prices = {
            'AAPL': 180,
            'MSFT': 380,
            'GOOGL': 140,
            'TSLA': 250,
            'SPY': 450
        }
        
        base = base_prices.get(symbol, 100)
        # Add small random movement
        return base * (1 + np.random.normal(0, 0.001))
    
    def update_market_prices(self, price_updates: Dict[str, float]):
        """Update positions with new market prices"""
        
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.average_cost) * pos.quantity
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        
        positions_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        
        portfolio_value = self.get_portfolio_value()
        total_pnl = portfolio_value - self.initial_capital
        total_return = total_pnl / self.initial_capital
        
        # Calculate win rate
        win_rate = self.metrics['winning_trades'] / max(self.metrics['total_trades'], 1)
        
        # Calculate drawdown
        if hasattr(self, 'portfolio_history'):
            peak = max(self.portfolio_history)
            drawdown = (peak - portfolio_value) / peak
        else:
            drawdown = 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': self.metrics['total_trades'],
            'win_rate': win_rate,
            'max_drawdown': max(self.metrics['max_drawdown'], drawdown),
            'positions': len(self.positions),
            'buying_power': self.cash
        }
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_open_orders(self) -> List[Order]:
        """Get open orders"""
        return list(self.orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.CANCELLED
            self.order_history.append(order)
            del self.orders[order_id]
            self.save_state()
            return True
        return False
    
    def save_state(self):
        """Save current state to file"""
        
        state = {
            'cash': self.cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'average_cost': pos.average_cost,
                    'current_price': pos.current_price,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.positions.items()
            },
            'metrics': self.metrics,
            'order_counter': self.order_counter,
            'trade_count': len(self.trade_history)
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load saved state from file"""
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.cash = state.get('cash', self.initial_capital)
                self.metrics = state.get('metrics', self.metrics)
                self.order_counter = state.get('order_counter', 0)
                
                # Restore positions
                for symbol, pos_data in state.get('positions', {}).items():
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=pos_data['quantity'],
                        average_cost=pos_data['average_cost'],
                        current_price=pos_data['current_price'],
                        unrealized_pnl=0,
                        realized_pnl=pos_data.get('realized_pnl', 0)
                    )
                
                print(f"Loaded paper trading state: ${self.get_portfolio_value():,.2f}")
                
            except Exception as e:
                print(f"Error loading state: {e}")
    
    def reset(self):
        """Reset paper trading account"""
        
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.order_history = []
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        self.save_state()
        print("Paper trading account reset")


# Example usage
def create_paper_trader():
    """Create and return a paper trading executor"""
    
    executor = PaperTradingExecutor(initial_capital=100000)
    
    print("\n=== PAPER TRADING EXECUTION SYSTEM ===")
    print(f"Initial Capital: ${executor.initial_capital:,.2f}")
    print(f"Current Value: ${executor.get_portfolio_value():,.2f}")
    print("Features:")
    print("- Realistic slippage simulation")
    print("- Market impact modeling")
    print("- Partial fill simulation")
    print("- Commission tracking")
    print("- State persistence")
    print("- No real money required!")
    
    return executor