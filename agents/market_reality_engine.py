"""
Market Reality Engine - Bridges the gap between simulation and real trading
Implements realistic market microstructure and adversarial dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
import time
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    ICEBERG = "ICEBERG"
    HIDDEN = "HIDDEN"

@dataclass
class Order:
    id: str
    side: str  # BUY/SELL
    price: float
    size: int
    type: OrderType
    timestamp: float
    hidden_size: int = 0  # For iceberg orders

@dataclass
class MarketImpact:
    permanent: float
    temporary: float
    total: float
    slippage: float

class MarketRealityEngine:
    """
    Realistic market simulator that accounts for:
    - Order book dynamics
    - Market impact
    - Adverse selection
    - Latency and queuing
    - Hidden liquidity
    - Predatory algorithms
    """
    
    def __init__(self):
        # Order book structure
        self.order_book = {
            'bids': {},  # price -> list of orders
            'asks': {}   # price -> list of orders
        }
        
        # Hidden order book (dark pools, hidden orders)
        self.hidden_book = {
            'bids': {},
            'asks': {}
        }
        
        # Market microstructure parameters
        self.tick_size = 0.01  # Minimum price increment
        self.lot_size = 100    # Minimum order size
        
        # Realistic market dynamics
        self.adverse_selection_prob = 0.3  # Probability of trading against informed flow
        self.hidden_liquidity_ratio = 0.25  # 25% of liquidity is hidden
        self.predatory_algo_prob = 0.15  # Chance of triggering predatory algos
        
        # Latency simulation (microseconds)
        self.latency_distribution = {
            'mean': 250,      # Average 250μs
            'std': 100,       # Standard deviation
            'min': 50,        # Minimum possible
            'spike_prob': 0.05,  # 5% chance of latency spike
            'spike_mult': 10   # Spike multiplier
        }
        
        # Market impact model (realistic parameters)
        self.impact_params = {
            'permanent_impact_coef': 0.1,    # λ in Almgren-Chriss
            'temporary_impact_coef': 0.01,   # η
            'nonlinearity': 0.6,             # Power law exponent
            'decay_rate': 0.5                # Impact decay
        }
        
        # Queue position tracking
        self.order_queues = defaultdict(deque)
        self.queue_priorities = {}
        
        # Adverse event tracking
        self.adverse_fills = 0
        self.total_fills = 0
        
        # Performance metrics
        self.metrics = {
            'avg_slippage_bps': deque(maxlen=1000),
            'fill_rates': deque(maxlen=1000),
            'adverse_selection_rate': 0,
            'queue_position_avg': deque(maxlen=1000)
        }
        
    def calculate_realistic_market_impact(self, order: Order, 
                                        current_price: float,
                                        market_state: Dict[str, Any]) -> MarketImpact:
        """Calculate realistic market impact including all microstructure effects"""
        
        size = order.size
        adv = market_state.get('adv', 1000000)  # Average daily volume
        volatility = market_state.get('volatility', 0.02)
        spread = market_state.get('spread', 0.0001)
        
        # Size as percentage of ADV
        size_pct = size / adv
        
        # 1. Permanent Impact (Information leakage)
        permanent = self.impact_params['permanent_impact_coef'] * \
                   (size_pct ** self.impact_params['nonlinearity']) * \
                   volatility * current_price
        
        # 2. Temporary Impact (Supply/demand imbalance)
        book_depth = self._get_book_depth(order.side, current_price)
        depth_ratio = size / max(book_depth, 1)
        
        temporary = self.impact_params['temporary_impact_coef'] * \
                   depth_ratio * spread * current_price
        
        # 3. Spread cost
        spread_cost = spread * current_price * 0.5
        
        # 4. Timing risk (adverse selection)
        if np.random.random() < self.adverse_selection_prob:
            # We're trading against informed flow
            adverse_impact = volatility * current_price * np.random.uniform(0.001, 0.005)
            self.adverse_fills += 1
        else:
            adverse_impact = 0
        
        self.total_fills += 1
        
        # 5. Predatory algorithm detection
        if size_pct > 0.001 and np.random.random() < self.predatory_algo_prob:
            # Predatory algos front-run our order
            predatory_impact = spread * current_price * np.random.uniform(1, 3)
        else:
            predatory_impact = 0
        
        # Total impact
        total_impact = permanent + temporary + spread_cost + adverse_impact + predatory_impact
        
        # Calculate slippage
        if order.side == 'BUY':
            execution_price = current_price * (1 + total_impact / current_price)
        else:
            execution_price = current_price * (1 - total_impact / current_price)
        
        slippage = abs(execution_price - current_price) / current_price
        
        # Track metrics
        self.metrics['avg_slippage_bps'].append(slippage * 10000)
        
        return MarketImpact(
            permanent=permanent,
            temporary=temporary,
            total=total_impact,
            slippage=slippage
        )
    
    def simulate_order_execution(self, order: Order, 
                               market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate realistic order execution with all market frictions"""
        
        current_price = market_state.get('price', 100)
        
        # 1. Latency simulation
        latency = self._simulate_latency()
        
        # 2. Price movement during latency
        price_drift = np.random.normal(0, market_state.get('volatility', 0.02) * 
                                     np.sqrt(latency / 1e6))  # Convert μs to seconds
        arrival_price = current_price * (1 + price_drift)
        
        # 3. Queue position for limit orders
        if order.type == OrderType.LIMIT:
            queue_position = self._get_queue_position(order, arrival_price)
            fill_probability = self._calculate_fill_probability(order, queue_position, market_state)
            
            if np.random.random() > fill_probability:
                return {
                    'status': 'UNFILLED',
                    'reason': 'Queue position',
                    'queue_position': queue_position,
                    'fill_probability': fill_probability
                }
        
        # 4. Calculate market impact
        impact = self.calculate_realistic_market_impact(order, arrival_price, market_state)
        
        # 5. Check for partial fills
        available_liquidity = self._get_available_liquidity(order.side, arrival_price)
        
        if available_liquidity < order.size:
            # Partial fill
            filled_size = available_liquidity
            unfilled_size = order.size - filled_size
            
            # Need to walk the book for remaining size
            walk_cost = self._calculate_book_walk_cost(unfilled_size, order.side, arrival_price)
            impact.total += walk_cost
        else:
            filled_size = order.size
            unfilled_size = 0
        
        # 6. Final execution price
        if order.side == 'BUY':
            execution_price = arrival_price * (1 + impact.total / arrival_price)
        else:
            execution_price = arrival_price * (1 - impact.total / arrival_price)
        
        # 7. Post-trade analysis
        post_trade_reversion = self._simulate_price_reversion(impact.temporary, market_state)
        
        return {
            'status': 'FILLED' if unfilled_size == 0 else 'PARTIAL',
            'filled_size': filled_size,
            'unfilled_size': unfilled_size,
            'arrival_price': arrival_price,
            'execution_price': execution_price,
            'latency_us': latency,
            'market_impact': impact,
            'slippage_bps': impact.slippage * 10000,
            'post_trade_reversion': post_trade_reversion,
            'adverse_selection': self.adverse_fills / max(self.total_fills, 1)
        }
    
    def _simulate_latency(self) -> float:
        """Simulate realistic execution latency"""
        
        if np.random.random() < self.latency_distribution['spike_prob']:
            # Latency spike (garbage collection, network congestion, etc.)
            base_latency = self.latency_distribution['mean'] * self.latency_distribution['spike_mult']
        else:
            # Normal latency
            base_latency = np.random.normal(
                self.latency_distribution['mean'],
                self.latency_distribution['std']
            )
        
        # Ensure minimum latency
        return max(base_latency, self.latency_distribution['min'])
    
    def _get_book_depth(self, side: str, price: float, levels: int = 5) -> float:
        """Get total size available in order book within levels"""
        
        book = self.order_book['asks' if side == 'BUY' else 'bids']
        hidden = self.hidden_book['asks' if side == 'BUY' else 'bids']
        
        total_size = 0
        
        # Visible liquidity
        prices = sorted(book.keys())
        if side == 'SELL':
            prices = reversed(prices)
        
        for i, book_price in enumerate(prices):
            if i >= levels:
                break
            
            if side == 'BUY' and book_price > price * 1.001:  # Within 10 bps
                break
            elif side == 'SELL' and book_price < price * 0.999:
                break
            
            total_size += sum(order.size for order in book[book_price])
        
        # Hidden liquidity estimate
        hidden_size = total_size * self.hidden_liquidity_ratio
        
        return total_size + hidden_size
    
    def _get_queue_position(self, order: Order, price: float) -> int:
        """Get position in queue for limit order"""
        
        queue_key = (order.side, price)
        
        if queue_key not in self.order_queues:
            return 1
        
        # Find position based on time priority
        position = 1
        for queued_order in self.order_queues[queue_key]:
            if queued_order.timestamp < order.timestamp:
                position += 1
        
        return position
    
    def _calculate_fill_probability(self, order: Order, queue_position: int,
                                  market_state: Dict[str, Any]) -> float:
        """Calculate probability of limit order fill"""
        
        # Base probability decreases with queue position
        base_prob = 1.0 / (1 + queue_position * 0.1)
        
        # Adjust for market conditions
        volatility = market_state.get('volatility', 0.02)
        trend = market_state.get('trend', 0)
        
        # Higher volatility increases fill probability
        vol_adjustment = min(1.5, 1 + volatility * 10)
        
        # Trend adjustment
        if order.side == 'BUY':
            trend_adjustment = 1 - trend * 0.5  # Harder to buy in uptrend
        else:
            trend_adjustment = 1 + trend * 0.5  # Harder to sell in downtrend
        
        fill_prob = base_prob * vol_adjustment * trend_adjustment
        
        return np.clip(fill_prob, 0.1, 0.95)
    
    def _get_available_liquidity(self, side: str, price: float) -> float:
        """Get immediately available liquidity at price level"""
        
        # Simulate realistic liquidity
        base_liquidity = np.random.lognormal(10, 1.5)  # Log-normal distribution
        
        # Adjust for hidden liquidity
        total_liquidity = base_liquidity * (1 + self.hidden_liquidity_ratio)
        
        return int(total_liquidity) * self.lot_size
    
    def _calculate_book_walk_cost(self, size: float, side: str, start_price: float) -> float:
        """Calculate cost of walking the order book for large orders"""
        
        remaining_size = size
        total_cost = 0
        current_level = 0
        
        while remaining_size > 0 and current_level < 10:
            # Liquidity decreases at each level
            level_liquidity = self._get_available_liquidity(side, start_price) * \
                            (0.7 ** current_level)
            
            # Price gets worse at each level
            if side == 'BUY':
                level_price = start_price * (1 + self.tick_size * current_level)
            else:
                level_price = start_price * (1 - self.tick_size * current_level)
            
            # Fill what we can at this level
            fill_size = min(remaining_size, level_liquidity)
            total_cost += fill_size * abs(level_price - start_price) / start_price
            
            remaining_size -= fill_size
            current_level += 1
        
        return total_cost
    
    def _simulate_price_reversion(self, temporary_impact: float,
                                market_state: Dict[str, Any]) -> float:
        """Simulate post-trade price reversion"""
        
        # Temporary impact decays over time
        decay_rate = self.impact_params['decay_rate']
        volatility = market_state.get('volatility', 0.02)
        
        # Add noise to reversion
        noise = np.random.normal(0, volatility * 0.1)
        
        # Typically 60-80% of temporary impact reverts
        reversion_pct = np.random.uniform(0.6, 0.8)
        
        return temporary_impact * reversion_pct * decay_rate + noise
    
    def get_execution_analysis(self) -> Dict[str, Any]:
        """Get comprehensive execution quality metrics"""
        
        if not self.metrics['avg_slippage_bps']:
            return {'status': 'No executions yet'}
        
        return {
            'avg_slippage_bps': np.mean(self.metrics['avg_slippage_bps']),
            'slippage_std': np.std(self.metrics['avg_slippage_bps']),
            'adverse_selection_rate': self.adverse_fills / max(self.total_fills, 1),
            'avg_queue_position': np.mean(self.metrics['queue_position_avg']) if self.metrics['queue_position_avg'] else 0,
            'total_executions': self.total_fills,
            'market_impact_breakdown': {
                'permanent': self.impact_params['permanent_impact_coef'],
                'temporary': self.impact_params['temporary_impact_coef'],
                'hidden_liquidity': self.hidden_liquidity_ratio
            }
        }