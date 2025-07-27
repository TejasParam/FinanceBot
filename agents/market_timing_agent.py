"""
Market Timing Agent - Specializes in identifying optimal entry/exit points
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import yfinance as yf

class MarketTimingAgent(BaseAgent):
    """
    Agent specialized in market timing and trend identification.
    Uses multiple timeframe analysis to identify high-probability setups.
    """
    
    def __init__(self):
        super().__init__("MarketTiming")
        self.data_collector = DataCollectionAgent()
        
    def analyze(self, ticker: str, period: str = "3mo", **kwargs) -> Dict[str, Any]:
        """
        Analyze market timing for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period for analysis
            
        Returns:
            Dictionary with timing analysis
        """
        try:
            # Get price data
            price_data = self.data_collector.fetch_stock_data(ticker, period=period)
            if price_data is None or len(price_data) < 50:
                return {
                    'error': 'Insufficient data for timing analysis',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data'
                }
            
            # Multi-timeframe analysis
            timing_score, confidence, signals = self._analyze_market_timing(price_data)
            
            # Generate reasoning
            reasoning = self._generate_timing_reasoning(signals, timing_score)
            
            return {
                'score': timing_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'timing_signals': signals,
                'market_phase': signals['market_phase'],
                'trend_strength': signals['trend_strength']
            }
            
        except Exception as e:
            return {
                'error': f'Market timing analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Timing analysis error: {str(e)[:100]}'
            }
    
    def _analyze_market_timing(self, data: pd.DataFrame) -> tuple:
        """Perform comprehensive market timing analysis - World-class implementation"""
        
        signals = {}
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
        open_price = data['Open']
        
        # 1. Trend Analysis
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        
        current_price = close.iloc[-1]
        
        # Trend alignment
        trend_score = 0
        sma10_val = float(sma10.iloc[-1])
        sma20_val = float(sma20.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        
        if current_price > sma10_val and sma10_val > sma20_val and sma20_val > sma50_val:
            trend_score = 1.0
            signals['trend'] = 'strong_uptrend'
        elif current_price > sma20_val and sma20_val > sma50_val:
            trend_score = 0.6
            signals['trend'] = 'uptrend'
        elif current_price < sma10_val and sma10_val < sma20_val and sma20_val < sma50_val:
            trend_score = -1.0
            signals['trend'] = 'strong_downtrend'
        elif current_price < sma20_val and sma20_val < sma50_val:
            trend_score = -0.6
            signals['trend'] = 'downtrend'
        else:
            trend_score = 0.0
            signals['trend'] = 'sideways'
        
        # 2. Momentum Analysis
        roc5 = (current_price / close.iloc[-6] - 1) if len(close) > 5 else 0
        roc10 = (current_price / close.iloc[-11] - 1) if len(close) > 10 else 0
        roc20 = (current_price / close.iloc[-21] - 1) if len(close) > 20 else 0
        
        momentum_score = 0
        if roc5 > 0.02 and roc10 > 0.04 and roc20 > 0.06:
            momentum_score = 0.8
            signals['momentum'] = 'strong_positive'
        elif roc5 > 0.01 and roc10 > 0.02:
            momentum_score = 0.4
            signals['momentum'] = 'positive'
        elif roc5 < -0.02 and roc10 < -0.04 and roc20 < -0.06:
            momentum_score = -0.8
            signals['momentum'] = 'strong_negative'
        elif roc5 < -0.01 and roc10 < -0.02:
            momentum_score = -0.4
            signals['momentum'] = 'negative'
        else:
            momentum_score = 0.0
            signals['momentum'] = 'neutral'
        
        # 3. Volume Analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        recent_volume = volume.iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        volume_score = 0
        if volume_ratio > 1.5 and trend_score > 0:
            volume_score = 0.3
            signals['volume'] = 'high_bullish'
        elif volume_ratio > 1.5 and trend_score < 0:
            volume_score = -0.3
            signals['volume'] = 'high_bearish'
        elif volume_ratio < 0.7:
            volume_score = -0.1
            signals['volume'] = 'low'
        else:
            signals['volume'] = 'normal'
        
        # 4. Support/Resistance Analysis
        high_20d = close.rolling(20).max().iloc[-1]
        low_20d = close.rolling(20).min().iloc[-1]
        price_position = (current_price - low_20d) / (high_20d - low_20d) if high_20d > low_20d else 0.5
        
        sr_score = 0
        if price_position > 0.8 and momentum_score > 0:
            sr_score = 0.2  # Near resistance but bullish
            signals['sr_level'] = 'near_resistance'
        elif price_position < 0.2 and momentum_score < 0:
            sr_score = -0.2  # Near support but bearish
            signals['sr_level'] = 'near_support'
        elif price_position > 0.8:
            sr_score = -0.3  # At resistance
            signals['sr_level'] = 'at_resistance'
        elif price_position < 0.2:
            sr_score = 0.3  # At support
            signals['sr_level'] = 'at_support'
        else:
            signals['sr_level'] = 'mid_range'
        
        # 5. Market Phase Detection
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        if trend_score > 0.5 and volatility < 0.02:
            signals['market_phase'] = 'trending_up'
        elif trend_score < -0.5 and volatility < 0.02:
            signals['market_phase'] = 'trending_down'
        elif volatility > 0.03:
            signals['market_phase'] = 'volatile'
        else:
            signals['market_phase'] = 'consolidating'
        
        # Calculate overall timing score
        total_score = (trend_score * 0.4 + momentum_score * 0.3 + 
                      volume_score * 0.2 + sr_score * 0.1)
        
        # Calculate confidence based on signal alignment
        aligned_signals = 0
        if abs(trend_score) > 0.5:
            aligned_signals += 1
        if abs(momentum_score) > 0.4:
            aligned_signals += 1
        if abs(volume_score) > 0.2:
            aligned_signals += 1
        
        confidence = 0.5 + (aligned_signals * 0.15)
        
        # Boost confidence for very strong setups
        if (trend_score > 0.8 and momentum_score > 0.6 and volume_ratio > 1.3):
            confidence = min(0.95, confidence + 0.15)
        elif (trend_score < -0.8 and momentum_score < -0.6 and volume_ratio > 1.3):
            confidence = min(0.95, confidence + 0.15)
        
        # 6. Market Microstructure Analysis
        microstructure = self._analyze_microstructure(data)
        signals.update(microstructure)
        
        # Adjust score based on microstructure
        if microstructure['bid_ask_imbalance'] > 0.2:
            total_score += 0.1
        elif microstructure['bid_ask_imbalance'] < -0.2:
            total_score -= 0.1
        
        # 7. Order Flow Analysis
        order_flow = self._analyze_order_flow(data)
        signals.update(order_flow)
        
        # Adjust for order flow
        if order_flow['buying_pressure'] > 0.6:
            total_score += 0.15
            confidence += 0.05
        elif order_flow['selling_pressure'] > 0.6:
            total_score -= 0.15
            confidence += 0.05
        
        # Store additional signals
        signals['trend_strength'] = abs(trend_score)
        signals['momentum_strength'] = abs(momentum_score)
        signals['volume_ratio'] = volume_ratio
        signals['price_position'] = price_position
        
        return total_score, confidence, signals
    
    def _generate_timing_reasoning(self, signals: Dict[str, Any], score: float) -> str:
        """Generate human-readable timing reasoning"""
        
        reasons = []
        
        # Trend reasoning
        if signals['trend'] == 'strong_uptrend':
            reasons.append("Strong uptrend with aligned moving averages")
        elif signals['trend'] == 'strong_downtrend':
            reasons.append("Strong downtrend with aligned moving averages")
        elif signals['trend'] == 'uptrend':
            reasons.append("Moderate uptrend in progress")
        elif signals['trend'] == 'downtrend':
            reasons.append("Moderate downtrend in progress")
        else:
            reasons.append("Sideways/consolidating market")
        
        # Momentum reasoning
        if signals['momentum'] == 'strong_positive':
            reasons.append("Strong positive momentum across timeframes")
        elif signals['momentum'] == 'strong_negative':
            reasons.append("Strong negative momentum across timeframes")
        elif signals['momentum'] == 'positive':
            reasons.append("Positive short-term momentum")
        elif signals['momentum'] == 'negative':
            reasons.append("Negative short-term momentum")
        
        # Volume reasoning
        if signals['volume'] == 'high_bullish':
            reasons.append("High volume confirms bullish move")
        elif signals['volume'] == 'high_bearish':
            reasons.append("High volume confirms bearish move")
        elif signals['volume'] == 'low':
            reasons.append("Low volume suggests lack of conviction")
        
        # Market phase
        if signals['market_phase'] == 'trending_up':
            reasons.append("Market in stable uptrend phase")
        elif signals['market_phase'] == 'trending_down':
            reasons.append("Market in stable downtrend phase")
        elif signals['market_phase'] == 'volatile':
            reasons.append("High volatility - caution advised")
        
        # Overall recommendation
        if score > 0.5:
            reasons.append("Timing favors bullish entry")
        elif score < -0.5:
            reasons.append("Timing favors bearish entry")
        else:
            reasons.append("No clear timing edge - wait for better setup")
        
        return ". ".join(reasons)
    
    def _analyze_microstructure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure for timing insights"""
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        open_price = data['Open']
        volume = data['Volume']
        
        # Bid-Ask Spread Proxy (using high-low)
        spread_proxy = (high - low) / ((high + low) / 2)
        avg_spread = float(spread_proxy.mean())
        recent_spread = float(spread_proxy.iloc[-5:].mean())
        
        # Price Impact Estimation
        price_moves = abs(close - open_price)
        volume_normalized = volume / volume.rolling(20).mean()
        price_impact = price_moves / volume_normalized
        avg_impact = float(price_impact.dropna().mean())
        
        # Bid-Ask Imbalance Proxy
        # Using close position within high-low range as proxy
        close_position = (close - low) / (high - low)
        bid_ask_imbalance = float((close_position.iloc[-5:].mean() - 0.5) * 2)
        
        # Liquidity Score
        liquidity_score = 1 / (avg_spread * avg_impact) if avg_impact > 0 else 1
        
        # Market Depth Proxy (using volume patterns)
        volume_volatility = float(volume.pct_change().std())
        depth_proxy = 1 / volume_volatility if volume_volatility > 0 else 1
        
        return {
            'avg_spread_proxy': avg_spread,
            'recent_spread_proxy': recent_spread,
            'bid_ask_imbalance': bid_ask_imbalance,
            'liquidity_score': min(10, liquidity_score),
            'market_depth_proxy': min(10, depth_proxy),
            'price_impact': avg_impact,
            'spread_widening': recent_spread > avg_spread * 1.2
        }
    
    def _analyze_order_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow patterns with HFT signals"""
        
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
        
        # Volume-Price Analysis
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive vs Negative Volume
        price_change = close.diff()
        positive_volume = volume.where(price_change > 0, 0)
        negative_volume = volume.where(price_change <= 0, 0)
        
        # Recent flow analysis (last 10 periods)
        recent_positive = positive_volume.iloc[-10:].sum()
        recent_negative = negative_volume.iloc[-10:].sum()
        total_recent = recent_positive + recent_negative
        
        buying_pressure = float(recent_positive / total_recent) if total_recent > 0 else 0.5
        selling_pressure = float(recent_negative / total_recent) if total_recent > 0 else 0.5
        
        # Large Trade Detection (volume spikes)
        volume_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        large_trades = (volume_zscore > 2).sum()
        
        # Accumulation/Distribution
        ad_line = ((close - low) - (high - close)) / (high - low) * volume
        ad_trend = float(ad_line.iloc[-5:].mean() - ad_line.iloc[-10:-5].mean())
        
        # HFT Signals
        hft_signals = self._detect_hft_patterns(data)
        
        return {
            'buying_pressure': buying_pressure,
            'selling_pressure': selling_pressure,
            'order_flow_imbalance': buying_pressure - selling_pressure,
            'large_trade_count': int(large_trades),
            'accumulation_distribution_trend': ad_trend,
            'net_order_flow': 'bullish' if buying_pressure > 0.6 else 'bearish' if selling_pressure > 0.6 else 'neutral',
            'hft_signals': hft_signals
        }
    
    def _detect_hft_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect High-Frequency Trading patterns and signals"""
        
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
        
        # Quote Stuffing Detection (rapid price changes)
        price_changes = close.pct_change()
        std_5 = price_changes.rolling(5).std()
        std_20 = price_changes.rolling(20).std()
        rapid_changes = (std_5 > std_20 * 2).sum()
        
        # Layering Detection (fake orders)
        spread = high - low
        avg_spread = spread.rolling(20).mean()
        spread_spikes = (spread > avg_spread * 1.5).sum()
        
        # Momentum Ignition (artificial momentum)
        momentum_5min = close.pct_change(5)
        momentum_reversals = ((momentum_5min > 0.01) & (momentum_5min.shift(-5) < -0.01)).sum()
        
        # Iceberg Order Detection
        volume_profile = volume.rolling(10).mean()
        consistent_volume = (volume > volume_profile * 0.8) & (volume < volume_profile * 1.2)
        iceberg_signals = (consistent_volume.rolling(5).sum() > 4).sum()
        
        # Microstructure Alpha Signals
        micro_alpha = self._calculate_microstructure_alpha(data)
        
        # Dark Pool Activity Estimation
        dark_pool_activity = self._estimate_dark_pool_activity(data)
        
        return {
            'quote_stuffing_detected': int(rapid_changes) > 5,
            'layering_signals': int(spread_spikes),
            'momentum_ignition': int(momentum_reversals),
            'iceberg_orders': int(iceberg_signals),
            'micro_alpha': micro_alpha,
            'dark_pool_activity': dark_pool_activity,
            'hft_risk_score': min(1.0, (rapid_changes + spread_spikes + momentum_reversals) / 30)
        }
    
    def _calculate_microstructure_alpha(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate microstructure-based alpha signals"""
        
        close = data['Close']
        volume = data['Volume']
        
        # Tick Rule
        tick_direction = np.sign(close.diff())
        tick_volume = tick_direction * volume
        cumulative_tick_volume = tick_volume.rolling(20).sum()
        
        # VPIN (Volume-Synchronized Probability of Informed Trading)
        # Simplified version
        total_volume = volume.rolling(20).sum()
        buy_volume = volume.where(close.diff() > 0, 0).rolling(20).sum()
        vpin = abs(buy_volume - (total_volume - buy_volume)) / total_volume
        
        # Time-Weighted Average Price (TWAP) deviation
        twap = close.rolling(20).mean()
        twap_deviation = (close - twap) / twap
        
        # Implementation Shortfall Signal
        # Measures the cost of delayed execution
        impl_shortfall = close.pct_change() * volume / volume.rolling(20).mean()
        
        return {
            'tick_pressure': float(cumulative_tick_volume.iloc[-1] / 1e6) if not np.isnan(cumulative_tick_volume.iloc[-1]) else 0,
            'vpin_score': float(vpin.iloc[-1]) if not np.isnan(vpin.iloc[-1]) else 0.5,
            'twap_signal': float(twap_deviation.iloc[-1]) if not np.isnan(twap_deviation.iloc[-1]) else 0,
            'impl_shortfall': float(impl_shortfall.iloc[-1]) if not np.isnan(impl_shortfall.iloc[-1]) else 0
        }
    
    def _estimate_dark_pool_activity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate dark pool and hidden liquidity activity"""
        
        volume = data['Volume']
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Volume clustering at round numbers (potential dark pool prints)
        round_prices = (close % 0.10 < 0.01) | (close % 0.10 > 0.09)
        round_volume = volume.where(round_prices, 0).sum() / volume.sum()
        
        # Large block detection
        avg_volume = volume.rolling(20).mean()
        block_trades = (volume > avg_volume * 3).sum()
        
        # Price improvement indicators
        midpoint = (high + low) / 2
        price_improvement = ((close - midpoint).abs() < (high - low) * 0.1).sum() / len(data)
        
        # Hidden liquidity estimation
        # When price moves through levels without much visible volume
        price_moves = close.pct_change().abs()
        volume_normalized = volume / volume.rolling(20).mean()
        hidden_liquidity = (price_moves > 0.001) & (volume_normalized < 0.5)
        
        return {
            'round_lot_percentage': float(round_volume),
            'block_trade_count': int(block_trades),
            'price_improvement_rate': float(price_improvement),
            'hidden_liquidity_signals': int(hidden_liquidity.sum()),
            'dark_pool_probability': min(1.0, round_volume + price_improvement)
        }