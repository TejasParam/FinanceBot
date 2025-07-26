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
        """Perform comprehensive market timing analysis"""
        
        signals = {}
        close = data['Close']
        volume = data['Volume']
        
        # 1. Trend Analysis
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        
        current_price = close.iloc[-1]
        
        # Trend alignment
        trend_score = 0
        if current_price > sma10.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
            trend_score = 1.0
            signals['trend'] = 'strong_uptrend'
        elif current_price > sma20.iloc[-1] > sma50.iloc[-1]:
            trend_score = 0.6
            signals['trend'] = 'uptrend'
        elif current_price < sma10.iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1]:
            trend_score = -1.0
            signals['trend'] = 'strong_downtrend'
        elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
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