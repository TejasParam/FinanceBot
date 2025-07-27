"""
Pattern Recognition Agent - Specializes in chart pattern detection and price patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import yfinance as yf

class PatternRecognitionAgent(BaseAgent):
    """
    Agent specialized in detecting chart patterns and price formations.
    Identifies bullish/bearish patterns to improve prediction accuracy.
    """
    
    def __init__(self):
        super().__init__("PatternRecognition")
        self.data_collector = DataCollectionAgent()
        
    def analyze(self, ticker: str, period: str = "3mo", **kwargs) -> Dict[str, Any]:
        """
        Analyze chart patterns for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period for analysis
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            # Get price data
            price_data = self.data_collector.fetch_stock_data(ticker, period=period)
            if price_data is None or len(price_data) < 50:
                return {
                    'error': 'Insufficient data for pattern analysis',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data'
                }
            
            # Analyze patterns
            pattern_score, confidence, signals = self._analyze_patterns(price_data)
            
            # Generate reasoning
            reasoning = self._generate_pattern_reasoning(signals, pattern_score)
            
            return {
                'score': pattern_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'patterns_detected': signals['patterns_detected'],
                'pattern_strength': signals['pattern_strength'],
                'key_levels': signals['key_levels']
            }
            
        except Exception as e:
            return {
                'error': f'Pattern analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Pattern analysis error: {str(e)[:100]}'
            }
    
    def _analyze_patterns(self, data: pd.DataFrame) -> tuple:
        """Perform comprehensive pattern analysis"""
        
        signals = {
            'patterns_detected': [],
            'pattern_scores': [],
            'key_levels': {}
        }
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # 1. Support and Resistance Levels
        support, resistance = self._find_support_resistance(close, high, low)
        signals['key_levels']['support'] = support
        signals['key_levels']['resistance'] = resistance
        
        current_price = float(close.iloc[-1])
        
        # Score based on proximity to key levels
        if support and resistance:
            price_position = (current_price - support) / (resistance - support)
            
            if price_position < 0.2:  # Near support
                signals['patterns_detected'].append('near_support')
                signals['pattern_scores'].append(0.3)
            elif price_position > 0.8:  # Near resistance
                signals['patterns_detected'].append('near_resistance')
                signals['pattern_scores'].append(-0.2)
        
        # 2. Trend Line Breaks
        trend_break = self._detect_trend_break(close)
        if trend_break['detected']:
            signals['patterns_detected'].append(f'{trend_break["type"]}_trend_break')
            score = 0.5 if trend_break['type'] == 'bullish' else -0.5
            signals['pattern_scores'].append(score)
        
        # 3. Moving Average Patterns
        ma_pattern = self._detect_ma_patterns(close)
        if ma_pattern['pattern'] != 'none':
            signals['patterns_detected'].append(ma_pattern['pattern'])
            signals['pattern_scores'].append(ma_pattern['score'])
        
        # 4. Candlestick Patterns (simplified)
        candle_patterns = self._detect_candlestick_patterns(data)
        for pattern in candle_patterns:
            signals['patterns_detected'].append(pattern['name'])
            signals['pattern_scores'].append(pattern['score'])
        
        # 5. Volume Patterns
        volume_pattern = self._detect_volume_patterns(close, volume)
        if volume_pattern['pattern'] != 'none':
            signals['patterns_detected'].append(volume_pattern['pattern'])
            signals['pattern_scores'].append(volume_pattern['score'])
        
        # 6. Price Action Patterns
        price_patterns = self._detect_price_patterns(close, high, low)
        for pattern in price_patterns:
            signals['patterns_detected'].append(pattern['name'])
            signals['pattern_scores'].append(pattern['score'])
        
        # 7. Breakout Detection
        breakout = self._detect_breakout(close, high, low, volume)
        if breakout['detected']:
            signals['patterns_detected'].append(f'{breakout["type"]}_breakout')
            score = 0.6 if breakout['type'] == 'bullish' else -0.6
            signals['pattern_scores'].append(score)
        
        # Calculate overall pattern score
        if signals['pattern_scores']:
            # Weight recent patterns more heavily
            weights = [1.0] * len(signals['pattern_scores'])
            pattern_score = np.average(signals['pattern_scores'], weights=weights)
            
            # Calculate pattern strength
            aligned_patterns = sum(1 for s in signals['pattern_scores'] if s * pattern_score > 0)
            signals['pattern_strength'] = aligned_patterns / len(signals['pattern_scores'])
            
            # Calculate confidence
            base_confidence = 0.5
            confidence = base_confidence + (signals['pattern_strength'] * 0.3)
            
            # Boost confidence for strong patterns
            if abs(pattern_score) > 0.4 and signals['pattern_strength'] > 0.7:
                confidence = min(0.9, confidence + 0.15)
        else:
            pattern_score = 0.0
            confidence = 0.5
            signals['pattern_strength'] = 0.0
        
        return pattern_score, confidence, signals
    
    def _find_support_resistance(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Tuple[float, float]:
        """Find key support and resistance levels"""
        
        # Use recent highs and lows
        recent_high = float(high.iloc[-20:].max())
        recent_low = float(low.iloc[-20:].min())
        
        # Find levels where price has bounced multiple times
        price_levels = pd.concat([high, low]).round(2)
        level_counts = price_levels.value_counts()
        
        # Get most frequent levels
        key_levels = level_counts[level_counts >= 2].index.tolist()
        
        current_price = float(close.iloc[-1])
        
        # Find nearest support and resistance
        support_levels = [level for level in key_levels if level < current_price]
        resistance_levels = [level for level in key_levels if level > current_price]
        
        support = max(support_levels) if support_levels else recent_low
        resistance = min(resistance_levels) if resistance_levels else recent_high
        
        return support, resistance
    
    def _detect_trend_break(self, close: pd.Series) -> Dict[str, Any]:
        """Detect trend line breaks"""
        
        # Simple trend detection using linear regression
        x = np.arange(len(close))
        
        # Check last 20 days for trend
        recent_x = x[-20:]
        recent_close = close.iloc[-20:].values
        
        # Calculate trend line
        z = np.polyfit(recent_x, recent_close, 1)
        slope = z[0]
        
        # Check if price has broken the trend
        trend_line = np.poly1d(z)
        expected_price = trend_line(x[-1])
        current_price = float(close.iloc[-1])
        
        if slope > 0 and current_price < expected_price * 0.98:  # Broke below uptrend
            return {'detected': True, 'type': 'bearish'}
        elif slope < 0 and current_price > expected_price * 1.02:  # Broke above downtrend
            return {'detected': True, 'type': 'bullish'}
        
        return {'detected': False}
    
    def _detect_ma_patterns(self, close: pd.Series) -> Dict[str, Any]:
        """Detect moving average patterns"""
        
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else sma20
        
        if len(close) < 50:
            return {'pattern': 'none', 'score': 0}
        
        current_price = float(close.iloc[-1])
        
        # Golden cross / Death cross
        recent_cross = None
        for i in range(-10, -1):
            if sma20.iloc[i-1] <= sma50.iloc[i-1] and sma20.iloc[i] > sma50.iloc[i]:
                recent_cross = 'golden_cross'
                break
            elif sma20.iloc[i-1] >= sma50.iloc[i-1] and sma20.iloc[i] < sma50.iloc[i]:
                recent_cross = 'death_cross'
                break
        
        if recent_cross == 'golden_cross':
            return {'pattern': 'golden_cross', 'score': 0.5}
        elif recent_cross == 'death_cross':
            return {'pattern': 'death_cross', 'score': -0.5}
        
        # Price above/below all MAs
        sma20_val = float(sma20.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        
        if current_price > sma20_val and sma20_val > sma50_val:
            return {'pattern': 'above_all_mas', 'score': 0.3}
        elif current_price < sma20_val and sma20_val < sma50_val:
            return {'pattern': 'below_all_mas', 'score': -0.3}
        
        return {'pattern': 'none', 'score': 0}
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect simple candlestick patterns"""
        
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        # Get last 3 days
        for i in [-3, -2, -1]:
            open_price = data['Open'].iloc[i]
            close_price = data['Close'].iloc[i]
            high_price = data['High'].iloc[i]
            low_price = data['Low'].iloc[i]
            
            body = abs(close_price - open_price)
            range_hl = high_price - low_price
            
            # Doji pattern
            if body < range_hl * 0.1:
                patterns.append({'name': 'doji', 'score': 0.0})  # Neutral
            
            # Hammer (bullish)
            if (close_price > open_price and 
                low_price < open_price - body and 
                high_price - close_price < body * 0.3):
                patterns.append({'name': 'hammer', 'score': 0.3})
            
            # Shooting star (bearish)
            if (close_price < open_price and 
                high_price > open_price + body and 
                close_price - low_price < body * 0.3):
                patterns.append({'name': 'shooting_star', 'score': -0.3})
        
        # Engulfing patterns
        if len(data) >= 2:
            prev_open = data['Open'].iloc[-2]
            prev_close = data['Close'].iloc[-2]
            curr_open = data['Open'].iloc[-1]
            curr_close = data['Close'].iloc[-1]
            
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous bearish
                curr_close > curr_open and  # Current bullish
                curr_open < prev_close and  # Opens below prev close
                curr_close > prev_open):     # Closes above prev open
                patterns.append({'name': 'bullish_engulfing', 'score': 0.4})
            
            # Bearish engulfing
            elif (prev_close > prev_open and  # Previous bullish
                  curr_close < curr_open and  # Current bearish
                  curr_open > prev_close and  # Opens above prev close
                  curr_close < prev_open):     # Closes below prev open
                patterns.append({'name': 'bearish_engulfing', 'score': -0.4})
        
        return patterns
    
    def _detect_volume_patterns(self, close: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        """Detect volume-based patterns"""
        
        avg_volume = volume.rolling(20).mean()
        recent_volume = volume.iloc[-5:].mean()
        
        price_change = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0
        
        # Volume breakout
        avg_volume_val = float(avg_volume.iloc[-1])
        current_close = float(close.iloc[-1])
        ma20_val = float(close.rolling(20).mean().iloc[-1])
        
        if recent_volume > avg_volume_val * 1.5:
            if price_change > 0.02:
                return {'pattern': 'volume_breakout_up', 'score': 0.4}
            elif price_change < -0.02:
                return {'pattern': 'volume_breakout_down', 'score': -0.4}
        
        # Low volume pullback
        elif recent_volume < avg_volume_val * 0.7:
            if price_change < -0.01 and current_close > ma20_val:
                return {'pattern': 'low_volume_pullback', 'score': 0.2}
        
        return {'pattern': 'none', 'score': 0}
    
    def _detect_price_patterns(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[Dict[str, Any]]:
        """Detect price action patterns"""
        
        patterns = []
        
        if len(close) < 20:
            return patterns
        
        # Higher highs and higher lows (uptrend)
        recent_highs = high.iloc[-20:].rolling(5).max()
        recent_lows = low.iloc[-20:].rolling(5).min()
        
        if len(recent_highs) > 10:
            hh_count = sum(recent_highs.iloc[i] > recent_highs.iloc[i-5] 
                          for i in range(-10, -1) if i-5 >= -len(recent_highs))
            hl_count = sum(recent_lows.iloc[i] > recent_lows.iloc[i-5] 
                          for i in range(-10, -1) if i-5 >= -len(recent_lows))
            
            if hh_count >= 2 and hl_count >= 2:
                patterns.append({'name': 'uptrend_continuation', 'score': 0.4})
            elif hh_count <= 0 and hl_count <= 0:
                patterns.append({'name': 'downtrend_continuation', 'score': -0.4})
        
        # Double bottom/top patterns
        if len(low) >= 20:
            recent_low_idx = low.iloc[-20:].idxmin()
            recent_low_val = low.iloc[-20:].min()
            
            # Look for another similar low
            for i in range(-20, -1):
                if abs(i) > 5 and abs(low.iloc[i] - recent_low_val) / recent_low_val < 0.01:
                    if close.iloc[-1] > close.iloc[i]:
                        patterns.append({'name': 'double_bottom', 'score': 0.5})
                        break
        
        return patterns
    
    def _detect_breakout(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        """Detect breakout patterns"""
        
        if len(close) < 20:
            return {'detected': False}
        
        # Define consolidation range
        recent_high = high.iloc[-20:-1].max()
        recent_low = low.iloc[-20:-1].min()
        range_size = (recent_high - recent_low) / recent_low
        
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-20:].mean()
        
        # Check for breakout
        if range_size < 0.05:  # Tight consolidation
            avg_volume_val = float(avg_volume) if not pd.isna(avg_volume) else 0
            if current_price > recent_high and current_volume > avg_volume_val * 1.2:
                return {'detected': True, 'type': 'bullish'}
            elif current_price < recent_low and current_volume > avg_volume_val * 1.2:
                return {'detected': True, 'type': 'bearish'}
        
        return {'detected': False}
    
    def _generate_pattern_reasoning(self, signals: Dict[str, Any], score: float) -> str:
        """Generate human-readable pattern reasoning"""
        
        reasons = []
        
        # List detected patterns
        if signals['patterns_detected']:
            pattern_list = ', '.join(signals['patterns_detected'][:3])  # Top 3 patterns
            reasons.append(f"Detected patterns: {pattern_list}")
        else:
            reasons.append("No significant patterns detected")
        
        # Pattern strength
        strength = signals.get('pattern_strength', 0)
        if strength > 0.7:
            reasons.append("Strong pattern alignment")
        elif strength > 0.5:
            reasons.append("Moderate pattern alignment")
        elif strength < 0.3:
            reasons.append("Weak or conflicting patterns")
        
        # Key levels
        if 'support' in signals['key_levels'] and 'resistance' in signals['key_levels']:
            reasons.append(f"Key levels: Support ${signals['key_levels']['support']:.2f}, Resistance ${signals['key_levels']['resistance']:.2f}")
        
        # Overall assessment
        if score > 0.3:
            reasons.append("Bullish pattern formation")
        elif score < -0.3:
            reasons.append("Bearish pattern formation")
        else:
            reasons.append("Neutral pattern outlook")
        
        return ". ".join(reasons)