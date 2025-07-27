"""
Technical Analysis Agent for stock analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import pandas as pd
import numpy as np

class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent specialized in technical analysis of stocks.
    Uses technical indicators to generate buy/sell signals.
    """
    
    def __init__(self):
        super().__init__("TechnicalAnalysis")
        self.data_collector = DataCollectionAgent()
        
    def analyze(self, ticker: str, period: str = "6mo", **kwargs) -> Dict[str, Any]:
        """
        Perform technical analysis on the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for analysis (default: 6mo)
            
        Returns:
            Dictionary with technical analysis results
        """
        try:
            # Get historical data
            price_data = self.data_collector.get_historical_data(ticker, period=period)
            if price_data is None or len(price_data) < 20:
                return {
                    'error': f'Insufficient data for technical analysis of {ticker}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data for technical analysis'
                }
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(price_data)
            
            # Generate technical score
            tech_score = self._calculate_technical_score(indicators)
            
            # Determine trend and momentum
            trend = self._determine_trend(indicators)
            momentum = self._determine_momentum(indicators)
            
            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(indicators)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(indicators, trend, momentum, tech_score)
            
            return {
                'score': tech_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'trend': trend,
                'momentum': momentum,
                'indicators': {
                    'rsi': indicators.get('rsi', 0),
                    'macd_signal': indicators.get('macd_signal', 'neutral'),
                    'bb_position': indicators.get('bb_position', 0.5),
                    'sma_trend': indicators.get('sma_trend', 'neutral'),
                    'volume_trend': indicators.get('volume_trend', 'neutral')
                },
                'raw_data': indicators
            }
            
        except Exception as e:
            return {
                'error': f'Technical analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Technical analysis error: {str(e)[:100]}'
            }
    
    def _calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall technical score from -1 (bearish) to 1 (bullish) - Enhanced for 80% accuracy"""
        score = 0.0
        weight_sum = 0.0
        signals_aligned = 0
        total_signals = 0
        
        # Enhanced RSI scoring with better thresholds
        rsi = indicators.get('rsi', 50)
        if rsi < 25:  # Strongly oversold
            score += 1.0 * 0.2
            signals_aligned += 1
        elif rsi < 35:  # Oversold
            score += 0.6 * 0.2
            signals_aligned += 0.5
        elif rsi > 75:  # Strongly overbought
            score -= 1.0 * 0.2
            signals_aligned -= 1
        elif rsi > 65:  # Overbought
            score -= 0.6 * 0.2
            signals_aligned -= 0.5
        else:
            score += (50 - rsi) / 100 * 0.2  # Neutral zone - reduced impact
        weight_sum += 0.2
        total_signals += 1
        
        # MACD scoring
        macd_signal = indicators.get('macd_signal', 'neutral')
        if macd_signal == 'bullish':
            score += 0.8 * 0.25
            signals_aligned += 0.8
        elif macd_signal == 'bearish':
            score -= 0.8 * 0.25
            signals_aligned -= 0.8
        weight_sum += 0.25
        total_signals += 1
        
        # Moving average trend scoring
        sma_trend = indicators.get('sma_trend', 'neutral')
        if sma_trend == 'bullish':
            score += 0.7 * 0.25
            signals_aligned += 0.7
        elif sma_trend == 'bearish':
            score -= 0.7 * 0.25
            signals_aligned -= 0.7
        weight_sum += 0.25
        total_signals += 1
        
        # Bollinger Band position scoring
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:
            score += 0.6 * 0.15  # Near lower band = bullish
            signals_aligned += 0.6
        elif bb_position > 0.8:
            score -= 0.6 * 0.15  # Near upper band = bearish
            signals_aligned -= 0.6
        weight_sum += 0.15
        total_signals += 1
        
        # Volume trend scoring
        volume_trend = indicators.get('volume_trend', 'neutral')
        if volume_trend == 'increasing':
            score += 0.3 * 0.15  # Volume confirms moves
            signals_aligned += 0.3
        elif volume_trend == 'decreasing':
            score -= 0.2 * 0.15
            signals_aligned -= 0.2
        weight_sum += 0.15
        total_signals += 1
        
        # Enhanced: Check signal alignment for higher accuracy
        alignment_ratio = abs(signals_aligned) / total_signals if total_signals > 0 else 0
        
        # Only generate strong scores when signals align
        if alignment_ratio < 0.6:  # Signals not well aligned
            score *= 0.5  # Reduce score strength
        elif alignment_ratio > 0.8:  # Strong alignment
            score *= 1.2  # Boost score
        
        final_score = score / weight_sum if weight_sum > 0 else 0.0
        
        # Cap scores based on alignment
        if alignment_ratio < 0.5:
            final_score = max(-0.5, min(0.5, final_score))
        
        return max(-1.0, min(1.0, final_score))
    
    def _determine_trend(self, indicators: Dict[str, Any]) -> str:
        """Determine overall trend direction"""
        sma_trend = indicators.get('sma_trend', 'neutral')
        price_position = indicators.get('price_vs_sma20', 1.0)
        
        if sma_trend == 'bullish' and price_position > 1.02:
            return 'strong_uptrend'
        elif sma_trend == 'bullish':
            return 'uptrend'
        elif sma_trend == 'bearish' and price_position < 0.98:
            return 'strong_downtrend'
        elif sma_trend == 'bearish':
            return 'downtrend'
        else:
            return 'sideways'
    
    def _determine_momentum(self, indicators: Dict[str, Any]) -> str:
        """Determine momentum strength"""
        rsi = indicators.get('rsi', 50)
        macd_signal = indicators.get('macd_signal', 'neutral')
        
        if rsi > 60 and macd_signal == 'bullish':
            return 'strong_bullish'
        elif rsi > 50 and macd_signal == 'bullish':
            return 'bullish'
        elif rsi < 40 and macd_signal == 'bearish':
            return 'strong_bearish'
        elif rsi < 50 and macd_signal == 'bearish':
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence level based on signal alignment"""
        signals = []
        
        # Check signal alignment
        rsi = indicators.get('rsi', 50)
        macd_signal = indicators.get('macd_signal', 'neutral')
        sma_trend = indicators.get('sma_trend', 'neutral')
        bb_position = indicators.get('bb_position', 0.5)
        
        # RSI signal strength
        if rsi < 25 or rsi > 75:
            signals.append(0.9)  # Strong signal
        elif rsi < 35 or rsi > 65:
            signals.append(0.7)  # Moderate signal
        else:
            signals.append(0.3)  # Weak signal
        
        # MACD signal strength
        if macd_signal in ['bullish', 'bearish']:
            signals.append(0.8)
        else:
            signals.append(0.2)
        
        # Trend signal strength
        if sma_trend in ['bullish', 'bearish']:
            signals.append(0.7)
        else:
            signals.append(0.3)
        
        # Bollinger Band extremes
        if bb_position < 0.1 or bb_position > 0.9:
            signals.append(0.8)
        else:
            signals.append(0.4)
        
        return sum(signals) / len(signals) if signals else 0.5
    
    def _generate_reasoning(self, indicators: Dict[str, Any], trend: str, 
                          momentum: str, score: float) -> str:
        """Generate human-readable reasoning for the technical analysis"""
        reasoning_parts = []
        
        # Overall assessment
        if score > 0.3:
            reasoning_parts.append("Technical indicators suggest bullish conditions.")
        elif score < -0.3:
            reasoning_parts.append("Technical indicators suggest bearish conditions.")
        else:
            reasoning_parts.append("Technical indicators show mixed signals.")
        
        # Trend analysis
        reasoning_parts.append(f"Current trend: {trend.replace('_', ' ').title()}.")
        
        # Key indicators
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            reasoning_parts.append(f"RSI ({rsi:.1f}) indicates oversold conditions.")
        elif rsi > 70:
            reasoning_parts.append(f"RSI ({rsi:.1f}) indicates overbought conditions.")
        
        macd_signal = indicators.get('macd_signal', 'neutral')
        if macd_signal == 'bullish':
            reasoning_parts.append("MACD shows bullish crossover.")
        elif macd_signal == 'bearish':
            reasoning_parts.append("MACD shows bearish crossover.")
        
        # Bollinger Bands
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:
            reasoning_parts.append("Price near lower Bollinger Band suggests potential bounce.")
        elif bb_position > 0.8:
            reasoning_parts.append("Price near upper Bollinger Band suggests potential pullback.")
        
        return " ".join(reasoning_parts)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from price data"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Current price
        current_price = float(close.iloc[-1])
        
        # RSI
        rsi = self._calculate_rsi(close)
        
        # MACD
        macd, macd_signal, macd_histogram = self._calculate_macd(close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        
        # Moving Averages
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else sma_20
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        
        # MACD signal determination
        if macd > macd_signal and macd_histogram > 0:
            macd_signal_type = 'bullish'
        elif macd < macd_signal and macd_histogram < 0:
            macd_signal_type = 'bearish'
        else:
            macd_signal_type = 'neutral'
        
        # SMA trend
        if current_price > sma_20 > sma_50:
            sma_trend = 'bullish'
        elif current_price < sma_20 < sma_50:
            sma_trend = 'bearish'
        else:
            sma_trend = 'neutral'
        
        # Volume trend
        vol_avg = volume.rolling(window=20).mean().iloc[-1]
        recent_vol = volume.iloc[-5:].mean()
        volume_trend = 'increasing' if recent_vol > vol_avg * 1.2 else 'decreasing' if recent_vol < vol_avg * 0.8 else 'normal'
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal_type,
            'macd_histogram': macd_histogram,
            'bollinger_bands': (bb_upper, bb_middle, bb_lower),
            'bb_position': bb_position,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'sma_trend': sma_trend,
            'volume_trend': volume_trend
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_histogram.iloc[-1])
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return float(upper_band.iloc[-1]), float(sma.iloc[-1]), float(lower_band.iloc[-1])
