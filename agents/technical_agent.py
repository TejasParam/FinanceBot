"""
Technical Analysis Agent for stock analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from .base_agent import BaseAgent
from technical_analysis import technical_analyst_agent
from data_collection import DataCollectionAgent

class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent specialized in technical analysis of stocks.
    Uses technical indicators to generate buy/sell signals.
    """
    
    def __init__(self):
        super().__init__("TechnicalAnalysis")
        self.data_collector = DataCollectionAgent()
        self.technical_analyzer = technical_analyst_agent()
        
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
            data = self.technical_analyzer.getData(ticker)
            if data is None or len(data) == 0 or len(data[0]) < 20:
                return {
                    'error': f'Insufficient data for technical analysis of {ticker}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data for technical analysis'
                }
            
            # Use the first DataFrame from the returned list
            price_data = data[0]
            
            # Calculate technical indicators using existing methods
            indicators = {
                'rsi': self.technical_analyzer.calculate_rsi(ticker),
                'macd': self.technical_analyzer.calculate_macd(ticker),
                'bollinger_bands': self.technical_analyzer.calculate_bollinger_bands(ticker),
                'sma_20': self.technical_analyzer.calculate_sma(ticker, 20),
                'ema_12': self.technical_analyzer.calculate_ema(ticker, 12),
                'current_price': float(price_data['Close'].iloc[-1])
            }
            
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
        """Calculate overall technical score from -1 (bearish) to 1 (bullish)"""
        score = 0.0
        weight_sum = 0.0
        
        # RSI scoring
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += 0.8 * 0.2  # Oversold = bullish
        elif rsi > 70:
            score -= 0.8 * 0.2  # Overbought = bearish
        else:
            score += (50 - rsi) / 50 * 0.2  # Neutral zone
        weight_sum += 0.2
        
        # MACD scoring
        macd_signal = indicators.get('macd_signal', 'neutral')
        if macd_signal == 'bullish':
            score += 0.8 * 0.25
        elif macd_signal == 'bearish':
            score -= 0.8 * 0.25
        weight_sum += 0.25
        
        # Moving average trend scoring
        sma_trend = indicators.get('sma_trend', 'neutral')
        if sma_trend == 'bullish':
            score += 0.7 * 0.25
        elif sma_trend == 'bearish':
            score -= 0.7 * 0.25
        weight_sum += 0.25
        
        # Bollinger Band position scoring
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:
            score += 0.6 * 0.15  # Near lower band = bullish
        elif bb_position > 0.8:
            score -= 0.6 * 0.15  # Near upper band = bearish
        weight_sum += 0.15
        
        # Volume trend scoring
        volume_trend = indicators.get('volume_trend', 'neutral')
        if volume_trend == 'increasing':
            score += 0.3 * 0.15  # Volume confirms moves
        elif volume_trend == 'decreasing':
            score -= 0.2 * 0.15
        weight_sum += 0.15
        
        return max(-1.0, min(1.0, score / weight_sum if weight_sum > 0 else 0.0))
    
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
