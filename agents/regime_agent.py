"""
Regime Detection Agent for market regime analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import numpy as np
import pandas as pd
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent

class RegimeDetectionAgent(BaseAgent):
    """
    Agent specialized in detecting market regimes (bull, bear, sideways).
    Analyzes market conditions to determine the current trading environment.
    """
    
    def __init__(self):
        super().__init__("RegimeDetection")
        self.data_collector = DataCollectionAgent()
        
    def analyze(self, ticker: str, period: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        Perform market regime analysis for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for analysis (default: 1y)
            
        Returns:
            Dictionary with regime analysis results
        """
        try:
            # Get historical data
            data = self.data_collector.get_historical_data(ticker, period=period)
            if data is None or len(data) < 60:
                return {
                    'error': f'Insufficient data for regime analysis of {ticker}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data for regime analysis'
                }
            
            # Calculate regime indicators
            regime_metrics = self._calculate_regime_metrics(data)
            
            # Determine current regime
            current_regime = self._determine_regime(regime_metrics)
            
            # Calculate regime score (-1 = bear, 0 = sideways, 1 = bull)
            regime_score = self._calculate_regime_score(current_regime, regime_metrics)
            
            # Calculate confidence in regime detection
            confidence = self._calculate_regime_confidence(regime_metrics)
            
            # Generate reasoning
            reasoning = self._generate_regime_reasoning(current_regime, regime_metrics)
            
            return {
                'score': regime_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'regime': current_regime,
                'regime_strength': abs(regime_score),
                'metrics': {
                    'volatility': regime_metrics['volatility'],
                    'trend_strength': regime_metrics['trend_strength'],
                    'price_range': regime_metrics['price_range'],
                    'momentum': regime_metrics['momentum'],
                    'volume_trend': regime_metrics['volume_trend']
                },
                'raw_metrics': regime_metrics
            }
            
        except Exception as e:
            return {
                'error': f'Regime analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Regime analysis error: {str(e)[:100]}'
            }
    
    def _calculate_regime_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various metrics for regime detection"""
        metrics = {}
        
        # Price-based metrics
        returns = data['Close'].pct_change().dropna()
        
        # Volatility (annualized)
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Trend strength using moving averages
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        sma_200 = data['Close'].rolling(200).mean() if len(data) >= 200 else sma_50
        
        # Current price vs moving averages
        current_price = data['Close'].iloc[-1]
        metrics['price_vs_sma20'] = current_price / sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else 1.0
        metrics['price_vs_sma50'] = current_price / sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else 1.0
        metrics['price_vs_sma200'] = current_price / sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else 1.0
        
        # Moving average trends
        sma20_trend = (sma_20.iloc[-1] / sma_20.iloc[-10] - 1) if len(sma_20) >= 10 and not pd.isna(sma_20.iloc[-10]) else 0
        sma50_trend = (sma_50.iloc[-1] / sma_50.iloc[-20] - 1) if len(sma_50) >= 20 and not pd.isna(sma_50.iloc[-20]) else 0
        
        metrics['sma20_trend'] = sma20_trend
        metrics['sma50_trend'] = sma50_trend
        
        # Overall trend strength
        trend_signals = [
            1 if metrics['price_vs_sma20'] > 1.02 else -1 if metrics['price_vs_sma20'] < 0.98 else 0,
            1 if metrics['price_vs_sma50'] > 1.02 else -1 if metrics['price_vs_sma50'] < 0.98 else 0,
            1 if metrics['price_vs_sma200'] > 1.02 else -1 if metrics['price_vs_sma200'] < 0.98 else 0,
            1 if sma20_trend > 0.01 else -1 if sma20_trend < -0.01 else 0,
            1 if sma50_trend > 0.01 else -1 if sma50_trend < -0.01 else 0
        ]
        metrics['trend_strength'] = sum(trend_signals) / len(trend_signals)
        
        # Price range analysis
        high_52w = data['High'].rolling(min(252, len(data))).max().iloc[-1]
        low_52w = data['Low'].rolling(min(252, len(data))).min().iloc[-1]
        metrics['price_range'] = (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
        
        # Recent momentum
        returns_20d = returns.tail(20)
        metrics['momentum'] = returns_20d.mean() * 20  # 20-day cumulative return
        
        # Volume analysis (if available)
        if 'Volume' in data.columns:
            volume_ma = data['Volume'].rolling(20).mean()
            recent_volume = data['Volume'].tail(10).mean()
            avg_volume = volume_ma.tail(50).mean()
            metrics['volume_trend'] = recent_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            metrics['volume_trend'] = 1.0
        
        # Drawdown analysis
        rolling_max = data['Close'].expanding().max()
        drawdown = (data['Close'] - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown.iloc[-1]
        
        return metrics
    
    def _determine_regime(self, metrics: Dict[str, Any]) -> str:
        """Determine the current market regime"""
        trend_strength = metrics.get('trend_strength', 0)
        volatility = metrics.get('volatility', 0.2)
        momentum = metrics.get('momentum', 0)
        price_range = metrics.get('price_range', 0.5)
        
        # Bull market criteria
        if (trend_strength > 0.4 and 
            momentum > 0.02 and 
            price_range > 0.7 and
            volatility < 0.4):
            return 'strong_bull'
        elif (trend_strength > 0.2 and 
              momentum > 0.01 and 
              price_range > 0.5):
            return 'bull'
        
        # Bear market criteria
        elif (trend_strength < -0.4 and 
              momentum < -0.02 and 
              price_range < 0.3 and
              metrics.get('current_drawdown', 0) < -0.2):
            return 'strong_bear'
        elif (trend_strength < -0.2 and 
              momentum < -0.01 and 
              price_range < 0.5):
            return 'bear'
        
        # High volatility regime
        elif volatility > 0.5:
            return 'high_volatility'
        
        # Sideways/consolidation
        else:
            return 'sideways'
    
    def _calculate_regime_score(self, regime: str, metrics: Dict[str, Any]) -> float:
        """Calculate regime score from -1 (bearish) to 1 (bullish)"""
        regime_scores = {
            'strong_bull': 0.8,
            'bull': 0.5,
            'sideways': 0.0,
            'high_volatility': -0.2,  # Slightly bearish due to uncertainty
            'bear': -0.5,
            'strong_bear': -0.8
        }
        
        base_score = regime_scores.get(regime, 0.0)
        
        # Adjust based on momentum and trend strength
        trend_strength = metrics.get('trend_strength', 0)
        momentum = metrics.get('momentum', 0)
        
        # Fine-tune score based on metrics
        adjustment = (trend_strength * 0.3 + momentum * 10 * 0.2)
        adjusted_score = base_score + adjustment
        
        return max(-1.0, min(1.0, adjusted_score))
    
    def _calculate_regime_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in regime detection"""
        confidence_factors = []
        
        # Trend consistency
        trend_strength = abs(metrics.get('trend_strength', 0))
        confidence_factors.append(min(1.0, trend_strength * 2))
        
        # Momentum consistency
        momentum = abs(metrics.get('momentum', 0))
        confidence_factors.append(min(1.0, momentum * 20))
        
        # Price position clarity (extreme positions = higher confidence)
        price_range = metrics.get('price_range', 0.5)
        range_confidence = abs(price_range - 0.5) * 2  # 0 at middle, 1 at extremes
        confidence_factors.append(range_confidence)
        
        # Volatility consistency (very high or very low = more confident)
        volatility = metrics.get('volatility', 0.2)
        if volatility < 0.15 or volatility > 0.6:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_regime_reasoning(self, regime: str, metrics: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for regime analysis"""
        reasoning_parts = []
        
        # Regime description
        regime_descriptions = {
            'strong_bull': 'Strong bull market conditions with robust uptrend.',
            'bull': 'Bull market conditions with positive momentum.',
            'sideways': 'Sideways market with consolidation patterns.',
            'high_volatility': 'High volatility environment with uncertain direction.',
            'bear': 'Bear market conditions with negative momentum.',
            'strong_bear': 'Strong bear market with significant downtrend.'
        }
        
        reasoning_parts.append(regime_descriptions.get(regime, 'Market regime unclear.'))
        
        # Key metrics
        trend_strength = metrics.get('trend_strength', 0)
        if trend_strength > 0.3:
            reasoning_parts.append("Strong bullish trend indicators.")
        elif trend_strength < -0.3:
            reasoning_parts.append("Strong bearish trend indicators.")
        else:
            reasoning_parts.append("Mixed trend signals.")
        
        # Volatility context
        volatility = metrics.get('volatility', 0.2)
        if volatility > 0.5:
            reasoning_parts.append(f"High volatility ({volatility:.1%}) suggests uncertain environment.")
        elif volatility < 0.15:
            reasoning_parts.append(f"Low volatility ({volatility:.1%}) suggests stable conditions.")
        
        # Price position
        price_range = metrics.get('price_range', 0.5)
        if price_range > 0.8:
            reasoning_parts.append("Price near 52-week highs suggests strength.")
        elif price_range < 0.2:
            reasoning_parts.append("Price near 52-week lows suggests weakness.")
        
        # Momentum
        momentum = metrics.get('momentum', 0)
        if momentum > 0.02:
            reasoning_parts.append("Positive momentum supports bullish outlook.")
        elif momentum < -0.02:
            reasoning_parts.append("Negative momentum supports bearish outlook.")
        
        return " ".join(reasoning_parts)
