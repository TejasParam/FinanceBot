"""
Volatility Analysis Agent - Specializes in volatility patterns and risk assessment
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

class VolatilityAnalysisAgent(BaseAgent):
    """
    Agent specialized in volatility analysis and risk assessment.
    Helps identify high-risk periods and adjust predictions accordingly.
    """
    
    def __init__(self):
        super().__init__("VolatilityAnalysis")
        self.data_collector = DataCollectionAgent()
        
    def analyze(self, ticker: str, period: str = "3mo", **kwargs) -> Dict[str, Any]:
        """
        Analyze volatility patterns for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period for analysis
            
        Returns:
            Dictionary with volatility analysis
        """
        try:
            # Get price data
            price_data = self.data_collector.fetch_stock_data(ticker, period=period)
            if price_data is None or len(price_data) < 30:
                return {
                    'error': 'Insufficient data for volatility analysis',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data'
                }
            
            # Analyze volatility patterns
            vol_score, confidence, signals = self._analyze_volatility(price_data)
            
            # Generate reasoning
            reasoning = self._generate_volatility_reasoning(signals, vol_score)
            
            return {
                'score': vol_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'volatility_signals': signals,
                'risk_level': signals['risk_level'],
                'volatility_regime': signals['volatility_regime']
            }
            
        except Exception as e:
            return {
                'error': f'Volatility analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Volatility analysis error: {str(e)[:100]}'
            }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> tuple:
        """Perform comprehensive volatility analysis"""
        
        signals = {}
        close = data['Close']
        
        # 1. Historical Volatility (multiple timeframes)
        returns = close.pct_change().dropna()
        
        # Short-term volatility (5-day)
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        current_vol_5d = float(vol_5d.iloc[-1])
        
        # Medium-term volatility (20-day)
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        current_vol_20d = float(vol_20d.iloc[-1])
        
        # Long-term volatility (60-day)
        vol_60d = returns.rolling(60).std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
        current_vol_60d = float(vol_60d.iloc[-1]) if len(returns) >= 60 else current_vol_20d
        
        signals['vol_5d'] = current_vol_5d
        signals['vol_20d'] = current_vol_20d
        signals['vol_60d'] = current_vol_60d
        
        # 2. Volatility Trend
        vol_trend = 0
        if current_vol_5d > current_vol_20d > current_vol_60d:
            vol_trend = 1  # Increasing volatility
            signals['vol_trend'] = 'increasing'
        elif current_vol_5d < current_vol_20d < current_vol_60d:
            vol_trend = -1  # Decreasing volatility
            signals['vol_trend'] = 'decreasing'
        else:
            signals['vol_trend'] = 'stable'
        
        # 3. Volatility Regime Detection
        if current_vol_20d < 0.15:
            signals['volatility_regime'] = 'low'
            regime_score = 0.3  # Low vol favors trend following
        elif current_vol_20d < 0.25:
            signals['volatility_regime'] = 'normal'
            regime_score = 0.0
        elif current_vol_20d < 0.35:
            signals['volatility_regime'] = 'elevated'
            regime_score = -0.3  # High vol reduces confidence
        else:
            signals['volatility_regime'] = 'extreme'
            regime_score = -0.6  # Extreme vol suggests avoiding trades
        
        # 4. Volatility Clustering (GARCH-like effect)
        recent_big_moves = abs(returns.iloc[-5:]) > returns.std() * 2
        volatility_clustering = recent_big_moves.sum() >= 2
        signals['volatility_clustering'] = volatility_clustering
        
        if volatility_clustering:
            regime_score -= 0.2  # Recent big moves suggest more to come
        
        # 5. Intraday Volatility (using High-Low)
        if 'High' in data.columns and 'Low' in data.columns:
            hl_ratio = (data['High'] - data['Low']) / data['Close']
            avg_hl_ratio = hl_ratio.rolling(20).mean().iloc[-1]
            recent_hl_ratio = hl_ratio.iloc[-5:].mean()
            
            signals['intraday_vol'] = float(recent_hl_ratio)
            signals['intraday_vol_trend'] = 'increasing' if recent_hl_ratio > avg_hl_ratio * 1.2 else 'stable'
            
            if recent_hl_ratio > avg_hl_ratio * 1.5:
                regime_score -= 0.2  # High intraday volatility
        
        # 6. Volatility of Volatility (vol of vol)
        if len(vol_20d) > 20:
            vol_of_vol = vol_20d.rolling(20).std().iloc[-1]
            signals['vol_of_vol'] = float(vol_of_vol)
            
            if vol_of_vol > 0.05:  # High vol of vol
                regime_score -= 0.1
                signals['vol_stability'] = 'unstable'
            else:
                signals['vol_stability'] = 'stable'
        
        # 7. Downside vs Upside Volatility
        down_returns = returns[returns < 0]
        up_returns = returns[returns > 0]
        
        if len(down_returns) > 5 and len(up_returns) > 5:
            down_vol = down_returns.std() * np.sqrt(252)
            up_vol = up_returns.std() * np.sqrt(252)
            
            vol_skew = down_vol / up_vol if up_vol > 0 else 1
            signals['vol_skew'] = float(vol_skew)
            
            if vol_skew > 1.3:  # More downside volatility
                regime_score -= 0.2
                signals['vol_asymmetry'] = 'bearish'
            elif vol_skew < 0.7:  # More upside volatility
                regime_score += 0.1
                signals['vol_asymmetry'] = 'bullish'
            else:
                signals['vol_asymmetry'] = 'neutral'
        
        # 8. Calculate overall volatility score
        # In low vol regimes, we can be more confident in trends
        # In high vol regimes, we should be more cautious
        volatility_score = regime_score
        
        # Adjust for current price trend
        sma20 = close.rolling(20).mean()
        price_trend = 1 if close.iloc[-1] > sma20.iloc[-1] else -1
        
        if signals['volatility_regime'] == 'low' and abs(price_trend) > 0:
            # Low volatility trending markets are good
            volatility_score = price_trend * 0.4
        elif signals['volatility_regime'] == 'extreme':
            # Extreme volatility markets are risky
            volatility_score = -0.5
        
        # Calculate confidence based on volatility stability
        base_confidence = 0.6
        
        if signals['volatility_regime'] == 'low':
            confidence = base_confidence + 0.2
        elif signals['volatility_regime'] == 'normal':
            confidence = base_confidence + 0.1
        elif signals['volatility_regime'] == 'elevated':
            confidence = base_confidence - 0.1
        else:  # extreme
            confidence = base_confidence - 0.2
        
        # Adjust confidence for volatility clustering
        if volatility_clustering:
            confidence -= 0.1
        
        # Determine risk level
        if current_vol_20d > 0.35 or volatility_clustering:
            signals['risk_level'] = 'high'
        elif current_vol_20d > 0.25:
            signals['risk_level'] = 'medium'
        else:
            signals['risk_level'] = 'low'
        
        return volatility_score, confidence, signals
    
    def _generate_volatility_reasoning(self, signals: Dict[str, Any], score: float) -> str:
        """Generate human-readable volatility reasoning"""
        
        reasons = []
        
        # Volatility regime
        regime = signals['volatility_regime']
        if regime == 'low':
            reasons.append("Low volatility environment favors trend following")
        elif regime == 'normal':
            reasons.append("Normal volatility conditions")
        elif regime == 'elevated':
            reasons.append("Elevated volatility suggests caution")
        else:  # extreme
            reasons.append("Extreme volatility - high risk environment")
        
        # Volatility trend
        if signals['vol_trend'] == 'increasing':
            reasons.append("Volatility is increasing across timeframes")
        elif signals['vol_trend'] == 'decreasing':
            reasons.append("Volatility is decreasing - market stabilizing")
        
        # Volatility clustering
        if signals.get('volatility_clustering', False):
            reasons.append("Recent large moves suggest continued volatility")
        
        # Risk level
        risk = signals['risk_level']
        if risk == 'high':
            reasons.append("High risk - consider reducing position size")
        elif risk == 'low':
            reasons.append("Low risk environment")
        
        # Volatility asymmetry
        if 'vol_asymmetry' in signals:
            if signals['vol_asymmetry'] == 'bearish':
                reasons.append("Higher downside volatility indicates fear")
            elif signals['vol_asymmetry'] == 'bullish':
                reasons.append("Higher upside volatility indicates optimism")
        
        # Overall assessment
        if score > 0.2:
            reasons.append("Volatility conditions favor bullish positions")
        elif score < -0.2:
            reasons.append("Volatility conditions suggest defensive stance")
        else:
            reasons.append("Mixed volatility signals")
        
        return ". ".join(reasons)