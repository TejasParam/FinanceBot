"""
Improved Agent Coordinator with better accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from .coordinator import AgentCoordinator

class ImprovedAgentCoordinator(AgentCoordinator):
    """
    Improved coordinator with better signal aggregation and filtering
    """
    
    def __init__(self, enable_ml: bool = True, enable_llm: bool = True, 
                 parallel_execution: bool = True):
        super().__init__(enable_ml, enable_llm, parallel_execution)
        
        # Agent weights based on historical performance
        self.agent_weights = {
            'TechnicalAnalysis': 0.25,
            'FundamentalAnalysis': 0.20,
            'SentimentAnalysis': 0.15,
            'MLPrediction': 0.20,
            'RegimeDetection': 0.10,
            'LLMExplanation': 0.10
        }
        
        # Minimum agreement threshold for high confidence
        self.min_agreement_threshold = 0.7
        
    def analyze_stock(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Enhanced analysis with better signal filtering
        """
        # Get base analysis
        base_result = super().analyze_stock(ticker, **kwargs)
        
        # Apply improvements
        improved_result = self._improve_analysis(ticker, base_result)
        
        return improved_result
    
    def _improve_analysis(self, ticker: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvements to base analysis"""
        
        # 1. Check market regime
        market_regime = self._check_market_regime(ticker)
        
        # 2. Get multi-timeframe confirmation
        timeframe_signals = self._get_timeframe_signals(ticker)
        
        # 3. Calculate improved scores
        agent_results = base_result.get('agent_results', {})
        improved_score, improved_confidence = self._calculate_improved_metrics(
            agent_results, market_regime, timeframe_signals
        )
        
        # 4. Generate improved recommendation
        recommendation = self._generate_improved_recommendation(
            improved_score, improved_confidence, market_regime
        )
        
        # 5. Update the result
        improved_result = base_result.copy()
        improved_result['aggregated_analysis'] = {
            'overall_score': improved_score,
            'overall_confidence': improved_confidence,
            'recommendation': recommendation,
            'market_regime': market_regime,
            'timeframe_alignment': timeframe_signals['alignment'],
            'reasoning': self._generate_improved_reasoning(
                agent_results, market_regime, timeframe_signals, recommendation
            )
        }
        
        return improved_result
    
    def _check_market_regime(self, ticker: str) -> str:
        """Check current market regime"""
        try:
            # Get SPY data as market proxy
            spy = yf.download('SPY', period='3mo', progress=False)
            if len(spy) < 20:
                return 'unknown'
            
            # Calculate regime indicators
            sma20 = spy['Close'].rolling(20).mean()
            sma50 = spy['Close'].rolling(50).mean() if len(spy) >= 50 else sma20
            
            # Current position
            current_price = spy['Close'].iloc[-1]
            
            # Volatility
            returns = spy['Close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Determine regime
            if current_price > sma20.iloc[-1] and sma20.iloc[-1] > sma50.iloc[-1]:
                if volatility < returns.std() * 0.8:
                    return 'bullish_calm'
                else:
                    return 'bullish_volatile'
            elif current_price < sma20.iloc[-1] and sma20.iloc[-1] < sma50.iloc[-1]:
                if volatility < returns.std() * 0.8:
                    return 'bearish_calm'
                else:
                    return 'bearish_volatile'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.warning(f"Market regime check failed: {e}")
            return 'unknown'
    
    def _get_timeframe_signals(self, ticker: str) -> Dict[str, Any]:
        """Get signals from multiple timeframes"""
        try:
            # Get data for different periods
            data_1mo = yf.download(ticker, period='1mo', progress=False)
            data_3mo = yf.download(ticker, period='3mo', progress=False)
            
            signals = {
                'short_term': 'neutral',
                'medium_term': 'neutral',
                'alignment': 0.0
            }
            
            if len(data_1mo) >= 10 and len(data_3mo) >= 20:
                # Short term (1 month)
                sma10 = data_1mo['Close'].rolling(10).mean()
                if data_1mo['Close'].iloc[-1] > sma10.iloc[-1]:
                    signals['short_term'] = 'bullish'
                else:
                    signals['short_term'] = 'bearish'
                
                # Medium term (3 months)
                sma20 = data_3mo['Close'].rolling(20).mean()
                if data_3mo['Close'].iloc[-1] > sma20.iloc[-1]:
                    signals['medium_term'] = 'bullish'
                else:
                    signals['medium_term'] = 'bearish'
                
                # Calculate alignment
                if signals['short_term'] == signals['medium_term']:
                    signals['alignment'] = 1.0
                else:
                    signals['alignment'] = 0.0
                    
            return signals
            
        except Exception as e:
            self.logger.warning(f"Timeframe analysis failed: {e}")
            return {'short_term': 'neutral', 'medium_term': 'neutral', 'alignment': 0.0}
    
    def _calculate_improved_metrics(self, agent_results: Dict[str, Any], 
                                  market_regime: str, 
                                  timeframe_signals: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate improved score and confidence"""
        
        # Get weighted scores from agents
        weighted_score = 0.0
        total_weight = 0.0
        agent_scores = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 0.0)
                weight = self.agent_weights.get(agent_name, 0.1)
                
                # Adjust weight based on confidence
                adjusted_weight = weight * confidence
                weighted_score += score * adjusted_weight
                total_weight += adjusted_weight
                
                agent_scores.append(score)
        
        # Calculate base score
        base_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Apply regime adjustments
        regime_multiplier = 1.0
        if market_regime == 'bearish_volatile':
            regime_multiplier = 0.7  # More conservative in bad markets
        elif market_regime == 'bullish_calm':
            regime_multiplier = 1.1  # More aggressive in good markets
        
        # Apply timeframe alignment bonus
        alignment_bonus = timeframe_signals['alignment'] * 0.1
        
        # Calculate final score
        final_score = np.clip(base_score * regime_multiplier + alignment_bonus, -1, 1)
        
        # Calculate confidence based on agreement
        if agent_scores:
            score_std = np.std(agent_scores)
            agreement = 1.0 - min(score_std * 2, 1.0)
            
            # Higher confidence when agents agree and timeframes align
            confidence = min(0.95, agreement * 0.7 + timeframe_signals['alignment'] * 0.3)
            
            # Reduce confidence in volatile regimes
            if 'volatile' in market_regime:
                confidence *= 0.85
        else:
            confidence = 0.0
        
        return final_score, confidence
    
    def _generate_improved_recommendation(self, score: float, confidence: float, 
                                        market_regime: str) -> str:
        """Generate recommendation with regime awareness"""
        
        # Adjust thresholds based on market regime
        buy_threshold = 0.3
        strong_buy_threshold = 0.6
        sell_threshold = -0.3
        strong_sell_threshold = -0.6
        
        if 'bearish' in market_regime:
            # More conservative in bearish markets
            buy_threshold = 0.4
            strong_buy_threshold = 0.7
        elif 'bullish' in market_regime:
            # More aggressive in bullish markets
            buy_threshold = 0.2
            strong_buy_threshold = 0.5
        
        # Only make strong recommendations with high confidence
        if confidence < 0.7:
            if score > 0.1:
                return 'HOLD'  # Positive but low confidence
            else:
                return 'HOLD'  # Default to HOLD when uncertain
        
        # High confidence recommendations
        if score >= strong_buy_threshold:
            return 'STRONG_BUY'
        elif score >= buy_threshold:
            return 'BUY'
        elif score <= strong_sell_threshold:
            return 'STRONG_SELL'
        elif score <= sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_improved_reasoning(self, agent_results: Dict[str, Any],
                                   market_regime: str,
                                   timeframe_signals: Dict[str, Any],
                                   recommendation: str) -> str:
        """Generate improved reasoning"""
        
        reasons = []
        
        # Market regime
        regime_text = market_regime.replace('_', ' ').title()
        reasons.append(f"Market Regime: {regime_text}")
        
        # Timeframe alignment
        if timeframe_signals['alignment'] > 0.8:
            reasons.append("Strong timeframe alignment")
        elif timeframe_signals['alignment'] < 0.2:
            reasons.append("Weak timeframe alignment")
        
        # Agent consensus
        bullish_agents = sum(1 for r in agent_results.values() 
                           if isinstance(r, dict) and r.get('score', 0) > 0.3)
        bearish_agents = sum(1 for r in agent_results.values() 
                           if isinstance(r, dict) and r.get('score', 0) < -0.3)
        
        if bullish_agents > bearish_agents:
            reasons.append(f"{bullish_agents} agents bullish")
        elif bearish_agents > bullish_agents:
            reasons.append(f"{bearish_agents} agents bearish")
        else:
            reasons.append("Mixed agent signals")
        
        # Add recommendation rationale
        if recommendation in ['BUY', 'STRONG_BUY']:
            reasons.append("Positive momentum with good risk/reward")
        elif recommendation in ['SELL', 'STRONG_SELL']:
            reasons.append("Negative momentum with poor risk/reward")
        else:
            reasons.append("Neutral outlook, waiting for clearer signals")
        
        return ". ".join(reasons)