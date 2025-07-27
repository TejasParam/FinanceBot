"""
Market Filter for High-Accuracy Trading
Only allows trading in favorable market conditions
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time

class MarketFilter:
    """Filters out unfavorable market conditions to improve accuracy"""
    
    def __init__(self):
        self.favorable_regimes = ['trending_up', 'trending_down', 'low_volatility']
        self.unfavorable_regimes = ['high_volatility', 'chaotic', 'news_driven', 'sideways']
        
    def should_trade(self, market_context: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if current conditions are favorable for trading
        
        Returns:
            (should_trade, reason)
        """
        
        # Check market regime
        regime = market_context.get('regime', 'unknown')
        if regime in self.unfavorable_regimes:
            return False, f"Unfavorable market regime: {regime}"
        
        # Check volatility
        volatility = market_context.get('volatility', 0.015)
        if volatility > 0.03:  # 3% daily volatility is too high
            return False, f"Volatility too high: {volatility:.1%}"
        
        # Check VIX if available
        vix = market_context.get('vix', 20)
        if vix > 30:
            return False, f"VIX too high: {vix}"
        
        # Check trend clarity
        trend = market_context.get('trend', 'unknown')
        trend_strength = market_context.get('trend_strength', 0)
        if trend == 'sideways' and trend_strength < 0.3:
            return False, "No clear trend direction"
        
        # Check time of day (avoid first/last 30 minutes)
        # Temporarily disabled for backtesting
        # if not self._is_good_trading_time():
        #     return False, "Outside optimal trading hours"
        
        # Check agent agreement
        agent_agreement = self._check_agent_agreement(analysis)
        if agent_agreement < 0.6:  # Less than 60% of agents agree
            return False, f"Low agent agreement: {agent_agreement:.0%}"
        
        # Check confidence levels
        overall_confidence = analysis.get('aggregated_analysis', {}).get('overall_confidence', 0)
        if overall_confidence < 0.70:  # Reasonable confidence threshold
            return False, f"Confidence too low: {overall_confidence:.0%}"
        
        # All checks passed
        return True, "All conditions favorable"
    
    def get_dynamic_confidence_threshold(self, market_context: Dict[str, Any]) -> float:
        """
        Calculate dynamic confidence threshold based on market conditions
        
        Higher threshold = more selective = higher accuracy
        """
        base_threshold = 0.70
        
        # Volatility adjustment
        volatility = market_context.get('volatility', 0.015)
        if volatility > 0.02:  # High volatility
            base_threshold += 0.10
        elif volatility > 0.025:  # Very high volatility
            base_threshold += 0.15
        
        # Trend adjustment
        trend = market_context.get('trend', 'unknown')
        if trend == 'sideways':
            base_threshold += 0.05  # Need more confidence in choppy markets
        elif trend in ['strong_up', 'strong_down']:
            base_threshold -= 0.05  # Can be slightly less selective in strong trends
        
        # Time of day adjustment
        current_hour = datetime.now().hour
        if current_hour < 10 or current_hour > 15:  # Outside core hours
            base_threshold += 0.05
        
        # Cap at reasonable levels
        return min(0.90, max(0.60, base_threshold))
    
    def _is_good_trading_time(self) -> bool:
        """Check if current time is good for trading"""
        now = datetime.now()
        current_time = now.time()
        
        # Avoid first 30 minutes (9:30-10:00 ET)
        if current_time < time(10, 0):
            return False
        
        # Avoid last 30 minutes (3:30-4:00 ET)
        if current_time > time(15, 30):
            return False
        
        # Avoid lunch hour (12:00-13:00 ET) - often lower volume
        if time(12, 0) <= current_time <= time(13, 0):
            return False
        
        # Check day of week (avoid Mondays and Fridays if possible)
        if now.weekday() in [0, 4]:  # Monday = 0, Friday = 4
            return False
        
        return True
    
    def _check_agent_agreement(self, analysis: Dict[str, Any]) -> float:
        """Calculate percentage of agents that agree on direction"""
        agent_results = analysis.get('agent_results', {})
        
        if not agent_results:
            return 0.0
        
        # Count bullish vs bearish agents
        bullish_count = 0
        bearish_count = 0
        total_valid = 0
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'score' in result:
                score = result['score']
                total_valid += 1
                
                if score > 0.1:
                    bullish_count += 1
                elif score < -0.1:
                    bearish_count += 1
        
        if total_valid == 0:
            return 0.0
        
        # Agreement = majority percentage
        majority = max(bullish_count, bearish_count)
        return majority / total_valid
    
    def get_position_size_multiplier(self, market_context: Dict[str, Any], confidence: float) -> float:
        """
        Calculate position size multiplier based on conditions
        
        Returns multiplier between 0.5 and 1.5
        """
        base_multiplier = 1.0
        
        # Confidence adjustment
        if confidence > 0.85:
            base_multiplier *= 1.2
        elif confidence < 0.70:
            base_multiplier *= 0.8
        
        # Volatility adjustment (inverse - smaller positions in volatile markets)
        volatility = market_context.get('volatility', 0.015)
        if volatility > 0.02:
            base_multiplier *= 0.7
        elif volatility < 0.01:
            base_multiplier *= 1.2
        
        # Trend strength adjustment
        trend_strength = market_context.get('trend_strength', 0.5)
        if trend_strength > 0.7:
            base_multiplier *= 1.1
        elif trend_strength < 0.3:
            base_multiplier *= 0.9
        
        # Cap multiplier
        return min(1.5, max(0.5, base_multiplier))