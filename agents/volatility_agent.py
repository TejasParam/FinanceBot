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
            
            # Add order book imbalance detection (institutional feature)
            order_flow_signals = self._detect_order_flow_imbalance(price_data, ticker)
            signals.update(order_flow_signals)
            
            # Add microstructure analysis
            microstructure_signals = self._analyze_microstructure(price_data)
            signals.update(microstructure_signals)
            
            # Adjust score based on order flow
            if order_flow_signals.get('order_imbalance_signal') == 'strong_buy_pressure':
                vol_score += 0.2
                confidence = min(0.9, confidence + 0.1)
            elif order_flow_signals.get('order_imbalance_signal') == 'strong_sell_pressure':
                vol_score -= 0.2
                confidence = min(0.9, confidence + 0.1)
            
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
        
        # Order flow signals
        if 'order_imbalance_signal' in signals:
            imbalance = signals['order_imbalance_signal']
            if 'strong_buy' in imbalance:
                reasons.append("Strong buy pressure detected in order flow")
            elif 'strong_sell' in imbalance:
                reasons.append("Strong sell pressure detected in order flow")
            elif 'moderate_buy' in imbalance:
                reasons.append("Moderate buy pressure in order flow")
            elif 'moderate_sell' in imbalance:
                reasons.append("Moderate sell pressure in order flow")
        
        # Microstructure signals
        if signals.get('institutional_activity') == 'likely':
            reasons.append("Institutional activity detected")
        
        if signals.get('market_depth') == 'shallow':
            reasons.append("Shallow market depth - larger price impact expected")
        elif signals.get('market_depth') == 'deep':
            reasons.append("Deep market liquidity detected")
        
        if signals.get('accumulation_distribution') == 'accumulation':
            reasons.append("Accumulation pattern detected")
        elif signals.get('accumulation_distribution') == 'distribution':
            reasons.append("Distribution pattern detected")
        
        # Overall assessment
        if score > 0.2:
            reasons.append("Volatility conditions favor bullish positions")
        elif score < -0.2:
            reasons.append("Volatility conditions suggest defensive stance")
        else:
            reasons.append("Mixed volatility signals")
        
        return ". ".join(reasons)
    
    def _detect_order_flow_imbalance(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Detect order book imbalance using price/volume dynamics - institutional feature"""
        
        signals = {}
        
        try:
            if 'Volume' not in data.columns:
                return {'order_imbalance_signal': 'unknown'}
            
            close = data['Close']
            volume = data['Volume']
            high = data['High']
            low = data['Low']
            open_price = data['Open']
            
            # 1. Volume-Weighted Price Analysis
            typical_price = (high + low + close) / 3
            
            # Buy vs Sell Volume estimation (using price movement)
            price_change = close - open_price
            
            # Estimate buy/sell volume based on price movement
            buy_volume = pd.Series(index=data.index, dtype=float)
            sell_volume = pd.Series(index=data.index, dtype=float)
            
            for i in range(len(data)):
                if price_change.iloc[i] > 0:
                    # Price went up - more buy volume
                    buy_ratio = 0.5 + (price_change.iloc[i] / (high.iloc[i] - low.iloc[i])) * 0.5 if high.iloc[i] > low.iloc[i] else 0.6
                    buy_volume.iloc[i] = volume.iloc[i] * buy_ratio
                    sell_volume.iloc[i] = volume.iloc[i] * (1 - buy_ratio)
                else:
                    # Price went down - more sell volume
                    sell_ratio = 0.5 + abs(price_change.iloc[i] / (high.iloc[i] - low.iloc[i])) * 0.5 if high.iloc[i] > low.iloc[i] else 0.6
                    sell_volume.iloc[i] = volume.iloc[i] * sell_ratio
                    buy_volume.iloc[i] = volume.iloc[i] * (1 - sell_ratio)
            
            # 2. Order Flow Imbalance Calculation
            # Recent imbalance (5 days)
            recent_buy = buy_volume.iloc[-5:].sum()
            recent_sell = sell_volume.iloc[-5:].sum()
            recent_imbalance = (recent_buy - recent_sell) / (recent_buy + recent_sell) if (recent_buy + recent_sell) > 0 else 0
            
            # Medium-term imbalance (20 days)
            medium_buy = buy_volume.iloc[-20:].sum()
            medium_sell = sell_volume.iloc[-20:].sum()
            medium_imbalance = (medium_buy - medium_sell) / (medium_buy + medium_sell) if (medium_buy + medium_sell) > 0 else 0
            
            signals['order_imbalance_recent'] = float(recent_imbalance)
            signals['order_imbalance_medium'] = float(medium_imbalance)
            
            # 3. Large Order Detection (institutional activity)
            avg_volume = volume.rolling(20).mean()
            large_volume_days = volume > avg_volume * 2
            
            # Check if large volume days had directional bias
            large_buy_days = large_volume_days & (price_change > 0)
            large_sell_days = large_volume_days & (price_change < 0)
            
            signals['large_buy_days_5d'] = int(large_buy_days.iloc[-5:].sum())
            signals['large_sell_days_5d'] = int(large_sell_days.iloc[-5:].sum())
            
            # 4. Price Impact Analysis
            # Higher volume should move price less if there's hidden liquidity
            volume_normalized = volume / avg_volume
            price_impact = abs(close.pct_change()) / volume_normalized
            price_impact = price_impact.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(price_impact) > 10:
                recent_impact = price_impact.iloc[-5:].mean()
                historical_impact = price_impact.iloc[-20:].mean()
                
                # Lower impact = more hidden liquidity
                if recent_impact < historical_impact * 0.7:
                    signals['hidden_liquidity'] = 'high'
                elif recent_impact > historical_impact * 1.3:
                    signals['hidden_liquidity'] = 'low'
                else:
                    signals['hidden_liquidity'] = 'normal'
            
            # 5. Accumulation/Distribution Line trend
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            adl = mfv.cumsum()
            
            # ADL trend
            if len(adl) > 20:
                adl_sma = adl.rolling(10).mean()
                if adl.iloc[-1] > adl_sma.iloc[-1] and adl.iloc[-5] > adl.iloc[-20]:
                    signals['accumulation_distribution'] = 'accumulation'
                elif adl.iloc[-1] < adl_sma.iloc[-1] and adl.iloc[-5] < adl.iloc[-20]:
                    signals['accumulation_distribution'] = 'distribution'
                else:
                    signals['accumulation_distribution'] = 'neutral'
            
            # 6. Generate Order Imbalance Signal
            if recent_imbalance > 0.3 and medium_imbalance > 0.2:
                signals['order_imbalance_signal'] = 'strong_buy_pressure'
            elif recent_imbalance > 0.15:
                signals['order_imbalance_signal'] = 'moderate_buy_pressure'
            elif recent_imbalance < -0.3 and medium_imbalance < -0.2:
                signals['order_imbalance_signal'] = 'strong_sell_pressure'
            elif recent_imbalance < -0.15:
                signals['order_imbalance_signal'] = 'moderate_sell_pressure'
            else:
                signals['order_imbalance_signal'] = 'balanced'
            
            # 7. Smart Money Flow Index
            # First hour vs rest of day volume
            if len(data) > 60:  # Need intraday data ideally
                # Simplified version using daily data
                smart_money_flow = (close.iloc[-1] - open_price.iloc[-1]) * volume.iloc[-1]
                signals['smart_money_direction'] = 'bullish' if smart_money_flow > 0 else 'bearish'
            
            return signals
            
        except Exception as e:
            return {'order_imbalance_signal': 'unknown', 'error': str(e)}
    
    def _analyze_microstructure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure patterns - institutional feature"""
        
        signals = {}
        
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # 1. Bid-Ask Spread Proxy (using high-low)
            hl_spread = (high - low) / close
            avg_spread = hl_spread.rolling(20).mean().iloc[-1]
            recent_spread = hl_spread.iloc[-5:].mean()
            
            signals['spread_proxy'] = float(recent_spread)
            signals['spread_widening'] = recent_spread > avg_spread * 1.2
            
            # 2. Price Efficiency Ratio
            # How much price moves vs how much it could have moved
            net_change = abs(close.iloc[-20] - close.iloc[-1])
            total_movement = abs(close.diff()).iloc[-20:].sum()
            efficiency_ratio = net_change / total_movement if total_movement > 0 else 0
            
            signals['price_efficiency'] = float(efficiency_ratio)
            signals['trending_market'] = efficiency_ratio > 0.3
            
            # 3. Microstructure Noise
            # High frequency reversals indicate noise
            returns = close.pct_change()
            reversals = (returns.shift(1) * returns < 0).sum()
            reversal_rate = reversals / len(returns) if len(returns) > 0 else 0.5
            
            signals['reversal_rate'] = float(reversal_rate)
            signals['noisy_market'] = reversal_rate > 0.6
            
            # 4. Kyle's Lambda (simplified price impact)
            # Price change per unit volume
            if len(data) > 20:
                price_changes = abs(close.pct_change())
                volume_ma = volume.rolling(20).mean()
                normalized_volume = volume / volume_ma
                
                # Remove zeros and infinities
                valid_idx = (normalized_volume > 0.1) & (price_changes < 0.1)
                if valid_idx.sum() > 10:
                    kyle_lambda = (price_changes[valid_idx] / normalized_volume[valid_idx]).mean()
                    signals['price_impact_coefficient'] = float(kyle_lambda)
                    signals['market_depth'] = 'deep' if kyle_lambda < 0.01 else 'shallow' if kyle_lambda > 0.03 else 'normal'
            
            # 5. Quote Stuffing Detection (abnormal volume spikes)
            volume_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
            quote_stuffing_days = (volume_zscore > 3).sum()
            signals['potential_quote_stuffing'] = int(quote_stuffing_days)
            
            # 6. Time-weighted Average Price (TWAP) deviation
            # Large deviations suggest institutional activity
            twap = close.rolling(20).mean()
            vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
            
            if len(data) > 20:
                twap_deviation = (close.iloc[-1] - twap.iloc[-1]) / twap.iloc[-1]
                vwap_deviation = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
                
                signals['twap_deviation'] = float(twap_deviation)
                signals['vwap_deviation'] = float(vwap_deviation)
                
                # Institutional activity detection
                if abs(vwap_deviation) > 0.02 and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                    signals['institutional_activity'] = 'likely'
                else:
                    signals['institutional_activity'] = 'normal'
            
            return signals
            
        except Exception as e:
            return {'microstructure_error': str(e)}