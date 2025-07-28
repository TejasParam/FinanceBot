"""
Technical Analysis Agent for stock analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent specialized in technical analysis of stocks.
    Uses technical indicators to generate buy/sell signals.
    """
    
    def __init__(self):
        super().__init__("TechnicalAnalysis")
        self.data_collector = DataCollectionAgent()
        
        # Advanced signal processing parameters
        self.wavelet_scales = np.arange(1, 128)  # For wavelet analysis
        self.fourier_components = 20  # Top frequency components to analyze
        self.kalman_process_noise = 0.01
        self.kalman_observation_noise = 0.1
        
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
        """Calculate overall technical score from -1 (bearish) to 1 (bullish) - Enhanced with signal processing"""
        score = 0.0
        weight_sum = 0.0
        signals_aligned = 0
        total_signals = 0
        
        # Add advanced signal processing components
        # Dominant cycle analysis
        cycle_period = indicators.get('dominant_cycle', 20)
        cycle_strength = indicators.get('cycle_strength', 0)
        if cycle_strength > 0.7:  # Strong cycle detected
            # Check where we are in the cycle
            current_phase = self._estimate_cycle_phase(indicators, cycle_period)
            if current_phase < 0.25:  # Bottom of cycle
                score += 0.8 * 0.15
                signals_aligned += 0.8
            elif current_phase > 0.75:  # Top of cycle
                score -= 0.8 * 0.15
                signals_aligned -= 0.8
            weight_sum += 0.15
            total_signals += 1
        
        # Kalman filter trend
        kalman_velocity = indicators.get('kalman_velocity', 0)
        if abs(kalman_velocity) > 0.001:  # Significant trend
            score += np.sign(kalman_velocity) * min(1, abs(kalman_velocity) * 100) * 0.2
            signals_aligned += np.sign(kalman_velocity) * 0.7
            weight_sum += 0.2
            total_signals += 1
        
        # Wavelet momentum
        wavelet_momentum = indicators.get('wavelet_momentum', 0)
        if abs(wavelet_momentum) > 0.3:
            score += wavelet_momentum * 0.15
            signals_aligned += wavelet_momentum * 0.6
            weight_sum += 0.15
            total_signals += 1
        
        # Market efficiency (fractal dimension)
        market_efficiency = indicators.get('market_efficiency', 0.5)
        if market_efficiency < 0.4:  # Trending market
            # Boost trend-following signals
            trend_boost = 1.3
        elif market_efficiency > 0.6:  # Mean-reverting market
            # Boost mean-reversion signals
            trend_boost = 0.7
        else:
            trend_boost = 1.0
        
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
        
        # Apply market efficiency adjustment
        score *= trend_boost
        
        # Enhanced: Check signal alignment for higher accuracy
        alignment_ratio = abs(signals_aligned) / total_signals if total_signals > 0 else 0
        
        # Advanced: Consider noise ratio
        noise_ratio = indicators.get('noise_ratio', 0.5)
        if noise_ratio > 0.7:  # High noise environment
            score *= 0.6  # Reduce confidence in noisy markets
        elif noise_ratio < 0.3:  # Clear signals
            score *= 1.2  # Boost confidence
        
        # Only generate strong scores when signals align
        if alignment_ratio < 0.6:  # Signals not well aligned
            score *= 0.5  # Reduce score strength
        elif alignment_ratio > 0.8:  # Strong alignment
            score *= 1.2  # Boost score
        
        # Microstructure pattern adjustment
        microstructure_score = indicators.get('microstructure_score', 0)
        if abs(microstructure_score) > 0.5:
            score += microstructure_score * 0.1
        
        final_score = score / weight_sum if weight_sum > 0 else 0.0
        
        # Cap scores based on alignment and noise
        if alignment_ratio < 0.5 or noise_ratio > 0.8:
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
        """Calculate confidence level based on signal alignment and quality"""
        signals = []
        
        # Advanced signal quality metrics
        noise_ratio = indicators.get('noise_ratio', 0.5)
        cycle_strength = indicators.get('cycle_strength', 0)
        trend_strength = indicators.get('trend_strength', 0)
        
        # Base confidence on signal quality
        signal_quality = (1 - noise_ratio) * 0.4 + cycle_strength * 0.3 + min(1, trend_strength) * 0.3
        signals.append(signal_quality)
        
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
        
        # Advanced indicators confidence
        if indicators.get('microstructure_score', 0) != 0:
            signals.append(0.7)
        
        if indicators.get('kalman_velocity', 0) != 0:
            signals.append(abs(indicators['kalman_velocity']) * 500)
        
        base_confidence = sum(signals) / len(signals) if signals else 0.5
        
        # Adjust for market efficiency
        market_efficiency = indicators.get('market_efficiency', 0.5)
        if market_efficiency < 0.3 or market_efficiency > 0.7:
            # More confident in trending or strongly mean-reverting markets
            base_confidence *= 1.2
        
        return min(0.95, base_confidence)
    
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
        
        # Advanced signal processing insights
        if indicators.get('dominant_cycle', 0) > 0:
            reasoning_parts.append(f"Dominant market cycle: {indicators['dominant_cycle']} periods.")
        
        noise_ratio = indicators.get('noise_ratio', 0.5)
        if noise_ratio > 0.7:
            reasoning_parts.append("High market noise detected - signals less reliable.")
        elif noise_ratio < 0.3:
            reasoning_parts.append("Clear market signals with low noise.")
        
        if indicators.get('microstructure_score', 0) > 0.5:
            patterns = indicators.get('detected_patterns', [])
            if patterns:
                reasoning_parts.append(f"Bullish microstructure: {', '.join(patterns[:2])}.")
        elif indicators.get('microstructure_score', 0) < -0.5:
            patterns = indicators.get('detected_patterns', [])
            if patterns:
                reasoning_parts.append(f"Bearish microstructure: {', '.join(patterns[:2])}.")
        
        market_efficiency = indicators.get('market_efficiency', 0.5)
        if market_efficiency < 0.4:
            reasoning_parts.append("Market showing strong trending behavior.")
        elif market_efficiency > 0.6:
            reasoning_parts.append("Market showing mean-reverting behavior.")
        
        return " ".join(reasoning_parts)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate institutional-grade technical indicators"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        open_price = data['Open']
        
        # Current price
        current_price = float(close.iloc[-1])
        
        # Standard indicators
        rsi = self._calculate_rsi(close)
        macd, macd_signal, macd_histogram = self._calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        
        # Moving Averages
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else sma_20
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        
        # Advanced indicators (institutional grade)
        # 1. VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        vwap_current = float(vwap.iloc[-1])
        
        # 2. Money Flow Index (MFI)
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        # 3. Keltner Channels
        atr = self._calculate_atr(high, low, close)
        kc_upper = ema_12 + (2 * atr)
        kc_lower = ema_12 - (2 * atr)
        kc_position = (current_price - kc_lower) / (kc_upper - kc_lower) if kc_upper > kc_lower else 0.5
        
        # 4. Ichimoku Cloud components
        period_9_high = high.rolling(window=9).max()
        period_9_low = low.rolling(window=9).min()
        tenkan_sen = (period_9_high + period_9_low) / 2
        
        period_26_high = high.rolling(window=26).max()
        period_26_low = low.rolling(window=26).min()
        kijun_sen = (period_26_high + period_26_low) / 2
        
        # 5. Market Profile (simplified)
        price_levels = pd.cut(close, bins=20)
        volume_profile = volume.groupby(price_levels).sum()
        poc_index = volume_profile.idxmax()  # Point of Control
        
        # 6. Order Flow indicators
        delta = close - open_price
        cumulative_delta = delta.cumsum()
        delta_divergence = (cumulative_delta.iloc[-1] - cumulative_delta.iloc[-20]) / 20
        
        # 7. Advanced Signal Processing
        signal_processing_results = self._advanced_signal_processing(close, volume)
        
        # 8. Microstructure patterns
        microstructure = self._detect_microstructure_patterns(close, high, low, volume)
        
        # 9. Entropy and Information Theory metrics
        entropy_metrics = self._calculate_entropy_metrics(close)
        
        # 10. Fractal Dimension
        fractal_dim = self._calculate_fractal_dimension(close)
        
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
            'volume_trend': volume_trend,
            # New institutional indicators
            'vwap': vwap_current,
            'price_vs_vwap': current_price / vwap_current,
            'mfi': float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50,
            'keltner_position': kc_position,
            'tenkan_sen': float(tenkan_sen.iloc[-1]),
            'kijun_sen': float(kijun_sen.iloc[-1]),
            'ichimoku_signal': 'bullish' if current_price > float(kijun_sen.iloc[-1]) else 'bearish',
            'delta_divergence': delta_divergence,
            'atr': atr,
            'atr_ratio': atr / current_price,  # Normalized ATR
            # Advanced signal processing results
            'dominant_cycle': signal_processing_results['dominant_cycle'],
            'cycle_strength': signal_processing_results['cycle_strength'],
            'trend_strength': signal_processing_results['trend_strength'],
            'noise_ratio': signal_processing_results['noise_ratio'],
            'kalman_trend': signal_processing_results['kalman_trend'],
            'kalman_velocity': signal_processing_results['kalman_velocity'],
            'wavelet_momentum': signal_processing_results['wavelet_momentum'],
            # Microstructure
            'microstructure_score': microstructure['score'],
            'detected_patterns': microstructure['patterns'],
            # Entropy metrics
            'price_entropy': entropy_metrics['price_entropy'],
            'volume_entropy': entropy_metrics['volume_entropy'],
            'information_ratio': entropy_metrics['information_ratio'],
            # Fractal analysis
            'fractal_dimension': fractal_dim,
            'market_efficiency': 2 - fractal_dim  # Closer to 1 = more efficient
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
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return float(atr.iloc[-1])
    
    def _advanced_signal_processing(self, prices: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        """Apply advanced signal processing techniques"""
        results = {}
        
        # 1. Fourier Transform for cycle detection
        try:
            # Detrend the data
            detrended = signal.detrend(prices.values)
            
            # Apply FFT
            fft_values = fft(detrended)
            frequencies = fftfreq(len(detrended))
            
            # Find dominant frequency (excluding DC component)
            positive_freq_idx = frequencies > 0
            magnitudes = np.abs(fft_values[positive_freq_idx])
            freqs_positive = frequencies[positive_freq_idx]
            
            dominant_freq_idx = np.argmax(magnitudes)
            dominant_frequency = freqs_positive[dominant_freq_idx]
            dominant_period = 1 / dominant_frequency if dominant_frequency > 0 else len(prices)
            
            # Calculate cycle strength
            total_power = np.sum(magnitudes ** 2)
            dominant_power = magnitudes[dominant_freq_idx] ** 2
            cycle_strength = dominant_power / total_power if total_power > 0 else 0
            
            results['dominant_cycle'] = int(dominant_period)
            results['cycle_strength'] = float(cycle_strength)
            
        except:
            results['dominant_cycle'] = 20
            results['cycle_strength'] = 0.0
        
        # 2. Kalman Filter for trend extraction
        try:
            kalman_trend, kalman_velocity = self._apply_kalman_filter(prices.values)
            results['kalman_trend'] = kalman_trend[-1]
            results['kalman_velocity'] = kalman_velocity[-1]
            results['trend_strength'] = abs(kalman_velocity[-1]) * 1000
        except:
            results['kalman_trend'] = float(prices.iloc[-1])
            results['kalman_velocity'] = 0.0
            results['trend_strength'] = 0.0
        
        # 3. Wavelet Analysis for multi-scale momentum
        try:
            wavelet_coeffs = self._wavelet_analysis(prices.values)
            # Calculate momentum from different scales
            short_term_momentum = np.mean(wavelet_coeffs[:5])
            medium_term_momentum = np.mean(wavelet_coeffs[5:20])
            long_term_momentum = np.mean(wavelet_coeffs[20:])
            
            # Weighted momentum
            wavelet_momentum = (
                short_term_momentum * 0.5 + 
                medium_term_momentum * 0.3 + 
                long_term_momentum * 0.2
            )
            results['wavelet_momentum'] = float(np.tanh(wavelet_momentum * 10))  # Normalize to [-1, 1]
        except:
            results['wavelet_momentum'] = 0.0
        
        # 4. Signal-to-Noise Ratio
        try:
            # Calculate using returns
            returns = prices.pct_change().dropna()
            signal_power = np.var(returns.rolling(20).mean())
            noise_power = np.var(returns - returns.rolling(20).mean())
            snr = signal_power / noise_power if noise_power > 0 else 1
            results['noise_ratio'] = 1 / (1 + snr)  # Convert to noise ratio [0, 1]
        except:
            results['noise_ratio'] = 0.5
        
        return results
    
    def _apply_kalman_filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Kalman filter for trend extraction"""
        n = len(prices)
        
        # State: [position, velocity]
        x = np.zeros((2, n))
        x[:, 0] = [prices[0], 0]
        
        # Covariance matrix
        P = np.eye(2)
        
        # Process noise
        Q = np.array([[self.kalman_process_noise, 0],
                      [0, self.kalman_process_noise]])
        
        # Observation matrix
        H = np.array([[1, 0]])
        
        # Observation noise
        R = np.array([[self.kalman_observation_noise]])
        
        # State transition matrix
        F = np.array([[1, 1],
                      [0, 1]])
        
        for i in range(1, n):
            # Predict
            x_pred = F @ x[:, i-1]
            P_pred = F @ P @ F.T + Q
            
            # Update
            y = prices[i] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            x[:, i] = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_pred
        
        return x[0, :], x[1, :]  # Return position and velocity
    
    def _wavelet_analysis(self, prices: np.ndarray) -> np.ndarray:
        """Perform wavelet analysis using continuous wavelet transform"""
        # Simple wavelet analysis using differences at multiple scales
        coefficients = []
        
        for scale in [1, 2, 4, 8, 16, 32, 64]:
            if scale < len(prices):
                # Calculate difference at this scale
                diff = prices[scale:] - prices[:-scale]
                coeff = np.mean(diff) / scale
                coefficients.append(coeff)
        
        return np.array(coefficients)
    
    def _detect_microstructure_patterns(self, close: pd.Series, high: pd.Series, 
                                       low: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        """Detect microstructure patterns using advanced techniques"""
        patterns = []
        score = 0.0
        
        # 1. Pin bar detection (rejection patterns)
        body = abs(close - close.shift(1))
        upper_wick = high - np.maximum(close, close.shift(1))
        lower_wick = np.minimum(close, close.shift(1)) - low
        
        # Bullish pin bar
        bullish_pin = (lower_wick > 2 * body) & (lower_wick > 2 * upper_wick)
        if bullish_pin.iloc[-5:].any():
            patterns.append('bullish_pin_bar')
            score += 0.3
        
        # Bearish pin bar
        bearish_pin = (upper_wick > 2 * body) & (upper_wick > 2 * lower_wick)
        if bearish_pin.iloc[-5:].any():
            patterns.append('bearish_pin_bar')
            score -= 0.3
        
        # 2. Volume spike analysis
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        volume_z_score = (volume - vol_mean) / vol_std
        
        if volume_z_score.iloc[-1] > 2:
            if close.iloc[-1] > close.iloc[-2]:
                patterns.append('bullish_volume_spike')
                score += 0.2
            else:
                patterns.append('bearish_volume_spike')
                score -= 0.2
        
        # 3. Price acceleration patterns
        returns = close.pct_change()
        acceleration = returns.diff()
        
        if acceleration.iloc[-3:].mean() > 0.001:
            patterns.append('positive_acceleration')
            score += 0.2
        elif acceleration.iloc[-3:].mean() < -0.001:
            patterns.append('negative_acceleration')
            score -= 0.2
        
        # 4. Compression patterns (low volatility before breakout)
        recent_atr = self._calculate_atr(high.iloc[-10:], low.iloc[-10:], close.iloc[-10:])
        historical_atr = self._calculate_atr(high.iloc[-50:-10], low.iloc[-50:-10], close.iloc[-50:-10])
        
        if recent_atr < historical_atr * 0.5:
            patterns.append('volatility_compression')
            # This is neutral but important for breakout prediction
        
        return {
            'patterns': patterns,
            'score': np.clip(score, -1, 1)
        }
    
    def _calculate_entropy_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate entropy-based metrics for market efficiency"""
        returns = prices.pct_change().dropna()
        
        # Shannon entropy of returns
        hist, bin_edges = np.histogram(returns, bins=20, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        price_entropy = -np.sum(hist * np.log2(hist))
        
        # Volume entropy (if available)
        volume_entropy = 0.0  # Placeholder
        
        # Information ratio (simplified)
        # Higher entropy = less predictable = lower information ratio
        max_entropy = np.log2(len(hist))
        information_ratio = 1 - (price_entropy / max_entropy)
        
        return {
            'price_entropy': float(price_entropy),
            'volume_entropy': float(volume_entropy),
            'information_ratio': float(information_ratio)
        }
    
    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Normalize prices to [0, 1]
            normalized = (prices - prices.min()) / (prices.max() - prices.min())
            
            # Create price-time pairs
            n = len(normalized)
            points = np.column_stack([np.arange(n) / n, normalized])
            
            # Box counting at different scales
            scales = [2, 4, 8, 16, 32]
            counts = []
            
            for scale in scales:
                if scale > n / 4:
                    continue
                    
                # Count occupied boxes
                grid_size = 1 / scale
                occupied = set()
                
                for point in points:
                    box_x = int(point[0] / grid_size)
                    box_y = int(point[1] / grid_size)
                    occupied.add((box_x, box_y))
                
                counts.append(len(occupied))
            
            if len(counts) >= 2:
                # Fit log-log relationship
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                
                # Linear regression
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dimension = -slope
            else:
                fractal_dimension = 1.5  # Default for random walk
                
            return float(np.clip(fractal_dimension, 1.0, 2.0))
            
        except:
            return 1.5  # Default fractal dimension
    
    def _estimate_cycle_phase(self, indicators: Dict[str, Any], period: int) -> float:
        """Estimate current phase in the dominant cycle (0 to 1)"""
        # Simplified phase estimation using RSI and price position
        rsi = indicators.get('rsi', 50)
        bb_position = indicators.get('bb_position', 0.5)
        
        # Combine multiple indicators to estimate phase
        # 0 = bottom, 0.5 = middle, 1 = top
        phase = (rsi / 100) * 0.5 + bb_position * 0.5
        
        return float(phase)
