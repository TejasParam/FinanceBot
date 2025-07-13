import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from data_collection import DataCollectionAgent

class EnhancedTechnicalAnalyst:
    """
    Enhanced technical analysis with advanced indicators, market microstructure,
    and intermarket analysis capabilities.
    """
    
    def __init__(self):
        self.data_collector = DataCollectionAgent()
        self.indicator_cache = {}
        self.market_profile_cache = {}
        
    def getData(self, ticker: str) -> List[pd.DataFrame]:
        """Get data using data collection agent"""
        try:
            data = []
            data.append(self.data_collector.fetch_stock_data(ticker=ticker))
            data.append(self.data_collector.fetch_realtime_data(ticker=ticker))
            data.append(self.data_collector.fetch_fundamentals(ticker=ticker))
            return data
        except Exception:
            return [pd.DataFrame()]
    
    def calculate_all_indicators(self, ticker: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive set of technical indicators"""
        if data is None:
            data_list = self.getData(ticker)
            data = data_list[0]
        
        if data.empty:
            return {'error': 'No data available'}
        
        indicators = {}
        
        # Price Action Indicators
        indicators['price_action'] = self._calculate_price_action_indicators(data)
        
        # Trend Indicators
        indicators['trend'] = self._calculate_trend_indicators(data)
        
        # Momentum Indicators
        indicators['momentum'] = self._calculate_momentum_indicators(data)
        
        # Volatility Indicators
        indicators['volatility'] = self._calculate_volatility_indicators(data)
        
        # Volume Indicators
        indicators['volume'] = self._calculate_volume_indicators(data)
        
        # Market Profile
        indicators['market_profile'] = self._calculate_market_profile(data)
        
        # Advanced Patterns
        indicators['patterns'] = self._detect_advanced_patterns(data)
        
        # Intermarket Analysis
        indicators['intermarket'] = self._calculate_intermarket_indicators(ticker)
        
        # Options Flow (if available)
        indicators['options_flow'] = self._analyze_options_flow(ticker)
        
        # Composite Score
        indicators['composite_score'] = self._calculate_composite_score(indicators)
        
        return indicators
    
    def _calculate_price_action_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price action based indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        open_price = data['Open'].values
        
        indicators = {}
        
        # Support and Resistance Levels
        indicators['support_resistance'] = self._calculate_support_resistance(data)
        
        # Pivot Points
        indicators['pivot_points'] = self._calculate_pivot_points(data)
        
        # Price Channels
        indicators['channels'] = {
            'donchian_upper': talib.MAX(high, timeperiod=20)[-1],
            'donchian_lower': talib.MIN(low, timeperiod=20)[-1],
            'keltner_upper': self._calculate_keltner_channels(data)['upper'],
            'keltner_lower': self._calculate_keltner_channels(data)['lower']
        }
        
        # Candlestick Patterns
        indicators['candlestick_patterns'] = self._detect_candlestick_patterns(data)
        
        # Price Action Momentum
        indicators['price_momentum'] = {
            'close_position': (close[-1] - low[-1]) / (high[-1] - low[-1]) if high[-1] != low[-1] else 0.5,
            'daily_range': (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0,
            'gap': (open_price[-1] - close[-2]) / close[-2] if len(close) > 1 else 0,
            'body_size': abs(close[-1] - open_price[-1]) / (high[-1] - low[-1]) if high[-1] != low[-1] else 0
        }
        
        return indicators
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend following indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        indicators = {}
        
        # Moving Averages
        indicators['sma'] = {
            'sma_10': talib.SMA(close, timeperiod=10)[-1],
            'sma_20': talib.SMA(close, timeperiod=20)[-1],
            'sma_50': talib.SMA(close, timeperiod=50)[-1],
            'sma_200': talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else None
        }
        
        indicators['ema'] = {
            'ema_12': talib.EMA(close, timeperiod=12)[-1],
            'ema_26': talib.EMA(close, timeperiod=26)[-1],
            'ema_50': talib.EMA(close, timeperiod=50)[-1]
        }
        
        # MACD
        macd, signal, hist = talib.MACD(close)
        indicators['macd'] = {
            'macd': macd[-1] if len(macd) > 0 else 0,
            'signal': signal[-1] if len(signal) > 0 else 0,
            'histogram': hist[-1] if len(hist) > 0 else 0,
            'crossover': self._detect_macd_crossover(macd, signal)
        }
        
        # ADX
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)[-1]
        indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
        indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
        
        # Supertrend
        indicators['supertrend'] = self._calculate_supertrend(data)
        
        # Ichimoku Cloud
        indicators['ichimoku'] = self._calculate_ichimoku_cloud(data)
        
        # Parabolic SAR
        indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)[-1]
        
        # Linear Regression
        indicators['linear_regression'] = self._calculate_linear_regression(close)
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        indicators = {}
        
        # RSI variants
        indicators['rsi'] = {
            'rsi_14': talib.RSI(close, timeperiod=14)[-1],
            'rsi_9': talib.RSI(close, timeperiod=9)[-1],
            'rsi_25': talib.RSI(close, timeperiod=25)[-1],
            'stoch_rsi': self._calculate_stochastic_rsi(close)
        }
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close)
        indicators['stochastic'] = {
            'k': slowk[-1] if len(slowk) > 0 else 0,
            'd': slowd[-1] if len(slowd) > 0 else 0,
            'signal': self._get_stochastic_signal(slowk[-1], slowd[-1])
        }
        
        # Williams %R
        indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
        
        # CCI
        indicators['cci'] = talib.CCI(high, low, close, timeperiod=20)[-1]
        
        # MFI
        indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
        
        # ROC
        indicators['roc'] = talib.ROC(close, timeperiod=10)[-1]
        
        # Momentum
        indicators['momentum'] = talib.MOM(close, timeperiod=10)[-1]
        
        # Ultimate Oscillator
        indicators['ultimate_oscillator'] = talib.ULTOSC(high, low, close)[-1]
        
        # TSI (True Strength Index)
        indicators['tsi'] = self._calculate_tsi(close)
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        indicators = {}
        
        # ATR
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
        indicators['atr_percent'] = indicators['atr'] / close[-1] * 100
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        indicators['bollinger'] = {
            'upper': upper[-1],
            'middle': middle[-1],
            'lower': lower[-1],
            'width': upper[-1] - lower[-1],
            'percent_b': (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
        }
        
        # Standard Deviation
        indicators['std_dev'] = talib.STDDEV(close, timeperiod=20)[-1]
        
        # Historical Volatility
        indicators['historical_volatility'] = self._calculate_historical_volatility(close)
        
        # Parkinson Volatility
        indicators['parkinson_volatility'] = self._calculate_parkinson_volatility(high, low)
        
        # Garman-Klass Volatility
        indicators['gk_volatility'] = self._calculate_garman_klass_volatility(data)
        
        # Chaikin Volatility
        indicators['chaikin_volatility'] = self._calculate_chaikin_volatility(high, low)
        
        return indicators
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        indicators = {}
        
        # OBV
        indicators['obv'] = talib.OBV(close, volume)[-1]
        
        # Volume SMA
        indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)[-1]
        indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
        
        # Accumulation/Distribution
        indicators['ad'] = talib.AD(high, low, close, volume)[-1]
        
        # Chaikin Money Flow
        indicators['cmf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)[-1]
        
        # Volume Price Trend
        indicators['vpt'] = self._calculate_vpt(close, volume)
        
        # Ease of Movement
        indicators['eom'] = self._calculate_ease_of_movement(high, low, volume)
        
        # Volume Profile
        indicators['volume_profile'] = self._calculate_volume_profile(data)
        
        # VWAP
        indicators['vwap'] = self._calculate_vwap(data)
        
        # Volume Weighted Moving Average
        indicators['vwma'] = self._calculate_vwma(close, volume)
        
        return indicators
    
    def _calculate_market_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market profile indicators"""
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values
        
        profile = {}
        
        # Value Area
        value_area = self._calculate_value_area(data)
        profile['value_area_high'] = value_area['vah']
        profile['value_area_low'] = value_area['val']
        profile['point_of_control'] = value_area['poc']
        
        # Market Structure
        profile['market_structure'] = self._analyze_market_structure(data)
        
        # Volume Nodes
        profile['volume_nodes'] = self._find_volume_nodes(data)
        
        # Time at Price
        profile['time_at_price'] = self._calculate_time_at_price(data)
        
        return profile
    
    def _detect_advanced_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect advanced chart patterns"""
        patterns = {}
        
        # Harmonic Patterns
        patterns['harmonic'] = self._detect_harmonic_patterns(data)
        
        # Elliott Wave
        patterns['elliott_wave'] = self._detect_elliott_waves(data)
        
        # Chart Patterns
        patterns['chart'] = self._detect_chart_patterns(data)
        
        # Candlestick Patterns (using TA-Lib)
        patterns['candlestick'] = self._detect_all_candlestick_patterns(data)
        
        # Volume Patterns
        patterns['volume'] = self._detect_volume_patterns(data)
        
        return patterns
    
    def _calculate_intermarket_indicators(self, ticker: str) -> Dict[str, Any]:
        """Calculate intermarket relationships"""
        indicators = {}
        
        try:
            # Get correlated assets
            correlations = self._calculate_correlations(ticker)
            indicators['correlations'] = correlations
            
            # Dollar strength impact
            indicators['dollar_impact'] = self._analyze_dollar_impact(ticker)
            
            # Bond yield impact
            indicators['yield_impact'] = self._analyze_yield_impact(ticker)
            
            # Commodity relationships
            indicators['commodity_impact'] = self._analyze_commodity_impact(ticker)
            
            # Sector rotation
            indicators['sector_rotation'] = self._analyze_sector_rotation(ticker)
            
        except Exception as e:
            indicators['error'] = f"Intermarket analysis failed: {str(e)}"
        
        return indicators
    
    def _analyze_options_flow(self, ticker: str) -> Dict[str, Any]:
        """Analyze options flow for sentiment"""
        try:
            stock = yf.Ticker(ticker)
            options_dates = stock.options
            
            if not options_dates:
                return {'available': False}
            
            # Get nearest expiration
            nearest_date = options_dates[0]
            opt_chain = stock.option_chain(nearest_date)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Calculate Put/Call ratio
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 1
            
            # Calculate skew
            call_iv = calls['impliedVolatility'].mean()
            put_iv = puts['impliedVolatility'].mean()
            iv_skew = put_iv - call_iv
            
            # Unusual options activity
            unusual_activity = self._detect_unusual_options_activity(calls, puts)
            
            return {
                'available': True,
                'put_call_ratio': pc_ratio,
                'iv_skew': iv_skew,
                'call_volume': total_call_volume,
                'put_volume': total_put_volume,
                'unusual_activity': unusual_activity,
                'sentiment': 'bullish' if pc_ratio < 0.7 else 'bearish' if pc_ratio > 1.3 else 'neutral'
            }
            
        except Exception:
            return {'available': False}
    
    def _calculate_composite_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite technical score"""
        scores = []
        weights = []
        
        # Trend score
        trend_score = self._calculate_trend_score(indicators.get('trend', {}))
        scores.append(trend_score)
        weights.append(0.3)
        
        # Momentum score
        momentum_score = self._calculate_momentum_score(indicators.get('momentum', {}))
        scores.append(momentum_score)
        weights.append(0.25)
        
        # Volume score
        volume_score = self._calculate_volume_score(indicators.get('volume', {}))
        scores.append(volume_score)
        weights.append(0.2)
        
        # Volatility score
        volatility_score = self._calculate_volatility_score(indicators.get('volatility', {}))
        scores.append(volatility_score)
        weights.append(0.15)
        
        # Pattern score
        pattern_score = self._calculate_pattern_score(indicators.get('patterns', {}))
        scores.append(pattern_score)
        weights.append(0.1)
        
        # Calculate weighted average
        total_weight = sum(weights)
        composite_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Determine signal strength
        if composite_score > 0.6:
            signal = 'STRONG BUY'
        elif composite_score > 0.2:
            signal = 'BUY'
        elif composite_score < -0.6:
            signal = 'STRONG SELL'
        elif composite_score < -0.2:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        
        return {
            'score': composite_score,
            'signal': signal,
            'components': {
                'trend': trend_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score,
                'patterns': pattern_score
            }
        }
    
    # Helper methods for advanced calculations
    
    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        # Find local peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(window, len(high) - window):
            if high[i] == max(high[i-window:i+window+1]):
                peaks.append(high[i])
            if low[i] == min(low[i-window:i+window+1]):
                troughs.append(low[i])
        
        # Cluster nearby levels
        resistance_levels = self._cluster_levels(peaks)
        support_levels = self._cluster_levels(troughs)
        
        # Add psychological levels
        current_price = close[-1]
        psychological_levels = [
            round(current_price / 10) * 10,
            round(current_price / 50) * 50,
            round(current_price / 100) * 100
        ]
        
        return {
            'resistance': sorted(resistance_levels, reverse=True)[:5],
            'support': sorted(support_levels, reverse=True)[:5],
            'psychological': psychological_levels
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points"""
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def _calculate_keltner_channels(self, data: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
        """Calculate Keltner Channels"""
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        ema = talib.EMA(close, timeperiod=period)
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return {
            'upper': upper[-1],
            'middle': ema[-1],
            'lower': lower[-1]
        }
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[str]:
        """Detect candlestick patterns"""
        patterns = []
        
        # Current candle properties
        open_price = data['Open'].iloc[-1]
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        body = abs(close - open_price)
        range_hl = high - low
        
        # Doji
        if body / range_hl < 0.1 and range_hl > 0:
            patterns.append('doji')
        
        # Hammer/Hanging Man
        if body > 0 and (low - min(open_price, close)) / body > 2:
            if close > open_price:
                patterns.append('hammer')
            else:
                patterns.append('hanging_man')
        
        # Shooting Star/Inverted Hammer
        if body > 0 and (max(open_price, close) - high) / body > 2:
            if close < open_price:
                patterns.append('shooting_star')
            else:
                patterns.append('inverted_hammer')
        
        return patterns
    
    def _detect_macd_crossover(self, macd: np.ndarray, signal: np.ndarray) -> str:
        """Detect MACD crossover"""
        if len(macd) < 2 or len(signal) < 2:
            return 'none'
        
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            return 'bullish_crossover'
        elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
            return 'bearish_crossover'
        
        return 'none'
    
    def _calculate_supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
        """Calculate Supertrend indicator"""
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        up = hl_avg - (multiplier * atr)
        dn = hl_avg + (multiplier * atr)
        
        # Initialize
        trend = np.zeros_like(close)
        trend[0] = 1
        
        # Calculate Supertrend
        for i in range(1, len(close)):
            if close[i] <= dn[i]:
                trend[i] = -1
            elif close[i] >= up[i]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]
        
        return {
            'trend': trend[-1],
            'value': up[-1] if trend[-1] == 1 else dn[-1],
            'signal': 'BUY' if trend[-1] == 1 else 'SELL'
        }
    
    def _calculate_ichimoku_cloud(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Ichimoku Cloud components"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Tenkan-sen (Conversion Line) - 9 periods
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line) - 26 periods
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou = close.shift(-26)
        
        return {
            'tenkan': tenkan.iloc[-1] if len(tenkan) > 0 else 0,
            'kijun': kijun.iloc[-1] if len(kijun) > 0 else 0,
            'senkou_a': senkou_a.iloc[-1] if len(senkou_a) > 0 else 0,
            'senkou_b': senkou_b.iloc[-1] if len(senkou_b) > 0 else 0,
            'chikou': chikou.iloc[-1] if len(chikou) > 0 else 0,
            'cloud_color': 'green' if senkou_a.iloc[-1] > senkou_b.iloc[-1] else 'red'
        }
    
    def _calculate_linear_regression(self, prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        """Calculate linear regression channel"""
        if len(prices) < period:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}
        
        x = np.arange(period)
        y = prices[-period:]
        
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'angle': np.degrees(np.arctan(slope)),
            'current_value': slope * (period - 1) + intercept
        }
    
    def _calculate_stochastic_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic RSI"""
        rsi = talib.RSI(close, timeperiod=period)
        
        if len(rsi) < period:
            return 0
        
        rsi_min = talib.MIN(rsi, timeperiod=period)
        rsi_max = talib.MAX(rsi, timeperiod=period)
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        return stoch_rsi[-1] * 100 if len(stoch_rsi) > 0 else 0
    
    def _get_stochastic_signal(self, k: float, d: float) -> str:
        """Get stochastic oscillator signal"""
        if k > 80 and d > 80:
            return 'overbought'
        elif k < 20 and d < 20:
            return 'oversold'
        elif k > d:
            return 'bullish'
        else:
            return 'bearish'
    
    def _calculate_tsi(self, close: np.ndarray, slow: int = 25, fast: int = 13) -> float:
        """Calculate True Strength Index"""
        momentum = np.diff(close)
        
        abs_momentum = np.abs(momentum)
        
        slow_ema_momentum = talib.EMA(momentum, timeperiod=slow)
        slow_ema_abs = talib.EMA(abs_momentum, timeperiod=slow)
        
        fast_ema_momentum = talib.EMA(slow_ema_momentum, timeperiod=fast)
        fast_ema_abs = talib.EMA(slow_ema_abs, timeperiod=fast)
        
        tsi = 100 * (fast_ema_momentum / fast_ema_abs)
        
        return tsi[-1] if len(tsi) > 0 else 0
    
    def _calculate_historical_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Calculate historical volatility"""
        returns = np.diff(np.log(close))
        return np.std(returns[-period:]) * np.sqrt(252) * 100
    
    def _calculate_parkinson_volatility(self, high: np.ndarray, low: np.ndarray, period: int = 20) -> float:
        """Calculate Parkinson volatility"""
        log_hl = np.log(high / low)
        return np.sqrt(252 / (4 * np.log(2))) * np.std(log_hl[-period:]) * 100
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Garman-Klass volatility"""
        high = data['High'].values[-period:]
        low = data['Low'].values[-period:]
        close = data['Close'].values[-period:]
        open_price = data['Open'].values[-period:]
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        gk = np.sqrt(252 / period * (0.5 * np.sum(log_hl**2) - (2*np.log(2) - 1) * np.sum(log_co**2)))
        
        return gk * 100
    
    def _calculate_chaikin_volatility(self, high: np.ndarray, low: np.ndarray, period: int = 10) -> float:
        """Calculate Chaikin Volatility"""
        hl_diff = high - low
        ema = talib.EMA(hl_diff, timeperiod=period)
        
        if len(ema) < period:
            return 0
        
        chaikin_vol = (ema[-1] - ema[-period-1]) / ema[-period-1] * 100
        return chaikin_vol
    
    def _calculate_vpt(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Volume Price Trend"""
        price_change = np.diff(close) / close[:-1]
        vpt = np.cumsum(price_change * volume[1:])
        return vpt[-1] if len(vpt) > 0 else 0
    
    def _calculate_ease_of_movement(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray, period: int = 14) -> float:
        """Calculate Ease of Movement"""
        distance_moved = (high + low) / 2 - np.roll((high + low) / 2, 1)
        emv = distance_moved / (volume / 10000000) / ((high - low))
        emv_ma = talib.SMA(emv[1:], timeperiod=period)
        return emv_ma[-1] if len(emv_ma) > 0 else 0
    
    def _calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Calculate volume profile"""
        close = data['Close'].values
        volume = data['Volume'].values
        
        price_min = data['Low'].min()
        price_max = data['High'].max()
        
        # Create price bins
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins)
        
        # Accumulate volume in each price bin
        for i in range(len(close)):
            bin_idx = np.digitize(close[i], price_bins) - 1
            if 0 <= bin_idx < bins:
                volume_profile[bin_idx] += volume[i]
        
        # Find value area (70% of volume)
        total_volume = np.sum(volume_profile)
        cumsum = np.cumsum(volume_profile)
        
        # Point of Control (highest volume price)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        return {
            'poc': poc_price,
            'profile': list(zip(price_bins[:-1], volume_profile)),
            'current_price_volume': volume_profile[np.digitize(close[-1], price_bins) - 1]
        }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate VWAP"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).sum() / data['Volume'].sum()
    
    def _calculate_vwma(self, close: np.ndarray, volume: np.ndarray, period: int = 20) -> float:
        """Calculate Volume Weighted Moving Average"""
        if len(close) < period:
            return close[-1]
        
        weights = volume[-period:]
        prices = close[-period:]
        
        return np.average(prices, weights=weights)
    
    def _calculate_value_area(self, data: pd.DataFrame, value_area_percent: float = 0.7) -> Dict[str, float]:
        """Calculate Value Area High, Low, and Point of Control"""
        volume_profile = self._calculate_volume_profile(data)
        
        # Implementation would calculate the price range containing 70% of volume
        # For now, return approximation
        return {
            'vah': data['High'].quantile(0.85),
            'val': data['Low'].quantile(0.15),
            'poc': volume_profile['poc']
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> str:
        """Analyze market structure (trending, ranging, etc.)"""
        close = data['Close'].values
        
        if len(close) < 50:
            return 'insufficient_data'
        
        # Calculate ADX
        adx = talib.ADX(data['High'].values, data['Low'].values, close)
        current_adx = adx[-1] if len(adx) > 0 else 0
        
        # Calculate linear regression
        lr = self._calculate_linear_regression(close)
        
        # Determine structure
        if current_adx > 25 and abs(lr['angle']) > 30:
            return 'strong_trend'
        elif current_adx > 20:
            return 'trending'
        elif current_adx < 20 and lr['r_squared'] < 0.3:
            return 'ranging'
        else:
            return 'consolidating'
    
    def _find_volume_nodes(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """Find high volume nodes (price levels with significant volume)"""
        volume_profile = self._calculate_volume_profile(data)
        
        nodes = []
        for price, vol in volume_profile['profile']:
            if vol > np.mean([v for _, v in volume_profile['profile']]) * 1.5:
                nodes.append({'price': price, 'volume': vol})
        
        return sorted(nodes, key=lambda x: x['volume'], reverse=True)[:5]
    
    def _calculate_time_at_price(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate time spent at different price levels"""
        # Simplified implementation
        close = data['Close'].values
        current_price = close[-1]
        
        time_above = np.sum(close > current_price) / len(close)
        time_below = np.sum(close < current_price) / len(close)
        
        return {
            'time_above_current': time_above,
            'time_below_current': time_below,
            'time_at_current': 1 - time_above - time_below
        }
    
    def _detect_harmonic_patterns(self, data: pd.DataFrame) -> List[str]:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        # Simplified implementation - would need complex pattern recognition
        patterns = []
        
        # This would involve finding XABCD points and checking Fibonacci ratios
        # For now, return empty list
        
        return patterns
    
    def _detect_elliott_waves(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Elliott Wave patterns"""
        # Simplified implementation
        return {
            'current_wave': 'unknown',
            'wave_count': 0,
            'trend': 'unknown'
        }
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[str]:
        """Detect chart patterns (head and shoulders, triangles, etc.)"""
        patterns = []
        
        # This would involve complex pattern recognition algorithms
        # For now, detect simple patterns
        
        close = data['Close'].values
        if len(close) > 20:
            # Detect potential double top/bottom
            recent_high = np.max(close[-20:])
            recent_low = np.min(close[-20:])
            
            high_count = np.sum(close[-20:] > recent_high * 0.98)
            low_count = np.sum(close[-20:] < recent_low * 1.02)
            
            if high_count >= 2:
                patterns.append('potential_double_top')
            if low_count >= 2:
                patterns.append('potential_double_bottom')
        
        return patterns
    
    def _detect_all_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Detect all TA-Lib candlestick patterns"""
        open_price = data['Open'].values
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        patterns = {}
        
        # Two Crows
        patterns['two_crows'] = talib.CDL2CROWS(open_price, high, low, close)[-1] != 0
        
        # Three Black Crows
        patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)[-1] != 0
        
        # Three Inside Up/Down
        patterns['three_inside'] = talib.CDL3INSIDE(open_price, high, low, close)[-1] != 0
        
        # Three-Line Strike
        patterns['three_line_strike'] = talib.CDL3LINESTRIKE(open_price, high, low, close)[-1] != 0
        
        # Three Outside Up/Down
        patterns['three_outside'] = talib.CDL3OUTSIDE(open_price, high, low, close)[-1] != 0
        
        # Three Stars In The South
        patterns['three_stars_south'] = talib.CDL3STARSINSOUTH(open_price, high, low, close)[-1] != 0
        
        # Three Advancing White Soldiers
        patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)[-1] != 0
        
        # Doji
        patterns['doji'] = talib.CDLDOJI(open_price, high, low, close)[-1] != 0
        
        # Engulfing
        patterns['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)[-1] != 0
        
        # Hammer
        patterns['hammer'] = talib.CDLHAMMER(open_price, high, low, close)[-1] != 0
        
        # Hanging Man
        patterns['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)[-1] != 0
        
        # Harami
        patterns['harami'] = talib.CDLHARAMI(open_price, high, low, close)[-1] != 0
        
        # Morning Star
        patterns['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)[-1] != 0
        
        # Evening Star
        patterns['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)[-1] != 0
        
        return {k: v for k, v in patterns.items() if v}
    
    def _detect_volume_patterns(self, data: pd.DataFrame) -> List[str]:
        """Detect volume patterns"""
        volume = data['Volume'].values
        close = data['Close'].values
        
        patterns = []
        
        if len(volume) < 20:
            return patterns
        
        # Volume spike
        avg_volume = np.mean(volume[-20:])
        if volume[-1] > avg_volume * 2:
            patterns.append('volume_spike')
        
        # Volume dry up
        if volume[-1] < avg_volume * 0.5:
            patterns.append('volume_dryup')
        
        # Volume trend
        volume_slope = self._calculate_linear_regression(volume[-20:])['slope']
        if volume_slope > 0 and close[-1] > close[-20]:
            patterns.append('volume_price_confirmation')
        elif volume_slope < 0 and close[-1] > close[-20]:
            patterns.append('volume_price_divergence')
        
        return patterns
    
    def _calculate_correlations(self, ticker: str) -> Dict[str, float]:
        """Calculate correlations with major indices and assets"""
        correlations = {}
        
        try:
            # Get ticker data
            ticker_data = yf.download(ticker, period='3mo', progress=False)['Close']
            
            # Major indices
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ',
                'DIA': 'Dow Jones',
                'IWM': 'Russell 2000',
                'VIX': 'Volatility Index'
            }
            
            for symbol, name in indices.items():
                try:
                    index_data = yf.download(symbol, period='3mo', progress=False)['Close']
                    if len(index_data) > 0 and len(ticker_data) > 0:
                        # Align dates
                        common_dates = ticker_data.index.intersection(index_data.index)
                        if len(common_dates) > 20:
                            corr = ticker_data[common_dates].corr(index_data[common_dates])
                            correlations[name] = corr
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return correlations
    
    def _analyze_dollar_impact(self, ticker: str) -> float:
        """Analyze US Dollar impact on stock"""
        try:
            # Get DXY (Dollar Index) data
            dxy_data = yf.download('DX-Y.NYB', period='3mo', progress=False)['Close']
            ticker_data = yf.download(ticker, period='3mo', progress=False)['Close']
            
            if len(dxy_data) > 0 and len(ticker_data) > 0:
                common_dates = ticker_data.index.intersection(dxy_data.index)
                if len(common_dates) > 20:
                    return ticker_data[common_dates].corr(dxy_data[common_dates])
        except Exception:
            pass
        
        return 0.0
    
    def _analyze_yield_impact(self, ticker: str) -> float:
        """Analyze bond yield impact on stock"""
        try:
            # Get 10-year Treasury yield data
            yield_data = yf.download('^TNX', period='3mo', progress=False)['Close']
            ticker_data = yf.download(ticker, period='3mo', progress=False)['Close']
            
            if len(yield_data) > 0 and len(ticker_data) > 0:
                common_dates = ticker_data.index.intersection(yield_data.index)
                if len(common_dates) > 20:
                    return ticker_data[common_dates].corr(yield_data[common_dates])
        except Exception:
            pass
        
        return 0.0
    
    def _analyze_commodity_impact(self, ticker: str) -> Dict[str, float]:
        """Analyze commodity relationships"""
        commodities = {}
        
        try:
            ticker_data = yf.download(ticker, period='3mo', progress=False)['Close']
            
            # Check correlations with major commodities
            commodity_symbols = {
                'GLD': 'Gold',
                'SLV': 'Silver',
                'USO': 'Oil',
                'UNG': 'Natural Gas'
            }
            
            for symbol, name in commodity_symbols.items():
                try:
                    commodity_data = yf.download(symbol, period='3mo', progress=False)['Close']
                    if len(commodity_data) > 0 and len(ticker_data) > 0:
                        common_dates = ticker_data.index.intersection(commodity_data.index)
                        if len(common_dates) > 20:
                            corr = ticker_data[common_dates].corr(commodity_data[common_dates])
                            commodities[name] = corr
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return commodities
    
    def _analyze_sector_rotation(self, ticker: str) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            stock_info = yf.Ticker(ticker).info
            sector = stock_info.get('sector', 'Unknown')
            
            # Get sector ETF
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Industrial': 'XLI',
                'Energy': 'XLE',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
                'Consumer Staples': 'XLP'
            }
            
            sector_etf = sector_etfs.get(sector)
            if sector_etf:
                # Compare sector performance vs market
                spy_data = yf.download('SPY', period='1mo', progress=False)['Close']
                sector_data = yf.download(sector_etf, period='1mo', progress=False)['Close']
                
                if len(spy_data) > 0 and len(sector_data) > 0:
                    spy_return = (spy_data[-1] - spy_data[0]) / spy_data[0]
                    sector_return = (sector_data[-1] - sector_data[0]) / sector_data[0]
                    
                    return {
                        'sector': sector,
                        'sector_performance': sector_return,
                        'market_performance': spy_return,
                        'relative_strength': sector_return - spy_return,
                        'rotation_signal': 'into_sector' if sector_return > spy_return else 'out_of_sector'
                    }
        except Exception:
            pass
        
        return {'sector': 'Unknown', 'rotation_signal': 'unknown'}
    
    def _detect_unusual_options_activity(self, calls: pd.DataFrame, puts: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual options activity"""
        unusual = []
        
        # Check for high volume relative to open interest
        for df, option_type in [(calls, 'call'), (puts, 'put')]:
            high_volume = df[df['volume'] > df['openInterest'] * 2]
            
            for _, row in high_volume.iterrows():
                unusual.append({
                    'type': option_type,
                    'strike': row['strike'],
                    'volume': row['volume'],
                    'open_interest': row['openInterest'],
                    'volume_oi_ratio': row['volume'] / row['openInterest'] if row['openInterest'] > 0 else 0
                })
        
        return sorted(unusual, key=lambda x: x['volume_oi_ratio'], reverse=True)[:5]
    
    def _calculate_trend_score(self, trend_indicators: Dict[str, Any]) -> float:
        """Calculate trend score from indicators"""
        score = 0.0
        count = 0
        
        # Moving average alignment
        sma = trend_indicators.get('sma', {})
        if sma.get('sma_10') and sma.get('sma_20') and sma.get('sma_50'):
            if sma['sma_10'] > sma['sma_20'] > sma['sma_50']:
                score += 1.0
            elif sma['sma_10'] < sma['sma_20'] < sma['sma_50']:
                score -= 1.0
            count += 1
        
        # MACD
        macd = trend_indicators.get('macd', {})
        if macd.get('macd') is not None:
            if macd['macd'] > 0 and macd.get('histogram', 0) > 0:
                score += 0.5
            elif macd['macd'] < 0 and macd.get('histogram', 0) < 0:
                score -= 0.5
            count += 1
        
        # ADX
        adx = trend_indicators.get('adx')
        if adx is not None and adx > 25:
            if trend_indicators.get('plus_di', 0) > trend_indicators.get('minus_di', 0):
                score += 0.5
            else:
                score -= 0.5
            count += 1
        
        # Supertrend
        supertrend = trend_indicators.get('supertrend', {})
        if supertrend.get('signal') == 'BUY':
            score += 1.0
            count += 1
        elif supertrend.get('signal') == 'SELL':
            score -= 1.0
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_momentum_score(self, momentum_indicators: Dict[str, Any]) -> float:
        """Calculate momentum score from indicators"""
        score = 0.0
        count = 0
        
        # RSI
        rsi = momentum_indicators.get('rsi', {})
        if rsi.get('rsi_14') is not None:
            rsi_value = rsi['rsi_14']
            if 30 < rsi_value < 70:
                score += (rsi_value - 50) / 50  # Normalize to -0.4 to 0.4
            elif rsi_value <= 30:
                score += 0.5  # Oversold = bullish
            else:
                score -= 0.5  # Overbought = bearish
            count += 1
        
        # Stochastic
        stoch = momentum_indicators.get('stochastic', {})
        if stoch.get('signal') == 'oversold':
            score += 0.5
            count += 1
        elif stoch.get('signal') == 'overbought':
            score -= 0.5
            count += 1
        
        # MFI
        mfi = momentum_indicators.get('mfi')
        if mfi is not None:
            if mfi < 20:
                score += 0.5
            elif mfi > 80:
                score -= 0.5
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_volume_score(self, volume_indicators: Dict[str, Any]) -> float:
        """Calculate volume score from indicators"""
        score = 0.0
        count = 0
        
        # Volume ratio
        volume_ratio = volume_indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 0.5
            count += 1
        elif volume_ratio < 0.5:
            score -= 0.5
            count += 1
        
        # OBV trend (would need historical OBV to calculate trend)
        # For now, just count it
        if 'obv' in volume_indicators:
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_volatility_score(self, volatility_indicators: Dict[str, Any]) -> float:
        """Calculate volatility score (inverse - lower volatility is better for trends)"""
        score = 0.0
        count = 0
        
        # Bollinger Band position
        bb = volatility_indicators.get('bollinger', {})
        if bb.get('percent_b') is not None:
            percent_b = bb['percent_b']
            if 0.2 < percent_b < 0.8:
                score += 0.5  # Within bands is good
            count += 1
        
        # ATR percentage (lower is better for established trends)
        atr_percent = volatility_indicators.get('atr_percent', 0)
        if atr_percent > 0:
            if atr_percent < 2:
                score += 0.5
            elif atr_percent > 5:
                score -= 0.5
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_pattern_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate pattern score"""
        score = 0.0
        count = 0
        
        # Candlestick patterns
        candlestick = patterns.get('candlestick', {})
        bullish_patterns = ['hammer', 'morning_star', 'three_white_soldiers', 'engulfing']
        bearish_patterns = ['hanging_man', 'evening_star', 'three_black_crows']
        
        for pattern in candlestick:
            if pattern in bullish_patterns:
                score += 1.0
                count += 1
            elif pattern in bearish_patterns:
                score -= 1.0
                count += 1
        
        # Chart patterns
        chart = patterns.get('chart', [])
        if 'potential_double_bottom' in chart:
            score += 0.5
            count += 1
        elif 'potential_double_top' in chart:
            score -= 0.5
            count += 1
        
        return score / count if count > 0 else 0.0