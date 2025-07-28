"""
Statistical Arbitrage Agent - Renaissance-Style Pairs and Basket Trading
Implements cointegration, mean reversion, and factor-based strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
try:
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    # Create dummy functions
    def coint(x, y):
        # Simple correlation-based test
        corr = np.corrcoef(x, y)[0, 1]
        p_value = 1 - abs(corr)  # Simplified
        return None, p_value
    
    def adfuller(x):
        # Simplified stationarity test
        mean_first_half = np.mean(x[:len(x)//2])
        mean_second_half = np.mean(x[len(x)//2:])
        p_value = abs(mean_first_half - mean_second_half) / np.std(x)
        return None, p_value

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple alternatives
    class LinearRegression:
        def __init__(self):
            self.coef_ = None
        
        def fit(self, X, y):
            # Simple least squares
            X = X.reshape(-1, 1) if len(X.shape) == 1 else X
            self.coef_ = np.array([np.cov(X[:, 0], y)[0, 1] / np.var(X[:, 0])])
            return self
        
        def predict(self, X):
            X = X.reshape(-1, 1) if len(X.shape) == 1 else X
            return X[:, 0] * self.coef_[0]
    
    class PCA:
        def __init__(self, n_components=2):
            self.n_components = n_components
            self.components_ = None
            self.explained_variance_ratio_ = None
        
        def fit_transform(self, X):
            # Simple PCA using SVD
            X_centered = X - np.mean(X, axis=0)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ratio_ = (S ** 2) / np.sum(S ** 2)
            return U[:, :self.n_components] @ np.diag(S[:self.n_components])

import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent
import yfinance as yf

class StatisticalArbitrageAgent(BaseAgent):
    """
    Agent specialized in statistical arbitrage strategies like those used by Renaissance Technologies.
    
    Key strategies:
    - Pairs trading with cointegration
    - Basket trading with PCA
    - Factor-neutral portfolios
    - Cross-sectional mean reversion
    - Lead-lag relationships
    """
    
    def __init__(self):
        super().__init__("StatisticalArbitrage")
        
        # Renaissance-style parameters
        self.lookback_window = 60  # days for analysis
        self.zscore_entry = 2.0    # Enter when zscore exceeds this
        self.zscore_exit = 0.5     # Exit when zscore returns to this
        self.min_half_life = 1     # Minimum half-life in days
        self.max_half_life = 30    # Maximum half-life in days
        
        # Cached data for efficiency
        self.market_data_cache = {}
        self.correlation_matrix = None
        self.cointegration_matrix = None
        
        # Known good pairs/baskets (would be discovered dynamically in production)
        self.known_pairs = [
            ('XLF', 'KRE'),  # Financials vs Regional Banks
            ('XLE', 'USO'),  # Energy sector vs Oil
            ('GLD', 'SLV'),  # Gold vs Silver
            ('EWJ', 'EWG'),  # Japan vs Germany
            ('IWM', 'SPY'),  # Small cap vs Large cap
        ]
        
        # Sector ETFs for basket trading
        self.sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU']
        
    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform statistical arbitrage analysis for the given ticker.
        """
        try:
            # 1. Find cointegrated pairs
            pairs_analysis = self._analyze_pairs_trading(ticker)
            
            # 2. Analyze sector/basket relationships
            basket_analysis = self._analyze_basket_trading(ticker)
            
            # 3. Factor analysis
            factor_analysis = self._analyze_factor_exposure(ticker)
            
            # 4. Lead-lag relationships
            lead_lag_analysis = self._analyze_lead_lag(ticker)
            
            # 5. Cross-sectional mean reversion
            cross_sectional = self._analyze_cross_sectional(ticker)
            
            # Aggregate signals
            signals = []
            confidences = []
            
            # Weight different strategies
            if pairs_analysis['has_opportunity']:
                signals.append(pairs_analysis['signal'])
                confidences.append(pairs_analysis['confidence'])
            
            if basket_analysis['has_opportunity']:
                signals.append(basket_analysis['signal'])
                confidences.append(basket_analysis['confidence'])
            
            if factor_analysis['has_opportunity']:
                signals.append(factor_analysis['signal'])
                confidences.append(factor_analysis['confidence'])
            
            # Calculate overall score
            if signals:
                overall_score = np.average(signals, weights=confidences)
                overall_confidence = np.mean(confidences)
            else:
                overall_score = 0.0
                overall_confidence = 0.0
            
            # Generate reasoning
            reasoning = self._generate_stat_arb_reasoning(
                pairs_analysis, basket_analysis, factor_analysis, 
                lead_lag_analysis, cross_sectional
            )
            
            return {
                'score': overall_score,
                'confidence': overall_confidence,
                'reasoning': reasoning,
                'pairs_trading': pairs_analysis,
                'basket_trading': basket_analysis,
                'factor_analysis': factor_analysis,
                'lead_lag': lead_lag_analysis,
                'cross_sectional': cross_sectional,
                'active_opportunities': len(signals),
                'stat_arb_metrics': {
                    'sharpe_estimate': self._estimate_sharpe(signals, confidences),
                    'expected_holding_days': self._estimate_holding_period(pairs_analysis, basket_analysis),
                    'market_neutral': True,
                    'leverage_recommendation': self._calculate_leverage(overall_confidence)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Statistical arbitrage analysis failed: {e}")
            return {
                'error': f'Statistical arbitrage analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Analysis error: {str(e)}'
            }
    
    def _analyze_pairs_trading(self, ticker: str) -> Dict[str, Any]:
        """Analyze pairs trading opportunities"""
        
        best_pair = None
        best_zscore = 0
        best_confidence = 0
        
        try:
            # Get ticker data
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None or len(ticker_data) < self.lookback_window:
                return {'has_opportunity': False}
            
            # Check known pairs first
            for pair_ticker in [p[1] if p[0] == ticker else p[0] if p[1] == ticker else None 
                               for p in self.known_pairs]:
                if pair_ticker:
                    pair_result = self._test_pair_cointegration(ticker, pair_ticker)
                    if pair_result['is_cointegrated']:
                        if abs(pair_result['current_zscore']) > abs(best_zscore):
                            best_zscore = pair_result['current_zscore']
                            best_pair = pair_ticker
                            best_confidence = pair_result['confidence']
            
            # Find other potential pairs
            if not best_pair:
                candidates = self._find_pair_candidates(ticker)
                for candidate in candidates[:5]:  # Test top 5
                    pair_result = self._test_pair_cointegration(ticker, candidate)
                    if pair_result['is_cointegrated']:
                        if abs(pair_result['current_zscore']) > abs(best_zscore):
                            best_zscore = pair_result['current_zscore']
                            best_pair = candidate
                            best_confidence = pair_result['confidence']
            
            if best_pair and abs(best_zscore) > self.zscore_exit:
                # Determine signal
                if best_zscore > self.zscore_entry:
                    signal = -0.8  # Short the spread (ticker is overvalued vs pair)
                elif best_zscore < -self.zscore_entry:
                    signal = 0.8   # Long the spread (ticker is undervalued vs pair)
                elif best_zscore > 0:
                    signal = -0.3  # Mild short
                else:
                    signal = 0.3   # Mild long
                
                return {
                    'has_opportunity': True,
                    'pair': best_pair,
                    'signal': signal,
                    'confidence': best_confidence,
                    'current_zscore': best_zscore,
                    'entry_threshold': self.zscore_entry,
                    'exit_threshold': self.zscore_exit,
                    'strategy': 'pairs_trading'
                }
            
            return {'has_opportunity': False}
            
        except Exception as e:
            self.logger.error(f"Pairs trading analysis failed: {e}")
            return {'has_opportunity': False, 'error': str(e)}
    
    def _test_pair_cointegration(self, ticker1: str, ticker2: str) -> Dict[str, Any]:
        """Test if two tickers are cointegrated"""
        
        try:
            # Get data for both tickers
            data1 = self._get_market_data(ticker1)
            data2 = self._get_market_data(ticker2)
            
            if data1 is None or data2 is None:
                return {'is_cointegrated': False}
            
            # Align data
            aligned_data = pd.DataFrame({
                ticker1: data1['Close'],
                ticker2: data2['Close']
            }).dropna()
            
            if len(aligned_data) < self.lookback_window:
                return {'is_cointegrated': False}
            
            # Use recent data
            recent_data = aligned_data.iloc[-self.lookback_window:]
            
            # Test for cointegration
            coint_result = coint(recent_data[ticker1], recent_data[ticker2])
            p_value = coint_result[1]
            
            if p_value < 0.05:  # Cointegrated at 95% confidence
                # Calculate spread
                model = LinearRegression()
                X = recent_data[ticker2].values.reshape(-1, 1)
                y = recent_data[ticker1].values
                model.fit(X, y)
                
                spread = y - model.predict(X)
                
                # Check if spread is stationary
                adf_result = adfuller(spread)
                if adf_result[1] < 0.05:  # Stationary spread
                    # Calculate current z-score
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread)
                    current_spread = recent_data[ticker1].iloc[-1] - model.predict([[recent_data[ticker2].iloc[-1]]])[0]
                    current_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
                    
                    # Calculate half-life
                    spread_lag = pd.Series(spread[:-1], index=range(len(spread)-1))
                    spread_diff = pd.Series(spread[1:], index=range(len(spread)-1)) - spread_lag
                    lag_model = LinearRegression()
                    lag_model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
                    half_life = -np.log(2) / lag_model.coef_[0] if lag_model.coef_[0] < 0 else 999
                    
                    # Check if half-life is reasonable
                    if self.min_half_life <= half_life <= self.max_half_life:
                        confidence = min(0.9, 1 - p_value)  # Higher confidence for lower p-value
                        
                        return {
                            'is_cointegrated': True,
                            'p_value': p_value,
                            'current_zscore': current_zscore,
                            'half_life': half_life,
                            'confidence': confidence,
                            'hedge_ratio': model.coef_[0]
                        }
            
            return {'is_cointegrated': False}
            
        except Exception as e:
            self.logger.error(f"Cointegration test failed: {e}")
            return {'is_cointegrated': False, 'error': str(e)}
    
    def _analyze_basket_trading(self, ticker: str) -> Dict[str, Any]:
        """Analyze basket trading opportunities using PCA"""
        
        try:
            # Get sector for the ticker
            ticker_info = yf.Ticker(ticker).info
            ticker_sector = ticker_info.get('sector', 'Unknown')
            
            # Get all sector ETF data
            sector_data = {}
            for etf in self.sector_etfs:
                data = self._get_market_data(etf)
                if data is not None and len(data) >= self.lookback_window:
                    sector_data[etf] = data['Close'].pct_change().dropna()
            
            if len(sector_data) < 5:
                return {'has_opportunity': False}
            
            # Create returns matrix
            returns_df = pd.DataFrame(sector_data).dropna()
            returns_df = returns_df.iloc[-self.lookback_window:]
            
            # Add ticker returns
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None:
                return {'has_opportunity': False}
            
            ticker_returns = ticker_data['Close'].pct_change().dropna()
            ticker_returns = ticker_returns.iloc[-self.lookback_window:]
            
            # Perform PCA
            pca = PCA(n_components=min(5, len(sector_data)))
            factors = pca.fit_transform(returns_df)
            
            # Project ticker onto factors
            ticker_projection = []
            for i in range(pca.n_components_):
                component = pca.components_[i]
                projection = np.dot(ticker_returns.iloc[-len(returns_df):], 
                                  returns_df.dot(component))
                ticker_projection.append(projection)
            
            # Find residual (alpha)
            predicted_return = np.zeros(len(ticker_returns.iloc[-len(returns_df):]))
            for i in range(pca.n_components_):
                predicted_return += ticker_projection[i] * factors[:, i]
            
            residuals = ticker_returns.iloc[-len(returns_df):] - predicted_return
            
            # Calculate z-score of residuals
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            current_residual = residuals.iloc[-1]
            residual_zscore = (current_residual - residual_mean) / residual_std if residual_std > 0 else 0
            
            # Generate signal
            if abs(residual_zscore) > 1.5:
                signal = -np.sign(residual_zscore) * min(0.7, abs(residual_zscore) / 3)
                confidence = min(0.85, pca.explained_variance_ratio_[:3].sum())  # Confidence based on variance explained
                
                return {
                    'has_opportunity': True,
                    'signal': signal,
                    'confidence': confidence,
                    'residual_zscore': residual_zscore,
                    'factors_used': pca.n_components_,
                    'variance_explained': pca.explained_variance_ratio_.sum(),
                    'strategy': 'basket_trading'
                }
            
            return {'has_opportunity': False}
            
        except Exception as e:
            self.logger.error(f"Basket trading analysis failed: {e}")
            return {'has_opportunity': False, 'error': str(e)}
    
    def _analyze_factor_exposure(self, ticker: str) -> Dict[str, Any]:
        """Analyze factor exposures and neutrality opportunities"""
        
        try:
            # Common factors (Fama-French style)
            factor_etfs = {
                'market': 'SPY',
                'size': 'IWM',  # Small cap
                'value': 'IVE',  # Value
                'momentum': 'MTUM',
                'quality': 'QUAL',
                'low_vol': 'USMV'
            }
            
            # Get ticker data
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None:
                return {'has_opportunity': False}
            
            ticker_returns = ticker_data['Close'].pct_change().dropna().iloc[-self.lookback_window:]
            
            # Calculate factor exposures
            exposures = {}
            factor_returns = {}
            
            for factor_name, factor_etf in factor_etfs.items():
                factor_data = self._get_market_data(factor_etf)
                if factor_data is not None:
                    factor_ret = factor_data['Close'].pct_change().dropna().iloc[-self.lookback_window:]
                    
                    # Align data
                    aligned_data = pd.DataFrame({
                        'ticker': ticker_returns,
                        'factor': factor_ret
                    }).dropna()
                    
                    if len(aligned_data) > 30:
                        # Calculate beta
                        cov = np.cov(aligned_data['ticker'], aligned_data['factor'])
                        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                        exposures[factor_name] = beta
                        factor_returns[factor_name] = aligned_data['factor']
            
            # Find over/under exposed factors
            if exposures:
                # Market-neutral alpha
                market_beta = exposures.get('market', 1.0)
                
                # Calculate residual returns after removing factor exposures
                residual_returns = ticker_returns.copy()
                for factor_name, beta in exposures.items():
                    if factor_name in factor_returns:
                        residual_returns -= beta * factor_returns[factor_name]
                
                # Check if there's alpha
                alpha_mean = residual_returns.mean() * 252  # Annualized
                alpha_std = residual_returns.std() * np.sqrt(252)
                sharpe = alpha_mean / alpha_std if alpha_std > 0 else 0
                
                if abs(sharpe) > 0.5:  # Meaningful alpha
                    signal = np.sign(alpha_mean) * min(0.6, abs(sharpe) / 2)
                    confidence = min(0.8, abs(sharpe) / 3)
                    
                    return {
                        'has_opportunity': True,
                        'signal': signal,
                        'confidence': confidence,
                        'alpha_annual': alpha_mean,
                        'alpha_sharpe': sharpe,
                        'factor_exposures': exposures,
                        'market_beta': market_beta,
                        'strategy': 'factor_neutral'
                    }
            
            return {'has_opportunity': False}
            
        except Exception as e:
            self.logger.error(f"Factor analysis failed: {e}")
            return {'has_opportunity': False, 'error': str(e)}
    
    def _analyze_lead_lag(self, ticker: str) -> Dict[str, Any]:
        """Analyze lead-lag relationships"""
        
        try:
            # Find potential leading indicators
            leading_candidates = {
                'futures': ['ES=F', 'NQ=F'],  # Index futures
                'etfs': ['SPY', 'QQQ', 'IWM'],
                'volatility': ['^VIX', 'VXX'],  # ^VIX is the correct ticker
                'sectors': self.sector_etfs
            }
            
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None:
                return {'lead_lag_relationships': []}
            
            ticker_returns = ticker_data['Close'].pct_change().dropna()
            
            relationships = []
            
            for category, candidates in leading_candidates.items():
                for candidate in candidates:
                    if candidate == ticker:
                        continue
                    
                    candidate_data = self._get_market_data(candidate)
                    if candidate_data is None:
                        continue
                    
                    candidate_returns = candidate_data['Close'].pct_change().dropna()
                    
                    # Test different lags
                    for lag in [1, 2, 3, 5]:
                        try:
                            # Align data with lag
                            lagged_data = pd.DataFrame({
                                'ticker': ticker_returns,
                                'leader': candidate_returns.shift(lag)
                            }).dropna()
                            
                            if len(lagged_data) > 50:
                                # Calculate correlation
                                correlation = lagged_data.corr().iloc[0, 1]
                                
                                # Test causality (simplified Granger test)
                                if abs(correlation) > 0.3:
                                    relationships.append({
                                        'leader': candidate,
                                        'lag': lag,
                                        'correlation': correlation,
                                        'category': category
                                    })
                        except:
                            continue
            
            # Sort by absolute correlation
            relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'lead_lag_relationships': relationships[:5],  # Top 5
                'has_predictive_leaders': len(relationships) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Lead-lag analysis failed: {e}")
            return {'lead_lag_relationships': [], 'error': str(e)}
    
    def _analyze_cross_sectional(self, ticker: str) -> Dict[str, Any]:
        """Analyze cross-sectional mean reversion opportunities"""
        
        try:
            # Get peer group (simplified - would use industry classification in production)
            ticker_info = yf.Ticker(ticker).info
            market_cap = ticker_info.get('marketCap', 0)
            
            # Define peer group by market cap
            if market_cap > 200e9:
                peers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'BRK-B', 'JPM']
            elif market_cap > 50e9:
                peers = ['AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CSCO', 'PEP']
            else:
                peers = ['ROKU', 'SQ', 'SNAP', 'PINS', 'DDOG', 'NET', 'COIN', 'HOOD']
            
            # Remove ticker from peers if present
            peers = [p for p in peers if p != ticker]
            
            # Calculate relative performance
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None:
                return {'cross_sectional_score': 0}
            
            ticker_return = float(ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-20] - 1)
            
            peer_returns = []
            for peer in peers:
                peer_data = self._get_market_data(peer)
                if peer_data is not None and len(peer_data) > 20:
                    peer_return = float(peer_data['Close'].iloc[-1] / peer_data['Close'].iloc[-20] - 1)
                    peer_returns.append(peer_return)
            
            if len(peer_returns) >= 3:
                # Calculate z-score of ticker vs peers
                peer_mean = np.mean(peer_returns)
                peer_std = np.std(peer_returns)
                
                if peer_std > 0:
                    z_score = (ticker_return - peer_mean) / peer_std
                    
                    return {
                        'cross_sectional_score': -z_score,  # Negative because we expect reversion
                        'ticker_return': ticker_return,
                        'peer_mean_return': peer_mean,
                        'peer_std_return': peer_std,
                        'z_score': z_score,
                        'peers_analyzed': len(peer_returns)
                    }
            
            return {'cross_sectional_score': 0}
            
        except Exception as e:
            self.logger.error(f"Cross-sectional analysis failed: {e}")
            return {'cross_sectional_score': 0, 'error': str(e)}
    
    def _get_market_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get market data with caching"""
        
        if ticker in self.market_data_cache:
            return self.market_data_cache[ticker]
        
        try:
            data = yf.download(ticker, period=f'{self.lookback_window + 30}d', progress=False)
            if len(data) > 0:
                self.market_data_cache[ticker] = data
                return data
        except:
            pass
        
        return None
    
    def _find_pair_candidates(self, ticker: str) -> List[str]:
        """Find potential pair trading candidates"""
        
        try:
            # Get ticker info
            ticker_info = yf.Ticker(ticker).info
            sector = ticker_info.get('sector', '')
            industry = ticker_info.get('industry', '')
            
            # Get similar companies (simplified)
            if sector == 'Technology':
                candidates = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM']
            elif sector == 'Financial Services':
                candidates = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC']
            elif sector == 'Consumer Cyclical':
                candidates = ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'WMT']
            else:
                # Use sector ETFs as candidates
                candidates = self.sector_etfs
            
            # Remove the ticker itself
            candidates = [c for c in candidates if c != ticker]
            
            # Calculate correlations
            ticker_data = self._get_market_data(ticker)
            if ticker_data is None:
                return []
            
            correlations = []
            for candidate in candidates:
                candidate_data = self._get_market_data(candidate)
                if candidate_data is not None:
                    # Calculate correlation
                    aligned_data = pd.DataFrame({
                        'ticker': ticker_data['Close'],
                        'candidate': candidate_data['Close']
                    }).dropna()
                    
                    if len(aligned_data) > 30:
                        corr = aligned_data.corr().iloc[0, 1]
                        correlations.append((candidate, corr))
            
            # Sort by correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Return top correlated tickers
            return [c[0] for c in correlations if abs(c[1]) > 0.7]
            
        except Exception as e:
            self.logger.error(f"Finding pair candidates failed: {e}")
            return []
    
    def _estimate_sharpe(self, signals: List[float], confidences: List[float]) -> float:
        """Estimate Sharpe ratio for the statistical arbitrage strategies"""
        
        if not signals:
            return 0.0
        
        # Simplified estimation based on signal strength and confidence
        expected_return = np.average(np.abs(signals), weights=confidences) * 0.1  # 10% return per unit signal
        expected_vol = 0.15 / np.sqrt(len(signals))  # Diversification benefit
        
        return expected_return / expected_vol if expected_vol > 0 else 0.0
    
    def _estimate_holding_period(self, pairs_analysis: Dict, basket_analysis: Dict) -> float:
        """Estimate expected holding period in days"""
        
        holding_periods = []
        
        if pairs_analysis.get('has_opportunity'):
            half_life = pairs_analysis.get('half_life', 10)
            holding_periods.append(half_life * 2)  # Typically hold for 2 half-lives
        
        if basket_analysis.get('has_opportunity'):
            holding_periods.append(5)  # Basket trades typically shorter
        
        return np.mean(holding_periods) if holding_periods else 10
    
    def _calculate_leverage(self, confidence: float) -> float:
        """Calculate recommended leverage for stat arb strategies"""
        
        # Renaissance-style leverage calculation
        base_leverage = 5.0  # Base leverage for market-neutral strategies
        
        # Adjust based on confidence
        if confidence > 0.8:
            return base_leverage * 1.5
        elif confidence > 0.6:
            return base_leverage
        else:
            return base_leverage * 0.5
    
    def _generate_stat_arb_reasoning(self, pairs: Dict, basket: Dict, factor: Dict,
                                    lead_lag: Dict, cross_section: Dict) -> str:
        """Generate reasoning for statistical arbitrage signals"""
        
        reasons = []
        
        if pairs.get('has_opportunity'):
            pair = pairs.get('pair', 'unknown')
            zscore = pairs.get('current_zscore', 0)
            reasons.append(f"Pairs trading opportunity with {pair} (z-score: {zscore:.2f})")
        
        if basket.get('has_opportunity'):
            zscore = basket.get('residual_zscore', 0)
            variance = basket.get('variance_explained', 0)
            reasons.append(f"Basket trading signal with {variance:.1%} variance explained (z-score: {zscore:.2f})")
        
        if factor.get('has_opportunity'):
            alpha = factor.get('alpha_annual', 0)
            sharpe = factor.get('alpha_sharpe', 0)
            reasons.append(f"Factor-neutral alpha of {alpha:.1%} annually (Sharpe: {sharpe:.2f})")
        
        if lead_lag.get('has_predictive_leaders'):
            top_leader = lead_lag['lead_lag_relationships'][0]
            reasons.append(f"{top_leader['leader']} leads by {top_leader['lag']} days (corr: {top_leader['correlation']:.2f})")
        
        cs_score = cross_section.get('cross_sectional_score', 0)
        if abs(cs_score) > 0.5:
            direction = "outperforming" if cs_score < 0 else "underperforming"
            reasons.append(f"Cross-sectionally {direction} peers")
        
        if not reasons:
            reasons.append("No statistical arbitrage opportunities detected")
        
        return " | ".join(reasons)