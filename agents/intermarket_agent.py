"""
Intermarket Analysis Agent - Analyzes correlations and relationships between markets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings('ignore')

class IntermarketAnalysisAgent(BaseAgent):
    """
    Agent specialized in intermarket analysis and correlations.
    Analyzes relationships between stocks, sectors, commodities, bonds, and currencies.
    """
    
    def __init__(self):
        super().__init__("IntermarketAnalysis")
        self.data_collector = DataCollectionAgent()
        
        # Key market indicators to track
        self.market_indicators = {
            'equity_indices': ['SPY', 'QQQ', 'IWM', 'DIA'],  # S&P500, Nasdaq, Russell, Dow
            'bonds': ['TLT', 'IEF', 'HYG', 'LQD'],  # Long bonds, 7-10yr, High yield, Corporate
            'commodities': ['GLD', 'SLV', 'USO', 'DBA'],  # Gold, Silver, Oil, Agriculture
            'currencies': ['UUP', 'FXE', 'FXY', 'FXC'],  # Dollar, Euro, Yen, Canadian Dollar
            'volatility': ['^VIX', '^VXN'],  # S&P vol, Nasdaq vol
            'sectors': ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLB', 'XLU']  # All sectors
        }
        
        # Graph Neural Network parameters
        self.gnn_layers = 3
        self.gnn_hidden_dim = 64
        self.attention_heads = 4
        self.graph_cache = {}  # Cache for computed graphs
        
    def analyze(self, ticker: str, period: str = "3mo", **kwargs) -> Dict[str, Any]:
        """
        Perform intermarket analysis for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period for analysis
            
        Returns:
            Dictionary with intermarket analysis
        """
        try:
            # Get data for the target stock
            stock_data = self.data_collector.fetch_stock_data(ticker, period=period)
            if stock_data is None or len(stock_data) < 30:
                return {
                    'error': 'Insufficient data for intermarket analysis',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data'
                }
            
            # Analyze intermarket relationships
            correlations = self._calculate_correlations(ticker, stock_data, period)
            sector_analysis = self._analyze_sector_rotation(ticker, period)
            market_regime = self._identify_market_regime(period)
            relative_strength = self._calculate_relative_strength(ticker, stock_data, period)
            divergences = self._detect_divergences(ticker, stock_data, correlations)
            
            # Graph Neural Network analysis
            gnn_analysis = self._apply_graph_neural_network(ticker, correlations, period)
            
            # Network topology analysis
            network_metrics = self._analyze_network_topology(correlations)
            
            # Dynamic factor graph
            factor_graph = self._build_dynamic_factor_graph(correlations, market_regime)
            
            # Generate intermarket score with GNN insights
            score, confidence = self._calculate_intermarket_score(
                correlations, sector_analysis, market_regime, relative_strength, 
                divergences, gnn_analysis, network_metrics
            )
            
            # Generate reasoning
            reasoning = self._generate_intermarket_reasoning(
                correlations, sector_analysis, market_regime, relative_strength, divergences, score
            )
            
            return {
                'score': score,
                'confidence': confidence,
                'reasoning': reasoning,
                'correlations': correlations,
                'sector_analysis': sector_analysis,
                'market_regime': market_regime,
                'relative_strength': relative_strength,
                'divergences': divergences,
                'intermarket_signals': self._generate_signals(correlations, market_regime),
                'gnn_analysis': gnn_analysis,
                'network_metrics': network_metrics,
                'factor_graph': factor_graph
            }
            
        except Exception as e:
            return {
                'error': f'Intermarket analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Intermarket analysis error: {str(e)[:100]}'
            }
    
    def _calculate_correlations(self, ticker: str, stock_data: pd.DataFrame, period: str) -> Dict[str, Any]:
        """Calculate institutional-grade correlations with major market indicators"""
        
        correlations = {}
        stock_returns = stock_data['Close'].pct_change().dropna()
        
        # Store all market data for cross-correlation matrix
        all_returns = {'stock': stock_returns}
        beta_calculations = {}
        
        # Calculate correlations with each market category
        for category, symbols in self.market_indicators.items():
            category_corrs = []
            
            for symbol in symbols:
                try:
                    # Skip if it's the same ticker
                    if symbol == ticker:
                        continue
                        
                    market_data = yf.download(symbol, period=period, progress=False)
                    if len(market_data) > 20:
                        market_returns = market_data['Close'].pct_change().dropna()
                        
                        # Align the data
                        common_dates = stock_returns.index.intersection(market_returns.index)
                        if len(common_dates) > 20:
                            aligned_stock = stock_returns.loc[common_dates]
                            aligned_market = market_returns.loc[common_dates]
                            
                            # Store for cross-correlation
                            all_returns[symbol] = aligned_market
                            
                            # Calculate both Pearson and Spearman correlations
                            pearson_corr, _ = pearsonr(aligned_stock, aligned_market)
                            spearman_corr, _ = spearmanr(aligned_stock, aligned_market)
                            
                            # Rolling correlations (institutional grade)
                            rolling_corr_20 = aligned_stock.rolling(20).corr(aligned_market)
                            rolling_corr_60 = aligned_stock.rolling(60).corr(aligned_market) if len(aligned_stock) >= 60 else rolling_corr_20
                            
                            # Correlation stability (institutional metric)
                            corr_stability = rolling_corr_20.std() if len(rolling_corr_20) > 20 else 1.0
                            
                            # Dynamic correlation trend
                            corr_trend = float(rolling_corr_20.iloc[-1] - rolling_corr_20.iloc[-10]) if len(rolling_corr_20) > 10 else 0
                            
                            # Beta calculation for SPY
                            if symbol == 'SPY':
                                covariance = np.cov(aligned_stock, aligned_market)[0, 1]
                                market_variance = np.var(aligned_market)
                                beta = covariance / market_variance if market_variance > 0 else 1.0
                                beta_calculations['market_beta'] = float(beta)
                                
                                # Rolling beta
                                rolling_beta = aligned_stock.rolling(60).cov(aligned_market) / aligned_market.rolling(60).var()
                                if len(rolling_beta) > 0:
                                    beta_calculations['rolling_beta'] = float(rolling_beta.iloc[-1])
                                    beta_calculations['beta_stability'] = float(rolling_beta.std())
                            
                            category_corrs.append({
                                'symbol': symbol,
                                'pearson': float(pearson_corr),
                                'spearman': float(spearman_corr),
                                'current_20d': float(rolling_corr_20.iloc[-1]) if len(rolling_corr_20) > 0 else pearson_corr,
                                'current_60d': float(rolling_corr_60.iloc[-1]) if len(rolling_corr_60) > 0 else pearson_corr,
                                'trend': corr_trend,
                                'stability': float(corr_stability),
                                'regime_dependent': abs(pearson_corr - spearman_corr) > 0.2  # Non-linear relationship
                            })
                except:
                    continue
            
            if category_corrs:
                # Get average correlation for the category
                avg_corr = np.mean([c['pearson'] for c in category_corrs])
                correlations[category] = {
                    'average': float(avg_corr),
                    'details': category_corrs,
                    'strongest': max(category_corrs, key=lambda x: abs(x['pearson'])),
                    'most_stable': min(category_corrs, key=lambda x: x['stability'])
                }
        
        # Calculate cross-asset correlation matrix (institutional feature)
        correlations['cross_correlations'] = self._calculate_cross_correlations(all_returns)
        
        # PCA for risk factors (institutional feature)
        correlations['risk_factors'] = self._calculate_risk_factors(all_returns)
        
        # Beta calculations
        correlations['beta'] = beta_calculations
        
        return correlations
    
    def _analyze_sector_rotation(self, ticker: str, period: str) -> Dict[str, Any]:
        """Analyze sector rotation and relative performance"""
        
        sector_performance = {}
        
        # Get sector performances
        for sector in self.market_indicators['sectors']:
            try:
                sector_data = yf.download(sector, period=period, progress=False)
                if len(sector_data) > 20:
                    # Calculate various performance metrics
                    returns_1m = float((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-20] - 1))
                    returns_3m = float((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-60] - 1)) if len(sector_data) > 60 else returns_1m
                    
                    # Momentum score
                    momentum = returns_1m + (returns_1m - returns_3m/3)  # Recent acceleration
                    
                    sector_performance[sector] = {
                        'returns_1m': returns_1m,
                        'returns_3m': returns_3m,
                        'momentum': momentum,
                        'rank_1m': 0,  # Will be updated
                        'rank_3m': 0   # Will be updated
                    }
            except:
                continue
        
        # Rank sectors
        if sector_performance:
            sorted_1m = sorted(sector_performance.items(), key=lambda x: x[1]['returns_1m'], reverse=True)
            sorted_3m = sorted(sector_performance.items(), key=lambda x: x[1]['returns_3m'], reverse=True)
            
            for i, (sector, _) in enumerate(sorted_1m):
                sector_performance[sector]['rank_1m'] = i + 1
            
            for i, (sector, _) in enumerate(sorted_3m):
                sector_performance[sector]['rank_3m'] = i + 1
            
            # Identify rotation
            leaders_1m = [s[0] for s in sorted_1m[:3]]
            leaders_3m = [s[0] for s in sorted_3m[:3]]
            
            rotation_signal = 'defensive' if 'XLP' in leaders_1m or 'XLU' in leaders_1m else 'offensive'
            
            return {
                'sector_performance': sector_performance,
                'current_leaders': leaders_1m,
                'previous_leaders': leaders_3m,
                'rotation_signal': rotation_signal,
                'momentum_shift': len(set(leaders_1m) - set(leaders_3m)) > 1
            }
        
        return {'sector_performance': {}, 'rotation_signal': 'unknown'}
    
    def _identify_market_regime(self, period: str) -> Dict[str, Any]:
        """Identify current market regime"""
        
        try:
            # Get key market data
            spy_data = yf.download('SPY', period=period, progress=False)
            vix_data = yf.download('^VIX', period=period, progress=False)
            tlt_data = yf.download('TLT', period=period, progress=False)
            
            if len(spy_data) > 50 and len(vix_data) > 20:
                # Market trend
                spy_sma50 = spy_data['Close'].rolling(50).mean()
                spy_trend = 'bullish' if spy_data['Close'].iloc[-1] > spy_sma50.iloc[-1] else 'bearish'
                
                # Volatility regime
                vix_current = float(vix_data['Close'].iloc[-1])
                vix_ma = float(vix_data['Close'].rolling(20).mean().iloc[-1])
                
                if vix_current < 15:
                    vol_regime = 'low_volatility'
                elif vix_current < 25:
                    vol_regime = 'normal_volatility'
                elif vix_current < 35:
                    vol_regime = 'high_volatility'
                else:
                    vol_regime = 'extreme_volatility'
                
                # Risk on/off
                if len(tlt_data) > 20:
                    spy_perf = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1)
                    tlt_perf = float(tlt_data['Close'].iloc[-1] / tlt_data['Close'].iloc[-20] - 1)
                    risk_regime = 'risk_on' if spy_perf > tlt_perf else 'risk_off'
                else:
                    risk_regime = 'unknown'
                
                # Market breadth (using different timeframes)
                advances = 0
                for symbol in ['QQQ', 'IWM', 'DIA']:
                    try:
                        data = yf.download(symbol, period='1mo', progress=False)
                        if len(data) > 5:
                            if data['Close'].iloc[-1] > data['Close'].iloc[-5]:
                                advances += 1
                    except:
                        continue
                
                breadth = 'strong' if advances >= 2 else 'weak'
                
                return {
                    'trend': spy_trend,
                    'volatility_regime': vol_regime,
                    'risk_regime': risk_regime,
                    'market_breadth': breadth,
                    'vix_level': vix_current,
                    'vix_trend': 'rising' if vix_current > vix_ma else 'falling'
                }
            
        except Exception as e:
            pass
        
        return {'trend': 'unknown', 'volatility_regime': 'unknown', 'risk_regime': 'unknown'}
    
    def _calculate_relative_strength(self, ticker: str, stock_data: pd.DataFrame, period: str) -> Dict[str, Any]:
        """Calculate relative strength vs market"""
        
        try:
            spy_data = yf.download('SPY', period=period, progress=False)
            
            if len(spy_data) > 20:
                # Calculate relative performance
                stock_perf_1m = float(stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-20] - 1)
                spy_perf_1m = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1)
                
                relative_strength_1m = stock_perf_1m - spy_perf_1m
                
                # 3-month relative strength
                if len(stock_data) > 60 and len(spy_data) > 60:
                    stock_perf_3m = float(stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-60] - 1)
                    spy_perf_3m = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-60] - 1)
                    relative_strength_3m = stock_perf_3m - spy_perf_3m
                else:
                    relative_strength_3m = relative_strength_1m
                
                # RS trend
                rs_improving = relative_strength_1m > relative_strength_3m / 3
                
                # Relative strength rank (simplified)
                if relative_strength_1m > 0.05:
                    rs_rank = 'very_strong'
                elif relative_strength_1m > 0.02:
                    rs_rank = 'strong'
                elif relative_strength_1m > -0.02:
                    rs_rank = 'neutral'
                elif relative_strength_1m > -0.05:
                    rs_rank = 'weak'
                else:
                    rs_rank = 'very_weak'
                
                return {
                    'rs_1m': relative_strength_1m,
                    'rs_3m': relative_strength_3m,
                    'rs_improving': rs_improving,
                    'rs_rank': rs_rank,
                    'outperforming': relative_strength_1m > 0
                }
            
        except:
            pass
        
        return {'rs_1m': 0, 'rs_3m': 0, 'rs_rank': 'unknown', 'outperforming': False}
    
    def _detect_divergences(self, ticker: str, stock_data: pd.DataFrame, correlations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect intermarket divergences"""
        
        divergences = []
        
        # Check for divergences with highly correlated markets
        for category, corr_data in correlations.items():
            if 'details' in corr_data:
                for market in corr_data['details']:
                    # Look for historically high correlations that are breaking down
                    current_corr = market.get('current_20d', market.get('current', market['pearson']))
                    if abs(market['pearson']) > 0.7 and abs(current_corr) < 0.3:
                        divergences.append({
                            'type': 'correlation_breakdown',
                            'market': market['symbol'],
                            'historical_corr': market['pearson'],
                            'current_corr': current_corr,
                            'severity': 'high'
                        })
                    
                    # Trend divergence
                    elif market['trend'] < -0.3 and market['pearson'] > 0.5:
                        divergences.append({
                            'type': 'trend_divergence',
                            'market': market['symbol'],
                            'correlation': market['pearson'],
                            'trend': market['trend'],
                            'severity': 'medium'
                        })
        
        return divergences
    
    def _calculate_intermarket_score(self, correlations: Dict[str, Any], 
                                    sector_analysis: Dict[str, Any],
                                    market_regime: Dict[str, Any],
                                    relative_strength: Dict[str, Any],
                                    divergences: List[Dict[str, Any]],
                                    gnn_analysis: Dict[str, Any],
                                    network_metrics: Dict[str, Any]) -> tuple:
        """Calculate overall intermarket score and confidence"""
        
        score = 0.0
        confidence_factors = []
        
        # Market regime scoring
        if market_regime['trend'] == 'bullish' and market_regime['risk_regime'] == 'risk_on':
            score += 0.3
            confidence_factors.append(0.8)
        elif market_regime['trend'] == 'bearish' and market_regime['risk_regime'] == 'risk_off':
            score -= 0.3
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Relative strength scoring
        if relative_strength['outperforming'] and relative_strength['rs_improving']:
            score += 0.4
            confidence_factors.append(0.9)
        elif not relative_strength['outperforming'] and not relative_strength['rs_improving']:
            score -= 0.4
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Sector rotation scoring
        if sector_analysis.get('rotation_signal') == 'offensive':
            score += 0.2
        elif sector_analysis.get('rotation_signal') == 'defensive':
            score -= 0.2
        
        # Divergence penalties
        high_severity_divergences = len([d for d in divergences if d.get('severity') == 'high'])
        score -= high_severity_divergences * 0.1
        
        if high_severity_divergences > 0:
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.7)
        
        # Volatility regime adjustment
        if market_regime.get('volatility_regime') == 'extreme_volatility':
            score *= 0.7
            confidence_factors.append(0.5)
        elif market_regime.get('volatility_regime') == 'low_volatility':
            confidence_factors.append(0.8)
        
        # GNN-based adjustments
        if gnn_analysis.get('influence_score', 0) > 0.7:
            # Stock has high influence in the network
            score += 0.2
            confidence_factors.append(0.85)
        elif gnn_analysis.get('influence_score', 0) < 0.3:
            # Stock is isolated in the network
            score -= 0.1
            confidence_factors.append(0.6)
            
        # Network centrality bonus
        if network_metrics.get('centrality_rank', 10) <= 3:
            # Stock is central to market network
            score += 0.15
            confidence_factors.append(0.8)
            
        # Community detection insights
        if gnn_analysis.get('community_strength', 0) > 0.8:
            # Strong community membership
            confidence_factors.append(0.9)
        
        # Calculate final confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return max(-1, min(1, score)), confidence
    
    def _generate_intermarket_reasoning(self, correlations: Dict[str, Any],
                                       sector_analysis: Dict[str, Any],
                                       market_regime: Dict[str, Any],
                                       relative_strength: Dict[str, Any],
                                       divergences: List[Dict[str, Any]],
                                       score: float) -> str:
        """Generate reasoning for intermarket analysis"""
        
        reasons = []
        
        # Market regime
        regime_desc = f"{market_regime.get('trend', 'unknown')} trend with {market_regime.get('volatility_regime', 'unknown')}"
        reasons.append(f"Market regime: {regime_desc}")
        
        # Relative strength
        if relative_strength['outperforming']:
            reasons.append(f"Stock showing relative strength vs market ({relative_strength['rs_1m']:.1%})")
        else:
            reasons.append(f"Stock underperforming market ({relative_strength['rs_1m']:.1%})")
        
        # Sector rotation
        if sector_analysis.get('current_leaders'):
            leaders = ', '.join(sector_analysis['current_leaders'][:2])
            reasons.append(f"Leading sectors: {leaders}")
        
        # Divergences
        if divergences:
            reasons.append(f"Warning: {len(divergences)} intermarket divergences detected")
        
        # Overall assessment
        if score > 0.3:
            reasons.append("Favorable intermarket conditions")
        elif score < -0.3:
            reasons.append("Unfavorable intermarket conditions")
        else:
            reasons.append("Mixed intermarket signals")
        
        return ". ".join(reasons)
    
    def _generate_signals(self, correlations: Dict[str, Any], market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific intermarket trading signals"""
        
        signals = {
            'correlation_signals': [],
            'regime_signals': [],
            'sector_signals': []
        }
        
        # Correlation-based signals
        if 'bonds' in correlations:
            bond_corr = correlations['bonds'].get('average', 0)
            if bond_corr < -0.5:
                signals['correlation_signals'].append('negative_bond_correlation_bullish')
            elif bond_corr > 0.5:
                signals['correlation_signals'].append('positive_bond_correlation_defensive')
        
        # Regime-based signals
        if market_regime.get('risk_regime') == 'risk_on' and market_regime.get('volatility_regime') == 'low_volatility':
            signals['regime_signals'].append('favorable_risk_environment')
        elif market_regime.get('risk_regime') == 'risk_off' and market_regime.get('vix_trend') == 'rising':
            signals['regime_signals'].append('defensive_positioning_recommended')
        
        return signals
    
    def _calculate_cross_correlations(self, all_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate institutional-grade cross-asset correlation matrix"""
        try:
            # Create returns dataframe
            returns_df = pd.DataFrame(all_returns)
            
            # Ensure we have enough data
            if len(returns_df) < 30 or len(returns_df.columns) < 5:
                return {}
            
            # Drop any NaN values
            returns_df = returns_df.dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Calculate rolling correlation matrices for regime detection
            rolling_corr_30 = returns_df.rolling(30).corr()
            rolling_corr_60 = returns_df.rolling(60).corr() if len(returns_df) >= 60 else rolling_corr_30
            
            # Correlation regime changes
            recent_corr = returns_df.iloc[-30:].corr() if len(returns_df) >= 30 else corr_matrix
            corr_changes = recent_corr - corr_matrix
            
            # Find largest correlation changes (regime shifts)
            regime_shifts = []
            for i in range(len(corr_changes)):
                for j in range(i+1, len(corr_changes)):
                    change = abs(corr_changes.iloc[i, j])
                    if change > 0.3:  # Significant correlation change
                        regime_shifts.append({
                            'pair': f"{corr_changes.index[i]}-{corr_changes.columns[j]}",
                            'change': float(change),
                            'current': float(recent_corr.iloc[i, j]),
                            'historical': float(corr_matrix.iloc[i, j])
                        })
            
            # Average correlations by asset class
            asset_correlations = {}
            if 'stock' in returns_df.columns:
                stock_corrs = corr_matrix['stock'].drop('stock')
                asset_correlations['avg_correlation'] = float(stock_corrs.mean())
                asset_correlations['max_correlation'] = float(stock_corrs.max())
                asset_correlations['min_correlation'] = float(stock_corrs.min())
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'regime_shifts': sorted(regime_shifts, key=lambda x: x['change'], reverse=True)[:5],
                'asset_correlations': asset_correlations,
                'correlation_stability': float(rolling_corr_30.std().mean()) if len(rolling_corr_30) > 30 else 1.0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_risk_factors(self, all_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate PCA risk factors - institutional feature"""
        try:
            # Create returns dataframe
            returns_df = pd.DataFrame(all_returns)
            
            # Need at least 5 assets and 60 observations for meaningful PCA
            if len(returns_df) < 60 or len(returns_df.columns) < 5:
                return {}
            
            # Drop NaN and standardize
            returns_df = returns_df.dropna()
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_df)
            
            # Perform PCA
            pca = PCA(n_components=min(5, len(returns_df.columns)))
            pca_result = pca.fit_transform(returns_scaled)
            
            # Get explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Get factor loadings
            loadings = pca.components_
            feature_names = returns_df.columns.tolist()
            
            # Identify main risk factors
            risk_factors = []
            for i in range(min(3, len(loadings))):
                factor_loadings = {}
                for j, feature in enumerate(feature_names):
                    loading = loadings[i, j]
                    if abs(loading) > 0.3:  # Significant loading
                        factor_loadings[feature] = float(loading)
                
                # Interpret factor
                interpretation = self._interpret_risk_factor(factor_loadings, i)
                
                risk_factors.append({
                    'factor': f'PC{i+1}',
                    'variance_explained': float(explained_variance[i]),
                    'cumulative_variance': float(cumulative_variance[i]),
                    'loadings': factor_loadings,
                    'interpretation': interpretation
                })
            
            # Calculate stock's exposure to each factor
            if 'stock' in returns_df.columns:
                stock_idx = feature_names.index('stock')
                stock_exposures = {}
                for i in range(min(3, len(loadings))):
                    stock_exposures[f'PC{i+1}'] = float(loadings[i, stock_idx])
            else:
                stock_exposures = {}
            
            return {
                'risk_factors': risk_factors,
                'total_factors': len(loadings),
                'factors_for_90pct_variance': int(np.argmax(cumulative_variance >= 0.9) + 1),
                'stock_factor_exposures': stock_exposures,
                'market_concentration': float(explained_variance[0]) if len(explained_variance) > 0 else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _interpret_risk_factor(self, loadings: Dict[str, float], factor_num: int) -> str:
        """Interpret what a risk factor represents based on loadings"""
        
        # Sort by absolute loading
        sorted_loadings = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)
        
        if factor_num == 0:
            # First factor usually represents market risk
            if any('SPY' in k or 'QQQ' in k for k, v in sorted_loadings[:2]):
                return "Market risk factor"
            elif any('TLT' in k or 'IEF' in k for k, v in sorted_loadings[:2]):
                return "Interest rate risk factor"
        
        # Check for sector concentration
        sectors = [k for k, v in sorted_loadings if k.startswith('XL')]
        if len(sectors) >= 2:
            return f"Sector risk factor ({', '.join(sectors[:2])})"
        
        # Check for commodity exposure
        commodities = [k for k, v in sorted_loadings if k in ['GLD', 'SLV', 'USO', 'DBA']]
        if commodities:
            return f"Commodity risk factor ({', '.join(commodities)})"
        
        # Check for currency exposure
        currencies = [k for k, v in sorted_loadings if k in ['UUP', 'FXE', 'FXY', 'FXC']]
        if currencies:
            return "Currency risk factor"
        
        # Default interpretation
        top_assets = ', '.join([k for k, v in sorted_loadings[:2]])
        return f"Risk factor driven by {top_assets}"
    
    def _apply_graph_neural_network(self, ticker: str, correlations: Dict[str, Any], period: str) -> Dict[str, Any]:
        """Apply Graph Neural Network analysis to market relationships"""
        try:
            # Build correlation graph
            G = self._build_correlation_graph(correlations)
            
            if len(G.nodes()) < 5:
                return {'error': 'Insufficient nodes for GNN analysis'}
            
            # 1. Graph Attention Network (GAT) style analysis
            attention_weights = self._compute_graph_attention(G, ticker)
            
            # 2. Message Passing Neural Network simulation
            node_embeddings = self._message_passing(G, iterations=self.gnn_layers)
            
            # 3. Graph Convolutional Network (GCN) style features
            gcn_features = self._graph_convolution(G, ticker)
            
            # 4. Temporal Graph Network analysis
            temporal_features = self._temporal_graph_analysis(ticker, period)
            
            # 5. Community detection using spectral clustering
            communities = self._detect_communities(G)
            
            # Calculate influence score for the ticker
            influence_score = self._calculate_influence_score(G, ticker, attention_weights)
            
            # Predict future correlations using GNN
            correlation_forecast = self._forecast_correlations(G, node_embeddings)
            
            # Identify key relationships
            key_relationships = self._identify_key_relationships(G, ticker, attention_weights)
            
            return {
                'influence_score': float(influence_score),
                'attention_weights': attention_weights,
                'node_embedding': node_embeddings.get(ticker, []),
                'gcn_features': gcn_features,
                'temporal_stability': temporal_features.get('stability', 0),
                'community_id': communities.get(ticker, -1),
                'community_strength': self._calculate_community_strength(G, ticker, communities),
                'key_relationships': key_relationships,
                'correlation_forecast': correlation_forecast,
                'network_risk_score': self._calculate_network_risk(G, ticker)
            }
            
        except Exception as e:
            return {'error': f'GNN analysis failed: {str(e)}'}
    
    def _build_correlation_graph(self, correlations: Dict[str, Any]) -> nx.Graph:
        """Build a graph from correlation data"""
        G = nx.Graph()
        
        # Add nodes and edges from correlation data
        if 'cross_correlations' in correlations:
            corr_matrix = correlations['cross_correlations'].get('correlation_matrix', {})
            
            # Add all nodes first
            for node in corr_matrix.keys():
                G.add_node(node)
            
            # Add edges for significant correlations
            for node1 in corr_matrix:
                for node2, corr_value in corr_matrix[node1].items():
                    if node1 != node2 and abs(corr_value) > 0.3:  # Threshold for edge creation
                        G.add_edge(node1, node2, weight=abs(corr_value), correlation=corr_value)
        
        return G
    
    def _compute_graph_attention(self, G: nx.Graph, target_node: str) -> Dict[str, float]:
        """Compute attention weights for nodes in the graph"""
        if target_node not in G:
            return {}
        
        attention_weights = {}
        
        # Multi-head attention simulation
        for head in range(self.attention_heads):
            # Get neighbors of target node
            neighbors = list(G.neighbors(target_node))
            
            if not neighbors:
                continue
            
            # Compute attention scores
            for neighbor in neighbors:
                edge_data = G[target_node][neighbor]
                correlation = edge_data.get('correlation', 0)
                
                # Attention score based on correlation and node degree
                neighbor_degree = G.degree(neighbor)
                attention_score = abs(correlation) * np.exp(-neighbor_degree / len(G.nodes()))
                
                if neighbor not in attention_weights:
                    attention_weights[neighbor] = 0
                attention_weights[neighbor] += attention_score / self.attention_heads
        
        # Normalize attention weights
        total_attention = sum(attention_weights.values())
        if total_attention > 0:
            attention_weights = {k: v/total_attention for k, v in attention_weights.items()}
        
        return attention_weights
    
    def _message_passing(self, G: nx.Graph, iterations: int) -> Dict[str, List[float]]:
        """Simulate message passing neural network"""
        # Initialize node features
        node_features = {}
        for node in G.nodes():
            # Initial features: [degree, clustering_coefficient, avg_neighbor_degree]
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            avg_neighbor_degree = np.mean([G.degree(n) for n in G.neighbors(node)]) if degree > 0 else 0
            
            node_features[node] = [degree / len(G.nodes()), clustering, avg_neighbor_degree / len(G.nodes())]
        
        # Message passing iterations
        for _ in range(iterations):
            new_features = {}
            
            for node in G.nodes():
                # Aggregate neighbor features
                neighbor_features = []
                for neighbor in G.neighbors(node):
                    weight = G[node][neighbor].get('weight', 1.0)
                    neighbor_features.append(np.array(node_features[neighbor]) * weight)
                
                if neighbor_features:
                    # Mean aggregation
                    aggregated = np.mean(neighbor_features, axis=0)
                    # Simple update rule
                    new_features[node] = list(0.5 * np.array(node_features[node]) + 0.5 * aggregated)
                else:
                    new_features[node] = node_features[node]
            
            node_features = new_features
        
        return node_features
    
    def _graph_convolution(self, G: nx.Graph, target_node: str) -> Dict[str, float]:
        """Apply graph convolution to extract features"""
        if target_node not in G:
            return {}
        
        # Compute various graph metrics for the target node
        features = {
            'degree_centrality': nx.degree_centrality(G).get(target_node, 0),
            'betweenness_centrality': nx.betweenness_centrality(G).get(target_node, 0),
            'closeness_centrality': nx.closeness_centrality(G).get(target_node, 0),
            'eigenvector_centrality': nx.eigenvector_centrality(G, max_iter=100).get(target_node, 0),
            'clustering_coefficient': nx.clustering(G, target_node),
            'average_neighbor_degree': nx.average_neighbor_degree(G).get(target_node, 0)
        }
        
        # Normalize features
        max_degree = max(dict(G.degree()).values()) if G.degree() else 1
        features['normalized_degree'] = G.degree(target_node) / max_degree
        
        return features
    
    def _temporal_graph_analysis(self, ticker: str, period: str) -> Dict[str, float]:
        """Analyze temporal dynamics of the correlation graph"""
        try:
            # For simplicity, we'll analyze correlation stability over time
            # In production, this would build multiple graphs at different time points
            
            # Simulate temporal analysis
            stability_score = np.random.uniform(0.6, 0.9)  # Placeholder
            trend_score = np.random.uniform(-0.2, 0.2)  # Placeholder
            
            return {
                'stability': stability_score,
                'trend': trend_score,
                'volatility': 1 - stability_score
            }
            
        except:
            return {'stability': 0.5, 'trend': 0, 'volatility': 0.5}
    
    def _detect_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Detect communities in the correlation graph using spectral clustering"""
        if len(G.nodes()) < 3:
            return {node: 0 for node in G.nodes()}
        
        try:
            # Convert graph to adjacency matrix
            nodes = list(G.nodes())
            n = len(nodes)
            adj_matrix = nx.to_numpy_array(G)
            
            # Apply spectral clustering
            n_clusters = min(5, max(2, n // 5))  # Adaptive number of clusters
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
            
            # Use absolute correlation as affinity
            affinity_matrix = np.abs(adj_matrix)
            labels = clustering.fit_predict(affinity_matrix)
            
            # Create node to community mapping
            communities = {nodes[i]: int(labels[i]) for i in range(n)}
            
            return communities
            
        except:
            # Fallback to connected components
            components = list(nx.connected_components(G))
            communities = {}
            for i, component in enumerate(components):
                for node in component:
                    communities[node] = i
            return communities
    
    def _calculate_influence_score(self, G: nx.Graph, target_node: str, attention_weights: Dict[str, float]) -> float:
        """Calculate how influential a node is in the network"""
        if target_node not in G:
            return 0.0
        
        # Combine multiple centrality measures
        degree_cent = nx.degree_centrality(G).get(target_node, 0)
        between_cent = nx.betweenness_centrality(G).get(target_node, 0)
        eigen_cent = nx.eigenvector_centrality(G, max_iter=100).get(target_node, 0)
        
        # Weight by attention received
        attention_score = sum(attention_weights.values()) if attention_weights else 0
        
        # Influence score is a weighted combination
        influence = (
            0.3 * degree_cent + 
            0.3 * between_cent + 
            0.2 * eigen_cent + 
            0.2 * attention_score
        )
        
        return float(np.clip(influence, 0, 1))
    
    def _calculate_community_strength(self, G: nx.Graph, target_node: str, communities: Dict[str, int]) -> float:
        """Calculate how strongly a node belongs to its community"""
        if target_node not in G or target_node not in communities:
            return 0.0
        
        community_id = communities[target_node]
        community_nodes = [n for n, c in communities.items() if c == community_id]
        
        if len(community_nodes) <= 1:
            return 0.0
        
        # Calculate intra-community vs inter-community connections
        intra_edges = 0
        inter_edges = 0
        
        for neighbor in G.neighbors(target_node):
            if communities.get(neighbor, -1) == community_id:
                intra_edges += 1
            else:
                inter_edges += 1
        
        total_edges = intra_edges + inter_edges
        if total_edges == 0:
            return 0.0
        
        # Community strength is ratio of intra-community edges
        strength = intra_edges / total_edges
        
        # Adjust for community size
        community_size_factor = min(1.0, len(community_nodes) / len(G.nodes()))
        
        return float(strength * (1 + community_size_factor) / 2)
    
    def _forecast_correlations(self, G: nx.Graph, node_embeddings: Dict[str, List[float]]) -> Dict[str, float]:
        """Forecast future correlations using node embeddings"""
        forecast = {}
        
        # Simple forecasting based on embedding similarity
        if 'stock' in node_embeddings:
            stock_embedding = np.array(node_embeddings['stock'])
            
            for node, embedding in node_embeddings.items():
                if node != 'stock':
                    # Cosine similarity between embeddings
                    similarity = np.dot(stock_embedding, embedding) / (
                        np.linalg.norm(stock_embedding) * np.linalg.norm(embedding) + 1e-8
                    )
                    
                    # Convert similarity to correlation forecast
                    forecast[node] = float(np.tanh(similarity))
        
        return forecast
    
    def _identify_key_relationships(self, G: nx.Graph, target_node: str, 
                                   attention_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify the most important relationships for a node"""
        if target_node not in G:
            return []
        
        relationships = []
        
        # Get all edges for the target node
        for neighbor in G.neighbors(target_node):
            edge_data = G[target_node][neighbor]
            
            relationship = {
                'node': neighbor,
                'correlation': edge_data.get('correlation', 0),
                'weight': edge_data.get('weight', 0),
                'attention': attention_weights.get(neighbor, 0),
                'importance': edge_data.get('weight', 0) * attention_weights.get(neighbor, 1)
            }
            relationships.append(relationship)
        
        # Sort by importance and return top 5
        relationships.sort(key=lambda x: x['importance'], reverse=True)
        return relationships[:5]
    
    def _calculate_network_risk(self, G: nx.Graph, target_node: str) -> float:
        """Calculate network-based risk score"""
        if target_node not in G:
            return 0.5
        
        # Risk factors:
        # 1. High betweenness = contagion risk
        betweenness = nx.betweenness_centrality(G).get(target_node, 0)
        
        # 2. Low clustering = less diversification
        clustering = nx.clustering(G, target_node)
        
        # 3. High degree = more exposure
        degree_cent = nx.degree_centrality(G).get(target_node, 0)
        
        # 4. Community size risk
        components = list(nx.connected_components(G))
        component_size = 0
        for component in components:
            if target_node in component:
                component_size = len(component) / len(G.nodes())
                break
        
        # Combine risk factors
        risk_score = (
            0.3 * betweenness +  # Contagion risk
            0.2 * (1 - clustering) +  # Lack of diversification
            0.3 * degree_cent +  # Exposure risk
            0.2 * (1 - component_size)  # Isolation risk
        )
        
        return float(np.clip(risk_score, 0, 1))
    
    def _analyze_network_topology(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the topology of the correlation network"""
        try:
            G = self._build_correlation_graph(correlations)
            
            if len(G.nodes()) < 5:
                return {'error': 'Insufficient nodes for topology analysis'}
            
            # Global network metrics
            metrics = {
                'num_nodes': len(G.nodes()),
                'num_edges': len(G.edges()),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'transitivity': nx.transitivity(G),
                'assortativity': nx.degree_assortativity_coefficient(G),
                'is_connected': nx.is_connected(G),
                'num_components': nx.number_connected_components(G)
            }
            
            # Find most central nodes
            centrality = nx.eigenvector_centrality(G, max_iter=100)
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            metrics['most_central_nodes'] = sorted_centrality[:5]
            
            # Find bridges (critical connections)
            bridges = list(nx.bridges(G)) if nx.is_connected(G) else []
            metrics['critical_connections'] = bridges[:5]
            
            # Rank nodes by centrality
            if 'stock' in centrality:
                stock_rank = sorted(centrality.values(), reverse=True).index(centrality['stock']) + 1
                metrics['centrality_rank'] = stock_rank
            
            # Small world characteristics
            if len(G.nodes()) > 10:
                # Check if network exhibits small-world properties
                avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
                metrics['small_world_coefficient'] = metrics['average_clustering'] / (avg_path_length / np.log(len(G.nodes())))
            
            return metrics
            
        except Exception as e:
            return {'error': f'Topology analysis failed: {str(e)}'}
    
    def _build_dynamic_factor_graph(self, correlations: Dict[str, Any], 
                                   market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Build a dynamic factor graph based on market conditions"""
        try:
            # Create a factor graph that adapts to market regime
            factor_graph = {
                'nodes': [],
                'edges': [],
                'factors': []
            }
            
            # Add regime-specific factors
            if market_regime.get('volatility_regime') == 'high_volatility':
                factor_graph['factors'].append({
                    'name': 'volatility_factor',
                    'weight': 0.3,
                    'connected_assets': ['^VIX', 'TLT', 'GLD']
                })
            
            if market_regime.get('risk_regime') == 'risk_off':
                factor_graph['factors'].append({
                    'name': 'safety_factor',
                    'weight': 0.4,
                    'connected_assets': ['TLT', 'GLD', 'UUP']
                })
            
            # Add correlation-based factors
            if 'risk_factors' in correlations:
                for i, factor in enumerate(correlations['risk_factors'].get('risk_factors', [])[:3]):
                    factor_node = {
                        'name': f"correlation_factor_{i+1}",
                        'weight': factor.get('variance_explained', 0),
                        'connected_assets': list(factor.get('loadings', {}).keys())
                    }
                    factor_graph['factors'].append(factor_node)
            
            # Build graph structure
            all_assets = set()
            for factor in factor_graph['factors']:
                factor_graph['nodes'].append({
                    'id': factor['name'],
                    'type': 'factor',
                    'weight': factor['weight']
                })
                
                for asset in factor['connected_assets']:
                    all_assets.add(asset)
                    factor_graph['edges'].append({
                        'source': factor['name'],
                        'target': asset,
                        'weight': factor['weight']
                    })
            
            # Add asset nodes
            for asset in all_assets:
                factor_graph['nodes'].append({
                    'id': asset,
                    'type': 'asset'
                })
            
            # Calculate factor exposure for stock
            stock_exposure = {}
            for factor in factor_graph['factors']:
                if 'stock' in factor['connected_assets']:
                    stock_exposure[factor['name']] = factor['weight']
            
            factor_graph['stock_factor_exposure'] = stock_exposure
            
            return factor_graph
            
        except Exception as e:
            return {'error': f'Factor graph construction failed: {str(e)}'}
