"""
Intermarket Analysis Agent - Analyzes correlations and relationships between markets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import yfinance as yf
from scipy.stats import pearsonr, spearmanr

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
            
            # Generate intermarket score
            score, confidence = self._calculate_intermarket_score(
                correlations, sector_analysis, market_regime, relative_strength, divergences
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
                'intermarket_signals': self._generate_signals(correlations, market_regime)
            }
            
        except Exception as e:
            return {
                'error': f'Intermarket analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'Intermarket analysis error: {str(e)[:100]}'
            }
    
    def _calculate_correlations(self, ticker: str, stock_data: pd.DataFrame, period: str) -> Dict[str, Any]:
        """Calculate correlations with major market indicators"""
        
        correlations = {}
        stock_returns = stock_data['Close'].pct_change().dropna()
        
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
                            
                            # Calculate both Pearson and Spearman correlations
                            pearson_corr, _ = pearsonr(aligned_stock, aligned_market)
                            spearman_corr, _ = spearmanr(aligned_stock, aligned_market)
                            
                            # Rolling correlation for trend
                            rolling_corr = aligned_stock.rolling(20).corr(aligned_market)
                            corr_trend = float(rolling_corr.iloc[-1] - rolling_corr.iloc[-10]) if len(rolling_corr) > 10 else 0
                            
                            category_corrs.append({
                                'symbol': symbol,
                                'pearson': float(pearson_corr),
                                'spearman': float(spearman_corr),
                                'current': float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else pearson_corr,
                                'trend': corr_trend
                            })
                except:
                    continue
            
            if category_corrs:
                # Get average correlation for the category
                avg_corr = np.mean([c['pearson'] for c in category_corrs])
                correlations[category] = {
                    'average': float(avg_corr),
                    'details': category_corrs,
                    'strongest': max(category_corrs, key=lambda x: abs(x['pearson']))
                }
        
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
                    if abs(market['pearson']) > 0.7 and abs(market['current']) < 0.3:
                        divergences.append({
                            'type': 'correlation_breakdown',
                            'market': market['symbol'],
                            'historical_corr': market['pearson'],
                            'current_corr': market['current'],
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
                                    divergences: List[Dict[str, Any]]) -> tuple:
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
