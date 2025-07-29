"""
Risk Management 2.0 Agent - Advanced risk analytics with extreme value theory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from .base_agent import BaseAgent
from data_collection import DataCollectionAgent
import warnings
warnings.filterwarnings('ignore')

class RiskManagementAgent(BaseAgent):
    """
    Advanced Risk Management Agent using institutional-grade risk models.
    
    Features:
    - Extreme Value Theory (EVT) for tail risk
    - Dynamic position sizing with Kelly Criterion
    - Value at Risk (VaR) and Conditional VaR (CVaR)
    - Correlation risk and contagion modeling
    - Regime-dependent risk adjustment
    - Real-time risk monitoring
    - Portfolio optimization with risk constraints
    """
    
    def __init__(self):
        super().__init__("RiskManagement")
        self.data_collector = DataCollectionAgent()
        
        # Risk parameters
        self.risk_free_rate = 0.05  # 5% annual
        self.max_leverage = 3.0
        self.base_position_size = 0.02  # 2% base position
        self.max_position_size = 0.10  # 10% max position
        self.max_portfolio_risk = 0.20  # 20% max portfolio risk
        
        # EVT parameters
        self.evt_threshold_percentile = 95  # For POT method
        self.evt_lookback = 252  # 1 year of data
        
        # Risk model parameters
        self.var_confidence_levels = [0.95, 0.99, 0.999]
        self.stress_scenarios = self._define_stress_scenarios()
        
        # Dynamic risk adjustment
        self.risk_regime = 'normal'
        self.risk_multiplier = 1.0
        
    def analyze(self, ticker: str, portfolio: Optional[Dict[str, float]] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis for a ticker and portfolio.
        
        Args:
            ticker: Stock ticker symbol
            portfolio: Current portfolio holdings {ticker: position_size}
            
        Returns:
            Dictionary with risk metrics and recommendations
        """
        try:
            # Get historical data
            data = self.data_collector.get_historical_data(ticker, period="2y")
            if data is None or len(data) < 100:
                return self._default_risk_response("Insufficient data for risk analysis")
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # 1. Basic risk metrics
            basic_metrics = self._calculate_basic_metrics(returns)
            
            # 2. Extreme Value Theory analysis
            evt_metrics = self._extreme_value_analysis(returns)
            
            # 3. Value at Risk and CVaR
            var_metrics = self._calculate_var_cvar(returns)
            
            # 4. Dynamic volatility modeling (GARCH)
            volatility_forecast = self._forecast_volatility(returns)
            
            # 5. Correlation and contagion risk
            if portfolio:
                correlation_risk = self._analyze_correlation_risk(ticker, portfolio)
            else:
                correlation_risk = {'portfolio_correlation': 0, 'contagion_risk': 0}
            
            # 6. Regime detection and adjustment
            regime_analysis = self._detect_risk_regime(returns, data)
            
            # 7. Position sizing with Kelly Criterion
            position_sizing = self._calculate_position_size(
                returns, basic_metrics, evt_metrics, regime_analysis
            )
            
            # 8. Stress testing
            stress_results = self._run_stress_tests(returns, position_sizing['recommended_size'])
            
            # 9. Portfolio optimization (if portfolio provided)
            if portfolio:
                optimization = self._optimize_portfolio_risk(ticker, portfolio, returns)
            else:
                optimization = None
            
            # 10. Generate risk score and recommendations
            risk_score = self._calculate_risk_score(
                basic_metrics, evt_metrics, var_metrics, 
                volatility_forecast, correlation_risk, regime_analysis
            )
            
            # Generate comprehensive response
            return {
                'score': risk_score['overall_score'],
                'confidence': risk_score['confidence'],
                'reasoning': self._generate_risk_reasoning(
                    basic_metrics, evt_metrics, var_metrics, 
                    regime_analysis, position_sizing
                ),
                'basic_metrics': basic_metrics,
                'extreme_value_metrics': evt_metrics,
                'var_metrics': var_metrics,
                'volatility_forecast': volatility_forecast,
                'correlation_risk': correlation_risk,
                'regime_analysis': regime_analysis,
                'position_sizing': position_sizing,
                'stress_test_results': stress_results,
                'portfolio_optimization': optimization,
                'risk_score': risk_score,
                'risk_limits': self._calculate_risk_limits(risk_score, regime_analysis),
                'real_time_alerts': self._generate_risk_alerts(risk_score, evt_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return self._default_risk_response(f"Risk analysis error: {str(e)}")
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic risk metrics"""
        
        # Volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)
        
        # Risk-adjusted returns
        mean_return = returns.mean() * 252
        sharpe_ratio = (mean_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'annual_volatility': float(annual_vol),
            'downside_volatility': float(downside_vol),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'daily_vol': float(daily_vol),
            'mean_return': float(mean_return)
        }
    
    def _extreme_value_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform Extreme Value Theory analysis"""
        
        # 1. Block Maxima Method (GEV)
        block_maxima = self._block_maxima_analysis(returns)
        
        # 2. Peaks Over Threshold (POT) Method
        pot_analysis = self._peaks_over_threshold(returns)
        
        # 3. Tail index estimation
        tail_index = self._estimate_tail_index(returns)
        
        # 4. Expected Shortfall in extreme scenarios
        extreme_es = self._calculate_extreme_expected_shortfall(returns, pot_analysis)
        
        # 5. Probability of extreme events
        extreme_probs = self._calculate_extreme_probabilities(returns, pot_analysis)
        
        return {
            'block_maxima': block_maxima,
            'pot_analysis': pot_analysis,
            'tail_index': tail_index,
            'extreme_expected_shortfall': extreme_es,
            'extreme_event_probabilities': extreme_probs,
            'tail_risk_score': self._calculate_tail_risk_score(tail_index, extreme_es)
        }
    
    def _block_maxima_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """Analyze using Block Maxima method (GEV distribution)"""
        
        # Create monthly blocks
        monthly_returns = returns.resample('M').apply(lambda x: x.min())  # Focus on losses
        monthly_losses = -monthly_returns.dropna()
        
        if len(monthly_losses) < 12:
            return {'gev_shape': 0, 'gev_location': 0, 'gev_scale': 0}
        
        # Fit GEV distribution
        try:
            params = stats.genextreme.fit(monthly_losses)
            shape, loc, scale = params
            
            # Calculate return levels (e.g., 1-in-100 month event)
            return_level_100 = stats.genextreme.ppf(0.99, shape, loc, scale)
            
            return {
                'gev_shape': float(shape),  # Xi parameter
                'gev_location': float(loc),
                'gev_scale': float(scale),
                'monthly_100_year_loss': float(return_level_100),
                'tail_type': 'heavy' if shape > 0 else 'light' if shape < 0 else 'exponential'
            }
        except:
            return {'gev_shape': 0, 'gev_location': 0, 'gev_scale': 0}
    
    def _peaks_over_threshold(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze using Peaks Over Threshold method (GPD)"""
        
        # Define threshold
        threshold_percentile = self.evt_threshold_percentile
        threshold = np.percentile(-returns, threshold_percentile)
        
        # Get exceedances
        losses = -returns
        exceedances = losses[losses > threshold] - threshold
        
        if len(exceedances) < 20:
            return {'gpd_shape': 0, 'gpd_scale': 0, 'threshold': threshold}
        
        try:
            # Fit Generalized Pareto Distribution
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            
            # Calculate expected shortfall beyond threshold
            if shape < 1:
                es_beyond_threshold = scale / (1 - shape)
            else:
                es_beyond_threshold = float('inf')
            
            # Estimate probability of extreme losses
            n_total = len(losses)
            n_exceed = len(exceedances)
            prob_exceed = n_exceed / n_total
            
            # Return period for different loss levels
            loss_levels = [0.05, 0.10, 0.20]  # 5%, 10%, 20% losses
            return_periods = {}
            
            for level in loss_levels:
                if level > threshold:
                    prob = prob_exceed * (1 - stats.genpareto.cdf(level - threshold, shape, scale=scale))
                    return_periods[f'{int(level*100)}pct_loss'] = 1 / prob if prob > 0 else float('inf')
            
            return {
                'gpd_shape': float(shape),
                'gpd_scale': float(scale),
                'threshold': float(threshold),
                'threshold_percentile': threshold_percentile,
                'exceedances_count': len(exceedances),
                'expected_shortfall_beyond_threshold': float(es_beyond_threshold),
                'return_periods': return_periods
            }
        except:
            return {'gpd_shape': 0, 'gpd_scale': 0, 'threshold': threshold}
    
    def _estimate_tail_index(self, returns: pd.Series) -> Dict[str, float]:
        """Estimate tail index using Hill estimator"""
        
        # Sort absolute returns
        sorted_returns = np.sort(np.abs(returns))[::-1]
        n = len(sorted_returns)
        
        # Hill estimator for different k values
        k_values = [int(n * p) for p in [0.05, 0.10, 0.15]]
        hill_estimates = []
        
        for k in k_values:
            if k > 0 and k < n:
                hill_sum = np.sum([np.log(sorted_returns[i] / sorted_returns[k]) 
                                  for i in range(k)])
                hill_estimate = k / hill_sum if hill_sum > 0 else 0
                hill_estimates.append(hill_estimate)
        
        # Average tail index
        avg_tail_index = np.mean(hill_estimates) if hill_estimates else 2.0
        
        return {
            'hill_tail_index': float(avg_tail_index),
            'tail_heaviness': 'heavy' if avg_tail_index < 2 else 'normal' if avg_tail_index < 3 else 'light',
            'infinite_variance': avg_tail_index < 2,
            'infinite_mean': avg_tail_index < 1
        }
    
    def _calculate_extreme_expected_shortfall(self, returns: pd.Series, 
                                            pot_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected shortfall for extreme scenarios"""
        
        results = {}
        
        # Use GPD parameters if available
        if 'gpd_shape' in pot_analysis and pot_analysis['gpd_shape'] != 0:
            shape = pot_analysis['gpd_shape']
            scale = pot_analysis['gpd_scale']
            threshold = pot_analysis['threshold']
            
            # ES for different probability levels
            for p in [0.99, 0.995, 0.999]:
                if shape < 1:
                    # Calculate ES using GPD
                    u = threshold
                    beta = scale
                    xi = shape
                    
                    if xi != 0:
                        es = u + beta * ((1-p)**(-xi) - 1) / (xi * (1-xi))
                    else:
                        es = u + beta * (-np.log(1-p))
                    
                    results[f'es_{int(p*100)}'] = float(es)
        
        # Empirical ES as fallback
        if not results:
            for p in [0.99, 0.995, 0.999]:
                cutoff = np.percentile(returns, (1-p)*100)
                es = returns[returns <= cutoff].mean()
                results[f'es_{int(p*100)}'] = float(abs(es))
        
        return results
    
    def _calculate_extreme_probabilities(self, returns: pd.Series, 
                                       pot_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate probabilities of extreme events"""
        
        # Historical frequencies
        probs = {
            'prob_5pct_daily_loss': float((returns <= -0.05).mean()),
            'prob_10pct_daily_loss': float((returns <= -0.10).mean()),
            'prob_20pct_daily_loss': float((returns <= -0.20).mean()),
        }
        
        # Model-based probabilities using EVT
        if 'gpd_shape' in pot_analysis and pot_analysis['gpd_shape'] != 0:
            shape = pot_analysis['gpd_shape']
            scale = pot_analysis['gpd_scale']
            threshold = pot_analysis['threshold']
            prob_exceed = pot_analysis.get('exceedances_count', 0) / len(returns)
            
            for loss_level in [0.05, 0.10, 0.20]:
                if loss_level > threshold:
                    # Use GPD for tail probability
                    excess = loss_level - threshold
                    tail_prob = 1 - stats.genpareto.cdf(excess, shape, scale=scale)
                    probs[f'evt_prob_{int(loss_level*100)}pct_loss'] = float(prob_exceed * tail_prob)
        
        return probs
    
    def _calculate_tail_risk_score(self, tail_index: Dict[str, float], 
                                  extreme_es: Dict[str, float]) -> float:
        """Calculate overall tail risk score"""
        
        score = 0.5  # Neutral base
        
        # Adjust for tail heaviness
        if tail_index.get('infinite_variance', False):
            score += 0.3
        elif tail_index.get('tail_heaviness') == 'heavy':
            score += 0.2
        
        # Adjust for extreme expected shortfall
        es_99 = extreme_es.get('es_99', 0)
        if es_99 > 0.10:  # More than 10% expected loss in 1% worst case
            score += 0.2
        elif es_99 > 0.05:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        
        results = {}
        
        for confidence in self.var_confidence_levels:
            # Historical VaR
            var = np.percentile(returns, (1 - confidence) * 100)
            results[f'var_{int(confidence*100)}'] = float(abs(var))
            
            # CVaR (Expected Shortfall)
            cvar = returns[returns <= var].mean()
            results[f'cvar_{int(confidence*100)}'] = float(abs(cvar))
            
            # Parametric VaR (assuming normal distribution for comparison)
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            parametric_var = mean + z_score * std
            results[f'parametric_var_{int(confidence*100)}'] = float(abs(parametric_var))
            
            # Cornish-Fisher VaR (adjusting for skewness and kurtosis)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns, fisher=True)
            
            cf_adjustment = (
                z_score + 
                (z_score**2 - 1) * skew / 6 + 
                (z_score**3 - 3*z_score) * kurt / 24 - 
                (2*z_score**3 - 5*z_score) * skew**2 / 36
            )
            
            cf_var = mean + cf_adjustment * std
            results[f'cf_var_{int(confidence*100)}'] = float(abs(cf_var))
        
        return results
    
    def _forecast_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Forecast volatility using GARCH and other models"""
        
        # Simple volatility forecasts
        current_vol = returns.iloc[-20:].std()  # 20-day volatility
        
        # EWMA volatility
        ewma_vol = returns.ewm(span=20).std().iloc[-1]
        
        # Realized volatility
        realized_vol = np.sqrt(np.sum(returns.iloc[-20:]**2))
        
        # Simple GARCH(1,1) simulation
        # In production, would use proper GARCH estimation
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        
        # One-step ahead forecast
        garch_forecast = np.sqrt(omega + alpha * returns.iloc[-1]**2 + beta * current_vol**2)
        
        # Volatility regime
        long_term_vol = returns.std()
        vol_regime = 'high' if current_vol > long_term_vol * 1.5 else 'low' if current_vol < long_term_vol * 0.5 else 'normal'
        
        return {
            'current_volatility': float(current_vol * np.sqrt(252)),
            'ewma_volatility': float(ewma_vol * np.sqrt(252)),
            'realized_volatility': float(realized_vol * np.sqrt(252)),
            'garch_forecast_1d': float(garch_forecast * np.sqrt(252)),
            'volatility_regime': vol_regime,
            'volatility_percentile': float(stats.percentileofscore(
                returns.rolling(20).std().dropna(), current_vol
            ))
        }
    
    def _analyze_correlation_risk(self, ticker: str, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation and contagion risk"""
        
        if not portfolio or len(portfolio) < 2:
            return {'portfolio_correlation': 0, 'contagion_risk': 0}
        
        # Get returns for all assets
        returns_data = {}
        for asset in list(portfolio.keys()) + [ticker]:
            if asset not in returns_data:
                data = self.data_collector.get_historical_data(asset, period="1y")
                if data is not None and len(data) > 100:
                    returns_data[asset] = data['Close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {'portfolio_correlation': 0, 'contagion_risk': 0}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Average correlation with portfolio
        if ticker in corr_matrix.columns:
            ticker_correlations = corr_matrix[ticker].drop(ticker)
            avg_correlation = ticker_correlations.mean()
        else:
            avg_correlation = 0
        
        # Dynamic conditional correlation (simplified)
        # In stress periods, correlations tend to increase
        stress_corr = returns_df[returns_df[ticker] < returns_df[ticker].quantile(0.05)].corr()
        stress_correlation = stress_corr[ticker].drop(ticker).mean() if ticker in stress_corr.columns else avg_correlation
        
        # Contagion risk score
        contagion_risk = max(0, stress_correlation - avg_correlation)
        
        # Correlation instability
        rolling_corr = returns_df.rolling(60).corr()
        if ticker in returns_df.columns:
            corr_stability = 1 - rolling_corr[ticker].std().mean()
        else:
            corr_stability = 0.5
        
        return {
            'portfolio_correlation': float(avg_correlation),
            'stress_correlation': float(stress_correlation),
            'contagion_risk': float(contagion_risk),
            'correlation_stability': float(corr_stability),
            'diversification_benefit': float(1 - avg_correlation),
            'concentration_risk': float(max(portfolio.values())) if portfolio else 0
        }
    
    def _detect_risk_regime(self, returns: pd.Series, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current risk regime"""
        
        # Volatility regime
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        vol_percentile = stats.percentileofscore(
            returns.rolling(20).std().dropna() * np.sqrt(252), current_vol
        )
        
        # Trend regime
        sma_50 = price_data['Close'].rolling(50).mean()
        sma_200 = price_data['Close'].rolling(200).mean()
        trend = 'bullish' if price_data['Close'].iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] else 'bearish'
        
        # Drawdown regime
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        current_dd = (cumulative.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
        
        # Determine overall regime
        if vol_percentile > 80 or current_dd < -0.10:
            regime = 'crisis'
            risk_multiplier = 0.5  # Reduce risk
        elif vol_percentile > 60 or current_dd < -0.05:
            regime = 'stressed'
            risk_multiplier = 0.7
        elif vol_percentile < 20 and trend == 'bullish':
            regime = 'calm'
            risk_multiplier = 1.3  # Can take more risk
        else:
            regime = 'normal'
            risk_multiplier = 1.0
        
        return {
            'risk_regime': regime,
            'risk_multiplier': float(risk_multiplier),
            'volatility_percentile': float(vol_percentile),
            'trend_regime': trend,
            'current_drawdown': float(current_dd),
            'regime_stability': self._calculate_regime_stability(returns)
        }
    
    def _calculate_regime_stability(self, returns: pd.Series) -> float:
        """Calculate how stable the current regime is"""
        
        # Use rolling volatility changes
        vol_changes = returns.rolling(20).std().pct_change().dropna()
        stability = 1 - vol_changes.iloc[-20:].std()
        
        return float(max(0, min(1, stability)))
    
    def _calculate_position_size(self, returns: pd.Series, basic_metrics: Dict[str, float],
                                evt_metrics: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size using Kelly Criterion and risk limits"""
        
        # Enhanced Kelly Criterion with multiple approaches
        
        # 1. Classic Kelly
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.001
        
        if avg_loss > 0:
            kelly_classic = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_classic = max(0, min(0.25, kelly_classic))  # Cap at 25%
        else:
            kelly_classic = 0
        
        # 2. Optimal f (Ralph Vince method)
        if len(returns) > 20:
            # Find f that maximizes terminal wealth growth
            returns_array = returns.values
            optimal_f = self._calculate_optimal_f(returns_array)
        else:
            optimal_f = kelly_classic
        
        # 3. Kelly with drawdown constraint
        max_dd = basic_metrics.get('max_drawdown', 0.2)
        kelly_dd_adjusted = kelly_classic * (1 - max_dd)
        
        # 4. Regime-adjusted Kelly
        if regime_analysis.get('regime') == 'bear':
            kelly_regime = kelly_classic * 0.5
        elif regime_analysis.get('regime') == 'high_volatility':
            kelly_regime = kelly_classic * 0.7
        else:
            kelly_regime = kelly_classic
        
        # Combine Kelly estimates (weighted average)
        kelly_fraction = (
            0.4 * kelly_classic +
            0.2 * optimal_f +
            0.2 * kelly_dd_adjusted +
            0.2 * kelly_regime
        )
        
        # Adjust for higher moments
        skew_adjustment = 1 - max(0, -basic_metrics['skewness']) * 0.1
        kurt_adjustment = 1 - max(0, basic_metrics['kurtosis'] - 3) * 0.05
        
        # Adjust for tail risk
        tail_adjustment = 1 - evt_metrics['tail_risk_score'] * 0.3
        
        # Regime adjustment
        regime_multiplier = regime_analysis['risk_multiplier']
        
        # Calculate final position size
        base_size = kelly_fraction * 0.5  # Use half-Kelly for safety
        adjusted_size = base_size * skew_adjustment * kurt_adjustment * tail_adjustment * regime_multiplier
        
        # Apply limits
        min_size = 0.001  # 0.1% minimum
        max_size = min(self.max_position_size, 1 / max(basic_metrics['annual_volatility'], 0.1))
        
        recommended_size = max(min_size, min(max_size, adjusted_size))
        
        # Calculate confidence in sizing
        sizing_confidence = min(0.9, (
            0.3 * (1 - evt_metrics['tail_risk_score']) +
            0.3 * regime_analysis['regime_stability'] +
            0.2 * (1 - basic_metrics['annual_volatility'] / 0.5) +
            0.2 * (basic_metrics['sharpe_ratio'] / 2)
        ))
        
        return {
            'kelly_fraction': float(kelly_fraction),
            'recommended_size': float(recommended_size),
            'sizing_confidence': float(sizing_confidence),
            'risk_budget_used': float(recommended_size * basic_metrics['annual_volatility']),
            'max_allowed_size': float(max_size),
            'adjustments': {
                'skew': float(skew_adjustment),
                'kurtosis': float(kurt_adjustment),
                'tail_risk': float(tail_adjustment),
                'regime': float(regime_multiplier)
            }
        }
    
    def _run_stress_tests(self, returns: pd.Series, position_size: float) -> Dict[str, float]:
        """Run stress tests on the position"""
        
        results = {}
        
        # Historical stress scenarios
        worst_day = returns.min()
        worst_week = returns.rolling(5).sum().min()
        worst_month = returns.rolling(20).sum().min()
        
        results['worst_day_loss'] = float(worst_day * position_size)
        results['worst_week_loss'] = float(worst_week * position_size)
        results['worst_month_loss'] = float(worst_month * position_size)
        
        # Synthetic stress scenarios
        for scenario_name, scenario in self.stress_scenarios.items():
            scenario_return = scenario['return']
            scenario_vol_mult = scenario.get('volatility_multiplier', 1)
            
            # Calculate potential loss
            scenario_loss = scenario_return * position_size
            
            # Adjust for increased volatility
            if scenario_vol_mult > 1:
                current_vol = returns.std()
                scenario_loss *= scenario_vol_mult
            
            results[f'{scenario_name}_loss'] = float(scenario_loss)
        
        # Maximum acceptable loss check
        max_acceptable_loss = self.max_portfolio_risk
        worst_scenario_loss = min(results.values())
        results['passes_risk_limits'] = abs(worst_scenario_loss) < max_acceptable_loss
        results['risk_capacity_used'] = float(abs(worst_scenario_loss) / max_acceptable_loss)
        
        return results
    
    def _define_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Define stress test scenarios"""
        
        return {
            'market_crash': {'return': -0.20, 'volatility_multiplier': 3},
            'flash_crash': {'return': -0.10, 'volatility_multiplier': 5},
            'black_swan': {'return': -0.30, 'volatility_multiplier': 4},
            'liquidity_crisis': {'return': -0.15, 'volatility_multiplier': 2},
            'correlation_breakdown': {'return': -0.12, 'correlation': 0.9}
        }
    
    def _optimize_portfolio_risk(self, ticker: str, portfolio: Dict[str, float],
                                ticker_returns: pd.Series) -> Dict[str, Any]:
        """Optimize portfolio considering new position"""
        
        # This is a simplified version
        # In production, would use proper portfolio optimization
        
        current_positions = sum(portfolio.values())
        available_capital = max(0, 1 - current_positions)
        
        # Simple risk parity approach
        ticker_vol = ticker_returns.std() * np.sqrt(252)
        
        # Calculate portfolio volatility contribution
        risk_contributions = {}
        total_risk = 0
        
        for asset, weight in portfolio.items():
            asset_data = self.data_collector.get_historical_data(asset, period="1y")
            if asset_data is not None:
                asset_returns = asset_data['Close'].pct_change().dropna()
                asset_vol = asset_returns.std() * np.sqrt(252)
                risk_contributions[asset] = weight * asset_vol
                total_risk += (weight * asset_vol) ** 2
        
        total_risk = np.sqrt(total_risk)
        
        # Target risk contribution for new asset
        target_contribution = total_risk / (len(portfolio) + 1)
        suggested_weight = min(available_capital, target_contribution / ticker_vol)
        
        return {
            'suggested_weight': float(suggested_weight),
            'portfolio_risk': float(total_risk),
            'marginal_risk_contribution': float(ticker_vol * suggested_weight),
            'available_capital': float(available_capital),
            'diversification_ratio': float(1 / (len(portfolio) + 1))
        }
    
    def _calculate_risk_score(self, basic_metrics: Dict[str, float], evt_metrics: Dict[str, Any],
                             var_metrics: Dict[str, float], volatility_forecast: Dict[str, float],
                             correlation_risk: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk score"""
        
        # Component scores (0 = low risk, 1 = high risk)
        
        # Volatility risk
        vol_score = min(1, basic_metrics['annual_volatility'] / 0.5)  # 50% vol = max risk
        
        # Tail risk
        tail_score = evt_metrics['tail_risk_score']
        
        # Drawdown risk
        dd_score = min(1, abs(basic_metrics['max_drawdown']) / 0.3)  # 30% DD = max risk
        
        # VaR risk
        var_score = min(1, var_metrics['var_99'] / 0.05)  # 5% daily VaR = max risk
        
        # Correlation risk
        corr_score = correlation_risk.get('contagion_risk', 0)
        
        # Regime risk
        regime_score = 0 if regime_analysis['risk_regime'] == 'calm' else \
                      0.5 if regime_analysis['risk_regime'] == 'normal' else \
                      0.7 if regime_analysis['risk_regime'] == 'stressed' else 1.0
        
        # Weight components
        weights = {
            'volatility': 0.20,
            'tail_risk': 0.25,
            'drawdown': 0.20,
            'var': 0.15,
            'correlation': 0.10,
            'regime': 0.10
        }
        
        # Calculate weighted score
        overall_score = (
            weights['volatility'] * vol_score +
            weights['tail_risk'] * tail_score +
            weights['drawdown'] * dd_score +
            weights['var'] * var_score +
            weights['correlation'] * corr_score +
            weights['regime'] * regime_score
        )
        
        # Calculate confidence
        confidence = 0.5 + 0.5 * regime_analysis.get('regime_stability', 0.5)
        
        return {
            'overall_score': float(overall_score),
            'confidence': float(confidence),
            'component_scores': {
                'volatility_risk': float(vol_score),
                'tail_risk': float(tail_score),
                'drawdown_risk': float(dd_score),
                'var_risk': float(var_score),
                'correlation_risk': float(corr_score),
                'regime_risk': float(regime_score)
            },
            'risk_level': 'low' if overall_score < 0.3 else 'medium' if overall_score < 0.7 else 'high'
        }
    
    def _calculate_risk_limits(self, risk_score: Dict[str, float], 
                              regime_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic risk limits"""
        
        base_var_limit = 0.02  # 2% daily VaR limit
        base_position_limit = self.max_position_size
        
        # Adjust for regime
        regime_multiplier = regime_analysis['risk_multiplier']
        
        # Adjust for overall risk
        risk_adjustment = 1 - risk_score['overall_score'] * 0.5
        
        return {
            'daily_var_limit': float(base_var_limit * regime_multiplier * risk_adjustment),
            'position_size_limit': float(base_position_limit * regime_multiplier * risk_adjustment),
            'leverage_limit': float(self.max_leverage * regime_multiplier * risk_adjustment),
            'concentration_limit': float(0.25 * regime_multiplier),  # 25% max in one position
            'sector_limit': float(0.40 * regime_multiplier)  # 40% max in one sector
        }
    
    def _generate_risk_alerts(self, risk_score: Dict[str, float], 
                             evt_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate real-time risk alerts"""
        
        alerts = []
        
        # High overall risk
        if risk_score['overall_score'] > 0.7:
            alerts.append({
                'level': 'high',
                'type': 'overall_risk',
                'message': f"High overall risk score: {risk_score['overall_score']:.2f}",
                'action': 'Consider reducing position size'
            })
        
        # Tail risk alert
        if evt_metrics['tail_risk_score'] > 0.7:
            alerts.append({
                'level': 'high',
                'type': 'tail_risk',
                'message': 'Elevated tail risk detected',
                'action': 'Review extreme loss scenarios'
            })
        
        # Check component scores
        for component, score in risk_score['component_scores'].items():
            if score > 0.8:
                alerts.append({
                    'level': 'medium',
                    'type': component,
                    'message': f'High {component} score: {score:.2f}',
                    'action': f'Monitor {component} closely'
                })
        
        return alerts
    
    def _generate_risk_reasoning(self, basic_metrics: Dict[str, float], evt_metrics: Dict[str, Any],
                                var_metrics: Dict[str, float], regime_analysis: Dict[str, Any],
                                position_sizing: Dict[str, Any]) -> str:
        """Generate human-readable risk assessment"""
        
        reasons = []
        
        # Overall risk level
        vol = basic_metrics['annual_volatility']
        if vol > 0.4:
            reasons.append(f"High volatility ({vol:.1%} annualized)")
        elif vol < 0.15:
            reasons.append(f"Low volatility ({vol:.1%} annualized)")
        
        # Tail risk
        if evt_metrics['tail_risk_score'] > 0.7:
            reasons.append("Significant tail risk detected")
        
        # Regime
        regime = regime_analysis['risk_regime']
        reasons.append(f"Market regime: {regime}")
        
        # Position sizing
        size = position_sizing['recommended_size']
        reasons.append(f"Recommended position size: {size:.1%}")
        
        # Risk-adjusted returns
        sharpe = basic_metrics['sharpe_ratio']
        if sharpe > 1:
            reasons.append(f"Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe < 0:
            reasons.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        return " | ".join(reasons)
    
    def _default_risk_response(self, error_msg: str) -> Dict[str, Any]:
        """Default response when analysis fails"""
        
        return {
            'error': error_msg,
            'score': 0.5,  # Neutral risk score
            'confidence': 0.0,
            'reasoning': error_msg,
            'position_sizing': {'recommended_size': self.base_position_size}
        }
    
    def _calculate_optimal_f(self, returns: np.ndarray) -> float:
        """Calculate optimal f using Ralph Vince method"""
        try:
            # Convert returns to profit/loss series
            initial_capital = 1.0
            trades = returns[returns != 0]  # Remove zero returns
            
            if len(trades) < 10:
                return 0.02  # Default small position
            
            # Find worst loss
            worst_loss = abs(min(trades)) if min(trades) < 0 else 0.01
            
            # Grid search for optimal f
            f_values = np.linspace(0.01, 0.5, 50)
            terminal_wealths = []
            
            for f in f_values:
                wealth = initial_capital
                for trade in trades:
                    # Calculate holding period return
                    hpr = 1 + f * (trade / worst_loss)
                    wealth *= max(0, hpr)  # Prevent negative wealth
                    
                    if wealth <= 0:
                        break
                
                terminal_wealths.append(wealth)
            
            # Find f that maximizes terminal wealth
            if terminal_wealths:
                optimal_idx = np.argmax(terminal_wealths)
                optimal_f = f_values[optimal_idx]
                
                # Apply safety factor
                return min(0.25, optimal_f * 0.5)  # Use half optimal f
            else:
                return 0.02
                
        except Exception as e:
            self.logger.warning(f"Optimal f calculation failed: {e}")
            return 0.02