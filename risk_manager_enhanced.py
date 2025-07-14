import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from data_collection import DataCollectionAgent
from technical_analysis_enhanced import EnhancedTechnicalAnalyst

class EnhancedRiskManager:
    """
    Advanced risk management with multiple VaR methods, Monte Carlo simulations,
    and comprehensive stress testing
    """
    
    def __init__(self):
        self.data_collector = DataCollectionAgent()
        self.tech_analyst = EnhancedTechnicalAnalyst()
        self.risk_free_rate = 0.05  # 5% annual risk-free rate
        
    def calculate_var_historical(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate VaR using historical simulation method
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_var_parametric(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate VaR using parametric (variance-covariance) method
        Assumes normal distribution
        """
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mean + z_score * std
    
    def calculate_var_monte_carlo(self, returns: pd.Series, confidence: float = 0.95, 
                                 n_simulations: int = 10000) -> float:
        """
        Calculate VaR using Monte Carlo simulation
        """
        mean = returns.mean()
        std = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR
        return np.percentile(simulated_returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        Average loss beyond VaR threshold
        """
        var = self.calculate_var_historical(returns, confidence)
        conditional_returns = returns[returns <= var]
        return conditional_returns.mean() if len(conditional_returns) > 0 else var
    
    def calculate_all_var_metrics(self, ticker: str, period: int = 252, 
                                 confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate all VaR metrics for a given ticker
        """
        try:
            # Get historical data
            data = self.tech_analyst.getData(ticker)[0].tail(period)
            returns = data['Close'].pct_change().dropna()
            
            # Calculate different VaR measures
            var_historical = self.calculate_var_historical(returns, confidence)
            var_parametric = self.calculate_var_parametric(returns, confidence)
            var_monte_carlo = self.calculate_var_monte_carlo(returns, confidence)
            cvar = self.calculate_cvar(returns, confidence)
            
            # Convert to percentage and make positive for readability
            results = {
                'var_historical': abs(var_historical) * 100,
                'var_parametric': abs(var_parametric) * 100,
                'var_monte_carlo': abs(var_monte_carlo) * 100,
                'cvar': abs(cvar) * 100,
                'confidence_level': confidence * 100,
                'period_days': period,
                'worst_daily_loss': abs(returns.min()) * 100,
                'best_daily_gain': returns.max() * 100
            }
            
            return results
            
        except Exception as e:
            print(f"Error calculating VaR metrics: {e}")
            return None
    
    def monte_carlo_portfolio_simulation(self, portfolio_weights: Dict[str, float],
                                       n_simulations: int = 10000,
                                       time_horizon: int = 252,
                                       initial_value: float = 100000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio risk assessment
        """
        try:
            tickers = list(portfolio_weights.keys())
            weights = np.array([portfolio_weights[ticker] for ticker in tickers])
            
            # Get historical data for all tickers
            returns_data = {}
            for ticker in tickers:
                data = self.tech_analyst.getData(ticker)[0].tail(252)
                returns_data[ticker] = data['Close'].pct_change().dropna()
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).dropna()
            
            # Calculate mean returns and covariance
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Portfolio parameters
            portfolio_mean = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Run simulations
            simulation_results = []
            
            for _ in range(n_simulations):
                # Generate random returns for the time horizon
                daily_returns = []
                
                for _ in range(time_horizon):
                    # Use multivariate normal to maintain correlations
                    random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
                    portfolio_return = np.dot(weights, random_returns)
                    daily_returns.append(portfolio_return)
                
                # Calculate cumulative return
                cumulative_return = np.prod([1 + r for r in daily_returns]) - 1
                final_value = initial_value * (1 + cumulative_return)
                simulation_results.append(final_value)
            
            simulation_results = np.array(simulation_results)
            
            # Calculate statistics
            results = {
                'mean_final_value': np.mean(simulation_results),
                'median_final_value': np.median(simulation_results),
                'std_final_value': np.std(simulation_results),
                'percentile_5': np.percentile(simulation_results, 5),
                'percentile_25': np.percentile(simulation_results, 25),
                'percentile_75': np.percentile(simulation_results, 75),
                'percentile_95': np.percentile(simulation_results, 95),
                'var_95': initial_value - np.percentile(simulation_results, 5),
                'cvar_95': initial_value - np.mean(simulation_results[simulation_results <= np.percentile(simulation_results, 5)]),
                'probability_loss': np.sum(simulation_results < initial_value) / n_simulations,
                'probability_gain_10pct': np.sum(simulation_results > initial_value * 1.1) / n_simulations,
                'probability_gain_20pct': np.sum(simulation_results > initial_value * 1.2) / n_simulations,
                'max_simulated_value': np.max(simulation_results),
                'min_simulated_value': np.min(simulation_results),
                'sharpe_ratio': (np.mean(simulation_results) / initial_value - 1 - self.risk_free_rate) / (np.std(simulation_results) / initial_value)
            }
            
            # Add distribution analysis
            results['distribution'] = {
                'skewness': stats.skew(simulation_results),
                'kurtosis': stats.kurtosis(simulation_results),
                'is_normal': self._test_normality(simulation_results)
            }
            
            return results
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return None
    
    def stress_test_portfolio(self, portfolio_weights: Dict[str, float],
                            scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio under various scenarios
        """
        if scenarios is None:
            # Default stress test scenarios
            scenarios = [
                {'name': 'Market Crash', 'market_drop': -0.20, 'volatility_spike': 2.0},
                {'name': 'Flash Crash', 'market_drop': -0.10, 'volatility_spike': 3.0},
                {'name': 'Recession', 'market_drop': -0.30, 'volatility_spike': 1.5, 'duration': 252},
                {'name': 'Interest Rate Shock', 'rate_change': 0.02, 'market_drop': -0.05},
                {'name': 'Black Swan', 'market_drop': -0.40, 'volatility_spike': 4.0},
                {'name': 'Sector Rotation', 'sector_impacts': {'tech': -0.15, 'finance': 0.10}},
                {'name': 'Inflation Surge', 'inflation': 0.05, 'market_drop': -0.08},
                {'name': 'Currency Crisis', 'fx_impact': -0.15, 'market_drop': -0.12}
            ]
        
        results = {}
        
        try:
            # Get current portfolio value and metrics
            tickers = list(portfolio_weights.keys())
            
            # Get historical data
            returns_data = {}
            current_prices = {}
            
            for ticker in tickers:
                data = self.tech_analyst.getData(ticker)[0].tail(252)
                returns_data[ticker] = data['Close'].pct_change().dropna()
                current_prices[ticker] = data['Close'].iloc[-1]
            
            # Calculate current portfolio metrics
            returns_df = pd.DataFrame(returns_data).dropna()
            weights = np.array([portfolio_weights[ticker] for ticker in tickers])
            
            portfolio_returns = returns_df.dot(weights)
            current_sharpe = self._calculate_sharpe(portfolio_returns)
            current_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Run each stress test scenario
            for scenario in scenarios:
                scenario_result = self._run_stress_scenario(
                    portfolio_weights,
                    returns_df,
                    current_prices,
                    scenario
                )
                
                if scenario_result:
                    # Compare to baseline
                    scenario_result['impact_vs_baseline'] = {
                        'return_change': scenario_result['stressed_return'] - portfolio_returns.mean() * 252,
                        'volatility_change': scenario_result['stressed_volatility'] - current_volatility,
                        'sharpe_change': scenario_result['stressed_sharpe'] - current_sharpe
                    }
                    
                    results[scenario['name']] = scenario_result
            
            # Add summary statistics
            results['summary'] = {
                'worst_scenario': min(results.items(), key=lambda x: x[1]['portfolio_loss'] if isinstance(x[1], dict) and 'portfolio_loss' in x[1] else 0)[0],
                'average_loss': np.mean([r['portfolio_loss'] for r in results.values() if isinstance(r, dict) and 'portfolio_loss' in r]),
                'scenarios_with_loss_over_10pct': sum(1 for r in results.values() if isinstance(r, dict) and r.get('portfolio_loss', 0) < -0.10)
            }
            
            return results
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            return None
    
    def _run_stress_scenario(self, portfolio_weights: Dict[str, float],
                           returns_df: pd.DataFrame,
                           current_prices: Dict[str, float],
                           scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single stress test scenario
        """
        try:
            tickers = list(portfolio_weights.keys())
            weights = np.array([portfolio_weights[ticker] for ticker in tickers])
            
            # Apply stress to returns
            stressed_returns = returns_df.copy()
            
            # Market-wide drop
            if 'market_drop' in scenario:
                drop = scenario['market_drop']
                # Apply with some randomness per asset
                for ticker in tickers:
                    asset_beta = self._estimate_beta(returns_df[ticker], returns_df.mean(axis=1))
                    asset_drop = drop * asset_beta * np.random.uniform(0.8, 1.2)
                    stressed_returns[ticker] = stressed_returns[ticker] + asset_drop / 252
            
            # Volatility spike
            if 'volatility_spike' in scenario:
                spike = scenario['volatility_spike']
                for ticker in tickers:
                    vol = stressed_returns[ticker].std()
                    noise = np.random.normal(0, vol * (spike - 1), len(stressed_returns))
                    stressed_returns[ticker] = stressed_returns[ticker] + noise
            
            # Calculate stressed portfolio metrics
            stressed_portfolio_returns = stressed_returns.dot(weights)
            
            # Calculate impact
            portfolio_value = sum(portfolio_weights[t] * current_prices[t] for t in tickers)
            stressed_value = portfolio_value * (1 + stressed_portfolio_returns.mean() * 252)
            
            result = {
                'scenario_name': scenario['name'],
                'portfolio_loss': (stressed_value - portfolio_value) / portfolio_value,
                'stressed_return': stressed_portfolio_returns.mean() * 252,
                'stressed_volatility': stressed_portfolio_returns.std() * np.sqrt(252),
                'stressed_sharpe': self._calculate_sharpe(stressed_portfolio_returns),
                'var_95': np.percentile(stressed_portfolio_returns, 5) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(stressed_portfolio_returns)
            }
            
            # Add asset-specific impacts
            asset_impacts = {}
            for ticker in tickers:
                original_return = returns_df[ticker].mean() * 252
                stressed_return = stressed_returns[ticker].mean() * 252
                asset_impacts[ticker] = {
                    'return_impact': stressed_return - original_return,
                    'weight': portfolio_weights[ticker]
                }
            
            result['asset_impacts'] = asset_impacts
            
            return result
            
        except Exception as e:
            print(f"Error running stress scenario: {e}")
            return None
    
    def calculate_risk_metrics(self, ticker: str, period: int = 252) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a single asset
        """
        try:
            # Get data
            data = self.tech_analyst.getData(ticker)[0].tail(period)
            returns = data['Close'].pct_change().dropna()
            
            # Basic statistics
            mean_return = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Downside risk measures
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Risk ratios
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate recovery time from max drawdown
            if max_drawdown < 0:
                drawdown_start = drawdown.idxmin()
                recovery_date = drawdown[drawdown_start:].loc[drawdown[drawdown_start:] >= -0.01].index
                recovery_time = len(drawdown[drawdown_start:recovery_date[0]]) if len(recovery_date) > 0 else None
            else:
                recovery_time = 0
            
            # Tail risk
            left_tail_prob = len(returns[returns < returns.mean() - 2 * returns.std()]) / len(returns)
            right_tail_prob = len(returns[returns > returns.mean() + 2 * returns.std()]) / len(returns)
            
            # Information ratio (assuming market as benchmark)
            try:
                market_data = yf.download("^GSPC", period=f"{period}d", progress=False)['Adj Close']
                market_returns = market_data.pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(market_returns.index)
                excess_returns = returns[common_dates] - market_returns[common_dates]
                
                information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
            except:
                information_ratio = None
            
            # Compile results
            metrics = {
                'ticker': ticker,
                'period_days': period,
                'basic_stats': {
                    'annual_return': mean_return,
                    'annual_volatility': volatility,
                    'daily_avg_return': returns.mean(),
                    'daily_volatility': returns.std()
                },
                'distribution': {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'is_normal': self._test_normality(returns),
                    'left_tail_prob': left_tail_prob,
                    'right_tail_prob': right_tail_prob
                },
                'risk_measures': {
                    'var_95': abs(self.calculate_var_historical(returns, 0.95)) * 100,
                    'cvar_95': abs(self.calculate_cvar(returns, 0.95)) * 100,
                    'max_drawdown': abs(max_drawdown),
                    'recovery_time_days': recovery_time,
                    'downside_deviation': downside_deviation
                },
                'risk_adjusted_returns': {
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'information_ratio': information_ratio,
                    'calmar_ratio': abs(mean_return / max_drawdown) if max_drawdown != 0 else None
                },
                'risk_score': self._calculate_comprehensive_risk_score(metrics)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return None
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        return (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def _estimate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Estimate beta coefficient"""
        covariance = asset_returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def _test_normality(self, returns: pd.Series) -> bool:
        """Test if returns follow normal distribution"""
        _, p_value = stats.jarque_bera(returns)
        return p_value > 0.05  # Accept normality if p-value > 0.05
    
    def _calculate_comprehensive_risk_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate a comprehensive risk score from 1-10
        Lower is better (less risky)
        """
        score_components = []
        
        # Volatility component (0-2.5 points)
        vol = metrics['basic_stats']['annual_volatility']
        vol_score = min(2.5, vol * 5)  # 50% annual vol = 2.5 points
        score_components.append(vol_score)
        
        # VaR component (0-2.5 points)
        var = metrics['risk_measures']['var_95']
        var_score = min(2.5, var / 4)  # 10% daily VaR = 2.5 points
        score_components.append(var_score)
        
        # Drawdown component (0-2.5 points)
        dd = metrics['risk_measures']['max_drawdown']
        dd_score = min(2.5, dd * 5)  # 50% drawdown = 2.5 points
        score_components.append(dd_score)
        
        # Risk-adjusted return component (0-2.5 points)
        sharpe = metrics['risk_adjusted_returns']['sharpe_ratio']
        sharpe_score = max(0, 2.5 - sharpe)  # Higher Sharpe = lower risk score
        score_components.append(sharpe_score)
        
        # Calculate total score
        total_score = sum(score_components)
        
        # Scale to 1-10
        return min(10, max(1, total_score))
    
    def recommend_risk_limits(self, ticker: str, portfolio_value: float,
                            risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Recommend position limits based on risk analysis
        """
        try:
            # Define risk tolerance parameters
            risk_params = {
                'conservative': {'max_position_pct': 0.05, 'var_limit': 0.02, 'max_volatility': 0.15},
                'moderate': {'max_position_pct': 0.10, 'var_limit': 0.05, 'max_volatility': 0.25},
                'aggressive': {'max_position_pct': 0.20, 'var_limit': 0.10, 'max_volatility': 0.40}
            }
            
            params = risk_params.get(risk_tolerance, risk_params['moderate'])
            
            # Get risk metrics
            metrics = self.calculate_risk_metrics(ticker)
            if not metrics:
                return None
            
            # Calculate position limits
            volatility = metrics['basic_stats']['annual_volatility']
            var_95 = metrics['risk_measures']['var_95'] / 100  # Convert to decimal
            
            # Position size based on volatility
            vol_based_limit = params['max_volatility'] / volatility * params['max_position_pct']
            
            # Position size based on VaR
            var_based_limit = params['var_limit'] / var_95 * params['max_position_pct']
            
            # Take the minimum (most conservative)
            recommended_position_pct = min(vol_based_limit, var_based_limit, params['max_position_pct'])
            recommended_position_value = portfolio_value * recommended_position_pct
            
            # Get current price for share calculation
            current_price = self.tech_analyst.getData(ticker)[0]['Close'].iloc[-1]
            recommended_shares = int(recommended_position_value / current_price)
            
            # Calculate stop loss levels
            stop_loss_pct = min(var_95 * 2, 0.10)  # 2x VaR or 10% max
            stop_loss_price = current_price * (1 - stop_loss_pct)
            
            return {
                'ticker': ticker,
                'risk_tolerance': risk_tolerance,
                'risk_score': metrics['risk_score'],
                'position_limits': {
                    'max_position_pct': recommended_position_pct * 100,
                    'max_position_value': recommended_position_value,
                    'max_shares': recommended_shares,
                    'current_price': current_price
                },
                'risk_controls': {
                    'stop_loss_pct': stop_loss_pct * 100,
                    'stop_loss_price': stop_loss_price,
                    'position_var_95': recommended_position_value * var_95,
                    'daily_risk_limit': recommended_position_value * params['var_limit']
                },
                'monitoring': {
                    'review_frequency': 'Daily' if volatility > 0.30 else 'Weekly',
                    'rebalance_trigger': f"{params['max_position_pct'] * 150:.1%} of portfolio",
                    'risk_alert_threshold': f"Loss > {params['var_limit'] * 100:.1%}"
                }
            }
            
        except Exception as e:
            print(f"Error calculating risk limits: {e}")
            return None


# Example usage
if __name__ == "__main__":
    risk_mgr = EnhancedRiskManager()
    
    # Example 1: Comprehensive VaR analysis
    print("=== VaR Analysis for AAPL ===")
    var_metrics = risk_mgr.calculate_all_var_metrics("AAPL")
    if var_metrics:
        print(f"Historical VaR (95%): {var_metrics['var_historical']:.2f}%")
        print(f"Parametric VaR (95%): {var_metrics['var_parametric']:.2f}%")
        print(f"Monte Carlo VaR (95%): {var_metrics['var_monte_carlo']:.2f}%")
        print(f"CVaR (95%): {var_metrics['cvar']:.2f}%")
    
    # Example 2: Portfolio Monte Carlo simulation
    print("\n=== Portfolio Monte Carlo Simulation ===")
    portfolio = {
        "AAPL": 0.3,
        "MSFT": 0.3,
        "GOOGL": 0.2,
        "AMZN": 0.2
    }
    
    mc_results = risk_mgr.monte_carlo_portfolio_simulation(portfolio, n_simulations=5000)
    if mc_results:
        print(f"Mean final value: ${mc_results['mean_final_value']:,.2f}")
        print(f"VaR (95%): ${mc_results['var_95']:,.2f}")
        print(f"Probability of loss: {mc_results['probability_loss']*100:.1f}%")
        print(f"Probability of 20% gain: {mc_results['probability_gain_20pct']*100:.1f}%")
    
    # Example 3: Stress testing
    print("\n=== Portfolio Stress Testing ===")
    stress_results = risk_mgr.stress_test_portfolio(portfolio)
    if stress_results and 'summary' in stress_results:
        print(f"Worst scenario: {stress_results['summary']['worst_scenario']}")
        print(f"Average loss across scenarios: {stress_results['summary']['average_loss']*100:.1f}%")
    
    # Example 4: Risk limits recommendation
    print("\n=== Risk Limits for AAPL ===")
    limits = risk_mgr.recommend_risk_limits("AAPL", 100000, "moderate")
    if limits:
        print(f"Recommended position: ${limits['position_limits']['max_position_value']:,.2f}")
        print(f"Stop loss at: ${limits['risk_controls']['stop_loss_price']:.2f}")
        print(f"Daily risk limit: ${limits['risk_controls']['daily_risk_limit']:,.2f}")