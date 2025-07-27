import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization imports
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("CVXPY not available. Advanced optimization features will be limited.")

# PyPortfolioOpt imports
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import BlackLittermanModel, plotting, objective_functions
    from pypfopt import HRPOpt, CLA
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("PyPortfolioOpt not available. Some optimization methods will be limited.")

class EnhancedPortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple strategies including
    Mean-Variance, Black-Litterman, Risk Parity, and Kelly Criterion.
    """
    
    def __init__(self):
        self.portfolio_cache = {}
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (Treasury)
        
    def optimize_portfolio(self, tickers: List[str], 
                         strategy: str = 'mean_variance',
                         views: Optional[Dict[str, float]] = None,
                         constraints: Optional[Dict[str, Any]] = None,
                         target_return: Optional[float] = None,
                         risk_aversion: float = 1.0) -> Dict[str, Any]:
        """
        Optimize portfolio using specified strategy
        
        Args:
            tickers: List of ticker symbols
            strategy: Optimization strategy ('mean_variance', 'black_litterman', 
                     'risk_parity', 'hrp', 'max_sharpe', 'min_volatility', 'kelly')
            views: Market views for Black-Litterman (e.g., {'AAPL': 0.10, 'GOOGL': 0.08})
            constraints: Additional constraints
            target_return: Target annual return for mean-variance
            risk_aversion: Risk aversion parameter (higher = more conservative)
        
        Returns:
            Dictionary with optimized weights and performance metrics
        """
        
        # Get historical data
        data = self._get_historical_data(tickers)
        if data.empty:
            return {'error': 'Failed to fetch historical data'}
        
        # Calculate returns and covariance
        returns = data.pct_change().dropna()
        
        # Choose optimization strategy
        if strategy == 'mean_variance':
            result = self._optimize_mean_variance(returns, target_return, constraints, risk_aversion)
        elif strategy == 'black_litterman':
            result = self._optimize_black_litterman(returns, data, views, constraints)
        elif strategy == 'risk_parity':
            result = self._optimize_risk_parity(returns)
        elif strategy == 'hrp':
            result = self._optimize_hierarchical_risk_parity(returns)
        elif strategy == 'max_sharpe':
            result = self._optimize_max_sharpe(returns, constraints)
        elif strategy == 'min_volatility':
            result = self._optimize_min_volatility(returns, constraints)
        elif strategy == 'kelly':
            result = self._optimize_kelly_criterion(returns, risk_aversion)
        else:
            return {'error': f'Unknown strategy: {strategy}'}
        
        # Add performance metrics
        if 'weights' in result:
            result['metrics'] = self._calculate_portfolio_metrics(result['weights'], returns)
            result['risk_decomposition'] = self._calculate_risk_decomposition(result['weights'], returns)
            result['diversification_ratio'] = self._calculate_diversification_ratio(result['weights'], returns)
        
        return result
    
    def _get_historical_data(self, tickers: List[str], period: str = '2y') -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            data = yf.download(tickers, period=period, progress=False)['Adj Close']
            if isinstance(data, pd.Series):
                data = pd.DataFrame({tickers[0]: data})
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _optimize_mean_variance(self, returns: pd.DataFrame, 
                              target_return: Optional[float],
                              constraints: Optional[Dict[str, Any]],
                              risk_aversion: float) -> Dict[str, Any]:
        """Classic Markowitz mean-variance optimization"""
        
        if PYPFOPT_AVAILABLE:
            # Use PyPortfolioOpt for advanced features
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            
            # Apply constraints
            if constraints:
                self._apply_pypfopt_constraints(ef, constraints)
            
            # Optimize
            if target_return:
                ef.efficient_return(target_return)
            else:
                # Maximize Sharpe ratio by default
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            
            weights = ef.clean_weights()
            
            # Get performance
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            
            return {
                'weights': weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        else:
            # Fallback to scipy optimization
            return self._optimize_mean_variance_scipy(returns, target_return, risk_aversion)
    
    def _optimize_mean_variance_scipy(self, returns: pd.DataFrame, 
                                    target_return: Optional[float],
                                    risk_aversion: float) -> Dict[str, Any]:
        """Mean-variance optimization using scipy"""
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Objective function (minimize negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate/252) / portfolio_vol
            return -sharpe
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, mean_returns) * 252 - target_return
            })
        
        # Bounds (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, mean_returns) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe
            }
        else:
            return {'error': 'Optimization failed'}
    
    def _optimize_black_litterman(self, returns: pd.DataFrame, 
                                prices: pd.DataFrame,
                                views: Optional[Dict[str, float]],
                                constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Black-Litterman optimization with investor views"""
        
        if not PYPFOPT_AVAILABLE:
            return {'error': 'PyPortfolioOpt required for Black-Litterman optimization'}
        
        # Market cap weights (using price * volume as proxy for market cap)
        # In practice, you'd use actual market cap data
        market_caps = prices.iloc[-1]
        
        # Get volume data for better market cap approximation
        volume_data = {}
        for ticker in prices.columns:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                shares = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 1e9))
                volume_data[ticker] = shares
            except:
                volume_data[ticker] = 1e9  # Default if data unavailable
        
        # Calculate market cap weights
        for ticker in prices.columns:
            market_caps[ticker] = prices[ticker].iloc[-1] * volume_data.get(ticker, 1e9)
        
        mcap_weights = market_caps / market_caps.sum()
        
        # Calculate market implied returns
        S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        
        # Dynamic risk aversion based on market conditions
        market_vol = returns.mean(axis=1).std() * np.sqrt(252)
        delta = 2.5 / (1 + market_vol)  # Adjust risk aversion based on market volatility
        
        market_implied_returns = delta * S.dot(mcap_weights)
        
        # Create Black-Litterman model
        bl = BlackLittermanModel(S, pi=market_implied_returns)
        
        # Enhanced view processing
        if views and isinstance(views, dict):
            # Support different view types
            if 'absolute' in views:
                # Absolute views: "AAPL will return 15%"
                viewdict = views['absolute']
                view_confidences = views.get('confidence', {})
                
                # Process each view
                processed_views = {}
                processed_confidences = []
                
                for ticker, view_return in viewdict.items():
                    if ticker in returns.columns:
                        processed_views[ticker] = view_return
                        # Use provided confidence or calculate based on view strength
                        conf = view_confidences.get(ticker, min(abs(view_return) * 5, 0.8))
                        processed_confidences.append(conf)
                
                if processed_views:
                    bl.bl_returns(processed_views, processed_confidences)
            
            elif 'relative' in views:
                # Relative views: "AAPL will outperform MSFT by 5%"
                relative_views = views['relative']
                
                for view in relative_views:
                    asset1 = view.get('asset1')
                    asset2 = view.get('asset2')
                    outperformance = view.get('outperformance', 0)
                    confidence = view.get('confidence', 0.5)
                    
                    if asset1 in returns.columns and asset2 in returns.columns:
                        # Create view matrix for relative view
                        view_matrix = pd.Series(0, index=returns.columns)
                        view_matrix[asset1] = 1
                        view_matrix[asset2] = -1
                        
                        bl.bl_returns(
                            {asset1: market_implied_returns[asset1] + outperformance,
                             asset2: market_implied_returns[asset2]},
                            [confidence, confidence]
                        )
            
            else:
                # Legacy simple views format
                viewdict = views
                view_confidences = []
                
                for ticker in viewdict:
                    confidence = min(abs(viewdict[ticker]) * 10, 1.0)
                    view_confidences.append(confidence)
                
                bl.bl_returns(viewdict, view_confidences)
        
        # Get posterior returns and covariance
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        # Optimize with posterior returns
        ef = EfficientFrontier(ret_bl, S_bl)
        
        if constraints:
            self._apply_pypfopt_constraints(ef, constraints)
        
        # Choose optimization based on investor preference
        optimization_method = constraints.get('bl_optimization', 'max_sharpe') if constraints else 'max_sharpe'
        
        if optimization_method == 'max_sharpe':
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif optimization_method == 'min_volatility':
            ef.min_volatility()
        elif optimization_method == 'efficient_return' and 'target_return' in constraints:
            ef.efficient_return(constraints['target_return'])
        else:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        
        # Calculate view impact
        prior_returns = dict(market_implied_returns)
        posterior_returns = dict(ret_bl)
        view_impact = {
            ticker: posterior_returns[ticker] - prior_returns[ticker] 
            for ticker in returns.columns
        }
        
        return {
            'weights': weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2],
            'market_implied_returns': prior_returns,
            'posterior_returns': posterior_returns,
            'view_impact': view_impact,
            'risk_aversion_parameter': delta
        }
    
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Risk parity optimization - equal risk contribution"""
        
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        
        # Objective: minimize portfolio variance subject to equal risk contribution
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target equal contribution
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared differences from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 1) for _ in range(n_assets))  # Min 1% to avoid zero weights
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_contribution, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            
            # Calculate metrics
            portfolio_return = np.dot(weights, returns.mean()) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            # Calculate actual risk contributions
            marginal_contrib = np.dot(cov_matrix * 252, weights) / portfolio_vol
            risk_contribs = weights * marginal_contrib
            
            return {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'risk_contributions': dict(zip(returns.columns, risk_contribs))
            }
        else:
            return {'error': 'Risk parity optimization failed'}
    
    def _optimize_hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Hierarchical Risk Parity optimization"""
        
        if not PYPFOPT_AVAILABLE:
            return self._optimize_risk_parity(returns)  # Fallback to regular risk parity
        
        # Use PyPortfolioOpt's HRP
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        
        # Calculate performance metrics
        portfolio_return = hrp.portfolio_performance(verbose=False)[0]
        portfolio_vol = hrp.portfolio_performance(verbose=False)[1]
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'clustering': 'hierarchical'
        }
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame, 
                           constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Maximize Sharpe ratio"""
        
        if PYPFOPT_AVAILABLE:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            
            if constraints:
                self._apply_pypfopt_constraints(ef, constraints)
            
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()
            
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            
            return {
                'weights': weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        else:
            return self._optimize_mean_variance_scipy(returns, None, 1.0)
    
    def _optimize_min_volatility(self, returns: pd.DataFrame,
                               constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Minimize portfolio volatility"""
        
        if PYPFOPT_AVAILABLE:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S)
            
            if constraints:
                self._apply_pypfopt_constraints(ef, constraints)
            
            ef.min_volatility()
            weights = ef.clean_weights()
            
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            
            return {
                'weights': weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
        else:
            # Scipy fallback
            n_assets = len(returns.columns)
            cov_matrix = returns.cov()
            
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((0, 1) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, returns.mean()) * 252
                portfolio_vol = result.fun * np.sqrt(252)
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                
                return {
                    'weights': dict(zip(returns.columns, weights)),
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe
                }
            else:
                return {'error': 'Optimization failed'}
    
    def _optimize_kelly_criterion(self, returns: pd.DataFrame, 
                                risk_aversion: float = 1.0) -> Dict[str, Any]:
        """Kelly Criterion optimization for position sizing"""
        
        # Calculate Kelly fractions for each asset
        kelly_fractions = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            # Calculate win/loss statistics
            positive_returns = asset_returns[asset_returns > 0]
            negative_returns = asset_returns[asset_returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                win_rate = len(positive_returns) / len(asset_returns)
                avg_win = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                
                # Kelly formula: f = (p*b - q) / b
                # where p = win rate, q = loss rate, b = win/loss ratio
                b = avg_win / avg_loss if avg_loss > 0 else 0
                q = 1 - win_rate
                
                kelly_fraction = (win_rate * b - q) / b if b > 0 else 0
                
                # Apply fractional Kelly (reduce by risk aversion)
                kelly_fraction = kelly_fraction / risk_aversion
                
                # Cap at reasonable levels
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% per position
            else:
                kelly_fraction = 0
            
            kelly_fractions[asset] = kelly_fraction
        
        # Normalize to sum to 1
        total_kelly = sum(kelly_fractions.values())
        if total_kelly > 0:
            weights = {asset: frac/total_kelly for asset, frac in kelly_fractions.items()}
        else:
            # Equal weights if Kelly fails
            n_assets = len(returns.columns)
            weights = {asset: 1/n_assets for asset in returns.columns}
        
        # Calculate performance metrics
        weights_array = np.array([weights[asset] for asset in returns.columns])
        portfolio_return = np.dot(weights_array, returns.mean()) * 252
        cov_matrix = returns.cov() * 252
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'kelly_fractions': kelly_fractions,
            'risk_aversion_applied': risk_aversion
        }
    
    def _apply_pypfopt_constraints(self, ef: Any, constraints: Dict[str, Any]):
        """Apply constraints to PyPortfolioOpt optimizer"""
        
        # Weight bounds (e.g., {'AAPL': (0.05, 0.30), 'GOOGL': (0.10, 0.25)})
        if 'weight_bounds' in constraints:
            if isinstance(constraints['weight_bounds'], dict):
                # Individual asset bounds
                for asset, (min_w, max_w) in constraints['weight_bounds'].items():
                    asset_idx = ef.tickers.index(asset) if asset in ef.tickers else None
                    if asset_idx is not None:
                        ef.add_constraint(lambda w, idx=asset_idx: w[idx] >= min_w)
                        ef.add_constraint(lambda w, idx=asset_idx: w[idx] <= max_w)
            else:
                # Global bounds
                ef.add_constraint(lambda w: w >= constraints['weight_bounds'][0])
                ef.add_constraint(lambda w: w <= constraints['weight_bounds'][1])
        
        # Sector constraints
        if 'sector_constraints' in constraints:
            for sector, (min_weight, max_weight) in constraints['sector_constraints'].items():
                sector_assets = constraints.get('sector_mapper', {}).get(sector, [])
                if sector_assets:
                    sector_indices = [ef.tickers.index(asset) for asset in sector_assets if asset in ef.tickers]
                    if sector_indices:
                        ef.add_constraint(lambda w, indices=sector_indices: sum(w[i] for i in indices) >= min_weight)
                        ef.add_constraint(lambda w, indices=sector_indices: sum(w[i] for i in indices) <= max_weight)
        
        # Maximum number of assets constraint
        if 'max_assets' in constraints:
            max_assets = constraints['max_assets']
            # This requires a binary variable for each asset (more complex)
            # For now, we'll use L2 regularization to encourage sparsity
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        
        # Turnover constraint (limit portfolio changes)
        if 'max_turnover' in constraints and 'current_weights' in constraints:
            current = constraints['current_weights']
            max_turnover = constraints['max_turnover']
            
            for i, ticker in enumerate(ef.tickers):
                current_weight = current.get(ticker, 0)
                ef.add_constraint(lambda w, idx=i, curr=current_weight, max_t=max_turnover: 
                                w[idx] - curr <= max_t)
                ef.add_constraint(lambda w, idx=i, curr=current_weight, max_t=max_turnover: 
                                curr - w[idx] <= max_t)
        
        # Risk budget constraints (contribution to portfolio risk)
        if 'risk_budget' in constraints:
            risk_budgets = constraints['risk_budget']
            # This is complex to implement exactly, using approximation
            for asset, max_risk_contrib in risk_budgets.items():
                asset_idx = ef.tickers.index(asset) if asset in ef.tickers else None
                if asset_idx is not None:
                    # Approximate risk contribution with weight * volatility
                    asset_vol = np.sqrt(ef.cov_matrix[asset_idx, asset_idx])
                    ef.add_constraint(lambda w, idx=asset_idx, vol=asset_vol, max_rc=max_risk_contrib: 
                                    w[idx] * vol <= max_rc)
        
        # ESG constraints
        if 'esg_scores' in constraints and 'min_esg_score' in constraints:
            esg_scores = constraints['esg_scores']
            min_score = constraints['min_esg_score']
            
            # Portfolio weighted ESG score must exceed minimum
            score_array = np.array([esg_scores.get(ticker, 0) for ticker in ef.tickers])
            ef.add_constraint(lambda w: np.dot(w, score_array) >= min_score)
        
        # Liquidity constraints (minimum daily volume in dollars)
        if 'min_liquidity' in constraints and 'daily_volumes' in constraints:
            min_liquidity = constraints['min_liquidity']
            daily_volumes = constraints['daily_volumes']
            
            for i, ticker in enumerate(ef.tickers):
                volume = daily_volumes.get(ticker, 0)
                if volume < min_liquidity:
                    # Force weight to 0 for illiquid assets
                    ef.add_constraint(lambda w, idx=i: w[idx] == 0)
        
        # Correlation constraints (limit correlation between holdings)
        if 'max_correlation' in constraints:
            max_corr = constraints['max_correlation']
            corr_matrix = ef.cov_matrix.copy()
            
            # Normalize to correlation matrix
            std_devs = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(std_devs, std_devs)
            
            # For highly correlated pairs, limit combined weight
            for i in range(len(ef.tickers)):
                for j in range(i+1, len(ef.tickers)):
                    if abs(corr_matrix[i, j]) > max_corr:
                        ef.add_constraint(lambda w, idx1=i, idx2=j: w[idx1] + w[idx2] <= 0.3)
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                   returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        
        # Returns
        portfolio_returns = returns.dot(weights_array)
        annual_return = portfolio_returns.mean() * 252
        
        # Volatility
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        
        # Conditional Value at Risk
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
    
    def _calculate_risk_decomposition(self, weights: Dict[str, float], 
                                    returns: pd.DataFrame) -> Dict[str, float]:
        """Decompose portfolio risk by asset contribution"""
        
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        cov_matrix = returns.cov() * 252
        
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        # Marginal contributions to risk
        marginal_contrib = np.dot(cov_matrix, weights_array) / portfolio_vol
        
        # Component contributions
        contrib = weights_array * marginal_contrib
        
        # Percentage contributions
        percent_contrib = contrib / portfolio_vol * 100
        
        return dict(zip(returns.columns, percent_contrib))
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float], 
                                       returns: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio"""
        
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        
        # Individual volatilities
        individual_vols = returns.std() * np.sqrt(252)
        
        # Weighted average of individual volatilities
        weighted_avg_vol = np.dot(weights_array, individual_vols)
        
        # Portfolio volatility
        cov_matrix = returns.cov() * 252
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        return div_ratio
    
    def efficient_frontier(self, tickers: List[str], 
                         n_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        
        # Get historical data
        data = self._get_historical_data(tickers)
        if data.empty:
            return pd.DataFrame()
        
        returns = data.pct_change().dropna()
        
        if PYPFOPT_AVAILABLE:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            # Generate efficient frontier
            ef = EfficientFrontier(mu, S)
            
            frontier_points = []
            
            # Get range of returns
            min_ret = mu.min()
            max_ret = mu.max()
            
            target_returns = np.linspace(min_ret, max_ret, n_portfolios)
            
            for target in target_returns:
                try:
                    ef_temp = EfficientFrontier(mu, S)
                    ef_temp.efficient_return(target)
                    weights = ef_temp.clean_weights()
                    perf = ef_temp.portfolio_performance(verbose=False)
                    
                    frontier_points.append({
                        'return': perf[0],
                        'volatility': perf[1],
                        'sharpe': perf[2]
                    })
                except:
                    continue
            
            return pd.DataFrame(frontier_points)
        else:
            # Simplified frontier using random portfolios
            return self._generate_random_portfolios(returns, n_portfolios)
    
    def _generate_random_portfolios(self, returns: pd.DataFrame, 
                                  n_portfolios: int) -> pd.DataFrame:
        """Generate random portfolios for frontier visualization"""
        
        n_assets = len(returns.columns)
        results = []
        
        for _ in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Calculate metrics
            portfolio_return = np.dot(weights, returns.mean()) * 252
            cov_matrix = returns.cov() * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            results.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe
            })
        
        return pd.DataFrame(results)
    
    def rebalance_portfolio(self, current_weights: Dict[str, float],
                          target_weights: Dict[str, float],
                          current_values: Dict[str, float],
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Calculate rebalancing trades"""
        
        total_value = sum(current_values.values())
        
        trades = {}
        costs = 0
        
        for asset in target_weights:
            current_weight = current_weights.get(asset, 0)
            target_weight = target_weights[asset]
            
            # Check if rebalancing needed
            if abs(current_weight - target_weight) > threshold:
                current_value = current_values.get(asset, 0)
                target_value = total_value * target_weight
                
                trade_value = target_value - current_value
                trades[asset] = {
                    'action': 'BUY' if trade_value > 0 else 'SELL',
                    'value': abs(trade_value),
                    'current_weight': current_weight,
                    'target_weight': target_weight
                }
                
                # Estimate transaction costs (0.1% assumed)
                costs += abs(trade_value) * 0.001
        
        return {
            'trades': trades,
            'estimated_costs': costs,
            'cost_percentage': costs / total_value * 100
        }
    
    def monte_carlo_simulation(self, weights: Dict[str, float],
                             returns: pd.DataFrame,
                             n_simulations: int = 1000,
                             time_horizon: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio"""
        
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        
        # Calculate parameters
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Portfolio parameters
        portfolio_return = np.dot(weights_array, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        
        # Run simulations
        final_values = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, time_horizon
            )
            
            # Calculate portfolio path
            portfolio_returns = np.dot(random_returns, weights_array)
            cumulative_return = np.prod(1 + portfolio_returns)
            
            final_values.append(cumulative_return)
        
        final_values = np.array(final_values)
        
        # Calculate statistics
        return {
            'mean_return': np.mean(final_values) - 1,
            'median_return': np.median(final_values) - 1,
            'percentile_5': np.percentile(final_values, 5) - 1,
            'percentile_95': np.percentile(final_values, 95) - 1,
            'probability_loss': np.sum(final_values < 1) / n_simulations,
            'expected_shortfall': np.mean(final_values[final_values < 1]) - 1 if np.any(final_values < 1) else 0
        }
    
    def calculate_correlation_matrix(self, tickers: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for assets"""
        
        data = self._get_historical_data(tickers)
        if data.empty:
            return pd.DataFrame()
        
        returns = data.pct_change().dropna()
        return returns.corr()
    
    def suggest_portfolio_improvements(self, current_weights: Dict[str, float],
                                     returns: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest improvements to current portfolio"""
        
        suggestions = []
        
        # Calculate current metrics
        current_metrics = self._calculate_portfolio_metrics(current_weights, returns)
        
        # Try different optimizations
        strategies = ['max_sharpe', 'min_volatility', 'risk_parity']
        
        for strategy in strategies:
            optimized = self.optimize_portfolio(
                list(current_weights.keys()), 
                strategy=strategy
            )
            
            if 'weights' in optimized:
                improvement = {
                    'strategy': strategy,
                    'current_sharpe': current_metrics['sharpe_ratio'],
                    'optimized_sharpe': optimized.get('sharpe_ratio', 0),
                    'sharpe_improvement': optimized.get('sharpe_ratio', 0) - current_metrics['sharpe_ratio'],
                    'volatility_change': optimized.get('volatility', 0) - current_metrics['annual_volatility'],
                    'return_change': optimized.get('expected_return', 0) - current_metrics['annual_return']
                }
                
                suggestions.append(improvement)
        
        # Sort by Sharpe improvement
        suggestions.sort(key=lambda x: x['sharpe_improvement'], reverse=True)
        
        return suggestions