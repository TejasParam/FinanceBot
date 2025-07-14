import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedBacktestEngine:
    """
    Advanced backtesting engine with walk-forward analysis, realistic transaction costs,
    and comprehensive performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = []
        
    def walk_forward_analysis(self,
                            price_data: pd.DataFrame,
                            strategy_func: Callable,
                            optimization_func: Optional[Callable] = None,
                            in_sample_period: int = 252,  # 1 year
                            out_sample_period: int = 63,   # 3 months
                            optimization_metric: str = 'sharpe_ratio',
                            transaction_costs: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform walk-forward analysis to test strategy robustness
        
        Args:
            price_data: Historical price data
            strategy_func: Strategy function that takes (data, params) and returns signals
            optimization_func: Function to optimize strategy parameters
            in_sample_period: Days for in-sample optimization
            out_sample_period: Days for out-of-sample testing
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            transaction_costs: Dict with cost parameters
        """
        
        print("🚀 Starting walk-forward analysis...")
        
        if transaction_costs is None:
            transaction_costs = self._get_default_transaction_costs()
        
        # Initialize results storage
        wf_results = {
            'periods': [],
            'in_sample_performance': [],
            'out_sample_performance': [],
            'parameters': [],
            'equity_curve': []
        }
        
        # Calculate number of walk-forward periods
        total_days = len(price_data)
        n_periods = (total_days - in_sample_period) // out_sample_period
        
        current_capital = self.initial_capital
        cumulative_equity = []
        
        for period in range(n_periods):
            # Define data windows
            in_sample_start = period * out_sample_period
            in_sample_end = in_sample_start + in_sample_period
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + out_sample_period
            
            if out_sample_end > total_days:
                break
            
            # Get data slices
            in_sample_data = price_data.iloc[in_sample_start:in_sample_end]
            out_sample_data = price_data.iloc[out_sample_start:out_sample_end]
            
            print(f"\n📅 Period {period + 1}/{n_periods}")
            print(f"  In-sample: {in_sample_data.index[0].date()} to {in_sample_data.index[-1].date()}")
            print(f"  Out-sample: {out_sample_data.index[0].date()} to {out_sample_data.index[-1].date()}")
            
            # Optimize parameters on in-sample data
            if optimization_func:
                optimal_params = self._optimize_parameters(
                    in_sample_data,
                    strategy_func,
                    optimization_func,
                    optimization_metric,
                    transaction_costs
                )
                print(f"  Optimal parameters: {optimal_params}")
            else:
                optimal_params = {}  # Use default parameters
            
            # Backtest on in-sample data
            in_sample_result = self._run_backtest_period(
                in_sample_data,
                strategy_func,
                optimal_params,
                current_capital,
                transaction_costs
            )
            
            # Backtest on out-of-sample data
            out_sample_result = self._run_backtest_period(
                out_sample_data,
                strategy_func,
                optimal_params,
                current_capital,
                transaction_costs
            )
            
            # Update capital for next period
            current_capital = out_sample_result['final_value']
            
            # Store results
            wf_results['periods'].append({
                'period': period + 1,
                'in_sample_dates': (in_sample_data.index[0], in_sample_data.index[-1]),
                'out_sample_dates': (out_sample_data.index[0], out_sample_data.index[-1])
            })
            
            wf_results['in_sample_performance'].append(in_sample_result['metrics'])
            wf_results['out_sample_performance'].append(out_sample_result['metrics'])
            wf_results['parameters'].append(optimal_params)
            
            # Build cumulative equity curve
            if period == 0:
                cumulative_equity.extend(out_sample_result['portfolio_history']['portfolio_value'].tolist())
            else:
                # Scale to continue from previous period's final value
                scaled_values = out_sample_result['portfolio_history']['portfolio_value'] * (
                    current_capital / out_sample_result['portfolio_history']['portfolio_value'].iloc[0]
                )
                cumulative_equity.extend(scaled_values.tolist())
        
        # Calculate aggregate statistics
        wf_results['aggregate_stats'] = self._calculate_wf_statistics(wf_results)
        wf_results['equity_curve'] = cumulative_equity
        wf_results['final_capital'] = current_capital
        wf_results['total_return'] = (current_capital - self.initial_capital) / self.initial_capital
        
        return wf_results
    
    def _optimize_parameters(self,
                           data: pd.DataFrame,
                           strategy_func: Callable,
                           optimization_func: Callable,
                           metric: str,
                           transaction_costs: Dict[str, float]) -> Dict[str, Any]:
        """Optimize strategy parameters on in-sample data"""
        
        # Use optimization function to find best parameters
        # This is a simplified version - in practice, you'd use more sophisticated optimization
        param_grid = optimization_func(data)
        
        best_params = None
        best_metric = -np.inf
        
        for params in param_grid:
            # Run backtest with these parameters
            result = self._run_backtest_period(
                data,
                lambda d: strategy_func(d, params),
                {},
                self.initial_capital,
                transaction_costs
            )
            
            # Extract optimization metric
            metric_value = result['metrics'].get(metric, -np.inf)
            
            if metric_value > best_metric:
                best_metric = metric_value
                best_params = params
        
        return best_params
    
    def _run_backtest_period(self,
                           data: pd.DataFrame,
                           strategy_func: Callable,
                           params: Dict[str, Any],
                           initial_capital: float,
                           transaction_costs: Dict[str, float]) -> Dict[str, Any]:
        """Run backtest for a specific period"""
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'shares': 0,
            'portfolio_value': initial_capital,
            'transactions': []
        }
        
        portfolio_history = []
        
        # Run backtest
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Get historical data up to current point
            historical_data = data.iloc[:i+1]
            
            # Get strategy signal
            try:
                if params:
                    signal = strategy_func(historical_data, params)
                else:
                    signal = strategy_func(historical_data)
                
                action = signal.get('recommendation', 'HOLD')
                position_size = signal.get('position_size', 1.0)  # Fraction of capital
                confidence = signal.get('confidence', 0.5)
            except Exception as e:
                action = 'HOLD'
                position_size = 1.0
                confidence = 0.5
            
            # Execute trades with enhanced cost model
            trade_executed = False
            
            if action == 'BUY' and portfolio['shares'] == 0:
                # Calculate position size
                available_capital = portfolio['cash'] * position_size
                
                # Apply transaction costs
                costs = self._calculate_transaction_costs(
                    available_capital,
                    current_price,
                    'BUY',
                    transaction_costs
                )
                
                shares_to_buy = int((available_capital - costs['total_cost']) / current_price)
                
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * current_price + costs['total_cost']
                    
                    if total_cost <= portfolio['cash']:
                        portfolio['cash'] -= total_cost
                        portfolio['shares'] += shares_to_buy
                        trade_executed = True
                        
                        portfolio['transactions'].append({
                            'date': current_date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'costs': costs,
                            'total_cost': total_cost,
                            'confidence': confidence
                        })
            
            elif action == 'SELL' and portfolio['shares'] > 0:
                # Sell position
                shares_to_sell = portfolio['shares']
                gross_revenue = shares_to_sell * current_price
                
                # Apply transaction costs
                costs = self._calculate_transaction_costs(
                    gross_revenue,
                    current_price,
                    'SELL',
                    transaction_costs
                )
                
                net_revenue = gross_revenue - costs['total_cost']
                
                portfolio['cash'] += net_revenue
                portfolio['shares'] = 0
                trade_executed = True
                
                portfolio['transactions'].append({
                    'date': current_date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'costs': costs,
                    'net_revenue': net_revenue,
                    'confidence': confidence
                })
            
            # Calculate current portfolio value
            portfolio['portfolio_value'] = portfolio['cash'] + (portfolio['shares'] * current_price)
            
            # Record portfolio state
            portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio['portfolio_value'],
                'cash': portfolio['cash'],
                'shares': portfolio['shares'],
                'stock_price': current_price,
                'action': action if trade_executed else 'HOLD',
                'confidence': confidence,
                'position': 'LONG' if portfolio['shares'] > 0 else 'CASH'
            })
        
        # Create DataFrame and calculate metrics
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        metrics = self._calculate_enhanced_metrics(portfolio_df, data, initial_capital)
        
        return {
            'portfolio_history': portfolio_df,
            'transactions': portfolio['transactions'],
            'final_value': portfolio['portfolio_value'],
            'total_return': (portfolio['portfolio_value'] - initial_capital) / initial_capital,
            'metrics': metrics
        }
    
    def _calculate_transaction_costs(self,
                                   trade_value: float,
                                   price: float,
                                   action: str,
                                   cost_params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate realistic transaction costs including:
        - Commission (fixed + percentage)
        - Slippage
        - Market impact
        - Bid-ask spread
        """
        
        costs = {
            'commission_fixed': cost_params.get('commission_fixed', 1.0),
            'commission_pct': trade_value * cost_params.get('commission_pct', 0.0005),
            'slippage': 0,
            'market_impact': 0,
            'spread_cost': 0
        }
        
        # Slippage (price movement during order execution)
        if 'slippage_pct' in cost_params:
            slippage_multiplier = 1 + cost_params['slippage_pct'] if action == 'BUY' else 1 - cost_params['slippage_pct']
            costs['slippage'] = abs(trade_value * cost_params['slippage_pct'])
        
        # Market impact (for large orders)
        if 'market_impact_factor' in cost_params:
            # Simplified square-root market impact model
            impact = cost_params['market_impact_factor'] * np.sqrt(trade_value / 1000000)  # Normalize by $1M
            costs['market_impact'] = trade_value * impact
        
        # Bid-ask spread cost
        if 'spread_bps' in cost_params:
            costs['spread_cost'] = trade_value * cost_params['spread_bps'] / 10000  # Convert bps to decimal
        
        costs['total_cost'] = sum(costs.values())
        
        return costs
    
    def _get_default_transaction_costs(self) -> Dict[str, float]:
        """Get default transaction cost parameters"""
        return {
            'commission_fixed': 1.0,        # $1 fixed commission
            'commission_pct': 0.0005,       # 0.05% of trade value
            'slippage_pct': 0.0001,        # 0.01% slippage
            'market_impact_factor': 0.0001, # Market impact factor
            'spread_bps': 5                 # 5 basis points bid-ask spread
        }
    
    def _calculate_enhanced_metrics(self,
                                  portfolio_df: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Portfolio returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Benchmark returns
        benchmark_value = (price_data['Close'] / price_data['Close'].iloc[0]) * initial_capital
        benchmark_returns = benchmark_value.pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        benchmark_return = (benchmark_value.iloc[-1] - initial_capital) / initial_capital
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted returns
        risk_free_rate = 0.05  # 5% annual
        sharpe_ratio = ((portfolio_returns.mean() * 252) - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = ((portfolio_returns.mean() * 252) - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_start = drawdown.idxmin()
        if max_drawdown < 0:
            recovery_dates = drawdown[drawdown_start:].loc[drawdown[drawdown_start:] >= -0.01].index
            drawdown_duration = len(drawdown[drawdown_start:recovery_dates[0]]) if len(recovery_dates) > 0 else len(drawdown[drawdown_start:])
        else:
            drawdown_duration = 0
        
        # Win/loss statistics
        winning_days = (portfolio_returns > 0).sum()
        losing_days = (portfolio_returns < 0).sum()
        total_days = len(portfolio_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        avg_win = portfolio_returns[portfolio_returns > 0].mean() if winning_days > 0 else 0
        avg_loss = portfolio_returns[portfolio_returns < 0].mean() if losing_days > 0 else 0
        profit_factor = abs(avg_win * winning_days) / abs(avg_loss * losing_days) if losing_days > 0 and avg_loss != 0 else np.inf
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (portfolio_returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        trades = portfolio_df[portfolio_df['action'].isin(['BUY', 'SELL'])]
        n_trades = len(trades)
        
        # Calculate holding periods
        buy_dates = trades[trades['action'] == 'BUY'].index
        sell_dates = trades[trades['action'] == 'SELL'].index
        
        holding_periods = []
        for i in range(min(len(buy_dates), len(sell_dates))):
            holding_period = (sell_dates[i] - buy_dates[i]).days
            holding_periods.append(holding_period)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            # Returns
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_df)) - 1,
            
            # Risk
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'drawdown_duration_days': drawdown_duration,
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            
            # Win/loss statistics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_day': portfolio_returns.max(),
            'worst_day': portfolio_returns.min(),
            
            # Trading statistics
            'total_trades': n_trades,
            'avg_holding_period_days': avg_holding_period,
            'turnover': n_trades / len(portfolio_df) * 252,  # Annualized
            
            # Distribution
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns)
        }
    
    def _calculate_wf_statistics(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate statistics for walk-forward analysis"""
        
        in_sample_metrics = pd.DataFrame(wf_results['in_sample_performance'])
        out_sample_metrics = pd.DataFrame(wf_results['out_sample_performance'])
        
        # Calculate consistency metrics
        in_sample_sharpe = in_sample_metrics['sharpe_ratio'].mean()
        out_sample_sharpe = out_sample_metrics['sharpe_ratio'].mean()
        
        # Correlation between in-sample and out-of-sample performance
        if len(in_sample_metrics) > 2:
            performance_correlation = in_sample_metrics['total_return'].corr(out_sample_metrics['total_return'])
        else:
            performance_correlation = 0
        
        # Stability of parameters (if numeric)
        param_stability = {}
        if wf_results['parameters'] and len(wf_results['parameters']) > 1:
            # Calculate parameter stability (simplified)
            param_keys = list(wf_results['parameters'][0].keys())
            for key in param_keys:
                try:
                    values = [p.get(key, 0) for p in wf_results['parameters'] if isinstance(p.get(key, 0), (int, float))]
                    if values:
                        param_stability[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                        }
                except:
                    pass
        
        return {
            'avg_in_sample_return': in_sample_metrics['total_return'].mean(),
            'avg_out_sample_return': out_sample_metrics['total_return'].mean(),
            'avg_in_sample_sharpe': in_sample_sharpe,
            'avg_out_sample_sharpe': out_sample_sharpe,
            'performance_degradation': (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe if in_sample_sharpe != 0 else 0,
            'performance_correlation': performance_correlation,
            'consistency_ratio': out_sample_sharpe / in_sample_sharpe if in_sample_sharpe != 0 else 0,
            'parameter_stability': param_stability,
            'n_profitable_periods': (out_sample_metrics['total_return'] > 0).sum(),
            'n_total_periods': len(out_sample_metrics)
        }
    
    def plot_walk_forward_results(self, wf_results: Dict[str, Any], ticker: str):
        """Plot walk-forward analysis results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Walk-Forward Analysis Results for {ticker}', fontsize=16)
        
        # Equity curve
        ax1.plot(wf_results['equity_curve'], linewidth=2)
        ax1.set_title('Cumulative Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # In-sample vs Out-of-sample returns
        periods = range(1, len(wf_results['in_sample_performance']) + 1)
        in_returns = [p['total_return'] for p in wf_results['in_sample_performance']]
        out_returns = [p['total_return'] for p in wf_results['out_sample_performance']]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax2.bar(x - width/2, in_returns, width, label='In-Sample', alpha=0.8)
        ax2.bar(x + width/2, out_returns, width, label='Out-of-Sample', alpha=0.8)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Return')
        ax2.set_title('Period Returns Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sharpe ratio comparison
        in_sharpe = [p['sharpe_ratio'] for p in wf_results['in_sample_performance']]
        out_sharpe = [p['sharpe_ratio'] for p in wf_results['out_sample_performance']]
        
        ax3.plot(periods, in_sharpe, 'o-', label='In-Sample', linewidth=2)
        ax3.plot(periods, out_sharpe, 's-', label='Out-of-Sample', linewidth=2)
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        stats = wf_results['aggregate_stats']
        stats_text = f"""
        Total Return: {wf_results['total_return']:.2%}
        Avg Out-of-Sample Return: {stats['avg_out_sample_return']:.2%}
        Avg Out-of-Sample Sharpe: {stats['avg_out_sample_sharpe']:.2f}
        Performance Correlation: {stats['performance_correlation']:.2f}
        Consistency Ratio: {stats['consistency_ratio']:.2f}
        Profitable Periods: {stats['n_profitable_periods']}/{stats['n_total_periods']}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax4.set_title('Walk-Forward Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def monte_carlo_analysis(self,
                           backtest_results: Dict[str, Any],
                           n_simulations: int = 1000,
                           confidence_levels: List[float] = [0.05, 0.25, 0.75, 0.95]) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis on backtest results to assess strategy robustness
        """
        
        portfolio_returns = backtest_results['portfolio_history']['portfolio_value'].pct_change().dropna()
        
        # Bootstrap returns
        simulation_results = []
        
        for _ in range(n_simulations):
            # Resample returns with replacement
            resampled_returns = np.random.choice(portfolio_returns, size=len(portfolio_returns), replace=True)
            
            # Calculate cumulative performance
            cumulative_return = np.prod(1 + resampled_returns) - 1
            
            # Calculate metrics
            volatility = np.std(resampled_returns) * np.sqrt(252)
            sharpe = (np.mean(resampled_returns) * 252 - 0.05) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + resampled_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            simulation_results.append({
                'total_return': cumulative_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            })
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(simulation_results)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for metric in results_df.columns:
            confidence_intervals[metric] = {}
            for level in confidence_levels:
                confidence_intervals[metric][f'{int(level*100)}%'] = np.percentile(results_df[metric], level * 100)
        
        return {
            'simulation_results': results_df,
            'confidence_intervals': confidence_intervals,
            'expected_metrics': results_df.mean().to_dict(),
            'metric_std': results_df.std().to_dict(),
            'probability_positive_return': (results_df['total_return'] > 0).mean(),
            'probability_beat_benchmark': (results_df['total_return'] > backtest_results['metrics']['benchmark_return']).mean()
        }


# Example usage
if __name__ == "__main__":
    # Initialize enhanced backtest engine
    engine = EnhancedBacktestEngine(initial_capital=100000)
    
    # Define a sample strategy
    def sample_strategy(data, params=None):
        """Simple MA crossover strategy"""
        if params is None:
            params = {'fast_ma': 20, 'slow_ma': 50}
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(params['fast_ma']).mean()
        slow_ma = data['Close'].rolling(params['slow_ma']).mean()
        
        # Generate signal
        if len(data) < params['slow_ma']:
            return {'recommendation': 'HOLD', 'confidence': 0.5}
        
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        # Crossover logic
        if current_fast > current_slow and prev_fast <= prev_slow:
            return {'recommendation': 'BUY', 'confidence': 0.8, 'position_size': 1.0}
        elif current_fast < current_slow and prev_fast >= prev_slow:
            return {'recommendation': 'SELL', 'confidence': 0.8}
        else:
            return {'recommendation': 'HOLD', 'confidence': 0.5}
    
    # Define optimization function
    def optimize_ma_params(data):
        """Generate parameter grid for MA strategy"""
        param_grid = []
        
        for fast in range(10, 30, 5):
            for slow in range(30, 60, 10):
                if fast < slow:
                    param_grid.append({'fast_ma': fast, 'slow_ma': slow})
        
        return param_grid
    
    # Load sample data (you would use real data here)
    import yfinance as yf
    ticker = "AAPL"
    data = yf.download(ticker, start="2020-01-01", end="2023-12-31", progress=False)
    
    # Run walk-forward analysis
    print("Running walk-forward analysis...")
    wf_results = engine.walk_forward_analysis(
        data,
        sample_strategy,
        optimize_ma_params,
        in_sample_period=252,
        out_sample_period=63
    )
    
    print(f"\nTotal Return: {wf_results['total_return']:.2%}")
    print(f"Final Capital: ${wf_results['final_capital']:,.2f}")
    
    # Plot results
    engine.plot_walk_forward_results(wf_results, ticker)