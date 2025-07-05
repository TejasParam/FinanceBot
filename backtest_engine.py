import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class BacktestEngine:
    """
    Backtesting engine to evaluate trading strategies and rules
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = []
        
    def backtest_strategy(self, 
                         price_data: pd.DataFrame, 
                         strategy_func, 
                         transaction_cost: float = 0.001,
                         start_date: str = None,
                         end_date: str = None) -> Dict[str, Any]:
        """
        Backtest a trading strategy on historical data
        """
        print("📈 Running backtest...")
        
        # Filter data by date range if provided
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        if len(price_data) < 30:
            return {"error": "Insufficient data for backtesting"}
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'shares': 0,
            'portfolio_value': self.initial_capital,
            'transactions': []
        }
        
        portfolio_history = []
        
        # Run backtest day by day
        for i in range(1, len(price_data)):
            current_date = price_data.index[i]
            current_price = price_data['Close'].iloc[i]
            
            # Get historical data up to current point for strategy
            historical_data = price_data.iloc[:i+1]
            
            # Get strategy recommendation
            try:
                recommendation = strategy_func(historical_data)
                action = recommendation.get('recommendation', 'HOLD')
                confidence = recommendation.get('confidence', 0.5)
            except Exception as e:
                action = 'HOLD'
                confidence = 0.5
            
            # Execute trades based on recommendation
            if action == 'BUY' and portfolio['cash'] > current_price * 100:
                # Buy as many shares as possible (minimum 100 shares)
                shares_to_buy = int(portfolio['cash'] / current_price / 100) * 100
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    if cost <= portfolio['cash']:
                        portfolio['cash'] -= cost
                        portfolio['shares'] += shares_to_buy
                        portfolio['transactions'].append({
                            'date': current_date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'confidence': confidence
                        })
            
            elif action == 'SELL' and portfolio['shares'] > 0:
                # Sell all shares
                shares_to_sell = portfolio['shares']
                revenue = shares_to_sell * current_price * (1 - transaction_cost)
                portfolio['cash'] += revenue
                portfolio['shares'] = 0
                portfolio['transactions'].append({
                    'date': current_date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue,
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
                'action': action,
                'confidence': confidence
            })
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        metrics = self._calculate_performance_metrics(portfolio_df, price_data)
        
        return {
            'portfolio_history': portfolio_df,
            'transactions': portfolio['transactions'],
            'final_value': portfolio['portfolio_value'],
            'total_return': (portfolio['portfolio_value'] - self.initial_capital) / self.initial_capital,
            'metrics': metrics
        }
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various performance metrics"""
        
        # Portfolio returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Benchmark (buy and hold) returns
        benchmark_value = (price_data['Close'] / price_data['Close'].iloc[0]) * self.initial_capital
        benchmark_returns = benchmark_value.pct_change().dropna()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        if len(portfolio_returns) == 0:
            return {}
        
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        benchmark_return = (benchmark_value.iloc[-1] - self.initial_capital) / self.initial_capital
        
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (portfolio_returns > 0).sum()
        total_days = len(portfolio_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len([t for t in portfolio_df['action'] if t in ['BUY', 'SELL']])
        }
    
    def plot_backtest_results(self, backtest_results: Dict[str, Any], ticker: str):
        """Plot backtest results"""
        portfolio_df = backtest_results['portfolio_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results for {ticker}', fontsize=16)
        
        # Portfolio value over time
        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], label='Strategy', linewidth=2)
        benchmark_value = (portfolio_df['stock_price'] / portfolio_df['stock_price'].iloc[0]) * self.initial_capital
        ax1.plot(portfolio_df.index, benchmark_value, label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stock price with buy/sell signals
        ax2.plot(portfolio_df.index, portfolio_df['stock_price'], label='Stock Price', alpha=0.7)
        buy_signals = portfolio_df[portfolio_df['action'] == 'BUY']
        sell_signals = portfolio_df[portfolio_df['action'] == 'SELL']
        ax2.scatter(buy_signals.index, buy_signals['stock_price'], color='green', marker='^', s=100, label='Buy')
        ax2.scatter(sell_signals.index, sell_signals['stock_price'], color='red', marker='v', s=100, label='Sell')
        ax2.set_title('Stock Price with Trading Signals')
        ax2.set_ylabel('Stock Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max * 100
        ax3.fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(portfolio_df.index, drawdown, color='red')
        ax3.set_title('Drawdown (%)')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics table
        metrics = backtest_results['metrics']
        metrics_text = f"""
        Total Return: {metrics.get('total_return', 0):.2%}
        Benchmark Return: {metrics.get('benchmark_return', 0):.2%}
        Excess Return: {metrics.get('excess_return', 0):.2%}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
        Win Rate: {metrics.get('win_rate', 0):.2%}
        Total Trades: {metrics.get('total_trades', 0)}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        ax4.set_title('Performance Metrics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_strategies(self, strategies: List[Tuple[str, callable]], price_data: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple strategies"""
        results = []
        
        for name, strategy_func in strategies:
            print(f"📊 Testing strategy: {name}")
            backtest_result = self.backtest_strategy(price_data, strategy_func)
            
            if 'error' not in backtest_result:
                metrics = backtest_result['metrics']
                metrics['strategy_name'] = name
                metrics['final_value'] = backtest_result['final_value']
                results.append(metrics)
        
        return pd.DataFrame(results).set_index('strategy_name')
