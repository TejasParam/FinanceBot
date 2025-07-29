#!/usr/bin/env python3
"""
Realistic backtesting with market microstructure
Shows TRUE expected performance with all market frictions
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from agents.market_reality_engine import MarketRealityEngine, Order, OrderType
import warnings
warnings.filterwarnings('ignore')

class RealisticBacktester:
    """Backtester that includes all real-world market frictions"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.market_engine = MarketRealityEngine()
        
        # Transaction costs (realistic)
        self.commission_per_share = 0.005  # $0.005 per share
        self.sec_fee = 0.0000229  # SEC fee
        self.taf_fee = 0.000119   # FINRA TAF
        
        # Risk limits
        self.max_position_size = 0.1  # Max 10% in one position
        self.max_daily_trades = 500   # Regulatory limits
        
    def calculate_market_features(self, data: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Calculate realistic market microstructure features"""
        
        if idx < 20:
            return None
            
        current = data.iloc[idx]
        history = data.iloc[max(0, idx-20):idx]
        
        # Basic metrics
        returns = history['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Microstructure metrics
        spread = (current['High'] - current['Low']) / current['Close']
        volume_ratio = current['Volume'] / history['Volume'].mean()
        
        # Price momentum
        momentum = (current['Close'] - history['Close'].iloc[0]) / history['Close'].iloc[0]
        
        # Intraday metrics (using high/low as proxy)
        intraday_range = (current['High'] - current['Low']) / current['Close']
        close_to_high = (current['High'] - current['Close']) / (current['High'] - current['Low'])
        
        return {
            'price': current['Close'],
            'volatility': volatility,
            'spread': spread,
            'volume': current['Volume'],
            'adv': history['Volume'].mean(),  # Average daily volume
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'trend': 1 if momentum > 0 else -1,
            'intraday_range': intraday_range,
            'close_position': close_to_high
        }
    
    def generate_signal_with_ml(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Generate trading signal using ensemble of strategies
        This simulates what our ML models would do
        """
        
        signals = []
        
        # 1. Mean Reversion Signal
        if features['close_position'] < 0.2:  # Close near low
            signals.append((1, 0.6))  # Buy signal
        elif features['close_position'] > 0.8:  # Close near high
            signals.append((-1, 0.6))  # Sell signal
        
        # 2. Momentum Signal
        if abs(features['momentum']) > 0.02:  # 2% move
            if features['momentum'] > 0 and features['volume_ratio'] > 1.2:
                signals.append((1, 0.7))
            elif features['momentum'] < 0 and features['volume_ratio'] > 1.2:
                signals.append((-1, 0.7))
        
        # 3. Volatility Regime
        if features['volatility'] < 0.15:  # Low volatility
            # Mean reversion in low vol
            if features['intraday_range'] > 0.02:
                signals.append((-np.sign(features['momentum']), 0.5))
        else:
            # Trend following in high vol
            if abs(features['momentum']) > 0.01:
                signals.append((np.sign(features['momentum']), 0.5))
        
        # 4. Microstructure Signal
        if features['spread'] > 0.002:  # Wide spread
            # Provide liquidity
            signals.append((0, 0.8))  # No trade in wide spreads
        
        # Aggregate signals
        if not signals:
            return 0, 0.5
        
        # Weighted average
        total_weight = sum(s[1] for s in signals)
        if total_weight == 0:
            return 0, 0.5
            
        weighted_signal = sum(s[0] * s[1] for s in signals) / total_weight
        avg_confidence = total_weight / len(signals)
        
        # Decision threshold (realistic)
        if abs(weighted_signal) > 0.3 and avg_confidence > 0.6:
            return int(np.sign(weighted_signal)), avg_confidence
        
        return 0, avg_confidence
    
    def execute_trade_realistically(self, signal: int, features: Dict[str, Any], 
                                  ticker: str, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Execute trade with realistic market impact and frictions"""
        
        current_price = features['price']
        
        # Calculate position size with risk management
        volatility = features['volatility']
        
        # Kelly criterion with safety factor
        edge_estimate = 0.02  # Realistic 2% edge
        kelly_fraction = edge_estimate / volatility
        safety_factor = 0.25  # Use 1/4 Kelly
        
        position_fraction = min(kelly_fraction * safety_factor, self.max_position_size)
        trade_value = self.cash * position_fraction
        shares = int(trade_value / current_price / 100) * 100  # Round to 100 shares
        
        if shares < 100:  # Minimum size
            return None
        
        # Create order
        order = Order(
            id=f"{ticker}_{timestamp}_{signal}",
            side='BUY' if signal > 0 else 'SELL',
            price=current_price,
            size=shares,
            type=OrderType.MARKET,  # Using market for simplicity
            timestamp=timestamp.timestamp()
        )
        
        # Execute with market reality engine
        execution = self.market_engine.simulate_order_execution(order, features)
        
        if execution['status'] in ['FILLED', 'PARTIAL']:
            filled_shares = execution['filled_size']
            exec_price = execution['execution_price']
            
            # Transaction costs
            commission = filled_shares * self.commission_per_share
            sec_fee = filled_shares * exec_price * self.sec_fee
            taf_fee = filled_shares * self.taf_fee
            total_cost = commission + sec_fee + taf_fee
            
            # Update cash and positions
            if signal > 0:  # Buy
                cost = filled_shares * exec_price + total_cost
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[ticker] = self.positions.get(ticker, 0) + filled_shares
                    
                    trade_record = {
                        'timestamp': timestamp,
                        'ticker': ticker,
                        'side': 'BUY',
                        'shares': filled_shares,
                        'price': current_price,
                        'exec_price': exec_price,
                        'slippage_bps': execution['slippage_bps'],
                        'commission': total_cost,
                        'market_impact': execution['market_impact'].total
                    }
                    self.trades.append(trade_record)
                    return trade_record
            
            else:  # Sell
                if ticker in self.positions and self.positions[ticker] >= filled_shares:
                    self.positions[ticker] -= filled_shares
                    self.cash += filled_shares * exec_price - total_cost
                    
                    trade_record = {
                        'timestamp': timestamp,
                        'ticker': ticker,
                        'side': 'SELL',
                        'shares': filled_shares,
                        'price': current_price,
                        'exec_price': exec_price,
                        'slippage_bps': execution['slippage_bps'],
                        'commission': total_cost,
                        'market_impact': execution['market_impact'].total
                    }
                    self.trades.append(trade_record)
                    return trade_record
        
        return None
    
    def run_backtest(self, ticker: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run realistic backtest on historical data"""
        
        print(f"\nTesting {ticker} with realistic market conditions...")
        
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(data) < 50:
            return None
        
        # Track metrics
        daily_returns = []
        win_trades = 0
        total_trades = 0
        
        # Main backtest loop
        for i in range(20, len(data)):
            # Calculate features
            features = self.calculate_market_features(data, i)
            if features is None:
                continue
            
            # Generate signal
            signal, confidence = self.generate_signal_with_ml(features)
            
            # Execute if we have a signal
            if signal != 0:
                trade = self.execute_trade_realistically(
                    signal, features, ticker, data.index[i]
                )
                
                if trade:
                    total_trades += 1
            
            # Track daily returns
            if i > 20:
                prev_value = self._calculate_portfolio_value(data.iloc[i-1])
                curr_value = self._calculate_portfolio_value(data.iloc[i])
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # Calculate final metrics
        final_value = self._calculate_portfolio_value(data.iloc[-1])
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Analyze trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Calculate win rate (comparing exec price to next day's close)
            for i, trade in enumerate(self.trades[:-1]):
                if trade['side'] == 'BUY':
                    # Find next sell or end price
                    next_price = self._find_exit_price(trade, i, data)
                    if next_price > trade['exec_price']:
                        win_trades += 1
            
            win_rate = win_trades / len(self.trades) if self.trades else 0
            
            # Execution analysis
            avg_slippage = trades_df['slippage_bps'].mean()
            total_commission = trades_df['commission'].sum()
            total_impact = trades_df['market_impact'].sum()
            
        else:
            win_rate = 0
            avg_slippage = 0
            total_commission = 0
            total_impact = 0
        
        # Risk metrics
        if daily_returns:
            returns_array = np.array(daily_returns)
            volatility = np.std(returns_array) * np.sqrt(252)
            sharpe = (np.mean(returns_array) * 252) / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(daily_returns)
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'ticker': ticker,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_slippage_bps': avg_slippage,
            'total_commission': total_commission,
            'total_market_impact': total_impact,
            'execution_analysis': self.market_engine.get_execution_analysis()
        }
    
    def _calculate_portfolio_value(self, price_data) -> float:
        """Calculate total portfolio value"""
        value = self.cash
        
        for ticker, shares in self.positions.items():
            value += shares * price_data['Close']
        
        return value
    
    def _find_exit_price(self, entry_trade: Dict, trade_idx: int, data: pd.DataFrame) -> float:
        """Find exit price for a trade"""
        # Look for corresponding exit trade
        for i in range(trade_idx + 1, len(self.trades)):
            if (self.trades[i]['ticker'] == entry_trade['ticker'] and 
                self.trades[i]['side'] == 'SELL'):
                return self.trades[i]['exec_price']
        
        # No exit found, use last price
        return data['Close'].iloc[-1]
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)


def main():
    """Run realistic backtests to show true expected performance"""
    
    print("\n" + "="*80)
    print("REALISTIC BACKTEST WITH MARKET MICROSTRUCTURE")
    print("="*80)
    print("Including: slippage, market impact, commissions, adverse selection")
    
    # Test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)  # 1 year
    
    test_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    
    all_results = []
    
    for ticker in test_tickers:
        backtester = RealisticBacktester()
        result = backtester.run_backtest(ticker, start_date, end_date)
        
        if result:
            all_results.append(result)
            
            print(f"\n{ticker} Results:")
            print(f"  Total Return: {result['total_return']:.2%}")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Total Trades: {result['total_trades']}")
            print(f"  Avg Slippage: {result['avg_slippage_bps']:.1f} bps")
            print(f"  Execution Costs: ${result['total_commission']:.2f}")
            
            exec_analysis = result['execution_analysis']
            if 'adverse_selection_rate' in exec_analysis:
                print(f"  Adverse Selection: {exec_analysis['adverse_selection_rate']:.2%}")
    
    if all_results:
        print("\n" + "="*80)
        print("REALISTIC PERFORMANCE SUMMARY")
        print("="*80)
        
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
        avg_slippage = np.mean([r['avg_slippage_bps'] for r in all_results])
        
        print(f"Average Return: {avg_return:.2%}")
        print(f"Average Win Rate: {avg_win_rate:.2%}")
        print(f"Average Sharpe: {avg_sharpe:.2f}")
        print(f"Average Slippage: {avg_slippage:.1f} bps")
        
        print("\n" + "="*80)
        print("COMPARISON TO CLAIMS")
        print("="*80)
        print(f"Simulated Win Rate: 76.4%")
        print(f"Realistic Win Rate: {avg_win_rate:.1%}")
        print(f"Gap: {76.4 - avg_win_rate*100:.1f} percentage points")
        
        print("\nWHY THE GAP EXISTS:")
        print("1. Market Impact: Every trade moves the market against you")
        print("2. Adverse Selection: ~30% of trades are against informed flow")
        print("3. Slippage: Execution rarely happens at desired price")
        print("4. Hidden Liquidity: Can't see 25% of available liquidity")
        print("5. Predatory Algos: Other HFTs hunt your stops and front-run")
        
        print("\nTO COMPETE WITH TOP QUANT FIRMS:")
        print("- Need sub-50μs latency (colocation)")
        print("- Access to full order book data (expensive)")
        print("- Hundreds of simultaneous strategies")
        print("- Machine learning on tick data (not daily)")
        print("- $10M+ technology infrastructure")

if __name__ == "__main__":
    main()