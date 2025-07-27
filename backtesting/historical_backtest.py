#!/usr/bin/env python3
"""
Historical Backtesting for AlphaBot AI
Tests the 7-agent agentic system on historical market data to validate accuracy claims
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from agents.coordinator import AgentCoordinator
from data_collection import DataCollectionAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalBacktest:
    """Run historical backtests on the AlphaBot AI system"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.coordinator = AgentCoordinator(
            enable_ml=True,
            enable_llm=False,  # Disable LLM for faster backtesting
            parallel_execution=True
        )
        self.data_collector = DataCollectionAgent()
        
    def backtest_period(self, 
                       start_date: str, 
                       end_date: str,
                       symbols: List[str],
                       confidence_threshold: float = 0.70) -> Dict:
        """
        Backtest the agentic system for a specific time period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of stock symbols to test
            confidence_threshold: Minimum confidence for trades
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING PERIOD: {start_date} to {end_date}")
        logger.info(f"{'='*80}")
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}
        trades = []
        predictions = []
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get all trading days in the period
        trading_days = pd.bdate_range(start=start_dt, end=end_dt)
        
        # Download all data upfront for efficiency
        logger.info(f"Downloading historical data for {len(symbols)} symbols...")
        all_data = {}
        for symbol in symbols:
            try:
                # Get extra data for indicators
                data_start = start_dt - timedelta(days=100)
                data = yf.download(symbol, start=data_start, end=end_dt + timedelta(days=10), progress=False)
                if len(data) > 0:
                    all_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
        
        logger.info(f"Successfully downloaded data for {len(all_data)} symbols")
        
        # Simulate each trading day
        daily_returns = []
        daily_values = []
        
        for day_idx, current_date in enumerate(trading_days):
            logger.info(f"\n--- {current_date.strftime('%Y-%m-%d')} ---")
            
            # Update portfolio value based on current prices
            portfolio_value = cash
            for symbol, shares in positions.items():
                if symbol in all_data:
                    try:
                        current_price = all_data[symbol].loc[current_date]['Close']
                        portfolio_value += shares * current_price
                    except:
                        pass
            
            daily_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': portfolio_value - cash
            })
            
            # Calculate daily return
            if day_idx > 0:
                prev_value = daily_values[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                daily_returns.append(daily_return)
            
            # Only analyze every 5 trading days (weekly rebalancing)
            if day_idx % 5 != 0:
                continue
                
            # Analyze each symbol using the agentic system
            signals = []
            
            for symbol in list(all_data.keys())[:10]:  # Limit to 10 symbols for speed
                if symbol not in all_data:
                    continue
                    
                try:
                    # Get historical data up to current date
                    symbol_data = all_data[symbol]
                    historical_data = symbol_data[symbol_data.index <= current_date]
                    
                    if len(historical_data) < 50:
                        continue
                    
                    # Run agentic analysis
                    logger.info(f"  Analyzing {symbol}...")
                    try:
                        analysis = self.coordinator.analyze_stock(
                            ticker=symbol,
                            price_data=historical_data
                        )
                    except Exception as agent_error:
                        logger.error(f"  Agent error for {symbol}: {str(agent_error)}")
                        import traceback
                        traceback.print_exc()
                        analysis = {'error': str(agent_error)}
                    
                    # Extract signals
                    if 'error' not in analysis:
                        aggregated = analysis.get('aggregated_analysis', {})
                        confidence = aggregated.get('overall_confidence', 0)
                        recommendation = aggregated.get('recommendation', 'HOLD')
                        
                        # Use dynamic confidence threshold from market filter
                        dynamic_threshold = analysis.get('confidence_threshold', confidence_threshold)
                        
                        # Check if we should trade based on market filter
                        should_trade = analysis.get('should_trade', True)
                        
                        if confidence >= dynamic_threshold and should_trade:
                            signals.append({
                                'symbol': symbol,
                                'recommendation': recommendation,
                                'confidence': confidence,
                                'composite_score': aggregated.get('overall_score', 0)
                            })
                            
                            # Track prediction for accuracy calculation
                            current_price = historical_data.iloc[-1]['Close']
                            
                            # Get future price (5 days ahead)
                            future_date = current_date + timedelta(days=7)
                            future_price = None
                            
                            try:
                                future_data = symbol_data[symbol_data.index > current_date]
                                if len(future_data) >= 5:
                                    future_price = future_data.iloc[4]['Close']
                            except:
                                pass
                            
                            if future_price is not None:
                                actual_return = float((future_price - current_price) / current_price)
                                predicted_direction = recommendation in ['BUY', 'STRONG_BUY']
                                actual_direction = actual_return > 0
                                
                                predictions.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'recommendation': recommendation,
                                    'confidence': confidence,
                                    'predicted_direction': predicted_direction,
                                    'actual_direction': actual_direction,
                                    'correct': predicted_direction == actual_direction,
                                    'actual_return': actual_return
                                })
                    
                except Exception as e:
                    logger.warning(f"  Error analyzing {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Execute trades based on signals
            if signals:
                # Sort by confidence
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Rebalance portfolio
                # Sell positions not in top signals
                for symbol in list(positions.keys()):
                    if symbol not in [s['symbol'] for s in signals[:5]]:
                        # Sell position
                        try:
                            current_price = all_data[symbol].loc[current_date]['Close']
                            cash += positions[symbol] * current_price
                            trades.append({
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': positions[symbol],
                                'price': current_price
                            })
                            del positions[symbol]
                            logger.info(f"  SOLD {symbol}")
                        except:
                            pass
                
                # Buy top signals
                available_cash = cash * 0.95  # Keep 5% cash buffer
                position_size = available_cash / min(5, len(signals))
                
                for signal in signals[:5]:
                    symbol = signal['symbol']
                    if symbol not in positions and signal['recommendation'] in ['BUY', 'STRONG_BUY']:
                        try:
                            current_price = all_data[symbol].loc[current_date]['Close']
                            shares = int(position_size / current_price)
                            if shares > 0 and cash >= shares * current_price:
                                positions[symbol] = shares
                                cash -= shares * current_price
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': current_price,
                                    'confidence': signal['confidence']
                                })
                                logger.info(f"  BOUGHT {shares} shares of {symbol} at ${current_price:.2f}")
                        except:
                            pass
        
        # Calculate final metrics
        final_value = cash
        for symbol, shares in positions.items():
            if symbol in all_data:
                try:
                    final_price = all_data[symbol].iloc[-1]['Close']
                    final_value += shares * final_price
                except:
                    pass
        
        # Calculate accuracy
        if predictions:
            accuracy = sum(1 for p in predictions if p['correct']) / len(predictions)
        else:
            accuracy = 0
        
        # Calculate performance metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio
        if daily_returns:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-6)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak = daily_values[0]['portfolio_value']
        max_drawdown = 0
        for dv in daily_values:
            peak = max(peak, dv['portfolio_value'])
            drawdown = (peak - dv['portfolio_value']) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'period': {
                'start': start_date,
                'end': end_date
            },
            'performance': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': total_return * 252 / len(trading_days) if len(trading_days) > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades)
            },
            'accuracy': {
                'prediction_accuracy': accuracy,
                'total_predictions': len(predictions),
                'correct_predictions': sum(1 for p in predictions if p['correct'])
            },
            'trades': trades,
            'predictions': predictions,
            'daily_values': daily_values
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """Save backtest results to files"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trades to CSV
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        
        # Save predictions to CSV
        if results['predictions']:
            predictions_df = pd.DataFrame(results['predictions'])
            predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Save daily values to CSV
        if results['daily_values']:
            daily_df = pd.DataFrame(results['daily_values'])
            daily_df.to_csv(os.path.join(output_dir, 'daily_portfolio_values.csv'), index=False)
        
        logger.info(f"Results saved to {output_dir}")

def main():
    """Run a sample backtest"""
    backtest = HistoricalBacktest()
    
    # Test on a recent period
    results = backtest.backtest_period(
        start_date='2023-01-01',
        end_date='2023-12-31',
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    )
    
    # Save results
    backtest.save_results(results, 'backtesting/results/2023')
    
    # Print summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"Period: {results['period']['start']} to {results['period']['end']}")
    print(f"Initial Capital: ${results['performance']['initial_capital']:,.2f}")
    print(f"Final Value: ${results['performance']['final_value']:,.2f}")
    print(f"Total Return: {results['performance']['total_return']:.2%}")
    print(f"Prediction Accuracy: {results['accuracy']['prediction_accuracy']:.1%}")
    print(f"Total Trades: {results['performance']['num_trades']}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()