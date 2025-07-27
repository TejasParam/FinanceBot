#!/usr/bin/env python3
"""Quick test of improved accuracy with minimal data"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest

print("Testing improved accuracy (quick test)...")
print("=" * 60)

# Initialize backtest
backtest = HistoricalBacktest(initial_capital=100000)

# Disable ML for speed
backtest.coordinator.enable_ml = False

# Test shorter period
results = backtest.backtest_period(
    start_date='2023-06-01',
    end_date='2023-06-30',  # Just June 2023
    symbols=['AAPL', 'MSFT', 'NVDA'],  # Just 3 stocks
    confidence_threshold=0.60
)

print(f"\nRESULTS:")
print(f"Predictions Made: {results['accuracy']['total_predictions']}")
print(f"Correct Predictions: {results['accuracy']['correct_predictions']}")
print(f"Accuracy: {results['accuracy']['prediction_accuracy']:.1%}")
print(f"Number of Trades: {results['performance']['num_trades']}")
print(f"Total Return: {results['performance']['total_return']:.2%}")

# Show filter impact
filtered_count = 0
if results['predictions']:
    print(f"\nPrediction Details:")
    for i, pred in enumerate(results['predictions'][:10]):
        status = "✓" if pred['correct'] else "✗"
        print(f"  {pred['date'].strftime('%m/%d')}: {pred['symbol']} {pred['recommendation']} "
              f"(conf: {pred['confidence']:.0%}) {status}")
        
# Check how many were filtered out
print(f"\nNOTE: Market filter is active - only trading in favorable conditions")
print(f"This reduces prediction count but should improve accuracy")