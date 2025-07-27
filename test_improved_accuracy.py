#!/usr/bin/env python3
"""Test the improved accuracy with market filtering and dynamic weights"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest
import pandas as pd

print("Testing improved accuracy with market filtering...")
print("=" * 80)

# Initialize backtest
backtest = HistoricalBacktest(initial_capital=100000)

# Run test on different periods to see improvement
test_periods = [
    ('2023-01-01', '2023-02-01', 'January 2023'),
    ('2023-03-01', '2023-04-01', 'March 2023'),
    ('2023-06-01', '2023-07-01', 'June 2023'),
]

overall_predictions = 0
overall_correct = 0

for start, end, label in test_periods:
    print(f"\nTesting {label}...")
    
    results = backtest.backtest_period(
        start_date=start,
        end_date=end,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        confidence_threshold=0.60  # Base threshold, will be adjusted dynamically
    )
    
    predictions = results['accuracy']['total_predictions']
    correct = results['accuracy']['correct_predictions']
    accuracy = results['accuracy']['prediction_accuracy']
    
    overall_predictions += predictions
    overall_correct += correct
    
    print(f"  Predictions: {predictions}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    
    # Show some trades if any
    if results['trades']:
        print(f"  Trades executed: {len(results['trades'])}")
        for trade in results['trades'][:3]:
            print(f"    {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} {trade['symbol']}")

print("\n" + "=" * 80)
print("OVERALL RESULTS:")
print(f"Total Predictions: {overall_predictions}")
print(f"Total Correct: {overall_correct}")
print(f"Overall Accuracy: {overall_correct/overall_predictions:.1%}" if overall_predictions > 0 else "No predictions")

print("\nNOTE: The market filter will reduce the number of predictions")
print("but should improve accuracy by avoiding unfavorable conditions.")