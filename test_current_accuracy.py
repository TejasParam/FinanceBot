#!/usr/bin/env python3
"""Quick test to get current system accuracy"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest

# Run minimal test
backtest = HistoricalBacktest(initial_capital=100000)

# Disable ML for speed
backtest.coordinator.enable_ml = False

print("Testing current system accuracy...")
print("Running backtest on 1 week of data...")

results = backtest.backtest_period(
    start_date='2023-06-01',
    end_date='2023-06-07',  # Just 1 week
    symbols=['AAPL', 'MSFT'],  # Just 2 stocks
    confidence_threshold=0.50  # Lower threshold
)

print(f"\n=== CURRENT SYSTEM ACCURACY ===")
print(f"Prediction Accuracy: {results['accuracy']['prediction_accuracy']:.1%}")
print(f"Total Predictions: {results['accuracy']['total_predictions']}")
print(f"Correct Predictions: {results['accuracy']['correct_predictions']}")

if results['accuracy']['total_predictions'] > 0:
    print(f"\nNote: This is based on a small sample of {results['accuracy']['total_predictions']} predictions")
    print("For more reliable accuracy estimates, run longer backtests")
else:
    print("\nNo predictions made - confidence threshold may be too high")