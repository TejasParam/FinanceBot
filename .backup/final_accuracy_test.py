#!/usr/bin/env python3
"""Final accuracy test with all fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest
import logging

# Reduce logging
logging.getLogger().setLevel(logging.WARNING)

print("FINAL ACCURACY TEST WITH ALL IMPROVEMENTS")
print("=" * 60)

# Initialize backtest
backtest = HistoricalBacktest(initial_capital=100000)

# Configure for reasonable speed
backtest.coordinator.enable_ml = False
backtest.coordinator.enable_llm = False

# Run on different market conditions
test_configs = [
    ('2023-01-01', '2023-01-31', 'January 2023 (Uptrend)'),
    ('2023-05-01', '2023-05-31', 'May 2023 (Mixed)'),
    ('2023-09-01', '2023-09-30', 'September 2023 (Downtrend)'),
]

all_predictions = []
all_correct = 0
all_total = 0

for start, end, label in test_configs:
    print(f"\nTesting {label}...")
    
    results = backtest.backtest_period(
        start_date=start,
        end_date=end,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        confidence_threshold=0.60
    )
    
    predictions = results['accuracy']['total_predictions']
    correct = results['accuracy']['correct_predictions']
    accuracy = results['accuracy']['prediction_accuracy'] if predictions > 0 else 0
    
    all_total += predictions
    all_correct += correct
    
    print(f"  Predictions: {predictions}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    
    # Store predictions
    if results['predictions']:
        all_predictions.extend(results['predictions'][:5])

# Overall results
print(f"\n{'='*60}")
print("OVERALL RESULTS:")
print(f"Total Predictions: {all_total}")
print(f"Total Correct: {all_correct}")
print(f"Overall Accuracy: {all_correct/all_total:.1%}" if all_total > 0 else "No predictions")

# Show sample predictions
if all_predictions:
    print(f"\nSample Predictions:")
    for i, pred in enumerate(all_predictions[:10]):
        status = "✓" if pred['correct'] else "✗"
        print(f"  {pred['symbol']:5s} {pred['recommendation']:12s} "
              f"(conf: {pred['confidence']:.0%}) {status}")

# Analysis
print(f"\n{'='*60}")
print("ACCURACY ANALYSIS:")

if all_total == 0:
    print("❌ NO PREDICTIONS MADE - Filters are too restrictive")
elif all_correct/all_total >= 0.60:
    print(f"✅ SUCCESS: Achieved {all_correct/all_total:.1%} accuracy (target: 60%)")
else:
    print(f"❌ BELOW TARGET: {all_correct/all_total:.1%} accuracy (target: 60%)")

print("\nNOTE: The reduced confidence caps and balanced thresholds should")
print("provide more realistic predictions with better accuracy.")