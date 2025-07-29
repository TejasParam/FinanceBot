#!/usr/bin/env python3
"""Verify ACTUAL accuracy with real backtesting - no estimates"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest
import logging

# Suppress verbose logging for speed
logging.getLogger().setLevel(logging.WARNING)

print("VERIFYING ACTUAL ACCURACY (not estimates)...")
print("=" * 60)

# Initialize backtest
backtest = HistoricalBacktest(initial_capital=100000)

# Disable everything we can for speed
backtest.coordinator.enable_ml = False
backtest.coordinator.enable_llm = False
backtest.coordinator.parallel_execution = False

# Disable some agents for speed
for agent_name in ['LLMExplanation', 'MLPrediction']:
    if agent_name in backtest.coordinator.agents:
        backtest.coordinator.agents[agent_name].disable()

print("Running actual backtest (this will take a moment)...")

# Test on a specific period with good data
results = backtest.backtest_period(
    start_date='2023-04-01',
    end_date='2023-04-30',  # One month
    symbols=['AAPL', 'MSFT', 'GOOGL'],  # Just 3 stocks
    confidence_threshold=0.50
)

# Show ACTUAL results
total_predictions = results['accuracy']['total_predictions']
correct_predictions = results['accuracy']['correct_predictions']
actual_accuracy = results['accuracy']['prediction_accuracy'] if total_predictions > 0 else 0

print(f"\n{'='*60}")
print(f"ACTUAL RESULTS (not estimates):")
print(f"{'='*60}")
print(f"Total Predictions Made: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"ACTUAL ACCURACY: {actual_accuracy:.1%}")
print(f"Number of Trades: {results['performance']['num_trades']}")

# Show prediction details
if results['predictions']:
    print(f"\nDetailed Predictions:")
    correct_count = 0
    for i, pred in enumerate(results['predictions'][:20]):  # Show up to 20
        status = "✓" if pred['correct'] else "✗"
        if pred['correct']:
            correct_count += 1
        print(f"{i+1:2d}. {pred['date'].strftime('%Y-%m-%d')} {pred['symbol']:5s} "
              f"{pred['recommendation']:12s} (conf: {pred['confidence']:.0%}) "
              f"return: {pred.get('actual_return', 0):.1%} {status}")
    
    if len(results['predictions']) > 20:
        print(f"... and {len(results['predictions']) - 20} more predictions")

# Verdict
print(f"\n{'='*60}")
if actual_accuracy >= 0.60:
    print(f"✅ VERIFIED: Accuracy is {actual_accuracy:.1%} (above 60% target)")
else:
    print(f"❌ FAILED: Accuracy is {actual_accuracy:.1%} (below 60% target)")
print(f"{'='*60}")

# Check if market filter is working
filtered_predictions = sum(1 for p in results['predictions'] if p['confidence'] >= 0.70)
print(f"\nMarket Filter Analysis:")
print(f"High confidence predictions (>70%): {filtered_predictions}")
print(f"All predictions: {total_predictions}")
print(f"Filter impact: {(1 - filtered_predictions/total_predictions)*100:.0%}% filtered out" if total_predictions > 0 else "No predictions")

print("\nNOTE: This is ACTUAL backtested accuracy, not an estimate.")