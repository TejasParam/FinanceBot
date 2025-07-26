#!/usr/bin/env python3
"""
Analyze why there's a discrepancy between claimed and actual accuracy
"""

import json
import pandas as pd

# Load the backtest results
with open('backtesting/results/working_backtest_results.json', 'r') as f:
    results = json.load(f)

predictions = pd.DataFrame(results['predictions'])

print("DISCREPANCY ANALYSIS")
print("="*60)

# 1. Check by date
print("\n1. Accuracy by Test Date:")
for date in predictions['date'].unique():
    date_preds = predictions[predictions['date'] == date]
    accuracy = date_preds['correct'].mean()
    print(f"   {date}: {accuracy:.1%} ({date_preds['correct'].sum()}/{len(date_preds)})")

# 2. Check by symbol
print("\n2. Accuracy by Symbol:")
for symbol in predictions['symbol'].unique():
    symbol_preds = predictions[predictions['symbol'] == symbol]
    accuracy = symbol_preds['correct'].mean()
    print(f"   {symbol}: {accuracy:.1%} ({symbol_preds['correct'].sum()}/{len(symbol_preds)})")

# 3. Check by recommendation type
print("\n3. Accuracy by Recommendation:")
for rec in predictions['recommendation'].unique():
    rec_preds = predictions[predictions['recommendation'] == rec]
    if len(rec_preds) > 0:
        accuracy = rec_preds['correct'].mean()
        print(f"   {rec}: {accuracy:.1%} ({rec_preds['correct'].sum()}/{len(rec_preds)})")

# 4. Check confidence levels
print("\n4. Confidence Distribution:")
print(f"   Average confidence: {predictions['confidence'].mean():.1%}")
print(f"   Min confidence: {predictions['confidence'].min():.1%}")
print(f"   Max confidence: {predictions['confidence'].max():.1%}")

# 5. Check returns
print("\n5. Return Analysis:")
print(f"   Average actual return: {predictions['actual_return'].mean():.2%}")
print(f"   Correct predictions avg return: {predictions[predictions['correct']]['actual_return'].mean():.2%}")
print(f"   Wrong predictions avg return: {predictions[~predictions['correct']]['actual_return'].mean():.2%}")

# 6. Direction bias
print("\n6. Direction Bias:")
up_predictions = (predictions['recommendation'].isin(['BUY', 'STRONG_BUY'])).sum()
down_predictions = (predictions['recommendation'].isin(['SELL', 'STRONG_SELL', 'HOLD'])).sum()
print(f"   UP predictions: {up_predictions}")
print(f"   DOWN/HOLD predictions: {down_predictions}")

actual_up = (predictions['actual_direction'] == 'UP').sum()
actual_down = (predictions['actual_direction'] == 'DOWN').sum()
print(f"   Actual UP moves: {actual_up}")
print(f"   Actual DOWN moves: {actual_down}")

print("\n7. Potential Issues:")
print("   • 5-day prediction window may be too short/noisy")
print("   • Test dates might be in volatile periods")
print("   • System may be overfit to recent data")
print("   • The 'theoretical' accuracy uses different assumptions")