#!/usr/bin/env python3
"""Direct accuracy test bypassing backtest infrastructure"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing accuracy improvements directly...")
print("=" * 60)

# Initialize coordinator with improvements
coordinator = AgentCoordinator(enable_ml=False, enable_llm=False, parallel_execution=False)

# Get some test data
print("\nDownloading test data...")
ticker = 'AAPL'
data = yf.download(ticker, start='2023-01-01', end='2023-06-01', progress=False)

# Test on multiple dates
test_dates = pd.date_range('2023-03-01', '2023-03-31', freq='7D')
predictions = []

print(f"\nAnalyzing {len(test_dates)} time periods...")

for test_date in test_dates:
    # Get data up to test date
    historical_data = data[data.index <= test_date]
    
    if len(historical_data) < 50:
        continue
    
    print(f"\nAnalyzing as of {test_date.strftime('%Y-%m-%d')}...")
    
    # Run analysis
    try:
        analysis = coordinator.analyze_stock(ticker, price_data=historical_data)
        
        # Check if we should trade
        should_trade = analysis.get('should_trade', True)
        filter_reason = analysis.get('filter_reason', 'N/A')
        confidence = analysis['aggregated_analysis']['overall_confidence']
        score = analysis['aggregated_analysis']['overall_score']
        recommendation = analysis['aggregated_analysis']['recommendation']
        
        print(f"  Should Trade: {should_trade}")
        print(f"  Filter Reason: {filter_reason}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Score: {score:.2f}")
        print(f"  Recommendation: {recommendation}")
        
        # Check actual outcome (5 days later)
        future_date = test_date + timedelta(days=7)
        if future_date in data.index:
            current_price = historical_data.iloc[-1]['Close']
            future_price = data.loc[future_date]['Close']
            actual_return = (future_price - current_price) / current_price
            
            predicted_bullish = recommendation in ['BUY', 'STRONG_BUY']
            actual_bullish = float(actual_return) > 0
            correct = predicted_bullish == actual_bullish
            
            if should_trade:  # Only count if we would have traded
                predictions.append({
                    'date': test_date,
                    'correct': correct,
                    'confidence': confidence,
                    'actual_return': actual_return
                })
                
                status = "✓" if correct else "✗"
                print(f"  Actual Return: {actual_return:.2%} {status}")
    
    except Exception as e:
        print(f"  Error: {str(e)}")

# Calculate accuracy
if predictions:
    correct_count = sum(1 for p in predictions if p['correct'])
    accuracy = correct_count / len(predictions)
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    
    print(f"\n{'='*60}")
    print(f"ACCURACY TEST RESULTS:")
    print(f"Total Predictions: {len(predictions)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average Confidence: {avg_confidence:.1%}")
    
    # Compare to expected baseline
    if accuracy >= 0.60:
        print(f"\n✅ SUCCESS: Accuracy improved to {accuracy:.1%} (target: 60%+)")
    else:
        print(f"\n❌ Below target: {accuracy:.1%} (target: 60%+)")
else:
    print("\nNo predictions made - filters may be too restrictive")