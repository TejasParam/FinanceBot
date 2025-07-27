#!/usr/bin/env python3
"""
Working backtest that properly tests the agentic system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from agentic_portfolio_manager import AgenticPortfolioManager

print("Initializing Agentic Portfolio Manager...")
manager = AgenticPortfolioManager(
    use_ml=True,
    use_llm=True,
    parallel_execution=False
)

# Test parameters
symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
test_dates = ['2023-06-15', '2023-09-15', '2023-11-15']

all_predictions = []

for test_date in test_dates:
    print(f"\n{'='*60}")
    print(f"Testing predictions for {test_date}")
    print('='*60)
    
    test_dt = pd.to_datetime(test_date)
    
    for symbol in symbols:
        try:
            print(f"\nAnalyzing {symbol}...")
            
            # Get historical data for the test date
            hist_data = yf.download(
                symbol,
                start=test_dt - timedelta(days=1),
                end=test_dt + timedelta(days=1),
                progress=False
            )
            
            if hist_data.empty:
                print(f"  No data available for {test_date}")
                continue
            
            # Get current price - extract scalar value properly
            current_price = float(hist_data['Close'].iloc[-1])
            
            # Run the agentic analysis
            analysis = manager.analyze_stock(symbol)
            
            # Extract results
            recommendation = analysis.get('recommendation', 'HOLD')
            confidence = analysis.get('confidence', 0.0)
            composite_score = analysis.get('composite_score', 0.0)
            
            # Get future price (5 trading days later)
            future_data = yf.download(
                symbol,
                start=test_dt,
                end=test_dt + timedelta(days=10),
                progress=False
            )
            
            if len(future_data) >= 6:
                future_price = float(future_data['Close'].iloc[5])
                actual_return = (future_price - current_price) / current_price
                
                # Check if prediction was correct
                predicted_up = recommendation in ['BUY', 'STRONG_BUY']
                actual_up = actual_return > 0
                correct = predicted_up == actual_up
                
                prediction = {
                    'date': test_date,
                    'symbol': symbol,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'composite_score': composite_score,
                    'current_price': current_price,
                    'future_price': future_price,
                    'actual_return': actual_return,
                    'predicted_direction': 'UP' if predicted_up else 'DOWN',
                    'actual_direction': 'UP' if actual_up else 'DOWN',
                    'correct': correct
                }
                
                all_predictions.append(prediction)
                
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Recommendation: {recommendation}")
                print(f"  Confidence: {confidence:.1%}")
                print(f"  Composite Score: {composite_score:.2f}")
                print(f"  Predicted: {'UP' if predicted_up else 'DOWN'}")
                print(f"  Actual: {'UP' if actual_up else 'DOWN'} ({actual_return:+.2%})")
                print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue

# Calculate final statistics
if all_predictions:
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS SUMMARY")
    print('='*60)
    
    total = len(all_predictions)
    correct = sum(1 for p in all_predictions if p['correct'])
    accuracy = correct / total
    
    # High confidence predictions
    high_conf = [p for p in all_predictions if p['confidence'] >= 0.7]
    high_conf_correct = sum(1 for p in high_conf if p['correct'])
    high_conf_accuracy = high_conf_correct / len(high_conf) if high_conf else 0
    
    # Very high confidence predictions
    very_high_conf = [p for p in all_predictions if p['confidence'] >= 0.8]
    very_high_conf_correct = sum(1 for p in very_high_conf if p['correct'])
    very_high_conf_accuracy = very_high_conf_correct / len(very_high_conf) if very_high_conf else 0
    
    print(f"\nTotal Predictions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Overall Accuracy: {accuracy:.1%}")
    
    print(f"\nHigh Confidence (≥70%) Predictions: {len(high_conf)}")
    print(f"High Confidence Accuracy: {high_conf_accuracy:.1%}")
    
    print(f"\nVery High Confidence (≥80%) Predictions: {len(very_high_conf)}")
    print(f"Very High Confidence Accuracy: {very_high_conf_accuracy:.1%}")
    
    # Save results
    os.makedirs('backtesting/results', exist_ok=True)
    with open('backtesting/results/working_backtest_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_predictions': total,
                'overall_accuracy': accuracy,
                'high_conf_accuracy': high_conf_accuracy,
                'very_high_conf_accuracy': very_high_conf_accuracy,
                'high_conf_count': len(high_conf),
                'very_high_conf_count': len(very_high_conf)
            },
            'predictions': all_predictions
        }, f, indent=2)
    
    print("\nResults saved to backtesting/results/working_backtest_results.json")
else:
    print("\nNo predictions were made successfully.")