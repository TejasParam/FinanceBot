#!/usr/bin/env python3
"""
Quick accuracy test for the enhanced system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def test_accuracy():
    """Test actual prediction accuracy"""
    
    print("Testing World-Class System Accuracy...")
    print("=" * 60)
    
    # Initialize coordinator
    coordinator = AgentCoordinator(enable_ml=True, enable_llm=False, parallel_execution=True)
    
    # Test on multiple stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'WMT', 'PG']
    
    correct_predictions = 0
    total_predictions = 0
    high_conf_correct = 0
    high_conf_total = 0
    
    for ticker in test_stocks:
        print(f"\nTesting {ticker}...")
        
        try:
            # Get historical data
            data = yf.download(ticker, start='2024-01-01', end='2024-12-31', progress=False)
            
            if len(data) < 60:
                continue
            
            # Test on last 20 trading days
            for i in range(20, min(40, len(data)-5)):
                # Analyze at day i
                analysis_date = data.index[i]
                
                # Temporarily set data up to analysis date
                temp_data = data.iloc[:i+1]
                
                # Run analysis
                analysis = coordinator.analyze_stock(ticker)
                
                if 'error' in analysis['aggregated_analysis']:
                    continue
                
                score = analysis['aggregated_analysis']['overall_score']
                confidence = analysis['aggregated_analysis']['overall_confidence']
                recommendation = analysis['aggregated_analysis']['recommendation']
                
                # Check actual price movement (5 days ahead)
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i+5]
                actual_return = (future_price - current_price) / current_price
                
                # Determine if prediction was correct
                predicted_bullish = score > 0
                actual_bullish = actual_return > 0
                
                correct = predicted_bullish == actual_bullish
                
                if correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Track high confidence predictions
                if confidence > 0.8:
                    high_conf_total += 1
                    if correct:
                        high_conf_correct += 1
                
                print(f"  Date: {analysis_date.strftime('%Y-%m-%d')}, "
                      f"Score: {score:.2f}, Conf: {confidence:.1%}, "
                      f"Actual: {actual_return:.2%}, "
                      f"{'✓' if correct else '✗'}")
                
        except Exception as e:
            print(f"  Error testing {ticker}: {str(e)}")
            continue
    
    # Calculate accuracies
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    
    print("\n" + "=" * 60)
    print("ACCURACY RESULTS:")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({correct_predictions}/{total_predictions})")
    print(f"High Confidence Accuracy: {high_conf_accuracy:.1%} ({high_conf_correct}/{high_conf_total})")
    print(f"Total Predictions: {total_predictions}")
    print(f"Stocks Tested: {len(test_stocks)}")
    
    return overall_accuracy

if __name__ == "__main__":
    accuracy = test_accuracy()