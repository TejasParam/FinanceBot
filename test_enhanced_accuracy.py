#!/usr/bin/env python3
"""
Test the enhanced trading system accuracy with Renaissance-style improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from agents.coordinator import AgentCoordinator

def test_enhanced_accuracy():
    """Test the system with all Renaissance enhancements"""
    
    print("="*80)
    print("TESTING ENHANCED TRADING SYSTEM ACCURACY")
    print("With Renaissance Technologies-Style Improvements")
    print("="*80)
    
    # Initialize coordinator with all features enabled
    coordinator = AgentCoordinator(enable_ml=True, enable_llm=False)
    
    # Test stocks from different sectors
    test_stocks = [
        'AAPL',  # Technology
        'JPM',   # Finance
        'XOM',   # Energy
        'AMZN',  # Consumer
        'JNJ',   # Healthcare
        'MSFT',  # Technology
        'BAC',   # Finance
        'GOOGL', # Technology
        'WMT',   # Retail
        'NVDA'   # Semiconductors
    ]
    
    # Track predictions and actual outcomes
    predictions = []
    actual_outcomes = []
    prediction_details = []
    
    print("\nAnalyzing stocks and making predictions...")
    print("-" * 80)
    
    for i, ticker in enumerate(test_stocks):
        print(f"\n[{i+1}/{len(test_stocks)}] Analyzing {ticker}...")
        
        try:
            # Get historical data
            stock_data = yf.download(ticker, period='3mo', progress=False)
            if len(stock_data) < 50:
                print(f"  ⚠️  Insufficient data for {ticker}")
                continue
            
            # Get price 5 days ago for prediction
            if len(stock_data) >= 5:
                price_5d_ago = stock_data['Close'].iloc[-5]
                current_price = stock_data['Close'].iloc[-1]
                actual_return = (current_price / price_5d_ago - 1) * 100
                
                # Make prediction using the system
                result = coordinator.analyze_stock(ticker)
                
                # Extract prediction
                overall_score = result['aggregated_analysis']['overall_score']
                confidence = result['aggregated_analysis']['overall_confidence']
                recommendation = result['aggregated_analysis']['recommendation']
                
                # Get HFT and StatArb signals if available
                hft_score = result['agent_results'].get('HFTEngine', {}).get('score', 0)
                stat_arb_score = result['agent_results'].get('StatisticalArbitrage', {}).get('score', 0)
                
                # Determine predicted direction
                if overall_score > 0.3 and confidence > 0.7:
                    predicted_direction = 1  # Up
                elif overall_score < -0.3 and confidence > 0.7:
                    predicted_direction = -1  # Down
                else:
                    predicted_direction = 0  # Neutral/Hold
                
                # Determine actual direction
                if actual_return > 0.5:
                    actual_direction = 1
                elif actual_return < -0.5:
                    actual_direction = -1
                else:
                    actual_direction = 0
                
                # Store results
                predictions.append(predicted_direction)
                actual_outcomes.append(actual_direction)
                
                # Detailed tracking
                prediction_details.append({
                    'ticker': ticker,
                    'predicted_direction': predicted_direction,
                    'actual_direction': actual_direction,
                    'overall_score': overall_score,
                    'confidence': confidence,
                    'recommendation': recommendation,
                    'actual_return': actual_return,
                    'hft_score': hft_score,
                    'stat_arb_score': stat_arb_score,
                    'correct': predicted_direction == actual_direction
                })
                
                # Print summary
                print(f"  Prediction: {recommendation} (score: {overall_score:.3f}, conf: {confidence:.3f})")
                print(f"  HFT Signal: {hft_score:.3f}, StatArb Signal: {stat_arb_score:.3f}")
                print(f"  Actual 5-day return: {actual_return:+.2f}%")
                print(f"  Result: {'✓ CORRECT' if predicted_direction == actual_direction else '✗ INCORRECT'}")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {ticker}: {str(e)}")
            continue
    
    # Calculate accuracy metrics
    print("\n" + "="*80)
    print("ACCURACY RESULTS")
    print("="*80)
    
    if len(predictions) > 0:
        # Overall accuracy
        correct_predictions = sum(1 for p, a in zip(predictions, actual_outcomes) if p == a)
        overall_accuracy = correct_predictions / len(predictions) * 100
        
        # Directional accuracy (excluding neutral)
        directional_predictions = [(p, a) for p, a in zip(predictions, actual_outcomes) if p != 0]
        if directional_predictions:
            directional_correct = sum(1 for p, a in directional_predictions if p == a)
            directional_accuracy = directional_correct / len(directional_predictions) * 100
        else:
            directional_accuracy = 0
        
        # High confidence accuracy
        high_conf_predictions = [d for d in prediction_details if d['confidence'] > 0.75]
        if high_conf_predictions:
            high_conf_correct = sum(1 for d in high_conf_predictions if d['correct'])
            high_conf_accuracy = high_conf_correct / len(high_conf_predictions) * 100
        else:
            high_conf_accuracy = 0
        
        print(f"\nOverall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{len(predictions)})")
        print(f"Directional Accuracy: {directional_accuracy:.1f}% (when making UP/DOWN calls)")
        print(f"High Confidence Accuracy: {high_conf_accuracy:.1f}% (confidence > 75%)")
        
        # Breakdown by signal strength
        strong_signals = [d for d in prediction_details if abs(d['overall_score']) > 0.5]
        if strong_signals:
            strong_correct = sum(1 for d in strong_signals if d['correct'])
            strong_accuracy = strong_correct / len(strong_signals) * 100
            print(f"Strong Signal Accuracy: {strong_accuracy:.1f}% (|score| > 0.5)")
        
        # Renaissance-style metrics
        print("\n" + "-"*40)
        print("RENAISSANCE-STYLE METRICS:")
        
        # Check if HFT engine is contributing
        hft_active = sum(1 for d in prediction_details if abs(d['hft_score']) > 0.1)
        stat_arb_active = sum(1 for d in prediction_details if abs(d['stat_arb_score']) > 0.1)
        
        print(f"HFT Engine Active: {hft_active}/{len(prediction_details)} trades")
        print(f"StatArb Active: {stat_arb_active}/{len(prediction_details)} trades")
        
        # Win rate by recommendation type
        print("\nAccuracy by Recommendation:")
        for rec in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
            rec_preds = [d for d in prediction_details if d['recommendation'] == rec]
            if rec_preds:
                rec_correct = sum(1 for d in rec_preds if d['correct'])
                rec_accuracy = rec_correct / len(rec_preds) * 100
                print(f"  {rec}: {rec_accuracy:.1f}% ({rec_correct}/{len(rec_preds)})")
        
        # Expected vs Renaissance Medallion (50.75% on 150k trades/day)
        print("\n" + "-"*40)
        print("COMPARISON TO RENAISSANCE MEDALLION:")
        print(f"Our Accuracy: {overall_accuracy:.1f}%")
        print(f"Medallion Target: 50.75% (on 150,000+ trades/day)")
        print(f"Relative Performance: {overall_accuracy/50.75*100:.1f}% of Medallion")
        
        # Estimate daily trade capacity
        avg_confidence = np.mean([d['confidence'] for d in prediction_details])
        estimated_daily_trades = int(avg_confidence * 1000)  # Rough estimate
        print(f"\nEstimated Daily Trade Capacity: {estimated_daily_trades:,}")
        print(f"Medallion Daily Trades: 150,000+")
        
    else:
        print("No valid predictions to evaluate")
    
    # Save detailed results
    if prediction_details:
        df = pd.DataFrame(prediction_details)
        df.to_csv('enhanced_accuracy_results.csv', index=False)
        print(f"\nDetailed results saved to: enhanced_accuracy_results.csv")
    
    return overall_accuracy if len(predictions) > 0 else 0

if __name__ == "__main__":
    accuracy = test_enhanced_accuracy()