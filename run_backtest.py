#!/usr/bin/env python3
"""
Enhanced Backtest System - Targeting 80%+ Accuracy
Tests the improved agentic system with enhanced agents
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
from typing import List, Dict
warnings.filterwarnings('ignore')

from agents.coordinator import AgentCoordinator

class EnhancedBacktest:
    """Enhanced backtest system with improved accuracy"""
    
    def __init__(self):
        self.manager = AgentCoordinator(
            enable_ml=True,
            enable_llm=False,  # Disable LLM for testing
            parallel_execution=False  # Sequential for consistency
        )
        
    def run_backtest(self, symbols: List[str], test_dates: List[str]) -> Dict:
        """Run enhanced backtest"""
        
        all_predictions = []
        trades_analyzed = 0
        trades_taken = 0
        
        for test_date in test_dates:
            print(f"\n{'='*60}")
            print(f"Testing {test_date}")
            print('='*60)
            
            test_dt = pd.to_datetime(test_date)
            
            for symbol in symbols:
                trades_analyzed += 1
                
                try:
                    print(f"\nAnalyzing {symbol}...")
                    
                    # Get historical data
                    hist_data = yf.download(
                        symbol,
                        start=test_dt - timedelta(days=100),
                        end=test_dt + timedelta(days=1),
                        progress=False
                    )
                    
                    if len(hist_data) < 50:
                        print(f"  Insufficient data")
                        continue
                    
                    current_price = float(hist_data['Close'].iloc[-1])
                    
                    # Run enhanced analysis
                    analysis = self.manager.analyze_stock(symbol)
                    
                    # Extract from aggregated analysis
                    aggregated = analysis.get('aggregated_analysis', {})
                    recommendation = aggregated.get('recommendation', 'HOLD')
                    confidence = aggregated.get('overall_confidence', 0.0)
                    composite_score = aggregated.get('overall_score', 0.0)
                    
                    # Apply enhanced filtering
                    # Only take trades with confidence >= 75%
                    if confidence < 0.75:
                        print(f"  Skipping - Low confidence ({confidence:.1%})")
                        continue
                    
                    # Skip weak signals
                    if abs(composite_score) < 0.35:
                        print(f"  Skipping - Weak signal ({composite_score:.2f})")
                        continue
                    
                    # Calculate simple trend for confirmation
                    sma20 = float(hist_data['Close'].rolling(20).mean().iloc[-1])
                    trend = 'up' if current_price > sma20 else 'down'
                    
                    # Skip counter-trend trades
                    if trend == 'up' and recommendation in ['SELL', 'STRONG_SELL']:
                        print(f"  Skipping - Against uptrend")
                        continue
                    if trend == 'down' and recommendation in ['BUY', 'STRONG_BUY']:
                        print(f"  Skipping - Against downtrend")
                        continue
                    
                    # Get future price (10-day window)
                    future_data = yf.download(
                        symbol,
                        start=test_dt,
                        end=test_dt + timedelta(days=15),
                        progress=False
                    )
                    
                    if len(future_data) >= 11:
                        future_price = float(future_data['Close'].iloc[10])
                        actual_return = (future_price - current_price) / current_price
                        
                        # Determine if prediction was correct
                        if recommendation in ['BUY', 'STRONG_BUY']:
                            correct = actual_return > 0.005  # Need 0.5% gain
                        elif recommendation in ['SELL', 'STRONG_SELL']:
                            correct = actual_return < -0.005  # Need 0.5% loss
                        else:  # HOLD
                            correct = abs(actual_return) < 0.02  # Less than 2% move
                        
                        trades_taken += 1
                        
                        prediction = {
                            'date': test_date,
                            'symbol': symbol,
                            'recommendation': recommendation,
                            'confidence': confidence,
                            'composite_score': composite_score,
                            'trend': trend,
                            'current_price': current_price,
                            'future_price': future_price,
                            'actual_return': actual_return,
                            'correct': correct
                        }
                        
                        all_predictions.append(prediction)
                        
                        print(f"  ✓ TRADE TAKEN")
                        print(f"  Recommendation: {recommendation} ({confidence:.1%})")
                        print(f"  Score: {composite_score:.2f}")
                        print(f"  10-day return: {actual_return:+.2%}")
                        print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
                        
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        print(f"\n\nTotal analyzed: {trades_analyzed}")
        print(f"Trades taken: {trades_taken}")
        print(f"Selectivity: {trades_taken/trades_analyzed*100:.1%}")
        
        # Calculate results
        if all_predictions:
            df = pd.DataFrame(all_predictions)
            
            total = len(df)
            correct = df['correct'].sum()
            accuracy = correct / total
            
            results = {
                'total_predictions': total,
                'correct_predictions': correct,
                'overall_accuracy': accuracy,
                'average_confidence': df['confidence'].mean(),
                'trades_analyzed': trades_analyzed,
                'selectivity': trades_taken / trades_analyzed,
                'by_recommendation': {},
                'all_predictions': all_predictions
            }
            
            # Accuracy by recommendation type
            for rec in df['recommendation'].unique():
                rec_df = df[df['recommendation'] == rec]
                results['by_recommendation'][rec] = {
                    'count': len(rec_df),
                    'accuracy': rec_df['correct'].mean()
                }
            
            return results
        else:
            return {'overall_accuracy': 0.0, 'total_predictions': 0}

def main():
    """Run enhanced backtest"""
    
    backtest = EnhancedBacktest()
    
    # Test on diverse stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 
               'JPM', 'V', 'JNJ', 'UNH', 'WMT', 'PG', 'HD', 'DIS']
    
    # Test dates across different market conditions
    test_dates = [
        '2023-02-15',  # Recovery
        '2023-04-15',  # Spring rally
        '2023-06-15',  # Summer
        '2023-08-15',  # Late summer
        '2023-10-15',  # Fall
        '2023-11-15',  # Year-end rally
    ]
    
    results = backtest.run_backtest(symbols, test_dates)
    
    # Print results
    print(f"\n{'='*60}")
    print("ENHANCED BACKTEST RESULTS")
    print('='*60)
    print(f"Total Trades: {results['total_predictions']}")
    print(f"Correct: {results.get('correct_predictions', 0)}")
    print(f"\n🎯 ACCURACY: {results['overall_accuracy']:.1%}")
    
    if results['total_predictions'] > 0:
        print(f"\nAverage Confidence: {results['average_confidence']:.1%}")
        print(f"Selectivity: {results['selectivity']:.1%}")
        
        print("\nBy Recommendation:")
        for rec, stats in results['by_recommendation'].items():
            print(f"  {rec}: {stats['accuracy']:.1%} ({stats['count']} trades)")
    
    # Save results
    os.makedirs('backtesting/results', exist_ok=True)
    with open('backtesting/results/enhanced_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to backtesting/results/enhanced_backtest_results.json")
    
    # Check if we achieved target
    if results['overall_accuracy'] >= 0.80:
        print(f"\n✅ TARGET ACHIEVED! {results['overall_accuracy']:.1%} accuracy!")
    else:
        print(f"\n❌ Current: {results['overall_accuracy']:.1%} (Target: 80%+)")

if __name__ == "__main__":
    main()