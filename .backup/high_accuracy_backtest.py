#!/usr/bin/env python3
"""
High accuracy backtest system - simplified approach
Target: 80%+ accuracy by being more selective
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

from agentic_portfolio_manager import AgenticPortfolioManager

class HighAccuracyBacktest:
    """
    Achieve 80%+ accuracy through:
    1. Only trade strongest signals
    2. Trade with the trend
    3. Use appropriate timeframes
    4. Filter out noise
    """
    
    def __init__(self):
        self.manager = AgenticPortfolioManager(
            use_ml=True,
            use_llm=True,
            parallel_execution=False
        )
        
    def run_selective_backtest(self, symbols: List[str], test_dates: List[str]) -> Dict:
        """
        Run highly selective backtest
        """
        
        all_predictions = []
        
        for test_date in test_dates:
            print(f"\n{'='*60}")
            print(f"Testing {test_date}")
            print('='*60)
            
            test_dt = pd.to_datetime(test_date)
            
            for symbol in symbols:
                try:
                    print(f"\nAnalyzing {symbol}...")
                    
                    # Get data with enough history
                    hist_data = yf.download(
                        symbol,
                        start=test_dt - timedelta(days=100),
                        end=test_dt + timedelta(days=1),
                        progress=False
                    )
                    
                    if len(hist_data) < 50:
                        print(f"  Insufficient data")
                        continue
                    
                    # Current price
                    current_price = float(hist_data['Close'].iloc[-1])
                    
                    # Calculate multiple trend indicators
                    sma20 = float(hist_data['Close'].rolling(20).mean().iloc[-1])
                    sma50 = float(hist_data['Close'].rolling(50).mean().iloc[-1]) if len(hist_data) >= 50 else sma20
                    
                    # Recent performance
                    week_return = (current_price - float(hist_data['Close'].iloc[-6])) / float(hist_data['Close'].iloc[-6])
                    month_return = (current_price - float(hist_data['Close'].iloc[-22])) / float(hist_data['Close'].iloc[-22])
                    
                    # Volatility check
                    returns = hist_data['Close'].pct_change()
                    recent_vol = float(returns.tail(20).std())
                    
                    # Define clear trend
                    strong_uptrend = (current_price > sma20 > sma50) and week_return > 0.02 and month_return > 0.05
                    strong_downtrend = (current_price < sma20 < sma50) and week_return < -0.02 and month_return < -0.05
                    
                    # Run analysis
                    analysis = self.manager.analyze_stock(symbol)
                    
                    recommendation = analysis.get('recommendation', 'HOLD')
                    confidence = analysis.get('confidence', 0.0)
                    composite_score = analysis.get('composite_score', 0.0)
                    
                    # STRICT FILTERING RULES FOR 80%+ ACCURACY
                    
                    # Rule 1: Only very high confidence trades
                    if confidence < 0.85:
                        print(f"  Skipping - confidence too low ({confidence:.1%})")
                        continue
                    
                    # Rule 2: Must align with strong trend
                    if strong_uptrend and recommendation not in ['BUY', 'STRONG_BUY']:
                        print(f"  Skipping - against uptrend")
                        continue
                    if strong_downtrend and recommendation not in ['SELL', 'STRONG_SELL', 'HOLD']:
                        print(f"  Skipping - against downtrend")
                        continue
                    
                    # Rule 3: Skip if no clear trend
                    if not strong_uptrend and not strong_downtrend:
                        print(f"  Skipping - no clear trend")
                        continue
                    
                    # Rule 4: Composite score must be strong
                    if abs(composite_score) < 0.4:
                        print(f"  Skipping - weak composite score ({composite_score:.2f})")
                        continue
                    
                    # Rule 5: Skip high volatility
                    if recent_vol > 0.03:  # 3% daily volatility
                        print(f"  Skipping - too volatile ({recent_vol:.1%})")
                        continue
                    
                    # Get future price (using 10 days for more stable results)
                    future_data = yf.download(
                        symbol,
                        start=test_dt,
                        end=test_dt + timedelta(days=15),
                        progress=False
                    )
                    
                    if len(future_data) >= 11:
                        future_price = float(future_data['Close'].iloc[10])
                        actual_return = (future_price - current_price) / current_price
                        
                        # For very selective trades, we expect stronger moves
                        if recommendation in ['BUY', 'STRONG_BUY']:
                            correct = actual_return > 0.01  # Need at least 1% gain
                        elif recommendation in ['SELL', 'STRONG_SELL']:
                            correct = actual_return < -0.01  # Need at least 1% loss
                        else:  # HOLD
                            correct = abs(actual_return) < 0.02  # Sideways movement
                        
                        prediction = {
                            'date': test_date,
                            'symbol': symbol,
                            'recommendation': recommendation,
                            'confidence': confidence,
                            'composite_score': composite_score,
                            'trend': 'up' if strong_uptrend else 'down',
                            'current_price': current_price,
                            'future_price': future_price,
                            'actual_return': actual_return,
                            'correct': correct,
                            'week_momentum': week_return,
                            'month_momentum': month_return
                        }
                        
                        all_predictions.append(prediction)
                        
                        print(f"  ✓ TRADE TAKEN")
                        print(f"  Price: ${current_price:.2f}")
                        print(f"  Trend: {'Strong Up' if strong_uptrend else 'Strong Down'}")
                        print(f"  Recommendation: {recommendation} ({confidence:.1%})")
                        print(f"  Composite Score: {composite_score:.2f}")
                        print(f"  10-day return: {actual_return:+.2%}")
                        print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
                        
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate results
        if all_predictions:
            return self._calculate_results(all_predictions)
        else:
            print("\nNo trades taken - all filtered out")
            return {'overall_accuracy': 0.0, 'total_predictions': 0}
    
    def _calculate_results(self, predictions: List[Dict]) -> Dict:
        """Calculate final results"""
        
        df = pd.DataFrame(predictions)
        
        total = len(df)
        correct = df['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        # Additional metrics
        avg_confidence = df['confidence'].mean()
        avg_return_when_correct = df[df['correct']]['actual_return'].mean() if correct > 0 else 0
        avg_return_when_wrong = df[~df['correct']]['actual_return'].mean() if (total - correct) > 0 else 0
        
        results = {
            'total_predictions': total,
            'correct_predictions': correct,
            'overall_accuracy': accuracy,
            'average_confidence': avg_confidence,
            'avg_return_correct': avg_return_when_correct,
            'avg_return_wrong': avg_return_when_wrong,
            'predictions_by_type': {},
            'all_predictions': predictions
        }
        
        # By recommendation type
        for rec in df['recommendation'].unique():
            rec_df = df[df['recommendation'] == rec]
            results['predictions_by_type'][rec] = {
                'count': len(rec_df),
                'accuracy': rec_df['correct'].mean() if len(rec_df) > 0 else 0
            }
        
        return results

def main():
    """Run high accuracy backtest"""
    
    backtest = HighAccuracyBacktest()
    
    # Test on major stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'JPM', 'V', 'JNJ']
    
    # Test dates with different market conditions
    test_dates = [
        '2023-03-15',  # Recovery period
        '2023-05-15',  # Tech rally
        '2023-07-15',  # Summer trading
        '2023-09-15',  # Pre-Q4
        '2023-11-15',  # Year-end rally start
    ]
    
    results = backtest.run_selective_backtest(symbols, test_dates)
    
    # Print summary
    print(f"\n{'='*60}")
    print("HIGH ACCURACY BACKTEST RESULTS")
    print('='*60)
    print(f"Total Trades Taken: {results['total_predictions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    
    if results['total_predictions'] > 0:
        print(f"\nAverage Confidence: {results['average_confidence']:.1%}")
        print(f"Avg Return When Correct: {results['avg_return_correct']:+.2%}")
        print(f"Avg Return When Wrong: {results['avg_return_wrong']:+.2%}")
        
        print("\nBy Recommendation Type:")
        for rec, stats in results['predictions_by_type'].items():
            if stats['count'] > 0:
                print(f"  {rec}: {stats['accuracy']:.1%} ({stats['count']} trades)")
    
    # Save results
    os.makedirs('backtesting/results', exist_ok=True)
    with open('backtesting/results/high_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to backtesting/results/high_accuracy_results.json")
    
    # Target check
    if results['overall_accuracy'] >= 0.80:
        print(f"\n✅ TARGET ACHIEVED: {results['overall_accuracy']:.1%} accuracy!")
    else:
        print(f"\n❌ Below target: {results['overall_accuracy']:.1%} (need 80%+)")
        if results['total_predictions'] < 10:
            print("   Note: Very few trades taken due to strict filtering")

if __name__ == "__main__":
    main()