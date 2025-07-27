#!/usr/bin/env python3
"""Quick test of the backtest system"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtesting.historical_backtest import HistoricalBacktest

def test_quick():
    """Run a quick test on one stock for one day"""
    print("Testing backtest system...")
    
    # Initialize backtest
    backtest = HistoricalBacktest(initial_capital=100000)
    
    # Test just one day with one stock
    results = backtest.backtest_period(
        start_date='2023-01-03',
        end_date='2023-01-05',
        symbols=['AAPL'],
        confidence_threshold=0.70
    )
    
    print(f"\nBacktest Results:")
    print(f"Total Predictions: {results['accuracy']['total_predictions']}")
    print(f"Accuracy: {results['accuracy']['prediction_accuracy']:.1%}")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_quick()