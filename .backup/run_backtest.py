#!/usr/bin/env python3
"""
Run historical backtest with proper data handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import the coordinator directly
from agents.coordinator import AgentCoordinator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleBacktest:
    """Simple backtest that tests the prediction accuracy"""
    
    def __init__(self):
        self.coordinator = AgentCoordinator(
            parallel_execution=True,
            enable_ml=True,
            enable_llm=True
        )
        
    def test_predictions(self, 
                        test_date: str,
                        symbols: List[str],
                        lookback_days: int = 90) -> Dict:
        """
        Test predictions for a specific date
        
        Args:
            test_date: Date to test predictions (YYYY-MM-DD)
            symbols: List of symbols to test
            lookback_days: Days of historical data to use
            
        Returns:
            Dictionary with results
        """
        logger.info(f"\nTesting predictions for {test_date}")
        logger.info("="*60)
        
        predictions = []
        test_dt = pd.to_datetime(test_date)
        
        # For each symbol, make a prediction
        for symbol in symbols:
            try:
                logger.info(f"\nAnalyzing {symbol}...")
                
                # Get historical data up to test date
                start_date = test_dt - timedelta(days=lookback_days)
                end_date = test_dt
                
                # Download data
                hist_data = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date,
                    progress=False
                )
                
                if len(hist_data) < 20:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Get the current price on test date
                current_price = hist_data['Close'].iloc[-1]
                
                # Run the analysis
                # The coordinator will fetch its own data internally
                analysis = self.coordinator.analyze_stock(symbol)
                
                # Get the prediction
                recommendation = analysis['aggregated_analysis']['recommendation']
                confidence = analysis['aggregated_analysis']['overall_confidence']
                composite_score = analysis['aggregated_analysis']['overall_score']
                
                # Now get the actual future price (5 days later)
                future_date = test_dt + timedelta(days=5)
                future_data = yf.download(
                    symbol,
                    start=test_dt,
                    end=future_date + timedelta(days=2),
                    progress=False
                )
                
                if len(future_data) >= 5:
                    future_price = future_data['Close'].iloc[4]
                    actual_return = (future_price - current_price) / current_price
                    
                    # Determine if prediction was correct
                    predicted_bullish = recommendation in ['BUY', 'STRONG_BUY']
                    actual_bullish = actual_return > 0
                    correct = predicted_bullish == actual_bullish
                    
                    prediction = {
                        'symbol': symbol,
                        'date': test_date,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'composite_score': composite_score,
                        'current_price': current_price,
                        'future_price': future_price,
                        'actual_return': actual_return,
                        'predicted_direction': 'UP' if predicted_bullish else 'DOWN',
                        'actual_direction': 'UP' if actual_bullish else 'DOWN',
                        'correct': correct
                    }
                    
                    predictions.append(prediction)
                    
                    logger.info(f"  Recommendation: {recommendation} (Confidence: {confidence:.1%})")
                    logger.info(f"  Predicted: {'UP' if predicted_bullish else 'DOWN'}")
                    logger.info(f"  Actual: {'UP' if actual_bullish else 'DOWN'} ({actual_return:.2%})")
                    logger.info(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Calculate overall accuracy
        if predictions:
            accuracy = sum(1 for p in predictions if p['correct']) / len(predictions)
            
            # High confidence predictions only
            high_conf_predictions = [p for p in predictions if p['confidence'] >= 0.7]
            high_conf_accuracy = 0
            if high_conf_predictions:
                high_conf_accuracy = sum(1 for p in high_conf_predictions if p['correct']) / len(high_conf_predictions)
        else:
            accuracy = 0
            high_conf_accuracy = 0
            
        return {
            'test_date': test_date,
            'total_predictions': len(predictions),
            'overall_accuracy': accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_count': len([p for p in predictions if p['confidence'] >= 0.7]),
            'predictions': predictions
        }

def main():
    """Run backtests on multiple dates"""
    backtest = SimpleBacktest()
    
    # Top tech stocks to test
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    # Test dates throughout 2023
    test_dates = [
        '2023-01-15',
        '2023-02-15', 
        '2023-03-15',
        '2023-04-15',
        '2023-05-15',
        '2023-06-15',
        '2023-07-15',
        '2023-08-15',
        '2023-09-15',
        '2023-10-15',
        '2023-11-15',
        '2023-12-15'
    ]
    
    all_results = []
    
    for test_date in test_dates:
        try:
            result = backtest.test_predictions(test_date, symbols)
            all_results.append(result)
            
            print(f"\n{test_date}: Accuracy = {result['overall_accuracy']:.1%} " +
                  f"(High Conf: {result['high_confidence_accuracy']:.1%})")
            
        except Exception as e:
            logger.error(f"Error testing {test_date}: {e}")
    
    # Calculate overall statistics
    if all_results:
        total_predictions = sum(r['total_predictions'] for r in all_results)
        total_correct = sum(len([p for p in r['predictions'] if p['correct']]) for r in all_results)
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        # High confidence stats
        total_high_conf = sum(r['high_confidence_count'] for r in all_results)
        total_high_conf_correct = sum(
            len([p for p in r['predictions'] if p['correct'] and p['confidence'] >= 0.7])
            for r in all_results
        )
        high_conf_accuracy = total_high_conf_correct / total_high_conf if total_high_conf > 0 else 0
        
        # Save results
        os.makedirs('backtesting/results', exist_ok=True)
        with open('backtesting/results/backtest_summary.json', 'w') as f:
            json.dump({
                'test_dates': test_dates,
                'symbols': symbols,
                'overall_accuracy': overall_accuracy,
                'high_confidence_accuracy': high_conf_accuracy,
                'total_predictions': total_predictions,
                'total_high_confidence': total_high_conf,
                'detailed_results': all_results
            }, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("OVERALL BACKTEST RESULTS")
        print("="*60)
        print(f"Total Predictions: {total_predictions}")
        print(f"Overall Accuracy: {overall_accuracy:.1%}")
        print(f"High Confidence Predictions: {total_high_conf}")
        print(f"High Confidence Accuracy: {high_conf_accuracy:.1%}")
        print("\nResults saved to backtesting/results/backtest_summary.json")

if __name__ == "__main__":
    main()