#!/usr/bin/env python3
"""
Backtest Runner for AlphaBot AI
Runs comprehensive backtests across multiple time periods and market conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from historical_backtest import HistoricalBacktest

class BacktestRunner:
    """Run and analyze multiple backtests"""
    
    def __init__(self):
        self.backtest = HistoricalBacktest()
        self.all_results = {}
        
    def run_multi_year_backtest(self, config_file: str = 'backtesting/configs/backtest_config.json'):
        """Run backtests for multiple years based on config"""
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n{'='*80}")
        print("ALPHABOT AI - MULTI-YEAR HISTORICAL BACKTEST")
        print(f"{'='*80}")
        
        # Run backtest for each period
        for period in config['test_periods']:
            year = period['year']
            description = period['description']
            
            print(f"\n📅 Testing {year}: {description}")
            
            # Run backtest
            results = self.backtest.backtest_period(
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                symbols=config['stocks_to_test'],
                confidence_threshold=config['confidence_threshold']
            )
            
            # Save results
            output_dir = f"backtesting/results/{year}"
            self.backtest.save_results(results, output_dir)
            
            # Store for summary
            self.all_results[year] = results
            
            # Print year summary
            self._print_year_summary(year, results)
        
        # Generate overall summary
        self._generate_overall_summary()
        
        # Create visualizations
        self._create_visualizations()
        
    def _print_year_summary(self, year: int, results: Dict):
        """Print summary for a single year"""
        perf = results['performance']
        acc = results['accuracy']
        
        print(f"\n{year} RESULTS:")
        print(f"  💰 Return: {perf['total_return']:.2%}")
        print(f"  🎯 Accuracy: {acc['prediction_accuracy']:.1%}")
        print(f"  📊 Trades: {perf['num_trades']}")
        print(f"  📈 Sharpe: {perf['sharpe_ratio']:.2f}")
        print(f"  📉 Max Drawdown: {perf['max_drawdown']:.2%}")
        
    def _generate_overall_summary(self):
        """Generate summary across all years"""
        summary = {
            'test_period': {
                'start': min(r['period']['start'] for r in self.all_results.values()),
                'end': max(r['period']['end'] for r in self.all_results.values())
            },
            'years_tested': list(self.all_results.keys()),
            'average_metrics': {},
            'yearly_performance': {},
            'overall_accuracy': {}
        }
        
        # Calculate averages
        total_predictions = 0
        correct_predictions = 0
        returns = []
        accuracies = []
        sharpes = []
        
        for year, results in self.all_results.items():
            perf = results['performance']
            acc = results['accuracy']
            
            returns.append(perf['total_return'])
            accuracies.append(acc['prediction_accuracy'])
            sharpes.append(perf['sharpe_ratio'])
            
            total_predictions += acc['total_predictions']
            correct_predictions += acc['correct_predictions']
            
            summary['yearly_performance'][year] = {
                'return': perf['total_return'],
                'accuracy': acc['prediction_accuracy'],
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': perf['max_drawdown'],
                'num_trades': perf['num_trades']
            }
        
        # Calculate overall metrics
        summary['average_metrics'] = {
            'avg_annual_return': np.mean(returns),
            'avg_accuracy': np.mean(accuracies),
            'avg_sharpe_ratio': np.mean(sharpes),
            'return_volatility': np.std(returns)
        }
        
        summary['overall_accuracy'] = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0
        }
        
        # Save summary
        with open('backtesting/results/summary/overall_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print("OVERALL BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"Years Tested: {', '.join(map(str, summary['years_tested']))}")
        print(f"\n📊 AVERAGE METRICS:")
        print(f"  Average Annual Return: {summary['average_metrics']['avg_annual_return']:.2%}")
        print(f"  Average Accuracy: {summary['average_metrics']['avg_accuracy']:.1%}")
        print(f"  Average Sharpe Ratio: {summary['average_metrics']['avg_sharpe_ratio']:.2f}")
        print(f"\n🎯 OVERALL ACCURACY:")
        print(f"  Total Predictions: {summary['overall_accuracy']['total_predictions']:,}")
        print(f"  Correct Predictions: {summary['overall_accuracy']['correct_predictions']:,}")
        print(f"  Overall Accuracy: {summary['overall_accuracy']['accuracy']:.1%}")
        
        # Check if 80% accuracy claim holds
        if summary['overall_accuracy']['accuracy'] >= 0.80:
            print(f"\n✅ 80%+ ACCURACY CLAIM VALIDATED!")
        else:
            print(f"\n⚠️  Actual accuracy: {summary['overall_accuracy']['accuracy']:.1%}")
        
        return summary
    
    def _create_visualizations(self):
        """Create visualization charts"""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Equity curves by year
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (year, results) in enumerate(self.all_results.items()):
            if idx < 4:  # Max 4 subplots
                daily_values = pd.DataFrame(results['daily_values'])
                daily_values['date'] = pd.to_datetime(daily_values['date'])
                
                ax = axes[idx]
                ax.plot(daily_values['date'], daily_values['portfolio_value'], label='Portfolio Value')
                ax.axhline(y=100000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
                ax.set_title(f'{year} Portfolio Performance')
                ax.set_xlabel('Date')
                ax.set_ylabel('Portfolio Value ($)')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('backtesting/visualizations/equity_curves.png', dpi=150)
        plt.close()
        
        # 2. Accuracy comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        years = []
        accuracies = []
        returns = []
        
        for year, results in sorted(self.all_results.items()):
            years.append(str(year))
            accuracies.append(results['accuracy']['prediction_accuracy'] * 100)
            returns.append(results['performance']['total_return'] * 100)
        
        # Accuracy by year
        ax1.bar(years, accuracies, color='green', alpha=0.7)
        ax1.axhline(y=80, color='r', linestyle='--', label='80% Target')
        ax1.set_title('Prediction Accuracy by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Returns by year
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(years, returns, color=colors, alpha=0.7)
        ax2.set_title('Annual Returns by Year')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('backtesting/visualizations/accuracy_returns.png', dpi=150)
        plt.close()
        
        # 3. Prediction accuracy distribution
        all_predictions = []
        for results in self.all_results.values():
            all_predictions.extend(results['predictions'])
        
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by confidence bins
            predictions_df['confidence_bin'] = pd.cut(
                predictions_df['confidence'], 
                bins=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                labels=['70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
            )
            
            accuracy_by_confidence = predictions_df.groupby('confidence_bin')['correct'].mean() * 100
            
            accuracy_by_confidence.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
            ax.set_title('Accuracy by Confidence Level')
            ax.set_xlabel('Confidence Range')
            ax.set_ylabel('Accuracy (%)')
            ax.set_ylim(0, 100)
            ax.axhline(y=80, color='r', linestyle='--', label='80% Target')
            ax.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('backtesting/visualizations/accuracy_by_confidence.png', dpi=150)
            plt.close()
        
        print(f"\n📊 Visualizations saved to backtesting/visualizations/")

def main():
    """Run the full backtest suite"""
    runner = BacktestRunner()
    
    # Create config if it doesn't exist
    config_path = 'backtesting/configs/backtest_config.json'
    if not os.path.exists(config_path):
        config = {
            "test_periods": [
                {"year": 2021, "description": "Bull market recovery, meme stock mania"},
                {"year": 2022, "description": "Bear market, high inflation, rate hikes"},
                {"year": 2023, "description": "AI boom, volatile recovery"}
            ],
            "stocks_to_test": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
                "META", "TSLA", "JPM", "V", "JNJ",
                "WMT", "PG", "UNH", "HD", "MA"
            ],
            "confidence_threshold": 0.70,
            "initial_capital": 100000
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Run backtests
    runner.run_multi_year_backtest(config_path)

if __name__ == "__main__":
    main()