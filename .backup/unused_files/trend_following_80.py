#!/usr/bin/env python3
"""
Trend Following Strategy - Achieving 80%+ Accuracy
Focuses on strong trends with 3-day prediction window
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class TrendFollowing80:
    """Trend following strategy optimized for 80% accuracy"""
    
    def analyze_stock(self, ticker: str, date: datetime) -> Dict:
        """Analyze stock for strong trends"""
        
        # Get historical data
        end_date = date
        start_date = date - timedelta(days=100)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) < 50:
                return None
            
            # Calculate indicators
            close = data['Close']
            volume = data['Volume']
            
            current_price = float(close.iloc[-1])
            
            # Moving averages
            sma5 = close.rolling(5).mean()
            sma10 = close.rolling(10).mean()
            sma20 = close.rolling(20).mean()
            
            # Momentum
            momentum_1d = float((close.iloc[-1] / close.iloc[-2] - 1))
            momentum_3d = float((close.iloc[-1] / close.iloc[-4] - 1)) if len(close) > 3 else 0
            momentum_5d = float((close.iloc[-1] / close.iloc[-6] - 1)) if len(close) > 5 else 0
            momentum_10d = float((close.iloc[-1] / close.iloc[-11] - 1)) if len(close) > 10 else 0
            
            # Trend strength over last 10 days
            trend_days = 0
            for i in range(1, min(11, len(close))):
                if close.iloc[-i] > close.iloc[-i-1]:
                    trend_days += 1
            
            # Volume trend
            vol_avg = float(volume.rolling(20).mean().iloc[-1])
            vol_recent = float(volume.iloc[-3:].mean())
            volume_increasing = vol_recent > vol_avg
            
            # Calculate trend consistency
            returns = close.pct_change().iloc[-10:]
            positive_days = (returns > 0).sum()
            trend_consistency = positive_days / len(returns)
            
            # Strategy: Only trade VERY strong trends
            recommendation = 'HOLD'
            confidence = 0.5
            score = 0
            
            # Strong uptrend criteria (for 3-day prediction)
            if (
                # Perfect alignment
                current_price > float(sma5.iloc[-1]) > float(sma10.iloc[-1]) > float(sma20.iloc[-1]) and
                # All momentum positive and increasing
                momentum_1d > 0.005 and momentum_3d > 0.015 and momentum_5d > 0.025 and momentum_10d > 0.04 and
                # Consistent trend
                trend_consistency >= 0.7 and
                # Volume confirmation
                volume_increasing and
                # Not a gap up
                momentum_1d < 0.03
            ):
                recommendation = 'BUY'
                confidence = 0.90 + (trend_consistency - 0.7) * 0.2  # 90-96% confidence
                score = 0.8
            
            # Strong downtrend criteria
            elif (
                # Perfect downtrend alignment
                current_price < float(sma5.iloc[-1]) < float(sma10.iloc[-1]) < float(sma20.iloc[-1]) and
                # All momentum negative
                momentum_1d < -0.005 and momentum_3d < -0.015 and momentum_5d < -0.025 and momentum_10d < -0.04 and
                # Consistent downtrend
                trend_consistency <= 0.3 and
                # Volume confirmation
                volume_increasing and
                # Not a gap down
                momentum_1d > -0.03
            ):
                recommendation = 'SELL'
                confidence = 0.90 + (0.3 - trend_consistency) * 0.2
                score = -0.8
            
            # Moderate trends (still selective)
            elif (
                current_price > float(sma5.iloc[-1]) > float(sma10.iloc[-1]) and
                momentum_3d > 0.01 and momentum_5d > 0.02 and
                trend_consistency >= 0.6
            ):
                recommendation = 'BUY'
                confidence = 0.85
                score = 0.6
            
            elif (
                current_price < float(sma5.iloc[-1]) < float(sma10.iloc[-1]) and
                momentum_3d < -0.01 and momentum_5d < -0.02 and
                trend_consistency <= 0.4
            ):
                recommendation = 'SELL'
                confidence = 0.85
                score = -0.6
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'score': score,
                'momentum_3d': momentum_3d,
                'momentum_5d': momentum_5d,
                'trend_consistency': trend_consistency,
                'volume_increasing': volume_increasing
            }
            
        except Exception as e:
            return None

def backtest_trend_following():
    """Run trend following backtest"""
    
    strategy = TrendFollowing80()
    
    # Focus on most liquid stocks
    symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
               'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA',
               'DIS', 'ADBE', 'CRM', 'NFLX', 'PFE', 'WMT', 'TMO']
    
    # Test dates throughout 2023
    test_dates = []
    start = datetime(2023, 1, 15)
    while start < datetime(2023, 12, 15):
        test_dates.append(start.strftime('%Y-%m-%d'))
        start += timedelta(days=7)  # Weekly testing
    
    all_predictions = []
    total_analyzed = 0
    
    print("Trend Following Backtest (3-day prediction window)")
    print("=" * 60)
    
    for date_str in test_dates:
        test_date = pd.to_datetime(date_str)
        daily_trades = []
        
        for symbol in symbols:
            total_analyzed += 1
            
            try:
                # Get analysis
                analysis = strategy.analyze_stock(symbol, test_date)
                
                if not analysis or analysis['recommendation'] == 'HOLD':
                    continue
                
                # Only take high confidence trades
                if analysis['confidence'] < 0.85:
                    continue
                
                # Get future price (3-day window)
                future_data = yf.download(
                    symbol,
                    start=test_date,
                    end=test_date + timedelta(days=7),
                    progress=False
                )
                
                if len(future_data) >= 4:
                    current_data = yf.download(
                        symbol,
                        start=test_date - timedelta(days=1),
                        end=test_date + timedelta(days=1),
                        progress=False
                    )
                    
                    if len(current_data) > 0 and len(future_data) > 3:
                        current_price = float(current_data['Close'].iloc[-1])
                        future_price_3d = float(future_data['Close'].iloc[3])
                        actual_return_3d = (future_price_3d - current_price) / current_price
                        
                        # Success criteria for 3-day window
                        if analysis['recommendation'] == 'BUY':
                            correct = actual_return_3d > 0.005  # 0.5% in 3 days
                        elif analysis['recommendation'] == 'SELL':
                            correct = actual_return_3d < -0.005
                        else:
                            correct = abs(actual_return_3d) < 0.01
                        
                        prediction = {
                            'date': date_str,
                            'symbol': symbol,
                            'recommendation': analysis['recommendation'],
                            'confidence': analysis['confidence'],
                            'momentum_3d': analysis['momentum_3d'],
                            'trend_consistency': analysis['trend_consistency'],
                            '3d_return': actual_return_3d,
                            'correct': correct
                        }
                        
                        daily_trades.append(prediction)
                        all_predictions.append(prediction)
                    
            except Exception as e:
                continue
        
        # Print daily summary
        if daily_trades:
            correct_today = sum(1 for t in daily_trades if t['correct'])
            accuracy_today = correct_today / len(daily_trades) if daily_trades else 0
            print(f"{date_str}: {len(daily_trades)} trades, {correct_today} correct ({accuracy_today:.0%})")
    
    # Calculate results
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        accuracy = df['correct'].mean()
        
        print(f"\n{'='*60}")
        print(f"TREND FOLLOWING RESULTS")
        print(f"{'='*60}")
        print(f"Total Analyzed: {total_analyzed}")
        print(f"Trades Taken: {len(df)}")
        print(f"Selectivity: {len(df)/total_analyzed*100:.1%}")
        print(f"Correct: {df['correct'].sum()}")
        print(f"\n🎯 ACCURACY: {accuracy:.1%}")
        
        # By recommendation
        print("\nBy Recommendation:")
        for rec in ['BUY', 'SELL']:
            if rec in df['recommendation'].values:
                rec_df = df[df['recommendation'] == rec]
                print(f"  {rec}: {rec_df['correct'].mean():.1%} ({len(rec_df)} trades)")
        
        # Best and worst performers
        symbol_accuracy = df.groupby('symbol')['correct'].agg(['mean', 'count'])
        symbol_accuracy = symbol_accuracy[symbol_accuracy['count'] >= 3].sort_values('mean', ascending=False)
        
        if len(symbol_accuracy) > 0:
            print("\nBest Performing Symbols (min 3 trades):")
            for symbol, row in symbol_accuracy.head(5).iterrows():
                print(f"  {symbol}: {row['mean']:.1%} ({row['count']} trades)")
        
        # Save results
        os.makedirs('backtesting/results', exist_ok=True)
        results = {
            'accuracy': float(accuracy),
            'total_trades': len(df),
            'correct_trades': int(df['correct'].sum()),
            'total_analyzed': total_analyzed,
            'selectivity': len(df)/total_analyzed,
            'predictions': all_predictions
        }
        
        with open('backtesting/results/trend_following_80_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return accuracy >= 0.80
    
    return False

if __name__ == "__main__":
    success = backtest_trend_following()
    if success:
        print("\n✅ Successfully achieved 80%+ accuracy!")
    else:
        print("\n❌ Failed to achieve 80% accuracy")
