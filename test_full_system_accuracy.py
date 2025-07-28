#!/usr/bin/env python3
"""
Full system accuracy test with all Renaissance-style components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from agents.coordinator import AgentCoordinator
import warnings
warnings.filterwarnings('ignore')

def test_full_system_accuracy():
    """Test the complete enhanced trading system"""
    
    print("="*80)
    print("FULL SYSTEM ACCURACY TEST - RENAISSANCE STYLE")
    print("Testing with all components: HFT, StatArb, ML, Quantum")
    print("="*80)
    
    # Initialize coordinator with all features
    coordinator = AgentCoordinator(enable_ml=True, enable_llm=False, parallel_execution=True)
    
    # Extended test set for better statistical significance
    test_stocks = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
        # Finance
        'JPM', 'BAC', 'GS', 'MS',
        # Energy
        'XOM', 'CVX',
        # Healthcare
        'JNJ', 'UNH',
        # Consumer
        'WMT', 'HD',
        # Industrials
        'BA', 'CAT',
        # ETFs for stat arb
        'SPY', 'QQQ'
    ]
    
    # Results tracking
    predictions = []
    timeframes = [1, 3, 5]  # Test multiple prediction horizons
    
    print(f"\nTesting {len(test_stocks)} stocks across {len(timeframes)} timeframes...")
    print("-" * 80)
    
    for i, ticker in enumerate(test_stocks):
        print(f"\n[{i+1}/{len(test_stocks)}] Analyzing {ticker}...")
        
        try:
            # Get historical data with extra buffer
            data = yf.download(ticker, period='6mo', progress=False)
            
            if len(data) < 100:
                print(f"  ⚠️  Insufficient data for {ticker}")
                continue
            
            # Analyze with full system
            result = coordinator.analyze_stock(ticker)
            
            # Extract all signals
            overall_score = result['aggregated_analysis']['overall_score']
            confidence = result['aggregated_analysis']['overall_confidence']
            recommendation = result['aggregated_analysis']['recommendation']
            risk_adjusted_score = result.get('risk_adjusted_score', overall_score)
            
            # Get agent-specific signals
            agents_results = result['agent_results']
            hft_result = agents_results.get('HFTEngine', {})
            stat_arb_result = agents_results.get('StatisticalArbitrage', {})
            ml_result = agents_results.get('MLPrediction', {})
            
            # Trading signals
            trading_signals = result.get('trading_signals', {})
            position_size = trading_signals.get('position_size', 0)
            
            # Market filter
            should_trade = result.get('should_trade', True)
            
            print(f"  Overall Score: {overall_score:.3f}, Confidence: {confidence:.3f}")
            print(f"  Recommendation: {recommendation}, Should Trade: {should_trade}")
            
            # Test predictions for different timeframes
            for days in timeframes:
                if len(data) >= days + 5:
                    # Calculate actual return
                    actual_return = float((data['Close'].iloc[-1] / data['Close'].iloc[-(days+1)] - 1) * 100)
                    
                    # Determine predicted direction based on score and confidence
                    if should_trade and confidence > 0.65:
                        if overall_score > 0.2:
                            pred_direction = 1
                        elif overall_score < -0.2:
                            pred_direction = -1
                        else:
                            pred_direction = 0
                    else:
                        pred_direction = 0  # No trade
                    
                    # Actual direction
                    threshold = 0.5 * days  # Scale threshold by timeframe
                    if actual_return > threshold:
                        actual_direction = 1
                    elif actual_return < -threshold:
                        actual_direction = -1
                    else:
                        actual_direction = 0
                    
                    # Check correctness
                    correct = pred_direction == actual_direction
                    
                    # Store detailed results
                    predictions.append({
                        'ticker': ticker,
                        'timeframe_days': days,
                        'overall_score': overall_score,
                        'confidence': confidence,
                        'risk_adjusted_score': risk_adjusted_score,
                        'recommendation': recommendation,
                        'predicted_direction': pred_direction,
                        'actual_direction': actual_direction,
                        'actual_return': actual_return,
                        'correct': correct,
                        'should_trade': should_trade,
                        'position_size': position_size,
                        # Agent details
                        'hft_score': hft_result.get('score', 0),
                        'hft_predictions': hft_result.get('predictions_per_day', 0),
                        'stat_arb_active': stat_arb_result.get('active_opportunities', 0),
                        'ml_accuracy': ml_result.get('model_accuracies', {}).get('ensemble', 0),
                        # Advanced features
                        'has_quantum': 'quantum_optimization' in result['aggregated_analysis'],
                        'agents_active': result.get('agents_successful', 0)
                    })
                    
                    if days == 5:  # Show details for 5-day predictions
                        print(f"    5-day: Pred={pred_direction}, Actual={actual_direction}, Return={actual_return:.2f}%, {'✓' if correct else '✗'}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:100]}")
    
    # Calculate comprehensive accuracy metrics
    print("\n" + "="*80)
    print("COMPREHENSIVE ACCURACY RESULTS")
    print("="*80)
    
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Overall accuracy
        overall_correct = df['correct'].sum()
        overall_total = len(df)
        overall_accuracy = overall_correct / overall_total * 100
        
        print(f"\nOverall Accuracy: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")
        
        # Accuracy by timeframe
        print("\nAccuracy by Timeframe:")
        for tf in timeframes:
            tf_df = df[df['timeframe_days'] == tf]
            if len(tf_df) > 0:
                tf_correct = tf_df['correct'].sum()
                tf_total = len(tf_df)
                tf_accuracy = tf_correct / tf_total * 100
                print(f"  {tf}-day: {tf_accuracy:.1f}% ({tf_correct}/{tf_total})")
        
        # High confidence trades only
        high_conf_df = df[df['confidence'] > 0.75]
        if len(high_conf_df) > 0:
            hc_correct = high_conf_df['correct'].sum()
            hc_total = len(high_conf_df)
            hc_accuracy = hc_correct / hc_total * 100
            print(f"\nHigh Confidence (>75%): {hc_accuracy:.1f}% ({hc_correct}/{hc_total})")
        
        # Directional trades only
        directional_df = df[df['predicted_direction'] != 0]
        if len(directional_df) > 0:
            dir_correct = directional_df['correct'].sum()
            dir_total = len(directional_df)
            dir_accuracy = dir_correct / dir_total * 100
            print(f"Directional Trades: {dir_accuracy:.1f}% ({dir_correct}/{dir_total})")
        
        # With market filter
        filtered_df = df[df['should_trade'] == True]
        if len(filtered_df) > 0:
            filt_correct = filtered_df['correct'].sum()
            filt_total = len(filtered_df)
            filt_accuracy = filt_correct / filt_total * 100
            print(f"With Market Filter: {filt_accuracy:.1f}% ({filt_correct}/{filt_total})")
        
        # Renaissance-style metrics
        print("\n" + "-"*40)
        print("RENAISSANCE-STYLE PERFORMANCE METRICS:")
        print("-"*40)
        
        # Check component usage
        hft_active = (df['hft_score'].abs() > 0.1).sum()
        stat_arb_active = (df['stat_arb_active'] > 0).sum()
        quantum_used = df['has_quantum'].sum()
        
        print(f"HFT Engine Active: {hft_active}/{len(df)} ({hft_active/len(df)*100:.1f}%)")
        print(f"StatArb Active: {stat_arb_active}/{len(df)} ({stat_arb_active/len(df)*100:.1f}%)")
        print(f"Quantum Optimization Used: {quantum_used}/{len(df)} ({quantum_used/len(df)*100:.1f}%)")
        
        avg_agents = df['agents_active'].mean()
        print(f"Average Active Agents: {avg_agents:.1f}")
        
        # Expected daily predictions (Renaissance-style)
        avg_hft_predictions = df['hft_predictions'].mean()
        print(f"\nHFT Daily Predictions Capacity: {avg_hft_predictions:,.0f}")
        print(f"Renaissance Medallion Target: 150,000+")
        
        # Accuracy by recommendation type
        print("\nAccuracy by Recommendation:")
        for rec in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
            rec_df = df[df['recommendation'] == rec]
            if len(rec_df) > 0:
                rec_correct = rec_df['correct'].sum()
                rec_total = len(rec_df)
                rec_accuracy = rec_correct / rec_total * 100
                print(f"  {rec}: {rec_accuracy:.1f}% ({rec_correct}/{rec_total})")
        
        # Risk-adjusted performance
        if len(directional_df) > 0:
            avg_position_size = directional_df['position_size'].mean()
            avg_return = directional_df[directional_df['correct']]['actual_return'].mean()
            avg_loss = directional_df[~directional_df['correct']]['actual_return'].mean()
            
            print(f"\nRisk Management:")
            print(f"Average Position Size: {avg_position_size:.1%}")
            print(f"Average Win: {avg_return:.2f}%")
            print(f"Average Loss: {avg_loss:.2f}%")
            
            if not directional_df[~directional_df['correct']].empty:
                win_loss_ratio = abs(avg_return / avg_loss)
                print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        # Final comparison
        print("\n" + "="*40)
        print("FINAL ASSESSMENT vs RENAISSANCE MEDALLION:")
        print("="*40)
        print(f"Our System Accuracy: {overall_accuracy:.1f}%")
        print(f"High Confidence Accuracy: {hc_accuracy:.1f}%" if len(high_conf_df) > 0 else "High Confidence Accuracy: N/A")
        print(f"Medallion Target: 50.75% (on 150k+ daily trades)")
        
        # Estimate Sharpe ratio
        if len(directional_df) > 0:
            returns = directional_df['actual_return'] * directional_df['position_size'] * (directional_df['predicted_direction'] == directional_df['actual_direction']).astype(int)
            sharpe_estimate = returns.mean() / returns.std() * np.sqrt(252/5) if returns.std() > 0 else 0
            print(f"\nEstimated Sharpe Ratio: {sharpe_estimate:.2f}")
            print(f"Medallion Sharpe: 4.0-7.0")
        
        # Save results
        df.to_csv('full_system_accuracy_results.csv', index=False)
        print(f"\nDetailed results saved to: full_system_accuracy_results.csv")
        
        return overall_accuracy
    
    else:
        print("No valid predictions generated")
        return 0

if __name__ == "__main__":
    accuracy = test_full_system_accuracy()