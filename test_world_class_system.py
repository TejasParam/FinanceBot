#!/usr/bin/env python3
"""
Test the World-Class Enhanced Agentic Trading System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import json
import pandas as pd
from datetime import datetime
import time

def test_enhanced_system():
    """Test the enhanced world-class trading system"""
    
    print("=" * 80)
    print("WORLD-CLASS AGENTIC TRADING SYSTEM TEST")
    print("Enhanced with Advanced Features")
    print("=" * 80)
    
    # Initialize the enhanced coordinator
    print("\n🌍 Initializing World-Class Multi-Agent System...")
    coordinator = AgentCoordinator(
        enable_ml=True,
        enable_llm=False,  # Disable LLM for speed
        parallel_execution=True
    )
    
    # Test stocks
    test_stocks = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'TSLA']
    
    print(f"\n🔍 Testing {len(test_stocks)} stocks with {len(coordinator.agents)} specialized agents:")
    for agent_name in coordinator.get_available_agents():
        print(f"  • {agent_name}")
    
    results = {}
    
    for ticker in test_stocks:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            # Run comprehensive analysis
            analysis = coordinator.analyze_stock(ticker)
            
            # Extract key results
            aggregated = analysis['aggregated_analysis']
            risk_metrics = analysis.get('risk_metrics', {})
            trading_signals = analysis.get('trading_signals', {})
            market_context = analysis.get('market_context', {})
            
            print(f"\n📊 Analysis Results:")
            print(f"  Overall Score: {aggregated['overall_score']:.3f}")
            print(f"  Confidence: {aggregated['overall_confidence']:.1%}")
            print(f"  Recommendation: {aggregated['recommendation']}")
            print(f"  System Confidence: {analysis.get('system_confidence', 0):.1%}")
            
            print(f"\n💰 Trading Signals:")
            print(f"  Entry Signal: {trading_signals.get('entry_signal', 'N/A')}")
            print(f"  Position Size: {trading_signals.get('position_size', 0):.0%}")
            print(f"  Stop Loss: {trading_signals.get('stop_loss_pct', 0):.1%}")
            print(f"  Take Profit: {trading_signals.get('take_profit_pct', 0):.1%}")
            print(f"  Risk/Reward: {trading_signals.get('risk_reward_ratio', 0):.1f}")
            print(f"  Time Horizon: {trading_signals.get('time_horizon', 'N/A')}")
            
            print(f"\n⚠️ Risk Metrics:")
            print(f"  Risk Score: {risk_metrics.get('risk_score', 0):.2f}")
            print(f"  Volatility (Annual): {risk_metrics.get('volatility_annualized', 0):.1%}")
            print(f"  95% VaR: {risk_metrics.get('var_95', 0):.1%}")
            print(f"  Agent Disagreement: {risk_metrics.get('agent_disagreement', 0):.2f}")
            
            print(f"\n📈 Market Context:")
            print(f"  Trend: {market_context.get('trend', 'unknown')}")
            print(f"  Momentum (5d): {market_context.get('momentum_5d', 0):.2%}")
            print(f"  Volume Surge: {market_context.get('volume_surge', False)}")
            
            # Show individual agent results
            print(f"\n🤖 Agent Results:")
            for agent_name, result in analysis['agent_results'].items():
                if 'error' not in result:
                    score = result.get('score', 0)
                    confidence = result.get('confidence', 0)
                    symbol = '✅' if score > 0.3 else '❌' if score < -0.3 else '⚠️'
                    print(f"  {symbol} {agent_name}: Score={score:.2f}, Conf={confidence:.1%}")
            
            print(f"\n⏱ Execution Time: {analysis['execution_time']:.2f}s")
            
            results[ticker] = {
                'score': aggregated['overall_score'],
                'confidence': aggregated['overall_confidence'],
                'recommendation': aggregated['recommendation'],
                'risk_score': risk_metrics.get('risk_score', 0),
                'signals': trading_signals
            }
            
        except Exception as e:
            print(f"\n❌ Error analyzing {ticker}: {str(e)}")
            results[ticker] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RECOMMENDATIONS")
    print('='*80)
    
    buy_signals = [t for t, r in results.items() if r.get('recommendation') in ['BUY', 'STRONG_BUY']]
    sell_signals = [t for t, r in results.items() if r.get('recommendation') in ['SELL', 'STRONG_SELL']]
    hold_signals = [t for t, r in results.items() if r.get('recommendation') == 'HOLD']
    
    if buy_signals:
        print(f"\n🟢 BUY Signals ({len(buy_signals)}):")
        for ticker in buy_signals:
            conf = results[ticker]['confidence']
            risk = results[ticker]['risk_score']
            print(f"  {ticker}: {conf:.1%} confidence, {risk:.2f} risk")
    
    if sell_signals:
        print(f"\n🔴 SELL Signals ({len(sell_signals)}):")
        for ticker in sell_signals:
            conf = results[ticker]['confidence']
            risk = results[ticker]['risk_score']
            print(f"  {ticker}: {conf:.1%} confidence, {risk:.2f} risk")
    
    if hold_signals:
        print(f"\n🟡 HOLD Signals ({len(hold_signals)}):")
        for ticker in hold_signals:
            print(f"  {ticker}")
    
    print(f"\n🎆 Key Features of This World-Class System:")
    print("• Adaptive agent weighting based on market conditions")
    print("• Advanced risk management with VaR and drawdown estimates")
    print("• Market microstructure analysis for better timing")
    print("• Intermarket correlation and divergence detection")
    print("• ML ensemble with XGBoost, LightGBM, and neural networks")
    print("• Pattern recognition for chart formations")
    print("• Volatility regime detection and adjustment")
    print("• Position sizing based on confidence and risk")
    print("• Dynamic stop-loss and take-profit levels")
    print("• Multi-timeframe analysis and signal alignment")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'world_class_analysis_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'total_stocks': len(test_stocks)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")
    print("\n✨ This system represents state-of-the-art in agentic trading analysis!")

if __name__ == "__main__":
    test_enhanced_system()
