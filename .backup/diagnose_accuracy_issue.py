#!/usr/bin/env python3
"""Diagnose why accuracy isn't improving"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import yfinance as yf
import pandas as pd

print("DIAGNOSING ACCURACY ISSUE...")
print("=" * 60)

# Get sample data
ticker = 'AAPL'
data = yf.download(ticker, start='2023-03-01', end='2023-04-01', progress=False)

# Initialize coordinator
coordinator = AgentCoordinator(enable_ml=False, enable_llm=False, parallel_execution=False)

# Analyze once
print("\nRunning single analysis...")
analysis = coordinator.analyze_stock(ticker, price_data=data)

# Check individual agent scores
print("\nIndividual Agent Scores:")
for agent_name, result in analysis['agent_results'].items():
    if isinstance(result, dict) and 'score' in result:
        score = result.get('score', 0)
        confidence = result.get('confidence', 0)
        print(f"  {agent_name:20s}: score={score:+.2f}, conf={confidence:.0%}")

# Check aggregation
agg = analysis['aggregated_analysis']
print(f"\nAggregated Result:")
print(f"  Overall Score: {agg['overall_score']:.2f}")
print(f"  Overall Confidence: {agg['overall_confidence']:.0%}")
print(f"  Recommendation: {agg['recommendation']}")

# Check market context
mc = analysis.get('market_context', {})
print(f"\nMarket Context:")
print(f"  Trend: {mc.get('trend', 'unknown')}")
print(f"  Volatility: {mc.get('volatility', 0):.1%}")
print(f"  Momentum: {mc.get('momentum_5d', 0):.1%}")

# Check filtering
print(f"\nFiltering:")
print(f"  Should Trade: {analysis.get('should_trade', True)}")
print(f"  Filter Reason: {analysis.get('filter_reason', 'N/A')}")
print(f"  Confidence Threshold: {analysis.get('confidence_threshold', 0.7):.0%}")

# The problem
print(f"\n{'='*60}")
print("PROBLEMS IDENTIFIED:")
print("1. System is too bullish (always STRONG_BUY)")
print("2. Confidence is always very high (80-95%)")
print("3. No bearish signals even in down markets")
print("4. Market context adjustments may be too aggressive")

# Check if we're in an uptrend that's causing bias
print(f"\nPrice Movement in Test Period:")
start_price = data.iloc[0]['Close']
end_price = data.iloc[-1]['Close']
period_return = (end_price - start_price) / start_price
print(f"  Start: ${start_price:.2f}")
print(f"  End: ${end_price:.2f}")
print(f"  Return: {period_return:.1%}")

if period_return > 0:
    print("\nNOTE: Test period had positive returns, which may bias results.")
    print("The momentum adjustments in the coordinator are amplifying bullish signals.")