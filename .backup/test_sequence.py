#!/usr/bin/env python3
"""
Test analyzing multiple stocks in sequence
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

from agentic_portfolio_manager import AgenticPortfolioManager

# Create manager
manager = AgenticPortfolioManager(
    use_ml=True,
    use_llm=True,
    parallel_execution=False
)

# Test multiple stocks in sequence
stocks = ['AAPL', 'MSFT', 'GOOGL']

for stock in stocks:
    try:
        print(f"\nAnalyzing {stock}...")
        result = manager.analyze_stock(stock)
        print(f"✓ {stock}: {result['recommendation']} (Confidence: {result['confidence']:.1%})")
    except Exception as e:
        print(f"✗ {stock} failed: {e}")
        import traceback
        traceback.print_exc()