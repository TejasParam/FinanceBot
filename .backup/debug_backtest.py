#!/usr/bin/env python3
"""
Debug the Series comparison error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traceback
from agentic_portfolio_manager import AgenticPortfolioManager

# Create manager with sequential execution
manager = AgenticPortfolioManager(
    use_ml=True,
    use_llm=True, 
    parallel_execution=False
)

# Try to analyze a stock and catch the full error
try:
    result = manager.analyze_stock('AAPL')
    print("Analysis successful!")
    print(f"Recommendation: {result.get('recommendation')}")
    print(f"Confidence: {result.get('confidence')}")
except Exception as e:
    print("Error occurred!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()