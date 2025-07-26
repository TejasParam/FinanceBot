#!/usr/bin/env python3
"""
Test a single prediction to debug the error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import traceback
from agentic_portfolio_manager import AgenticPortfolioManager

# Create manager
manager = AgenticPortfolioManager(
    use_ml=True,
    use_llm=True,
    parallel_execution=False
)

# Test date
test_date = '2023-06-15'
symbol = 'AAPL'

try:
    print(f"Testing {symbol} on {test_date}")
    
    # Get historical data
    test_dt = pd.to_datetime(test_date)
    hist_data = yf.download(
        symbol, 
        start=test_dt - timedelta(days=1),
        end=test_dt + timedelta(days=1),
        progress=False
    )
    
    print(f"Historical data shape: {hist_data.shape}")
    print(f"Current price: {hist_data['Close'].iloc[-1]}")
    
    # Run analysis - this is where the error might be
    print("\nRunning agentic analysis...")
    analysis = manager.analyze_stock(symbol)
    
    print("\nAnalysis successful!")
    print(f"Recommendation: {analysis.get('recommendation')}")
    print(f"Confidence: {analysis.get('confidence'):.1%}")
    
except Exception as e:
    print("\nError occurred!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()