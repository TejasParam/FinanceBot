#!/usr/bin/env python3
"""Debug the backtest Series error"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Get historical data like the backtest does
symbol = 'AAPL'
start_dt = pd.to_datetime('2023-01-01')
end_dt = pd.to_datetime('2023-01-10')
current_date = pd.to_datetime('2023-01-03')

# Get extra data for indicators
data_start = start_dt - timedelta(days=100)
print(f"Downloading data from {data_start} to {end_dt}...")
data = yf.download(symbol, start=data_start, end=end_dt + timedelta(days=10), progress=False)

# Get historical data up to current date
historical_data = data[data.index <= current_date]
print(f"Historical data shape: {historical_data.shape}")

# Create coordinator and analyze
coordinator = AgentCoordinator(enable_ml=True, enable_llm=False, parallel_execution=False)

print("\nAnalyzing with coordinator...")
try:
    analysis = coordinator.analyze_stock(
        ticker=symbol,
        price_data=historical_data
    )
    print("Analysis completed successfully!")
    print(f"Aggregated score: {analysis['aggregated_analysis']['overall_score']}")
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Print agent results
    if 'agent_results' in locals():
        print("\nAgent results before error:")
        for agent, result in analysis.get('agent_results', {}).items():
            print(f"  {agent}: {result.get('error', 'OK')}")