#!/usr/bin/env python3
"""
Simple test with just one stock and date to isolate the issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

# Try importing and testing each component separately
print("Testing individual components...")

# Test 1: Direct agent coordinator
try:
    from agents.coordinator import AgentCoordinator
    coordinator = AgentCoordinator(
        enable_ml=True,
        enable_llm=True,
        parallel_execution=False
    )
    print("✓ Agent Coordinator initialized")
    
    # Test analyze_stock directly
    result = coordinator.analyze_stock('AAPL')
    print(f"✓ Direct coordinator analysis works: {result['aggregated_analysis']['recommendation']}")
except Exception as e:
    print(f"✗ Coordinator failed: {e}")

# Test 2: Through portfolio manager
try:
    from agentic_portfolio_manager import AgenticPortfolioManager
    manager = AgenticPortfolioManager(
        use_ml=True,
        use_llm=True,
        parallel_execution=False
    )
    print("✓ Portfolio Manager initialized")
    
    # Test analyze_stock
    result = manager.analyze_stock('AAPL')
    print(f"✓ Manager analysis works: {result['recommendation']}")
except Exception as e:
    print(f"✗ Manager failed: {e}")

# Test 3: Check each agent individually
print("\nTesting individual agents...")
agents_to_test = [
    ('technical', 'agents.technical_agent', 'TechnicalAnalysisAgent'),
    ('sentiment', 'agents.sentiment_agent', 'SentimentAnalysisAgent'),
    ('fundamental', 'agents.fundamental_agent', 'FundamentalAnalysisAgent'),
    ('regime', 'agents.regime_agent', 'RegimeDetectionAgent'),
    ('ml', 'agents.ml_agent', 'MLPredictionAgent'),
]

for name, module_path, class_name in agents_to_test:
    try:
        exec(f"from {module_path} import {class_name}")
        exec(f"agent = {class_name}()")
        exec(f"result = agent.analyze('AAPL')")
        exec(f"print(f'✓ {name} agent works')")
    except Exception as e:
        print(f"✗ {name} agent failed: {e}")