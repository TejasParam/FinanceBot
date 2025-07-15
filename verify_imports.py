#!/usr/bin/env python3
"""
Verify all imports in the trading bot files
"""

import sys
import importlib

def test_imports():
    """Test importing all modules"""
    modules_to_test = [
        ('alpaca_integration', 'AlpacaTradingBot'),
        ('automated_portfolio_manager', 'AutomatedPortfolioManager'),
        ('main_trading_bot', 'TradingBotRunner'),
        ('dashboard_api', 'app'),
        ('sentiment_analyzer', 'MarketSentimentAnalyzer'),
        ('ml_predictor_enhanced', 'EnhancedMLPredictor'),
        ('data_collection', 'DataCollectionAgent'),
        ('portfolio_optimizer', 'EnhancedPortfolioOptimizer'),
        ('news_sentiment_enhanced', 'EnhancedNewsSentimentAnalyzer'),
    ]
    
    print("Testing imports...\n")
    
    failed = []
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✓ {module_name}.{class_name}")
            else:
                print(f"✗ {module_name}.{class_name} - class not found")
                failed.append((module_name, class_name))
        except ImportError as e:
            print(f"✗ {module_name} - {str(e)}")
            failed.append((module_name, str(e)))
        except Exception as e:
            print(f"✗ {module_name} - Unexpected error: {str(e)}")
            failed.append((module_name, str(e)))
    
    print("\n" + "="*50)
    if failed:
        print(f"Failed imports: {len(failed)}")
        for item in failed:
            print(f"  - {item}")
    else:
        print("All imports successful!")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)