#!/usr/bin/env python3
"""
Verify all imports are working correctly after migration to enhanced modules
"""

import sys

def test_import(module_name, class_name=None):
    """Test importing a module and optionally a class"""
    try:
        if class_name:
            exec(f"from {module_name} import {class_name}")
            print(f"✅ {module_name}.{class_name}")
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name}: Unexpected error - {e}")
        return False

print("🔍 Verifying Enhanced Module Imports")
print("=" * 50)

# Test enhanced modules
enhanced_modules = [
    ("portfolio_optimizer", "EnhancedPortfolioOptimizer"),
    ("risk_manager_enhanced", "EnhancedRiskManager"),
    ("backtest_engine_enhanced", "EnhancedBacktestEngine"),
    ("ml_predictor_enhanced", "EnhancedMLPredictor"),
    ("technical_analysis_enhanced", "EnhancedTechnicalAnalyst"),
    ("news_sentiment_enhanced", "EnhancedNewsSentimentAnalyst"),
    ("data_validation_pipeline", "DataValidationPipeline"),
    ("realtime_data_processor", "RealTimeDataProcessor"),
    ("parallel_agent_executor", "ParallelAgentExecutor"),
]

print("\n📦 Enhanced Modules:")
success_count = 0
for module, class_name in enhanced_modules:
    if test_import(module, class_name):
        success_count += 1

# Test agents
print("\n🤖 Agent Modules:")
agent_modules = [
    ("agents.technical_agent", "TechnicalAnalysisAgent"),
    ("agents.fundamental_agent", "FundamentalAnalysisAgent"),
    ("agents.sentiment_agent", "SentimentAnalysisAgent"),
    ("agents.ml_agent", "MLPredictionAgent"),
    ("agents.ml_agent_enhanced", "EnhancedMLPredictionAgent"),
    ("agents.regime_agent", "MarketRegimeAgent"),
    ("agents.coordinator", "AgentCoordinator"),
]

for module, class_name in agent_modules:
    if test_import(module, class_name):
        success_count += 1

# Test main components
print("\n🎯 Main Components:")
main_components = [
    ("agentic_portfolio_manager", "AgenticPortfolioManager"),
    ("data_collection", "DataCollectionAgent"),
]

for module, class_name in main_components:
    if test_import(module, class_name):
        success_count += 1

# Test demos
print("\n📊 Demo Modules:")
demo_modules = [
    ("demos.market_scanner", "MarketScanner"),
    ("demos.portfolio_builder", "PortfolioBuilder"),
    ("demos.complete_market_analysis", "CompleteMarketAnalysis"),
]

for module, class_name in demo_modules:
    if test_import(module, class_name):
        success_count += 1

# Summary
total_tests = len(enhanced_modules) + len(agent_modules) + len(main_components) + len(demo_modules)
print("\n" + "=" * 50)
print(f"📊 IMPORT VERIFICATION SUMMARY")
print(f"Total tests: {total_tests}")
print(f"Successful: {success_count}")
print(f"Failed: {total_tests - success_count}")

if success_count == total_tests:
    print("\n🎉 All imports successful! The project is ready to use.")
else:
    print(f"\n⚠️ {total_tests - success_count} imports failed. Please check the errors above.")
    sys.exit(1)

# Test instantiation of key components
print("\n🔧 Testing Component Instantiation:")

try:
    from portfolio_optimizer import EnhancedPortfolioOptimizer
    optimizer = EnhancedPortfolioOptimizer()
    print("✅ Portfolio Optimizer instantiated successfully")
except Exception as e:
    print(f"❌ Portfolio Optimizer: {e}")

try:
    from risk_manager_enhanced import EnhancedRiskManager
    risk_mgr = EnhancedRiskManager()
    print("✅ Risk Manager instantiated successfully")
except Exception as e:
    print(f"❌ Risk Manager: {e}")

try:
    from agentic_portfolio_manager import AgenticPortfolioManager
    manager = AgenticPortfolioManager(use_ml=False, use_llm=False)
    print("✅ Agentic Portfolio Manager instantiated successfully")
except Exception as e:
    print(f"❌ Agentic Portfolio Manager: {e}")

print("\n✅ Import verification complete!")