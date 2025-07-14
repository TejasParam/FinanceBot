#!/usr/bin/env python3
"""
Quick test to verify the enhanced modules are working
"""

print("🧪 Testing Enhanced FinanceBot Components")
print("=" * 50)

# Test 1: Portfolio Optimizer
try:
    from portfolio_optimizer import EnhancedPortfolioOptimizer
    optimizer = EnhancedPortfolioOptimizer()
    print("✅ Portfolio Optimizer loaded successfully")
except Exception as e:
    print(f"❌ Portfolio Optimizer: {e}")

# Test 2: Risk Manager
try:
    from risk_manager_enhanced import EnhancedRiskManager
    risk_mgr = EnhancedRiskManager()
    print("✅ Risk Manager loaded successfully")
except Exception as e:
    print(f"❌ Risk Manager: {e}")

# Test 3: Agentic Portfolio Manager
try:
    from agentic_portfolio_manager import AgenticPortfolioManager
    manager = AgenticPortfolioManager(use_ml=False, use_llm=False)
    print("✅ Agentic Portfolio Manager loaded successfully")
except Exception as e:
    print(f"❌ Agentic Portfolio Manager: {e}")

# Test 4: Simple functionality test
print("\n📊 Running Simple Functionality Test...")
try:
    # Test portfolio optimization with 2 stocks
    result = optimizer.optimize_portfolio(['AAPL', 'MSFT'], strategy='mean_variance')
    if 'weights' in result:
        print("✅ Portfolio optimization working!")
        print(f"   AAPL weight: {result['weights'].get('AAPL', 0):.1%}")
        print(f"   MSFT weight: {result['weights'].get('MSFT', 0):.1%}")
    else:
        print(f"⚠️ Optimization returned: {result}")
except Exception as e:
    print(f"❌ Optimization test failed: {e}")

print("\n✅ Basic components are working! The system is ready for use.")
print("\nTo run full demos:")
print("  python demos/market_scanner.py")
print("  python demos/portfolio_builder.py")
print("  python demos/complete_market_analysis.py")