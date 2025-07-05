#!/usr/bin/env python3
"""
Quick Start Guide for Enhanced Finance Bot

This script shows how to use the new advanced features:
1. Basic enhanced analysis
2. ML model training
3. Strategy backtesting
4. Strategy optimization
5. Market regime analysis
"""

from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent

def main():
    print("🚀 Enhanced Finance Bot - Quick Start")
    print("=" * 50)
    
    # Initialize the advanced portfolio manager
    manager = AdvancedPortfolioManagerAgent(use_ml=True)
    
    # Example 1: Basic Enhanced Analysis
    print("\n1️⃣ Basic Enhanced Analysis")
    result = manager.analyze_stock("AAPL")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Composite Score: {result.get('composite_score', 'N/A'):.3f}")
    
    # Example 2: Train ML Models (optional)
    print("\n2️⃣ ML Model Training")
    if not manager.ml_predictor.is_trained:
        print("   Training ML models... (this may take a moment)")
        training_result = manager.train_ml_models("AAPL")
        if "error" not in training_result:
            print("   ✅ ML models trained successfully!")
        else:
            print(f"   ⚠️ Training failed: {training_result['error']}")
    else:
        print("   ✅ ML models already trained!")
    
    # Example 3: Strategy Backtesting
    print("\n3️⃣ Strategy Backtesting")
    backtest_result = manager.backtest_strategy("AAPL")
    if "error" not in backtest_result:
        print(f"   Strategy Return: {backtest_result.get('total_return', 'N/A')}")
        print(f"   Sharpe Ratio: {backtest_result.get('sharpe_ratio', 'N/A')}")
    else:
        print(f"   ⚠️ Backtesting failed: {backtest_result['error']}")
    
    # Example 4: Strategy Optimization
    print("\n4️⃣ Strategy Optimization")
    optimization_result = manager.optimize_strategy("AAPL")
    if "error" not in optimization_result:
        print("   ✅ Strategy optimized!")
        print(f"   Best weights: {optimization_result['best_weights']}")
    else:
        print(f"   ⚠️ Optimization failed: {optimization_result['error']}")
    
    # Example 5: Market Regime Analysis
    print("\n5️⃣ Market Regime Analysis")
    regime_result = manager.get_market_regime_analysis("AAPL")
    if "error" not in regime_result:
        print(f"   Current Regime: {regime_result['regime']}")
        print(f"   Recommendation: {regime_result['recommendation']}")
    else:
        print(f"   ⚠️ Regime analysis failed: {regime_result['error']}")
    
    print("\n🎉 Quick start completed!")
    print("\n💡 Next Steps:")
    print("   - Try different stocks (MSFT, GOOGL, TSLA, etc.)")
    print("   - Experiment with different analysis parameters")
    print("   - Use backtesting to validate strategies")
    print("   - Train ML models on multiple stocks for better predictions")

if __name__ == "__main__":
    main()
