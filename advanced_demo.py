#!/usr/bin/env python3
"""
Advanced Finance Bot Demo - Showcasing ML, Backtesting, and Enhanced Analysis

This script demonstrates the enhanced capabilities of the finance bot including:
1. Machine learning predictions
2. Strategy backtesting
3. Strategy optimization
4. Market regime analysis
5. Comprehensive analysis with multiple data sources

Run this script to see the full power of the enhanced finance bot.
"""

import os
import sys
from datetime import datetime
import traceback

# Import our enhanced portfolio manager
try:
    from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent
    print("✅ Successfully imported AdvancedPortfolioManagerAgent")
except ImportError as e:
    print(f"❌ Failed to import AdvancedPortfolioManagerAgent: {e}")
    sys.exit(1)

def demo_basic_analysis(manager, ticker="AAPL"):
    """Demonstrate basic enhanced analysis"""
    print(f"\n{'='*50}")
    print(f"🔍 BASIC ENHANCED ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        result = manager.analyze_stock(ticker)
        
        print(f"\n📊 Analysis Results:")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Risk Assessment: {result['risk_assessment']}")
        
        if 'composite_score' in result:
            print(f"\n🧮 Scoring Breakdown:")
            print(f"Composite Score: {result['composite_score']:.2f}")
            for component, score in result['component_scores'].items():
                print(f"  {component.title()}: {score:.2f}")
        
        if result.get('ml_analysis') and 'error' not in result['ml_analysis']:
            ml = result['ml_analysis']
            print(f"\n🤖 ML Prediction:")
            print(f"Direction: {'UP' if ml['ensemble_prediction'] == 1 else 'DOWN'}")
            print(f"Probability Up: {ml['probability_up']:.1%}")
            print(f"ML Confidence: {ml['confidence']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        traceback.print_exc()
        return False

def demo_ml_training(manager, ticker="AAPL"):
    """Demonstrate ML model training"""
    print(f"\n{'='*50}")
    print(f"🎓 ML MODEL TRAINING FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        if not manager.use_ml:
            print("⚠️ ML components not available. Install required packages.")
            return False
        
        result = manager.train_ml_models(ticker, period="2y")
        
        if "error" in result:
            print(f"❌ Training failed: {result['error']}")
            return False
        
        print(f"✅ Training completed successfully!")
        print(f"Data points used: {result['data_points']}")
        print(f"Models saved to: {result['models_saved']}")
        
        if 'training_results' in result:
            print(f"\n📈 Model Performance:")
            for model_name, metrics in result['training_results'].items():
                print(f"  {model_name.title()}:")
                print(f"    Test Accuracy: {metrics['test_accuracy']:.1%}")
                print(f"    Cross Validation: {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
        traceback.print_exc()
        return False

def demo_backtesting(manager, ticker="AAPL"):
    """Demonstrate strategy backtesting"""
    print(f"\n{'='*50}")
    print(f"📈 STRATEGY BACKTESTING FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        if not manager.use_ml:
            print("⚠️ Backtesting requires ML components.")
            return False
        
        # Run backtest for the last year
        result = manager.backtest_strategy(ticker, start_date="2024-01-01")
        
        if "error" in result:
            print(f"❌ Backtesting failed: {result['error']}")
            return False
        
        print(f"✅ Backtest completed!")
        print(f"Final Portfolio Value: ${result['final_value']:,.2f}")
        print(f"Total Return: {result['total_return']:.1%}")
        print(f"Number of Transactions: {len(result['transactions'])}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n📊 Performance Metrics:")
            print(f"  Strategy Return: {metrics.get('total_return', 0):.1%}")
            print(f"  Benchmark Return: {metrics.get('benchmark_return', 0):.1%}")
            print(f"  Excess Return: {metrics.get('excess_return', 0):.1%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
        
        print(f"\n📋 Recent Transactions:")
        for transaction in result['transactions'][-5:]:  # Show last 5 transactions
            print(f"  {transaction['date'].strftime('%Y-%m-%d')}: {transaction['action']} "
                  f"{transaction.get('shares', 'N/A')} shares at ${transaction['price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        traceback.print_exc()
        return False

def demo_strategy_optimization(manager, ticker="AAPL"):
    """Demonstrate strategy optimization"""
    print(f"\n{'='*50}")
    print(f"🔧 STRATEGY OPTIMIZATION FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        if not manager.use_ml:
            print("⚠️ Strategy optimization requires ML components.")
            return False
        
        result = manager.optimize_strategy(ticker)
        
        if "error" in result:
            print(f"❌ Optimization failed: {result['error']}")
            return False
        
        print(f"✅ Strategy optimization completed!")
        print(f"Best Score: {result['best_score']:.3f}")
        
        print(f"\n🎯 Optimized Weights:")
        for component, weight in result['best_weights'].items():
            print(f"  {component.title()}: {weight:.1%}")
        
        print(f"\n📊 All Tested Combinations:")
        for i, test_result in enumerate(result['all_results'], 1):
            print(f"  Test {i}: Return={test_result['return']:.1%}, "
                  f"Sharpe={test_result['sharpe']:.2f}, Score={test_result['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy optimization failed: {e}")
        traceback.print_exc()
        return False

def demo_market_regime_analysis(manager, ticker="AAPL"):
    """Demonstrate market regime analysis"""
    print(f"\n{'='*50}")
    print(f"📊 MARKET REGIME ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        result = manager.get_market_regime_analysis(ticker)
        
        if "error" in result:
            print(f"❌ Regime analysis failed: {result['error']}")
            return False
        
        print(f"✅ Market regime analysis completed!")
        print(f"Current Regime: {result['regime']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Recommendation: {result['recommendation']}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n📈 Market Metrics:")
            print(f"  Volatility (Annualized): {metrics['volatility']:.1%}")
            print(f"  Trend Strength: {metrics['trend_strength']:.4f}")
            print(f"  Price Range: {metrics['price_range']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market regime analysis failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("🚀 ADVANCED FINANCE BOT DEMONSTRATION")
    print("=====================================")
    
    # Get ticker from user or use default
    ticker = input("Enter ticker symbol (default: AAPL): ").upper() or "AAPL"
    
    print(f"\n🎯 Running demonstrations for {ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the advanced portfolio manager
    print(f"\n🔧 Initializing Advanced Portfolio Manager...")
    try:
        manager = AdvancedPortfolioManagerAgent(use_ml=True)
        print(f"✅ Manager initialized. ML enabled: {manager.use_ml}")
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return
    
    # Run demonstrations
    demos = [
        ("Basic Enhanced Analysis", demo_basic_analysis),
        ("ML Model Training", demo_ml_training),
        ("Strategy Backtesting", demo_backtesting),
        ("Strategy Optimization", demo_strategy_optimization),
        ("Market Regime Analysis", demo_market_regime_analysis),
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        print(f"\n{'🔄' if manager.use_ml else '⚠️'} Running {demo_name}...")
        try:
            success = demo_func(manager, ticker)
            results[demo_name] = "✅ Success" if success else "❌ Failed"
        except Exception as e:
            print(f"❌ {demo_name} failed with exception: {e}")
            results[demo_name] = f"❌ Exception: {str(e)[:50]}..."
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 DEMONSTRATION SUMMARY")
    print(f"{'='*50}")
    print(f"Ticker: {ticker}")
    print(f"ML Components Available: {manager.use_ml}")
    print()
    
    for demo_name, result in results.items():
        print(f"{result} {demo_name}")
    
    print(f"\n💡 Tips:")
    if not manager.use_ml:
        print("- Install scikit-learn, matplotlib, and joblib for full ML functionality")
    print("- Train ML models on sufficient historical data (2+ years recommended)")
    print("- Backtest strategies before using them with real money")
    print("- This system provides analysis, not guaranteed predictions")
    print("- Always consider market conditions and do your own research")
    
    print(f"\n🎉 Demo completed! The enhanced finance bot is ready for use.")

if __name__ == "__main__":
    main()
