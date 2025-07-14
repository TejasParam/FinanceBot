#!/usr/bin/env python3
"""
Simple demo script to test portfolio creation functionality
Run this to verify that your portfolio optimization is working correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import your modules
from portfolio_optimizer import EnhancedPortfolioOptimizer
from risk_manager_enhanced import EnhancedRiskManager

def test_basic_portfolio_creation():
    """Test basic portfolio creation workflow"""
    print("🚀 Testing Basic Portfolio Creation")
    print("=" * 50)
    
    # Initialize components
    optimizer = EnhancedPortfolioOptimizer()
    risk_manager = EnhancedRiskManager()
    
    # Define test portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    print(f"📈 Testing with tickers: {tickers}")
    
    # Test 1: Mean-Variance Optimization
    print("\n1️⃣ Testing Mean-Variance Optimization...")
    try:
        mv_result = optimizer.optimize_portfolio(
            tickers=tickers,
            strategy='mean_variance'
        )
        
        if 'weights' in mv_result:
            print("✅ Success!")
            print("📊 Optimal weights:")
            for ticker, weight in mv_result['weights'].items():
                print(f"   {ticker}: {weight:.1%}")
            
            print(f"\n📈 Expected Annual Return: {mv_result.get('expected_return', 0):.2%}")
            print(f"📉 Annual Volatility: {mv_result.get('volatility', 0):.2%}")
            print(f"⚡ Sharpe Ratio: {mv_result.get('sharpe_ratio', 0):.2f}")
            
            if 'metrics' in mv_result:
                metrics = mv_result['metrics']
                print(f"📊 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"🎯 VaR (95%): {metrics.get('var_95', 0):.2%}")
        else:
            print(f"❌ Failed: {mv_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Risk Parity
    print("\n2️⃣ Testing Risk Parity Optimization...")
    try:
        rp_result = optimizer.optimize_portfolio(
            tickers=tickers,
            strategy='risk_parity'
        )
        
        if 'weights' in rp_result:
            print("✅ Success!")
            print("📊 Risk parity weights:")
            for ticker, weight in rp_result['weights'].items():
                print(f"   {ticker}: {weight:.1%}")
            
            if 'risk_contributions' in rp_result:
                print("📊 Risk contributions:")
                for ticker, contrib in rp_result['risk_contributions'].items():
                    print(f"   {ticker}: {contrib:.3f}")
        else:
            print(f"❌ Failed: {rp_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Black-Litterman with Views
    print("\n3️⃣ Testing Black-Litterman with Market Views...")
    try:
        # Define some market views
        views = {
            'absolute': {
                'AAPL': 0.15,  # Expect 15% return
                'MSFT': 0.12,  # Expect 12% return
                'GOOGL': 0.10  # Expect 10% return
            },
            'confidence': {
                'AAPL': 0.8,
                'MSFT': 0.7,
                'GOOGL': 0.6
            }
        }
        
        bl_result = optimizer.optimize_portfolio(
            tickers=tickers,
            strategy='black_litterman',
            views=views
        )
        
        if 'weights' in bl_result:
            print("✅ Success!")
            print("📊 Black-Litterman weights:")
            for ticker, weight in bl_result['weights'].items():
                print(f"   {ticker}: {weight:.1%}")
            
            if 'view_impact' in bl_result:
                print("📊 View impact on returns:")
                for ticker, impact in bl_result['view_impact'].items():
                    print(f"   {ticker}: {impact:+.2%}")
        else:
            print(f"❌ Failed: {bl_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Portfolio with Constraints
    print("\n4️⃣ Testing Constrained Optimization...")
    try:
        constraints = {
            'weight_bounds': {
                'AAPL': (0.10, 0.40),  # Between 10-40%
                'MSFT': (0.15, 0.35),  # Between 15-35%
                'GOOGL': (0.05, 0.30), # Between 5-30%
                'AMZN': (0.05, 0.25)   # Between 5-25%
            }
        }
        
        const_result = optimizer.optimize_portfolio(
            tickers=tickers,
            strategy='max_sharpe',
            constraints=constraints
        )
        
        if 'weights' in const_result:
            print("✅ Success!")
            print("📊 Constrained weights:")
            for ticker, weight in const_result['weights'].items():
                print(f"   {ticker}: {weight:.1%}")
            
            # Verify constraints
            print("✅ Constraint verification:")
            for ticker, (min_w, max_w) in constraints['weight_bounds'].items():
                actual_weight = const_result['weights'].get(ticker, 0)
                status = "✅" if min_w <= actual_weight <= max_w else "❌"
                print(f"   {ticker}: {actual_weight:.1%} (range: {min_w:.1%}-{max_w:.1%}) {status}")
        else:
            print(f"❌ Failed: {const_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Risk Analysis
    print("\n5️⃣ Testing Risk Analysis...")
    try:
        # Use the mean-variance weights for risk analysis
        if 'weights' in mv_result:
            portfolio_weights = mv_result['weights']
            
            # Monte Carlo simulation
            mc_results = risk_manager.monte_carlo_portfolio_simulation(
                portfolio_weights,
                n_simulations=1000,
                time_horizon=63,  # 3 months
                initial_value=100000
            )
            
            if mc_results:
                print("✅ Monte Carlo simulation successful!")
                print(f"📊 Expected Value: ${mc_results['mean_final_value']:,.0f}")
                print(f"📉 VaR (95%): ${mc_results['var_95']:,.0f}")
                print(f"📊 Probability of Loss: {mc_results['probability_loss']:.1%}")
                print(f"📈 Probability of 20% Gain: {mc_results['probability_gain_20pct']:.1%}")
            
            # Stress testing
            stress_results = risk_manager.stress_test_portfolio(portfolio_weights)
            
            if stress_results and 'summary' in stress_results:
                print("\n✅ Stress testing successful!")
                print(f"📊 Worst scenario: {stress_results['summary']['worst_scenario']}")
                print(f"📉 Average loss: {stress_results['summary']['average_loss']:.1%}")
        
    except Exception as e:
        print(f"❌ Risk analysis error: {e}")
    
    print("\n🎉 Portfolio creation test completed!")
    return True


def test_efficient_frontier():
    """Test efficient frontier generation"""
    print("\n📈 Testing Efficient Frontier Generation")
    print("=" * 50)
    
    optimizer = EnhancedPortfolioOptimizer()
    tickers = ["AAPL", "MSFT"]  # Use fewer tickers for faster testing
    
    try:
        frontier_data = optimizer.efficient_frontier(tickers, n_portfolios=10)
        
        if not frontier_data.empty:
            print("✅ Efficient frontier generated successfully!")
            print(f"📊 Generated {len(frontier_data)} portfolio points")
            print("\nSample efficient portfolios:")
            print("Risk (Vol) | Return | Sharpe")
            print("-" * 30)
            
            for i, row in frontier_data.head().iterrows():
                vol = row['volatility']
                ret = row['return']
                sharpe = row['sharpe']
                print(f"{vol:8.1%} | {ret:6.1%} | {sharpe:6.2f}")
            
            # Find best Sharpe ratio portfolio
            best_sharpe_idx = frontier_data['sharpe'].idxmax()
            best_portfolio = frontier_data.loc[best_sharpe_idx]
            
            print(f"\n🏆 Best Sharpe Ratio Portfolio:")
            print(f"   Return: {best_portfolio['return']:.2%}")
            print(f"   Risk: {best_portfolio['volatility']:.2%}")
            print(f"   Sharpe: {best_portfolio['sharpe']:.2f}")
            
        else:
            print("❌ Failed to generate efficient frontier")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def test_rebalancing():
    """Test portfolio rebalancing functionality"""
    print("\n⚖️ Testing Portfolio Rebalancing")
    print("=" * 50)
    
    optimizer = EnhancedPortfolioOptimizer()
    
    try:
        # Current portfolio allocation
        current_weights = {
            'AAPL': 0.45,  # Overweight
            'MSFT': 0.35,
            'GOOGL': 0.20
        }
        
        # Target allocation
        target_weights = {
            'AAPL': 0.35,  # Need to reduce
            'MSFT': 0.40,  # Need to increase
            'GOOGL': 0.25  # Need to increase
        }
        
        # Current values
        current_values = {
            'AAPL': 45000,
            'MSFT': 35000,
            'GOOGL': 20000
        }
        
        rebalance_result = optimizer.rebalance_portfolio(
            current_weights,
            target_weights,
            current_values,
            threshold=0.02  # 2% threshold
        )
        
        print("✅ Rebalancing analysis completed!")
        print(f"\n📊 Current vs Target Allocation:")
        print("Ticker | Current | Target | Difference")
        print("-" * 40)
        
        for ticker in current_weights:
            current = current_weights[ticker]
            target = target_weights[ticker]
            diff = target - current
            print(f"{ticker:6} | {current:7.1%} | {target:6.1%} | {diff:+7.1%}")
        
        if rebalance_result['trades']:
            print(f"\n💰 Recommended Trades:")
            for ticker, trade in rebalance_result['trades'].items():
                action = trade['action']
                value = trade['value']
                print(f"   {action} ${value:,.0f} of {ticker}")
            
            print(f"\n💸 Estimated Transaction Costs: ${rebalance_result['estimated_costs']:,.2f}")
            print(f"   Cost as % of portfolio: {rebalance_result['cost_percentage']:.3f}%")
        else:
            print("\n✅ No rebalancing needed - portfolio is within tolerance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def main():
    """Run all portfolio tests"""
    print("🌟 FinanceBot Portfolio Creation Test Suite")
    print("=" * 60)
    print("This will test the core portfolio optimization functionality")
    print("Note: Tests use live market data, so results may vary\n")
    
    # Run tests
    tests = [
        ("Basic Portfolio Creation", test_basic_portfolio_creation),
        ("Efficient Frontier", test_efficient_frontier),
        ("Portfolio Rebalancing", test_rebalancing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Your portfolio system is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print("\n💡 TIP: Run individual functions to debug specific issues:")
    print("   python demo_portfolio_test.py")


if __name__ == "__main__":
    main()