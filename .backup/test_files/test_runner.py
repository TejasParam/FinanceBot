#!/usr/bin/env python3
"""
Test Runner for FinanceBot Portfolio System
Choose from different testing options
"""

import sys
import os

def print_menu():
    """Display the test menu"""
    print("🧪 FinanceBot Portfolio Test Menu")
    print("=" * 40)
    print("1. Quick Demo Test (5 minutes)")
    print("2. Full Test Suite (15 minutes)")
    print("3. Portfolio Creation Only")
    print("4. Risk Management Only")
    print("5. Performance Test")
    print("6. Exit")
    print("-" * 40)

def run_quick_demo():
    """Run the quick demo test"""
    print("🚀 Running Quick Demo Test...")
    try:
        from demo_portfolio_test import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are installed")
    except Exception as e:
        print(f"❌ Test error: {e}")

def run_full_test_suite():
    """Run the comprehensive test suite"""
    print("🧪 Running Full Test Suite...")
    try:
        from test_portfolio_creation import run_portfolio_tests
        run_portfolio_tests()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are installed")
    except Exception as e:
        print(f"❌ Test error: {e}")

def run_portfolio_only():
    """Test only portfolio creation"""
    print("📈 Testing Portfolio Creation Only...")
    try:
        from demo_portfolio_test import test_basic_portfolio_creation
        test_basic_portfolio_creation()
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")

def run_risk_only():
    """Test only risk management"""
    print("⚠️ Testing Risk Management Only...")
    try:
        from risk_manager_enhanced import EnhancedRiskManager
        
        print("Initializing risk manager...")
        risk_manager = EnhancedRiskManager()
        
        # Test VaR calculation
        print("Testing VaR calculation...")
        var_result = risk_manager.calculate_all_var_metrics('AAPL', period=100)
        if var_result:
            print("✅ VaR calculation successful")
            print(f"   Historical VaR: {var_result['var_historical']:.2f}%")
            print(f"   Parametric VaR: {var_result['var_parametric']:.2f}%")
            print(f"   Monte Carlo VaR: {var_result['var_monte_carlo']:.2f}%")
        else:
            print("❌ VaR calculation failed")
        
        # Test Monte Carlo simulation
        print("Testing Monte Carlo simulation...")
        portfolio = {'AAPL': 0.6, 'MSFT': 0.4}
        mc_result = risk_manager.monte_carlo_portfolio_simulation(
            portfolio, n_simulations=1000
        )
        if mc_result:
            print("✅ Monte Carlo simulation successful")
            print(f"   Mean final value: ${mc_result['mean_final_value']:,.0f}")
            print(f"   VaR (95%): ${mc_result['var_95']:,.0f}")
        else:
            print("❌ Monte Carlo simulation failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")

def run_performance_test():
    """Run performance benchmarks"""
    print("⚡ Running Performance Tests...")
    import time
    
    try:
        from portfolio_optimizer import EnhancedPortfolioOptimizer
        from parallel_agent_executor import ParallelAgentExecutor
        
        # Test optimization speed
        optimizer = EnhancedPortfolioOptimizer()
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        print("Testing portfolio optimization speed...")
        start_time = time.time()
        
        result = optimizer.optimize_portfolio(tickers, strategy='mean_variance')
        
        end_time = time.time()
        duration = end_time - start_time
        
        if 'weights' in result:
            print(f"✅ Optimization completed in {duration:.2f} seconds")
            print("📊 Results:")
            for ticker, weight in result['weights'].items():
                print(f"   {ticker}: {weight:.1%}")
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
        
        # Test parallel execution if available
        try:
            print("\nTesting parallel execution...")
            executor = ParallelAgentExecutor(use_ray=False)  # Use multiprocessing
            
            # This would test parallel agent execution
            print("✅ Parallel execution framework loaded")
            
        except Exception as e:
            print(f"⚠️ Parallel execution not available: {e}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Performance test error: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking Dependencies...")
    
    required_modules = [
        'pandas', 'numpy', 'yfinance', 'scipy', 'matplotlib'
    ]
    
    optional_modules = [
        'ray', 'pypfopt', 'cvxpy'
    ]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_required.append(module)
            print(f"❌ {module} (REQUIRED)")
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (optional)")
        except ImportError:
            missing_optional.append(module)
            print(f"⚠️ {module} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required modules: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional modules: {', '.join(missing_optional)}")
        print("Some features may not work. Install with: pip install " + " ".join(missing_optional))
    
    print("\n✅ All required dependencies are available!")
    return True

def main():
    """Main test runner"""
    print("🌟 Welcome to FinanceBot Portfolio Testing!")
    print("This tool helps you test your portfolio optimization system.\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before testing.")
        return
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                run_quick_demo()
            elif choice == '2':
                run_full_test_suite()
            elif choice == '3':
                run_portfolio_only()
            elif choice == '4':
                run_risk_only()
            elif choice == '5':
                run_performance_test()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
            
            input("\nPress Enter to continue...")
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()