#!/usr/bin/env python3
"""
Complete Market Analysis Demo

This script demonstrates the full capabilities of the Enhanced Finance Bot
for market-wide analysis, including:
1. Market scanning and stock ranking
2. Portfolio construction and optimization
3. Risk analysis and recommendations
4. Performance reporting and visualization
"""

import os
import sys
from datetime import datetime
import time

# Import our market analysis modules
try:
    from market_scanner import MarketScanner
    from portfolio_builder import PortfolioBuilder
    from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent
    print("✅ Successfully imported all market analysis modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

class CompleteMarketAnalysis:
    """
    Comprehensive market analysis combining all enhanced features
    """
    
    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self.scanner = MarketScanner(use_ml=use_ml, max_workers=3)
        self.builder = PortfolioBuilder(self.scanner)
        self.manager = AdvancedPortfolioManagerAgent(use_ml=use_ml)
        
    def run_complete_analysis(self, portfolio_value: float = 100000, universe: str = "large_cap"):
        """Run complete market analysis workflow"""
        
        print(f"🚀 COMPLETE MARKET ANALYSIS")
        print(f"{'='*60}")
        print(f"Portfolio Value: ${portfolio_value:,.0f}")
        print(f"Stock Universe: {universe}")
        print(f"ML Enhancement: {'Enabled' if self.use_ml else 'Disabled'}")
        print(f"Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        results = {
            "start_time": datetime.now(),
            "portfolio_value": portfolio_value,
            "universe": universe,
            "success": False
        }
        
        try:
            # Step 1: Market Scanning
            print("📊 STEP 1: MARKET SCANNING")
            print("-" * 40)
            
            scan_start = time.time()
            scan_results = self.scanner.scan_market(universe, parallel=True)
            scan_time = time.time() - scan_start
            
            print(f"✅ Market scan completed in {scan_time:.1f} seconds")
            print(f"📈 Analyzed {len(scan_results)} stocks successfully")
            
            results["scan_results"] = len(scan_results)
            results["scan_time"] = scan_time
            
            # Step 2: Generate Market Report
            print(f"\n📋 STEP 2: MARKET ANALYSIS REPORT")
            print("-" * 40)
            
            market_report = self.scanner.generate_market_report(save_to_file=True)
            self.scanner.save_results_to_csv()
            
            print("✅ Market report generated and saved")
            
            # Step 3: Portfolio Construction
            print(f"\n🏗️ STEP 3: PORTFOLIO CONSTRUCTION")
            print("-" * 40)
            
            portfolios = []
            
            # Build Growth Portfolio
            print("Building Growth Portfolio...")
            growth_portfolio = self.builder.build_growth_portfolio(portfolio_value)
            if 'error' not in growth_portfolio:
                portfolios.append(growth_portfolio)
                print(f"✅ Growth portfolio: {len(growth_portfolio['positions'])} positions")
            else:
                print(f"⚠️ Growth portfolio: {growth_portfolio['error']}")
            
            # Build Balanced Portfolio
            print("Building Balanced Portfolio...")
            balanced_portfolio = self.builder.build_balanced_portfolio(portfolio_value)
            if 'error' not in balanced_portfolio:
                portfolios.append(balanced_portfolio)
                print(f"✅ Balanced portfolio: {len(balanced_portfolio['positions'])} positions")
            else:
                print(f"⚠️ Balanced portfolio: {balanced_portfolio['error']}")
            
            # Build Conservative Portfolio
            print("Building Conservative Portfolio...")
            conservative_portfolio = self.builder.build_conservative_portfolio(portfolio_value)
            if 'error' not in conservative_portfolio:
                portfolios.append(conservative_portfolio)
                print(f"✅ Conservative portfolio: {len(conservative_portfolio['positions'])} positions")
            else:
                print(f"⚠️ Conservative portfolio: {conservative_portfolio['error']}")
            
            results["portfolios_built"] = len(portfolios)
            
            # Step 4: Portfolio Analysis Report
            if portfolios:
                print(f"\n📊 STEP 4: PORTFOLIO ANALYSIS")
                print("-" * 40)
                
                portfolio_report = self.builder.generate_portfolio_report(portfolios, save_to_file=True)
                self.builder.save_portfolios_to_json(portfolios)
                
                print("✅ Portfolio analysis completed and saved")
                
                # Display summary of best recommendations
                self.display_quick_summary(portfolios)
            
            # Step 5: Individual Stock Deep Dive (optional)
            print(f"\n🔍 STEP 5: INDIVIDUAL STOCK ANALYSIS")
            print("-" * 40)
            
            # Get top recommendations for detailed analysis
            buy_recs = self.scanner.get_buy_recommendations(top_n=3)
            if buy_recs:
                print("Performing detailed analysis on top picks...")
                for stock in buy_recs[:2]:  # Analyze top 2
                    ticker = stock['ticker']
                    print(f"\n📈 Detailed Analysis: {ticker}")
                    
                    # Train ML model if not already trained
                    if self.use_ml:
                        try:
                            training_result = self.manager.train_ml_models(ticker)
                            if 'error' not in training_result:
                                print(f"   ML models trained for {ticker}")
                        except Exception as e:
                            print(f"   ML training failed for {ticker}: {str(e)}")
                    
                    # Backtest strategy
                    try:
                        backtest_result = self.manager.backtest_strategy(ticker)
                        if 'error' not in backtest_result:
                            total_return = backtest_result.get('total_return', 0)
                            print(f"   Backtest return: {total_return:.1%}")
                        else:
                            print(f"   Backtesting failed: {backtest_result['error']}")
                    except Exception as e:
                        print(f"   Backtesting error: {str(e)}")
            
            # Completion
            results["success"] = True
            results["end_time"] = datetime.now()
            results["total_time"] = (results["end_time"] - results["start_time"]).total_seconds()
            
            print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Total Analysis Time: {results['total_time']:.1f} seconds")
            print(f"Stocks Analyzed: {results['scan_results']}")
            print(f"Portfolios Built: {results['portfolios_built']}")
            print(f"Reports Generated: Multiple files saved")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR DURING ANALYSIS: {str(e)}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            return results
    
    def display_quick_summary(self, portfolios):
        """Display quick summary of portfolio recommendations"""
        print(f"\n💡 QUICK PORTFOLIO SUMMARY")
        print("-" * 30)
        
        for portfolio in portfolios:
            strategy = portfolio.get('strategy', 'Unknown')
            positions = portfolio.get('positions', [])
            metrics = portfolio.get('metrics', {})
            
            # Get top 3 holdings (excluding cash)
            stock_positions = [p for p in positions if p.get('ticker') != 'CASH'][:3]
            
            print(f"\n{strategy} Portfolio:")
            print(f"  Risk Level: {metrics.get('risk_level', 'Unknown')}")
            print(f"  Top Holdings: {', '.join([p['ticker'] for p in stock_positions])}")
            if metrics.get('avg_composite_score'):
                print(f"  Avg Score: {metrics['avg_composite_score']:.3f}")
    
    def interactive_analysis(self):
        """Interactive market analysis with user choices"""
        print("🎯 INTERACTIVE MARKET ANALYSIS")
        print("=" * 50)
        
        # Get user preferences
        print("\n💰 Portfolio Configuration:")
        portfolio_input = input("Enter portfolio value (default: $100,000): ").strip()
        try:
            portfolio_value = float(portfolio_input.replace('$', '').replace(',', '')) if portfolio_input else 100000
        except ValueError:
            portfolio_value = 100000
        
        print("\n📊 Select analysis scope:")
        print("1. Quick Analysis (5 stocks) - Fast results")
        print("2. Standard Analysis (16 stocks) - Balanced depth")
        print("3. Comprehensive Analysis (40 stocks) - Full market view")
        
        scope_choice = input("Enter choice (1-3, default=2): ").strip() or "2"
        scope_map = {"1": "test_small", "2": "large_cap", "3": "sp500_sample"}
        universe = scope_map.get(scope_choice, "large_cap")
        
        print(f"\n🚀 Starting analysis with ${portfolio_value:,.0f} across {universe} universe...")
        
        # Run analysis
        results = self.run_complete_analysis(portfolio_value, universe)
        
        if results.get("success"):
            print(f"\n✨ Analysis completed successfully!")
            print(f"Check the generated files for detailed results:")
            print(f"   📄 market_report_*.txt")
            print(f"   📄 portfolio_report_*.txt") 
            print(f"   📊 market_scan_*.csv")
            print(f"   💾 portfolios_*.json")
        else:
            print(f"\n⚠️ Analysis completed with issues. Check error messages above.")
        
        return results


def demo_quick_analysis():
    """Quick demo of market analysis capabilities"""
    print("⚡ QUICK MARKET ANALYSIS DEMO")
    print("=" * 40)
    
    analyzer = CompleteMarketAnalysis(use_ml=True)
    
    # Run quick analysis on small universe
    results = analyzer.run_complete_analysis(
        portfolio_value=50000,
        universe="test_small"
    )
    
    return results


def main():
    """Main entry point with user choice"""
    print("🎯 Enhanced Finance Bot - Complete Market Analysis")
    print("=" * 60)
    print("This tool provides comprehensive market analysis including:")
    print("• Market scanning and stock ranking")
    print("• ML-powered predictions and recommendations") 
    print("• Multi-strategy portfolio construction")
    print("• Risk analysis and performance metrics")
    print("• Detailed reporting and data export")
    
    print("\n🎮 Choose analysis mode:")
    print("1. Interactive Analysis - Customize your analysis")
    print("2. Quick Demo - Fast demonstration with sample data")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        analyzer = CompleteMarketAnalysis(use_ml=True)
        analyzer.interactive_analysis()
    elif choice == "2":
        demo_quick_analysis()
    elif choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("Invalid choice. Running interactive analysis...")
        analyzer = CompleteMarketAnalysis(use_ml=True)
        analyzer.interactive_analysis()
    
    print(f"\n🎉 Thank you for using the Enhanced Finance Bot!")
    print(f"💡 Remember: All analysis is for educational purposes only.")
    print(f"📚 Always conduct your own research before making investment decisions.")


if __name__ == "__main__":
    main()
