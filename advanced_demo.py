#!/usr/bin/env python3
"""
Advanced Finance Bot Demo - Showcasing Agentic AI, ML, Backtesting, and Enhanced Analysis

This script demonstrates the enhanced capabilities of the finance bot including:
1. Multi-agent agentic AI analysis
2. Machine learning predictions
3. Strategy backtesting
4. Strategy optimization
5. Market regime analysis
6. Specialized agent analysis (Technical, Fundamental, Sentiment, etc.)
7. LLM-powered explanations and insights

Run this script to see the full power of the agentic finance bot.
"""

import os
import sys
from datetime import datetime
import traceback

# Import our agentic portfolio manager
try:
    from agentic_portfolio_manager import AgenticPortfolioManager
    print("✅ Successfully imported AgenticPortfolioManager")
    AGENTIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agentic system not available: {e}")
    AGENTIC_AVAILABLE = False

# Fallback to legacy manager if agentic not available
if not AGENTIC_AVAILABLE:
    try:
        from portfolio_manager_rule_based import AdvancedPortfolioManagerAgent
        print("✅ Using legacy AdvancedPortfolioManagerAgent")
    except ImportError as e:
        print(f"❌ Failed to import any portfolio manager: {e}")
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
        
        # Check if we have agentic analysis data
        if 'agents_used' in result:
            print(f"\n🤖 Agentic Features:")
            print(f"Agents Used: {result['agents_used']}")
            print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        traceback.print_exc()
        return False

def demo_agentic_analysis(manager, ticker="AAPL"):
    """Demonstrate multi-agent agentic analysis"""
    print(f"\n{'='*50}")
    print(f"🤖 AGENTIC AI MULTI-AGENT ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        result = manager.analyze_stock(ticker)
        
        print(f"\n📊 Agentic Analysis Results:")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Risk Assessment: {result['risk_assessment']}")
        
        if 'composite_score' in result:
            print(f"\n🧮 Overall Scoring:")
            print(f"Composite Score: {result['composite_score']:.2f}")
            print(f"Agents Used: {result.get('agents_used', 0)}")
            print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if 'agent_consensus' in result:
            consensus = result['agent_consensus']
            print(f"\n🤝 Agent Consensus:")
            print(f"Consensus Level: {consensus.get('level', 'unknown').title()}")
            print(f"Direction: {consensus.get('direction', 'unknown').title()}")
            print(f"Strength: {consensus.get('strength', 0):.1%}")
        
        if 'component_scores' in result:
            print(f"\n📈 Component Breakdown:")
            for component, score in result['component_scores'].items():
                print(f"  {component.title()}: {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agentic analysis failed: {e}")
        traceback.print_exc()
        return False
    """Demonstrate multi-agent agentic analysis"""
    print(f"\n{'='*50}")
    print(f"🤖 AGENTIC AI MULTI-AGENT ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        result = manager.analyze_stock(ticker)
        
        print(f"\n📊 Agentic Analysis Results:")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Risk Assessment: {result['risk_assessment']}")
        
        if 'composite_score' in result:
            print(f"\n🧮 Overall Scoring:")
            print(f"Composite Score: {result['composite_score']:.2f}")
            print(f"Agents Used: {result.get('agents_used', 0)}")
            print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if 'agent_consensus' in result:
            consensus = result['agent_consensus']
            print(f"\n🤝 Agent Consensus:")
            print(f"Consensus Level: {consensus.get('level', 'unknown').title()}")
            print(f"Direction: {consensus.get('direction', 'unknown').title()}")
            print(f"Strength: {consensus.get('strength', 0):.1%}")
        
        if 'component_scores' in result:
            print(f"\n📈 Component Breakdown:")
            for component, score in result['component_scores'].items():
                print(f"  {component.title()}: {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agentic analysis failed: {e}")
        traceback.print_exc()
        return False

def demo_detailed_agent_analysis(manager, ticker="AAPL"):
    """Demonstrate detailed analysis by individual agents"""
    print(f"\n{'='*50}")
    print(f"🔍 DETAILED AGENT-BY-AGENT ANALYSIS FOR {ticker}")
    print(f"{'='*50}")
    
    try:
        # First run the analysis
        manager.analyze_stock(ticker)
        
        # Get detailed breakdown
        detailed = manager.get_detailed_analysis(ticker)
        
        if 'error' in detailed:
            print(f"❌ Could not get detailed analysis: {detailed['error']}")
            return False
        
        print(f"\n🤖 Individual Agent Results:")
        print(f"Execution Time: {detailed['execution_time']:.2f}s")
        print(f"Agents Used: {len(detailed['agents_used'])}")
        
        for agent_name, agent_result in detailed['by_agent'].items():
            print(f"\n  📋 {agent_name.replace('Analysis', '').replace('Detection', '')} Agent:")
            print(f"    Success: {'✅' if agent_result['success'] else '❌'}")
            if agent_result['success']:
                print(f"    Score: {agent_result['score']:.2f}")
                print(f"    Confidence: {agent_result['confidence']:.1%}")
                print(f"    Reasoning: {agent_result['reasoning'][:100]}...")
                
                # Show agent-specific details
                if 'indicators' in agent_result:
                    print(f"    Key Indicators: RSI={agent_result['indicators'].get('rsi', 'N/A'):.1f}, "
                          f"MACD={agent_result['indicators'].get('macd_signal', 'N/A')}")
                elif 'regime' in agent_result:
                    print(f"    Market Regime: {agent_result['regime']}")
                elif 'sources' in agent_result:
                    print(f"    Sources Analyzed: {agent_result['sources']}")
                elif 'model_accuracies' in agent_result:
                    accuracies = agent_result['model_accuracies']
                    if accuracies:
                        ensemble_acc = accuracies.get('ensemble', 0)
                        print(f"    Model Accuracy: {ensemble_acc:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed agent analysis failed: {e}")
        traceback.print_exc()
        return False

def demo_agent_comparison(manager, tickers=["AAPL", "MSFT", "GOOGL"]):
    """Demonstrate comparison across multiple stocks"""
    print(f"\n{'='*50}")
    print(f"📊 MULTI-STOCK AGENT COMPARISON")
    print(f"{'='*50}")
    
    try:
        print(f"Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
        
        # Analyze all tickers
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            manager.analyze_stock(ticker)
        
        # Get comparison
        comparison_df = manager.compare_analyses(tickers)
        
        if comparison_df.empty:
            print(f"❌ No comparison data available")
            return False
        
        print(f"\n📈 Comparison Results:")
        print(f"{'Ticker':<8} {'Score':<6} {'Conf':<5} {'Rec':<12} {'Consensus':<10}")
        print("-" * 50)
        
        for _, row in comparison_df.iterrows():
            ticker = row['Ticker']
            score = row['Overall_Score']
            conf = row['Confidence']
            rec = row['Recommendation']
            consensus = row['Consensus']
            
            print(f"{ticker:<8} {score:>5.2f} {conf:>4.1%} {rec:<12} {consensus:<10}")
        
        # Find best and worst
        best_idx = comparison_df['Overall_Score'].idxmax()
        worst_idx = comparison_df['Overall_Score'].idxmin()
        
        best_ticker = comparison_df.loc[best_idx, 'Ticker']
        worst_ticker = comparison_df.loc[worst_idx, 'Ticker']
        
        print(f"\n🏆 Best: {best_ticker} (Score: {comparison_df.loc[best_idx, 'Overall_Score']:.2f})")
        print(f"🔻 Worst: {worst_ticker} (Score: {comparison_df.loc[worst_idx, 'Overall_Score']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent comparison failed: {e}")
        traceback.print_exc()
        return False

def demo_agent_performance(manager):
    """Demonstrate agent performance statistics"""
    print(f"\n{'='*50}")
    print(f"📈 AGENT PERFORMANCE STATISTICS")
    print(f"{'='*50}")
    
    try:
        performance = manager.get_agent_performance()
        status = manager.get_agent_status()
        
        print(f"\n🔧 Overall Statistics:")
        print(f"Total Analyses: {performance['total_analyses']}")
        print(f"Successful: {performance['successful_analyses']}")
        print(f"Failed: {performance['failed_analyses']}")
        
        success_rate = (performance['successful_analyses'] / 
                       performance['total_analyses'] if performance['total_analyses'] > 0 else 0)
        print(f"Success Rate: {success_rate:.1%}")
        
        print(f"\n🤖 Individual Agent Performance:")
        print(f"{'Agent':<20} {'Success':<8} {'Failure':<8} {'Rate':<6} {'Status'}")
        print("-" * 60)
        
        for agent_name, agent_status in status.items():
            success = agent_status['success_count']
            failure = agent_status['failure_count']
            rate = agent_status['success_rate']
            enabled = "Enabled" if agent_status['enabled'] else "Disabled"
            
            display_name = agent_name.replace('Analysis', '').replace('Detection', '')
            print(f"{display_name:<20} {success:<8} {failure:<8} {rate:<5.1%} {enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent performance analysis failed: {e}")
        traceback.print_exc()
        return False
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
    print("🚀 ADVANCED AGENTIC FINANCE BOT DEMONSTRATION")
    print("=============================================")
    
    # Get ticker from user or use default
    ticker = input("Enter ticker symbol (default: AAPL): ").upper() or "AAPL"
    
    print(f"\n🎯 Running demonstrations for {ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the appropriate portfolio manager
    print(f"\n🔧 Initializing Portfolio Manager...")
    try:
        if AGENTIC_AVAILABLE:
            manager = AgenticPortfolioManager(use_ml=True, use_llm=True, parallel_execution=True)
            print(f"✅ Agentic Portfolio Manager initialized")
            use_agentic = True
        else:
            manager = AdvancedPortfolioManagerAgent(use_ml=True)
            print(f"✅ Legacy Portfolio Manager initialized (ML enabled: {manager.use_ml})")
            use_agentic = False
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return
    
    # Define demonstrations based on available features
    if use_agentic:
        demos = [
            ("Basic Enhanced Analysis", demo_basic_analysis),
            ("Agentic AI Multi-Agent Analysis", demo_agentic_analysis),
            ("Detailed Agent-by-Agent Analysis", demo_detailed_agent_analysis),
            ("Multi-Stock Agent Comparison", demo_agent_comparison),
            ("Agent Performance Statistics", demo_agent_performance),
            ("ML Model Training", demo_ml_training),
            ("Strategy Backtesting", demo_backtesting),
            ("Strategy Optimization", demo_strategy_optimization),
            ("Market Regime Analysis", demo_market_regime_analysis),
        ]
    else:
        demos = [
            ("Basic Enhanced Analysis", demo_basic_analysis),
            ("ML Model Training", demo_ml_training),
            ("Strategy Backtesting", demo_backtesting),
            ("Strategy Optimization", demo_strategy_optimization),
            ("Market Regime Analysis", demo_market_regime_analysis),
        ]
    
    # Run demonstrations
    results = {}
    for demo_name, demo_func in demos:
        print(f"\n{'🔄' if use_agentic else '⚠️'} Running {demo_name}...")
        try:
            if demo_name in ["Multi-Stock Agent Comparison", "Agent Performance Statistics"]:
                # Special handling for demos that don't take ticker
                if demo_name == "Agent Performance Statistics":
                    success = demo_func(manager)
                else:
                    success = demo_func(manager)
            else:
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
    print(f"System Type: {'Agentic AI' if use_agentic else 'Legacy'}")
    if use_agentic:
        print(f"Features: Multi-agent AI, LLM explanations, parallel execution")
    else:
        print(f"Features: ML analysis, backtesting, optimization")
    print()
    
    for demo_name, result in results.items():
        print(f"{result} {demo_name}")
    
    print(f"\n💡 Tips:")
    if use_agentic:
        print("- The agentic system uses specialized AI agents for comprehensive analysis")
        print("- Each agent contributes expertise in technical, fundamental, sentiment, etc.")
        print("- LLM agent provides natural language explanations and insights")
        print("- Agent consensus helps identify high-confidence opportunities")
    else:
        print("- Install the agents module for full agentic AI functionality")
    print("- Train ML models on sufficient historical data (2+ years recommended)")
    print("- Backtest strategies before using them with real money")
    print("- This system provides analysis, not guaranteed predictions")
    print("- Always consider market conditions and do your own research")
    
    if use_agentic:
        print(f"\n🎉 Agentic AI demo completed! The multi-agent finance system is ready.")
    else:
        print(f"\n🎉 Demo completed! The enhanced finance bot is ready for use.")

if __name__ == "__main__":
    main()
