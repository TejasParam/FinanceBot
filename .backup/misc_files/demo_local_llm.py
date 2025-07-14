#!/usr/bin/env python3
"""
Demonstration of FinanceBot with Local LLM Integration (Ollama)

This script shows how to use the agentic FinanceBot system with a local LLM
for generating natural language explanations and insights.

Prerequisites:
1. Ollama installed and running: brew install ollama && brew services start ollama
2. LLM model downloaded: ollama pull llama3.2:3b
3. Python environment with required packages

Usage:
python demo_local_llm.py [TICKER]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import Dict, Any
from agentic_portfolio_manager import AgenticPortfolioManager

def test_local_llm_connection():
    """Test if local LLM (Ollama) is available"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print("✅ Ollama is running with the following models:")
                for model in models:
                    print(f"   - {model['name']} ({model.get('size', 'unknown size')})")
                return True
            else:
                print("❌ Ollama is running but no models are installed")
                print("   Install a model: ollama pull llama3.2:3b")
                return False
        else:
            print("❌ Ollama is not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Start Ollama: brew services start ollama")
        return False

def run_local_llm_analysis(ticker: str = "AAPL"):
    """Run comprehensive analysis with local LLM explanations"""
    
    print(f"🤖 FinanceBot Local LLM Demo - Analyzing {ticker}")
    print("=" * 60)
    
    # Test Ollama connection first
    if not test_local_llm_connection():
        print("\n⚠️ Local LLM not available. Running without LLM explanations.")
        use_llm = False
    else:
        use_llm = True
    
    # Initialize portfolio manager
    print(f"\n📊 Initializing Agentic Portfolio Manager...")
    manager = AgenticPortfolioManager()
    
    # Enable local LLM if available
    if use_llm:
        print("🔧 Enabling local LLM integration...")
        manager.agent_coordinator.agents['LLMExplanation'].enable_local_llm()
    
    print(f"\n🔍 Running comprehensive analysis for {ticker}...")
    
    # Run full agentic analysis
    try:
        results = manager.analyze_stock(ticker, detailed=True)
        
        print(f"\n📈 Analysis Results for {ticker}")
        print("=" * 50)
        
        # Overall recommendation
        overall_score = results.get('composite_score', 0.0)
        recommendation = results.get('recommendation', 'HOLD')
        confidence = results.get('confidence', 0.0)
        
        print(f"Overall Score: {overall_score:.3f}")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence:.1%}")
        
        # Agent breakdown
        print(f"\n🤖 Agent Analysis Breakdown:")
        print("-" * 40)
        
        # Get the full agentic analysis
        agentic_analysis = results.get('agentic_analysis', {})
        agent_results = agentic_analysis.get('agent_results', {})
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                score = result.get('score', 0.0)
                signal = result.get('signal', 'HOLD')
                conf = result.get('confidence', 0.0)
                print(f"{agent_name:20} | {score:6.3f} | {signal:4} | {conf:5.1%}")
        
        # LLM Explanation (if available)
        llm_result = agent_results.get('LLMExplanation', {})
        if isinstance(llm_result, dict) and 'error' not in llm_result:
            print(f"\n🧠 AI Explanation & Insights:")
            print("=" * 50)
            
            # Main explanation
            reasoning = llm_result.get('reasoning', '')
            if reasoning:
                print(f"Summary: {reasoning}")
            
            # Detailed analysis
            detailed = llm_result.get('detailed_analysis', {})
            if detailed:
                print(f"\n📋 Detailed Analysis:")
                
                investment_thesis = detailed.get('investment_thesis', '')
                if investment_thesis:
                    print(f"\nInvestment Thesis:")
                    print(investment_thesis)
                
                key_insights = detailed.get('key_insights', '')
                if key_insights:
                    print(f"\nKey Insights:")
                    print(key_insights)
                
                recommendations = detailed.get('recommendations', '')
                if recommendations:
                    print(f"\nActionable Recommendations:")
                    print(recommendations)
            
            # Check if this was a real LLM or simulation
            llm_model = llm_result.get('llm_model', 'simulation')
            local_llm_used = llm_result.get('local_llm_used', False)
            
            if use_llm and 'real' in str(llm_model):
                if local_llm_used:
                    print(f"\n🤖 Analysis powered by local LLM (Ollama)")
                else:
                    print(f"\n☁️ Analysis powered by cloud LLM API")
            else:
                print(f"\n⚠️ Using simulated LLM analysis")
        
        else:
            print(f"\n⚠️ LLM explanation not available")
            if 'error' in llm_result:
                print(f"   Error: {llm_result['error']}")
        
        # Risk assessment
        print(f"\n⚖️ Risk Assessment:")
        print("-" * 30)
        
        # Try to get risk metrics from agentic analysis or legacy manager
        risk_assessment = results.get('risk_assessment', 'No specific risk factors identified')
        print(f"Risk Factors: {risk_assessment}")
        
        # Show execution metrics
        execution_time = results.get('execution_time', 0.0)
        agents_used = results.get('agents_used', 0)
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Agents Used: {agents_used}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main demonstration function"""
    
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print("🤖 FinanceBot Local LLM Integration Demo")
    print("=" * 60)
    print("This demo shows how to use FinanceBot with local LLM (Ollama)")
    print("for intelligent financial analysis and explanations.")
    
    # Run the analysis
    results = run_local_llm_analysis(ticker)
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"\nTo analyze another stock, run:")
        print(f"python {os.path.basename(__file__)} TSLA")
        
        print(f"\n📚 Available features:")
        print("- Multi-agent analysis (Technical, Fundamental, Sentiment, ML, Regime)")
        print("- Local LLM explanations (powered by Ollama)")
        print("- Risk assessment and portfolio metrics")
        print("- Backtesting and optimization")
        
    else:
        print(f"\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
