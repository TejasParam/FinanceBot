#!/usr/bin/env python3
"""
LLM Integration Test - Demonstrates how to enable real LLM capabilities
"""

import os
from agentic_portfolio_manager import AgenticPortfolioManager

def test_llm_integration():
    """Test LLM integration with different providers"""
    
    print("🤖 LLM Integration Test")
    print("=" * 50)
    
    # Initialize agentic manager
    manager = AgenticPortfolioManager(use_llm=True)
    
    # Test data for LLM analysis
    test_agent_results = {
        "TechnicalAnalysis": {
            "score": 0.7,
            "confidence": 0.8,
            "signal": "BUY",
            "reasoning": "RSI oversold, bullish MACD crossover"
        },
        "SentimentAnalysis": {
            "score": 0.6,
            "confidence": 0.9,
            "signal": "BUY",
            "reasoning": "Positive news sentiment, analyst upgrades"
        },
        "FundamentalAnalysis": {
            "score": 0.5,
            "confidence": 0.7,
            "signal": "HOLD",
            "reasoning": "Fair valuation, steady fundamentals"
        }
    }
    
    llm_agent = manager.agent_coordinator.agents['LLMExplanation']
    
    print("\n1. Testing Simulation Mode (Default)")
    print("-" * 40)
    
    result = llm_agent.analyze("AAPL", test_agent_results)
    print(f"✅ Simulation Result:")
    print(f"   Score: {result['score']:.2f}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reasoning: {result['reasoning'][:100]}...")
    
    print("\n2. OpenAI Integration Test")
    print("-" * 40)
    
    # Check for OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ OpenAI API key found")
        try:
            llm_agent.enable_real_llm(openai_key, "openai")
            result = llm_agent.analyze("AAPL", test_agent_results)
            
            if 'error' not in result:
                print("✅ OpenAI integration successful!")
                print(f"   Full analysis: {result.get('detailed_analysis', {}).get('full_text', '')[:200]}...")
            else:
                print(f"⚠️ OpenAI test failed: {result['error']}")
                
        except Exception as e:
            print(f"⚠️ OpenAI integration error: {e}")
    else:
        print("ℹ️ No OpenAI API key found (set OPENAI_API_KEY environment variable)")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
    
    print("\n3. Anthropic Integration Test")
    print("-" * 40)
    
    # Check for Anthropic API key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        print("✅ Anthropic API key found")
        try:
            # Reset to simulation mode first
            llm_agent.disable_real_llm()
            llm_agent.enable_real_llm(anthropic_key, "anthropic")
            result = llm_agent.analyze("AAPL", test_agent_results)
            
            if 'error' not in result:
                print("✅ Anthropic integration successful!")
                print(f"   Full analysis: {result.get('detailed_analysis', {}).get('full_text', '')[:200]}...")
            else:
                print(f"⚠️ Anthropic test failed: {result['error']}")
                
        except Exception as e:
            print(f"⚠️ Anthropic integration error: {e}")
    else:
        print("ℹ️ No Anthropic API key found (set ANTHROPIC_API_KEY environment variable)")
        print("   Example: export ANTHROPIC_API_KEY='your-api-key-here'")
    
    print("\n4. Local LLM Integration Test (Ollama)")
    print("-" * 40)
    
    try:
        import requests
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama service detected")
            try:
                llm_agent.disable_real_llm()
                llm_agent.enable_real_llm("local", "local")
                result = llm_agent.analyze("AAPL", test_agent_results)
                
                if 'error' not in result:
                    print("✅ Local LLM integration successful!")
                    print(f"   Analysis: {result.get('reasoning', '')[:200]}...")
                else:
                    print(f"⚠️ Local LLM test failed: {result['error']}")
                    
            except Exception as e:
                print(f"⚠️ Local LLM integration error: {e}")
        else:
            print("ℹ️ Ollama not running on localhost:11434")
            print("   Install: brew install ollama && ollama serve")
            print("   Then: ollama pull llama2")
            
    except Exception as e:
        print("ℹ️ Ollama not available")
        print("   Install: brew install ollama && ollama serve")
    
    print("\n5. Production Example")
    print("-" * 40)
    
    print("Example production usage:")
    print("""
# Set up environment
export OPENAI_API_KEY="your-api-key"

# Python code
from agentic_portfolio_manager import AgenticPortfolioManager

manager = AgenticPortfolioManager(use_llm=True)

# Enable real LLM
manager.agent_coordinator.agents['LLMExplanation'].enable_real_llm(
    api_key=os.getenv('OPENAI_API_KEY'),
    provider="openai"
)

# Analyze with real LLM explanations
result = manager.analyze_stock("AAPL")
llm_analysis = result['agentic_analysis']['agent_results']['LLMExplanation']
print("LLM Explanation:", llm_analysis['detailed_analysis']['full_text'])
""")
    
    print("\n✨ LLM Integration Test Complete!")
    print("The system works in simulation mode by default and can be")
    print("enhanced with real LLM capabilities when API keys are provided.")

if __name__ == "__main__":
    test_llm_integration()
