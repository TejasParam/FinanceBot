#!/usr/bin/env python3
"""
Simple test of LLM agent with Ollama
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMExplanationAgent
import json

def test_llm_agent():
    print("🧪 Testing LLM Agent with Ollama")
    print("=" * 40)
    
    # Create LLM agent
    llm_agent = LLMExplanationAgent()
    
    # Enable local LLM
    llm_agent.enable_local_llm()
    
    # Test data
    test_agent_results = {
        "TechnicalAnalysis": {
            "score": 0.7,
            "confidence": 0.8,
            "signal": "BUY",
            "reasoning": "Strong upward trend with RSI at 60, moving averages aligned bullishly."
        },
        "FundamentalAnalysis": {
            "score": 0.5,
            "confidence": 0.7,
            "signal": "HOLD",
            "reasoning": "Fair valuation with P/E ratio of 25, decent revenue growth."
        },
        "SentimentAnalysis": {
            "score": 0.3,
            "confidence": 0.6,
            "signal": "HOLD",
            "reasoning": "Mixed sentiment from news and social media sources."
        }
    }
    
    print(f"📊 Running LLM analysis for AAPL...")
    
    # Run analysis
    result = llm_agent.analyze("AAPL", test_agent_results)
    
    print(f"\n📋 LLM Analysis Result:")
    print("=" * 40)
    
    # Print key results
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Signal: {result.get('signal', 'N/A')}")
    print(f"LLM Model: {result.get('llm_model', 'N/A')}")
    print(f"Local LLM Used: {result.get('local_llm_used', 'N/A')}")
    
    print(f"\n🧠 Main Reasoning:")
    print("-" * 30)
    reasoning = result.get('reasoning', 'No reasoning provided')
    print(reasoning)
    
    # Print detailed analysis if available
    detailed = result.get('detailed_analysis', {})
    if detailed:
        print(f"\n📖 Detailed Analysis:")
        print("-" * 30)
        
        full_text = detailed.get('full_text', '')
        if full_text:
            print("Full LLM Response:")
            print(full_text)
        
        thesis = detailed.get('investment_thesis', '')
        if thesis:
            print(f"\nInvestment Thesis: {thesis}")
        
        insights = detailed.get('key_insights', '')
        if insights:
            print(f"\nKey Insights: {insights}")
        
        recommendations = detailed.get('recommendations', '')
        if recommendations:
            print(f"\nRecommendations: {recommendations}")
    
    # Check for errors
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    test_llm_agent()
