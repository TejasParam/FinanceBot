#!/usr/bin/env python3
"""
Test that all fixes are working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator import AgentCoordinator
import warnings
warnings.filterwarnings('ignore')

def test_fixes():
    """Test that all components work without errors"""
    
    print("Testing fixes...")
    print("-" * 40)
    
    # Initialize coordinator
    coordinator = AgentCoordinator(enable_ml=True, enable_llm=False, parallel_execution=False)
    
    # Test a single stock
    ticker = 'AAPL'
    
    try:
        print(f"Analyzing {ticker}...")
        result = coordinator.analyze_stock(ticker)
        
        # Check if all agents ran
        agents_results = result['agent_results']
        
        print(f"\nTotal agents in results: {len(agents_results)}")
        print("\nAgent Results:")
        for agent_name, agent_result in agents_results.items():
            if 'error' in agent_result:
                print(f"  {agent_name}: ❌ ERROR - {agent_result['error'][:50]}")
            else:
                score = agent_result.get('score', 'N/A')
                print(f"  {agent_name}: ✓ Score={score}")
        
        # Check overall results
        overall = result['aggregated_analysis']
        print(f"\nOverall Score: {overall['overall_score']:.3f}")
        print(f"Overall Confidence: {overall['overall_confidence']:.3f}")
        print(f"Recommendation: {overall['recommendation']}")
        
        # Check if HFT and StatArb are working
        if 'HFTEngine' in agents_results and 'error' not in agents_results['HFTEngine']:
            print("\n✅ HFT Engine is working!")
        else:
            print("\n❌ HFT Engine has issues")
            
        if 'StatisticalArbitrage' in agents_results and 'error' not in agents_results['StatisticalArbitrage']:
            print("✅ Statistical Arbitrage is working!")
        else:
            print("❌ Statistical Arbitrage has issues")
            
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixes()