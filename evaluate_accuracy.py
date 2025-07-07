#!/usr/bin/env python3
"""
FinanceBot Model Accuracy Evaluation

This script evaluates the accuracy of all components in the FinanceBot system:
1. ML Prediction models (Random Forest, Gradient Boosting, Logistic Regression)
2. Agent analysis consistency
3. LLM explanation quality
4. Overall system performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import time
from datetime import datetime, timedelta

from agentic_portfolio_manager import AgenticPortfolioManager
from ml_predictor import MLPredictor
import joblib

def evaluate_ml_models():
    """Evaluate ML model accuracy"""
    print("🤖 Evaluating ML Model Accuracy")
    print("=" * 50)
    
    # Check if models are already trained
    models_path = "models/portfolio_models.pkl"
    if os.path.exists(models_path):
        print("✅ Loading existing trained models...")
        models_data = joblib.load(models_path)
        ml_predictor = models_data.get('ml_predictor')
        
        if ml_predictor and hasattr(ml_predictor, 'model_accuracies'):
            print("📊 Model Performance Summary:")
            print("-" * 30)
            
            for model_name, accuracy in ml_predictor.model_accuracies.items():
                print(f"{model_name:20} | {accuracy:7.1%}")
            
            # Get cross-validation scores if available
            if hasattr(ml_predictor, 'cv_scores'):
                print(f"\n📈 Cross-Validation Scores:")
                print("-" * 30)
                for model_name, cv_scores in ml_predictor.cv_scores.items():
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                    print(f"{model_name:20} | {mean_score:6.1%} ± {std_score:5.1%}")
            
            return ml_predictor.model_accuracies
        else:
            print("⚠️ Models found but no accuracy data available")
    else:
        print("❌ No trained models found. Train models first using advanced_demo.py")
    
    return {}

def evaluate_agent_consistency():
    """Evaluate consistency of agent analyses across multiple runs"""
    print("\n🔍 Evaluating Agent Consistency")
    print("=" * 50)
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    manager = AgenticPortfolioManager()
    
    consistency_results = {}
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker} consistency...")
        
        # Run analysis multiple times
        runs = []
        for i in range(3):
            try:
                result = manager.analyze_stock(ticker)
                agentic_analysis = result.get('agentic_analysis', {})
                agent_results = agentic_analysis.get('agent_results', {})
                
                run_data = {}
                for agent_name, agent_result in agent_results.items():
                    if isinstance(agent_result, dict) and 'error' not in agent_result:
                        run_data[agent_name] = {
                            'score': agent_result.get('score', 0.0),
                            'confidence': agent_result.get('confidence', 0.0),
                            'signal': agent_result.get('signal', 'HOLD')
                        }
                
                runs.append(run_data)
                print(f"  Run {i+1}: {len(run_data)} agents successful")
                
            except Exception as e:
                print(f"  Run {i+1}: Failed - {e}")
        
        # Calculate consistency metrics
        if len(runs) >= 2:
            consistency_scores = {}
            
            for agent_name in runs[0].keys():
                if all(agent_name in run for run in runs):
                    scores = [run[agent_name]['score'] for run in runs]
                    confidences = [run[agent_name]['confidence'] for run in runs]
                    
                    score_std = np.std(scores)
                    confidence_std = np.std(confidences)
                    
                    # Lower standard deviation = higher consistency
                    consistency_score = max(0, 1.0 - score_std)
                    consistency_scores[agent_name] = consistency_score
            
            consistency_results[ticker] = consistency_scores
            
            print(f"  Consistency scores:")
            for agent, score in consistency_scores.items():
                print(f"    {agent:20} | {score:6.1%}")
    
    return consistency_results

def evaluate_llm_quality():
    """Evaluate LLM explanation quality"""
    print("\n🧠 Evaluating LLM Explanation Quality")
    print("=" * 50)
    
    # Test if local LLM is available
    manager = AgenticPortfolioManager()
    llm_agent = manager.agent_coordinator.agents['LLMExplanation']
    
    # Try to enable local LLM
    try:
        llm_agent.enable_local_llm()
        print("✅ Local LLM (Ollama) enabled for testing")
        llm_available = True
    except:
        print("⚠️ Local LLM not available, using simulation")
        llm_available = False
    
    test_tickers = ['AAPL', 'TSLA']
    llm_quality_metrics = {}
    
    for ticker in test_tickers:
        print(f"\n📝 Testing LLM explanations for {ticker}...")
        
        try:
            result = manager.analyze_stock(ticker)
            agentic_analysis = result.get('agentic_analysis', {})
            agent_results = agentic_analysis.get('agent_results', {})
            llm_result = agent_results.get('LLMExplanation', {})
            
            if 'error' not in llm_result:
                # Quality metrics
                reasoning = llm_result.get('reasoning', '')
                detailed = llm_result.get('detailed_analysis', {})
                
                quality_metrics = {
                    'reasoning_length': len(reasoning),
                    'has_detailed_analysis': bool(detailed),
                    'has_investment_thesis': bool(detailed.get('investment_thesis', '')),
                    'has_key_insights': bool(detailed.get('key_insights', '')),
                    'has_recommendations': bool(detailed.get('recommendations', '')),
                    'local_llm_used': llm_result.get('local_llm_used', False),
                    'confidence': llm_result.get('confidence', 0.0)
                }
                
                llm_quality_metrics[ticker] = quality_metrics
                
                print(f"  Reasoning length: {quality_metrics['reasoning_length']} chars")
                print(f"  Has detailed analysis: {quality_metrics['has_detailed_analysis']}")
                print(f"  LLM confidence: {quality_metrics['confidence']:.1%}")
                print(f"  Local LLM used: {quality_metrics['local_llm_used']}")
                
            else:
                print(f"  ❌ LLM analysis failed: {llm_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
    
    return llm_quality_metrics

def evaluate_system_performance():
    """Evaluate overall system performance"""
    print("\n⚡ Evaluating System Performance")
    print("=" * 50)
    
    manager = AgenticPortfolioManager()
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    performance_metrics = {
        'execution_times': [],
        'success_rates': [],
        'agent_success_counts': {}
    }
    
    for ticker in test_tickers:
        print(f"\n⏱️ Performance test for {ticker}...")
        
        start_time = time.time()
        try:
            result = manager.analyze_stock(ticker)
            execution_time = time.time() - start_time
            
            agentic_analysis = result.get('agentic_analysis', {})
            agent_results = agentic_analysis.get('agent_results', {})
            
            # Count successful agents
            successful_agents = 0
            total_agents = len(agent_results)
            
            for agent_name, agent_result in agent_results.items():
                if isinstance(agent_result, dict) and 'error' not in agent_result:
                    successful_agents += 1
                    
                    if agent_name not in performance_metrics['agent_success_counts']:
                        performance_metrics['agent_success_counts'][agent_name] = 0
                    performance_metrics['agent_success_counts'][agent_name] += 1
            
            success_rate = successful_agents / total_agents if total_agents > 0 else 0
            
            performance_metrics['execution_times'].append(execution_time)
            performance_metrics['success_rates'].append(success_rate)
            
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Success rate: {success_rate:.1%} ({successful_agents}/{total_agents})")
            
        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
    
    if performance_metrics['execution_times']:
        avg_time = np.mean(performance_metrics['execution_times'])
        avg_success = np.mean(performance_metrics['success_rates'])
        
        print(f"\n📊 Average Performance:")
        print(f"  Execution time: {avg_time:.2f}s")
        print(f"  Success rate: {avg_success:.1%}")
    
    return performance_metrics

def generate_accuracy_report(ml_accuracies, consistency_results, llm_quality, performance_metrics):
    """Generate comprehensive accuracy report"""
    print("\n" + "=" * 60)
    print("📊 FINANCEBOT ACCURACY REPORT")
    print("=" * 60)
    
    # ML Model Summary
    print("\n🤖 ML Model Accuracy Summary:")
    print("-" * 40)
    if ml_accuracies:
        for model_name, accuracy in ml_accuracies.items():
            status = "✅ Excellent" if accuracy > 0.7 else "⚠️ Good" if accuracy > 0.6 else "❌ Needs Improvement"
            print(f"{model_name:20} | {accuracy:7.1%} | {status}")
        
        best_model = max(ml_accuracies.items(), key=lambda x: x[1])
        print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]:.1%})")
    else:
        print("No ML accuracy data available")
    
    # Agent Consistency Summary
    print("\n🔍 Agent Consistency Summary:")
    print("-" * 40)
    if consistency_results:
        all_consistency_scores = []
        for ticker, scores in consistency_results.items():
            for agent, score in scores.items():
                all_consistency_scores.append(score)
        
        if all_consistency_scores:
            avg_consistency = np.mean(all_consistency_scores)
            status = "✅ Highly Consistent" if avg_consistency > 0.8 else "⚠️ Moderately Consistent" if avg_consistency > 0.6 else "❌ Inconsistent"
            print(f"Average Consistency: {avg_consistency:.1%} | {status}")
    
    # LLM Quality Summary
    print("\n🧠 LLM Quality Summary:")
    print("-" * 40)
    if llm_quality:
        llm_available = any(metrics.get('local_llm_used', False) for metrics in llm_quality.values())
        if llm_available:
            print("✅ Local LLM integration working")
        else:
            print("⚠️ Using LLM simulation mode")
        
        avg_confidence = np.mean([metrics.get('confidence', 0) for metrics in llm_quality.values()])
        print(f"Average LLM confidence: {avg_confidence:.1%}")
    
    # Performance Summary
    print("\n⚡ System Performance Summary:")
    print("-" * 40)
    if performance_metrics['execution_times']:
        avg_time = np.mean(performance_metrics['execution_times'])
        avg_success = np.mean(performance_metrics['success_rates'])
        
        time_status = "✅ Fast" if avg_time < 15 else "⚠️ Acceptable" if avg_time < 30 else "❌ Slow"
        success_status = "✅ Excellent" if avg_success > 0.9 else "⚠️ Good" if avg_success > 0.7 else "❌ Poor"
        
        print(f"Average execution time: {avg_time:.2f}s | {time_status}")
        print(f"Average success rate: {avg_success:.1%} | {success_status}")
    
    # Overall System Rating
    print("\n🏆 OVERALL SYSTEM RATING:")
    print("-" * 40)
    
    ratings = []
    if ml_accuracies and max(ml_accuracies.values()) > 0.6:
        ratings.append("ML Models: Good")
    if consistency_results:
        ratings.append("Agent Consistency: Functional")
    if llm_quality:
        ratings.append("LLM Integration: Working")
    if performance_metrics['execution_times'] and np.mean(performance_metrics['success_rates']) > 0.7:
        ratings.append("System Performance: Good")
    
    if len(ratings) >= 3:
        print("🎉 PRODUCTION READY")
        print("The FinanceBot system is ready for real-world use!")
    elif len(ratings) >= 2:
        print("⚠️ MOSTLY FUNCTIONAL")
        print("The system works but may need some improvements.")
    else:
        print("❌ NEEDS WORK")
        print("The system requires significant improvements.")
    
    print(f"\nWorking components: {', '.join(ratings)}")

def main():
    """Main evaluation function"""
    print("🧪 FinanceBot Accuracy Evaluation")
    print("=" * 60)
    print("This will evaluate all system components...")
    
    try:
        # Evaluate each component
        ml_accuracies = evaluate_ml_models()
        consistency_results = evaluate_agent_consistency()
        llm_quality = evaluate_llm_quality()
        performance_metrics = evaluate_system_performance()
        
        # Generate comprehensive report
        generate_accuracy_report(ml_accuracies, consistency_results, llm_quality, performance_metrics)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
