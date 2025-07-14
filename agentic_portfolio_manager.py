"""
Agentic Portfolio Manager - Enhanced version using specialized AI agents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

# Import enhanced components
from portfolio_optimizer import EnhancedPortfolioOptimizer
from risk_manager_enhanced import EnhancedRiskManager
from backtest_engine_enhanced import EnhancedBacktestEngine

# Import agentic system
from agents import AgentCoordinator

class AgenticPortfolioManager:
    """
    Enhanced Portfolio Manager using specialized AI agents.
    
    This manager orchestrates multiple AI agents to provide comprehensive
    stock analysis and portfolio management capabilities.
    """
    
    def __init__(self, use_ml: bool = True, use_llm: bool = True, 
                 parallel_execution: bool = True):
        self.logger = logging.getLogger("AgenticPortfolioManager")
        
        # Initialize agent coordinator
        self.agent_coordinator = AgentCoordinator(
            enable_ml=use_ml,
            enable_llm=use_llm,
            parallel_execution=parallel_execution
        )
        
        # Initialize enhanced components
        self.portfolio_optimizer = EnhancedPortfolioOptimizer()
        self.risk_manager = EnhancedRiskManager()
        self.backtest_engine = EnhancedBacktestEngine()
        
        # Store configuration for compatibility
        self.use_ml = use_ml
        self.use_llm = use_llm
        self.parallel_execution = parallel_execution
        
        self.analysis_history = {}
        
    def analyze_stock(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive multi-agent analysis of a stock.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            # Get multi-agent analysis
            agentic_result = self.agent_coordinator.analyze_stock(ticker, **kwargs)
            
            # Store in history
            self.analysis_history[ticker] = {
                'timestamp': datetime.now(),
                'result': agentic_result
            }
            
            # Format for compatibility with existing interfaces
            formatted_result = self._format_agentic_result(agentic_result)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Agentic analysis failed for {ticker}: {e}")
            return {
                'error': f'Agentic analysis failed: {str(e)}',
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'Analysis error: {str(e)[:100]}',
                'composite_score': 0.0
            }
    
    def _format_agentic_result(self, agentic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format agentic result for compatibility with existing interfaces"""
        if 'error' in agentic_result:
            return {
                'error': agentic_result['error'],
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Analysis failed',
                'composite_score': 0.0
            }
        
        aggregated = agentic_result.get('aggregated_analysis', {})
        agent_results = agentic_result.get('agent_results', {})
        
        # Extract main metrics
        overall_score = aggregated.get('overall_score', 0.0)
        overall_confidence = aggregated.get('overall_confidence', 0.0)
        recommendation = aggregated.get('recommendation', 'HOLD')
        reasoning = aggregated.get('reasoning', 'Multi-agent analysis completed')
        
        # Get LLM explanation if available
        llm_result = agent_results.get('LLMExplanation', {})
        if 'error' not in llm_result:
            detailed_analysis = llm_result.get('detailed_analysis', {})
            if detailed_analysis:
                reasoning = detailed_analysis.get('main_explanation', reasoning)
        
        # Build component scores for compatibility
        component_scores = {}
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Map agent names to legacy component names
                component_name = self._map_agent_to_component(agent_name)
                component_scores[component_name] = result.get('score', 0.0)
        
        return {
            'recommendation': recommendation,
            'confidence': overall_confidence,
            'reasoning': reasoning,
            'composite_score': overall_score,
            'component_scores': component_scores,
            'risk_assessment': self._generate_risk_assessment(aggregated, agent_results),
            'agentic_analysis': agentic_result,  # Include full agentic result
            'agent_consensus': aggregated.get('agent_consensus', {}),
            'execution_time': agentic_result.get('execution_time', 0.0),
            'agents_used': agentic_result.get('agents_successful', 0)
        }
    
    def _map_agent_to_component(self, agent_name: str) -> str:
        """Map agent names to legacy component names"""
        mapping = {
            'TechnicalAnalysis': 'technical',
            'FundamentalAnalysis': 'fundamental', 
            'SentimentAnalysis': 'sentiment',
            'MLPrediction': 'ml',
            'RegimeDetection': 'regime',
            'LLMExplanation': 'llm'
        }
        return mapping.get(agent_name, agent_name.lower())
    
    def _generate_risk_assessment(self, aggregated: Dict[str, Any], 
                                agent_results: Dict[str, Any]) -> str:
        """Generate risk assessment from agent results"""
        risk_factors = []
        
        # Check for high volatility regime
        regime_result = agent_results.get('RegimeDetection', {})
        if 'error' not in regime_result:
            regime = regime_result.get('regime', '')
            if 'high_volatility' in regime or 'bear' in regime:
                risk_factors.append("High volatility market conditions")
        
        # Check for negative sentiment
        sentiment_result = agent_results.get('SentimentAnalysis', {})
        if 'error' not in sentiment_result:
            sentiment_score = sentiment_result.get('score', 0.0)
            if sentiment_score < -0.3:
                risk_factors.append("Negative market sentiment")
        
        # Check for poor fundamentals
        fundamental_result = agent_results.get('FundamentalAnalysis', {})
        if 'error' not in fundamental_result:
            fundamental_score = fundamental_result.get('score', 0.0)
            if fundamental_score < -0.3:
                risk_factors.append("Weak fundamental metrics")
        
        # Check consensus
        consensus = aggregated.get('agent_consensus', {})
        if consensus.get('level') == 'low':
            risk_factors.append("Low consensus among analysis methods")
        
        if risk_factors:
            return f"Risk factors identified: {', '.join(risk_factors)}"
        else:
            return "Low risk profile with supportive conditions"
    
    def get_detailed_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get detailed analysis breakdown by agent"""
        if ticker not in self.analysis_history:
            return {'error': f'No analysis history for {ticker}'}
        
        agentic_result = self.analysis_history[ticker]['result']
        agent_results = agentic_result.get('agent_results', {})
        
        detailed = {
            'ticker': ticker,
            'timestamp': self.analysis_history[ticker]['timestamp'],
            'execution_time': agentic_result.get('execution_time', 0.0),
            'agents_used': list(agent_results.keys()),
            'by_agent': {}
        }
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict):
                detailed['by_agent'][agent_name] = {
                    'score': result.get('score', 0.0),
                    'confidence': result.get('confidence', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'success': 'error' not in result
                }
                
                # Add agent-specific details
                if agent_name == 'TechnicalAnalysis':
                    detailed['by_agent'][agent_name]['indicators'] = result.get('indicators', {})
                elif agent_name == 'FundamentalAnalysis':
                    detailed['by_agent'][agent_name]['breakdown'] = result.get('breakdown', {})
                elif agent_name == 'SentimentAnalysis':
                    detailed['by_agent'][agent_name]['sources'] = result.get('sources_analyzed', 0)
                elif agent_name == 'MLPrediction':
                    detailed['by_agent'][agent_name]['model_accuracies'] = result.get('model_accuracies', {})
                elif agent_name == 'RegimeDetection':
                    detailed['by_agent'][agent_name]['regime'] = result.get('regime', 'unknown')
                elif agent_name == 'LLMExplanation':
                    detailed['by_agent'][agent_name]['summary'] = result.get('summary', '')
        
        return detailed
    
    def compare_analyses(self, tickers: List[str]) -> pd.DataFrame:
        """Compare analyses across multiple tickers"""
        comparison_data = []
        
        for ticker in tickers:
            if ticker in self.analysis_history:
                result = self.analysis_history[ticker]['result']
                aggregated = result.get('aggregated_analysis', {})
                agent_results = result.get('agent_results', {})
                
                row = {
                    'Ticker': ticker,
                    'Overall_Score': aggregated.get('overall_score', 0.0),
                    'Confidence': aggregated.get('overall_confidence', 0.0),
                    'Recommendation': aggregated.get('recommendation', 'HOLD'),
                    'Consensus': aggregated.get('agent_consensus', {}).get('level', 'unknown'),
                    'Agents_Used': result.get('agents_successful', 0)
                }
                
                # Add individual agent scores
                for agent_name, agent_result in agent_results.items():
                    if isinstance(agent_result, dict) and 'error' not in agent_result:
                        row[f'{agent_name}_Score'] = agent_result.get('score', 0.0)
                        row[f'{agent_name}_Confidence'] = agent_result.get('confidence', 0.0)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all agents"""
        return self.agent_coordinator.get_execution_stats()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return self.agent_coordinator.get_agent_status()
    
    def enable_agent(self, agent_name: str):
        """Enable a specific agent"""
        self.agent_coordinator.enable_agent(agent_name)
    
    def disable_agent(self, agent_name: str):
        """Disable a specific agent"""
        self.agent_coordinator.disable_agent(agent_name)
    
    # Delegate methods to legacy manager for compatibility
    def train_ml_models(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
        """Train ML models (delegates to legacy manager)"""
        return self.legacy_manager.train_ml_models(ticker, period)
    
    def backtest_strategy(self, ticker: str, start_date: str = None, 
                         end_date: str = None) -> Dict[str, Any]:
        """Backtest strategy (delegates to legacy manager)"""
        return self.legacy_manager.backtest_strategy(ticker, start_date, end_date)
    
    def optimize_strategy(self, ticker: str) -> Dict[str, Any]:
        """Optimize strategy (delegates to legacy manager)"""
        return self.legacy_manager.optimize_strategy(ticker)
    
    def get_market_regime_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get market regime analysis from agents"""
        # Use agent-based regime analysis instead of legacy
        result = self.agent_coordinator.analyze_stock(ticker)
        if 'error' in result:
            return result
        
        regime_result = result.get('agent_results', {}).get('RegimeDetection', {})
        if 'error' in regime_result:
            return regime_result
        
        return {
            'regime': regime_result.get('regime', 'unknown'),
            'confidence': regime_result.get('confidence', 0.0),
            'recommendation': self._regime_to_recommendation(regime_result.get('regime', 'unknown')),
            'metrics': regime_result.get('metrics', {}),
            'reasoning': regime_result.get('reasoning', '')
        }
    
    def _regime_to_recommendation(self, regime: str) -> str:
        """Convert regime to recommendation"""
        if 'bull' in regime:
            return "Favor long positions and growth strategies"
        elif 'bear' in regime:
            return "Consider defensive positions and risk management"
        elif 'high_volatility' in regime:
            return "Use smaller position sizes and tight stops"
        else:
            return "Maintain balanced approach with careful stock selection"
    
    def batch_analyze(self, tickers: List[str], **kwargs) -> Dict[str, Any]:
        """Analyze multiple tickers using agentic system"""
        results = {}
        
        for ticker in tickers:
            try:
                result = self.analyze_stock(ticker, **kwargs)
                results[ticker] = result
            except Exception as e:
                results[ticker] = {
                    'error': f'Analysis failed: {str(e)}',
                    'recommendation': 'HOLD',
                    'confidence': 0.0
                }
        
        return results
