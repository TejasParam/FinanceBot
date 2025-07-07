"""
Agent Coordinator for managing and orchestrating all financial analysis agents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import BaseAgent
from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .ml_agent import MLPredictionAgent
from .regime_agent import RegimeDetectionAgent
from .fundamental_agent import FundamentalAnalysisAgent
from .llm_agent import LLMExplanationAgent

class AgentCoordinator:
    """
    Coordinates multiple specialized agents for comprehensive stock analysis.
    
    This class orchestrates the execution of various agents, aggregates their results,
    and provides a unified interface for multi-agent financial analysis.
    """
    
    def __init__(self, enable_ml: bool = True, enable_llm: bool = True, 
                 parallel_execution: bool = True):
        self.logger = logging.getLogger("AgentCoordinator")
        self.parallel_execution = parallel_execution
        
        # Initialize all agents
        self.agents = {
            'TechnicalAnalysis': TechnicalAnalysisAgent(),
            'SentimentAnalysis': SentimentAnalysisAgent(),
            'RegimeDetection': RegimeDetectionAgent(),
            'FundamentalAnalysis': FundamentalAnalysisAgent(),
        }
        
        # Optional agents
        if enable_ml:
            self.agents['MLPrediction'] = MLPredictionAgent()
        
        if enable_llm:
            self.agents['LLMExplanation'] = LLMExplanationAgent()
        
        self.execution_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'agent_performance': {name: {'success': 0, 'failure': 0} 
                                for name in self.agents.keys()}
        }
        
    def analyze_stock(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive multi-agent analysis of a stock.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters for specific agents
            
        Returns:
            Dictionary containing results from all agents plus aggregated analysis
        """
        self.execution_stats['total_analyses'] += 1
        
        start_time = time.time()
        
        try:
            # Execute all agents (except LLM first)
            core_agents = {name: agent for name, agent in self.agents.items() 
                          if name != 'LLMExplanation'}
            
            if self.parallel_execution:
                agent_results = self._execute_agents_parallel(core_agents, ticker, **kwargs)
            else:
                agent_results = self._execute_agents_sequential(core_agents, ticker, **kwargs)
            
            # Execute LLM agent with results from other agents
            if 'LLMExplanation' in self.agents:
                llm_result = self.agents['LLMExplanation'].safe_analyze(
                    ticker, agent_results=agent_results, **kwargs
                )
                agent_results['LLMExplanation'] = llm_result
            
            # Aggregate results
            aggregated = self._aggregate_results(agent_results, ticker)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update statistics
            self.execution_stats['successful_analyses'] += 1
            
            return {
                'ticker': ticker,
                'timestamp': time.time(),
                'execution_time': execution_time,
                'agent_results': agent_results,
                'aggregated_analysis': aggregated,
                'agents_executed': len(agent_results),
                'agents_successful': len([r for r in agent_results.values() 
                                        if isinstance(r, dict) and 'error' not in r])
            }
            
        except Exception as e:
            self.execution_stats['failed_analyses'] += 1
            self.logger.error(f"Multi-agent analysis failed for {ticker}: {e}")
            
            return {
                'ticker': ticker,
                'timestamp': time.time(),
                'execution_time': time.time() - start_time,
                'error': f'Multi-agent analysis failed: {str(e)}',
                'agent_results': {},
                'aggregated_analysis': {
                    'overall_score': 0.0,
                    'overall_confidence': 0.0,
                    'recommendation': 'HOLD',
                    'reasoning': f'Analysis failed: {str(e)[:100]}'
                }
            }
    
    def _execute_agents_parallel(self, agents: Dict[str, BaseAgent], 
                               ticker: str, **kwargs) -> Dict[str, Any]:
        """Execute agents in parallel for faster analysis"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(agents), 4)) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(agent.safe_analyze, ticker, **kwargs): name
                for name, agent in agents.items()
                if agent.is_enabled()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per agent
                    results[agent_name] = result
                    
                    # Update performance stats
                    if 'error' in result:
                        self.execution_stats['agent_performance'][agent_name]['failure'] += 1
                    else:
                        self.execution_stats['agent_performance'][agent_name]['success'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    results[agent_name] = {
                        'error': f'Agent execution failed: {str(e)}',
                        'score': 0.0,
                        'confidence': 0.0,
                        'reasoning': f'Agent {agent_name} execution error'
                    }
                    self.execution_stats['agent_performance'][agent_name]['failure'] += 1
        
        return results
    
    def _execute_agents_sequential(self, agents: Dict[str, BaseAgent], 
                                 ticker: str, **kwargs) -> Dict[str, Any]:
        """Execute agents sequentially (safer but slower)"""
        results = {}
        
        for name, agent in agents.items():
            if not agent.is_enabled():
                continue
                
            try:
                result = agent.safe_analyze(ticker, **kwargs)
                results[name] = result
                
                # Update performance stats
                if 'error' in result:
                    self.execution_stats['agent_performance'][name]['failure'] += 1
                else:
                    self.execution_stats['agent_performance'][name]['success'] += 1
                    
            except Exception as e:
                self.logger.error(f"Agent {name} failed: {e}")
                results[name] = {
                    'error': f'Agent execution failed: {str(e)}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': f'Agent {name} execution error'
                }
                self.execution_stats['agent_performance'][name]['failure'] += 1
        
        return results
    
    def _aggregate_results(self, agent_results: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Aggregate results from all agents into unified analysis"""
        
        # Extract scores and confidences
        valid_results = {name: result for name, result in agent_results.items()
                        if isinstance(result, dict) and 'error' not in result}
        
        if not valid_results:
            return {
                'overall_score': 0.0,
                'overall_confidence': 0.0,
                'recommendation': 'HOLD',
                'reasoning': 'No valid agent results available',
                'agent_consensus': 'unknown'
            }
        
        # Calculate weighted scores
        weighted_scores = []
        total_confidence = 0.0
        
        # Agent weights (can be customized based on performance)
        agent_weights = {
            'TechnicalAnalysis': 0.25,
            'FundamentalAnalysis': 0.30,
            'MLPrediction': 0.25,
            'SentimentAnalysis': 0.15,
            'RegimeDetection': 0.15,
            'LLMExplanation': 0.10  # Lower weight as it's synthetic
        }
        
        for agent_name, result in valid_results.items():
            score = result.get('score', 0.0)
            confidence = result.get('confidence', 0.5)
            weight = agent_weights.get(agent_name, 0.1)
            
            # Weight by both agent importance and confidence
            effective_weight = weight * confidence
            weighted_scores.append(score * effective_weight)
            total_confidence += effective_weight
        
        # Calculate overall metrics
        overall_score = sum(weighted_scores) / total_confidence if total_confidence > 0 else 0.0
        overall_confidence = total_confidence / sum(agent_weights.get(name, 0.1) 
                                                  for name in valid_results.keys())
        
        # Determine recommendation
        recommendation = self._score_to_recommendation(overall_score, overall_confidence)
        
        # Analyze agent consensus
        consensus_analysis = self._analyze_consensus(valid_results)
        
        # Generate combined reasoning
        reasoning = self._generate_combined_reasoning(
            overall_score, overall_confidence, valid_results, consensus_analysis
        )
        
        return {
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'agent_consensus': consensus_analysis,
            'component_scores': {name: result.get('score', 0.0) 
                               for name, result in valid_results.items()},
            'component_confidences': {name: result.get('confidence', 0.0) 
                                    for name, result in valid_results.items()},
            'agents_contributing': len(valid_results)
        }
    
    def _score_to_recommendation(self, score: float, confidence: float) -> str:
        """Convert aggregated score to recommendation"""
        # Adjust thresholds based on confidence
        high_threshold = 0.4 if confidence > 0.7 else 0.5
        low_threshold = -0.4 if confidence > 0.7 else -0.5
        
        if score > high_threshold:
            return 'STRONG BUY' if score > 0.6 else 'BUY'
        elif score < low_threshold:
            return 'STRONG SELL' if score < -0.6 else 'SELL'
        else:
            return 'HOLD'
    
    def _analyze_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus among agents"""
        scores = [result.get('score', 0.0) for result in results.values()]
        
        if not scores:
            return {'level': 'none', 'direction': 'unknown', 'strength': 0.0}
        
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        score_std = score_variance ** 0.5
        
        # Determine consensus level
        if score_std < 0.2:
            consensus_level = 'high'
        elif score_std < 0.4:
            consensus_level = 'medium'
        else:
            consensus_level = 'low'
        
        # Determine direction
        if avg_score > 0.2:
            direction = 'bullish'
        elif avg_score < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'level': consensus_level,
            'direction': direction,
            'strength': max(0, 1 - score_std),
            'average_score': avg_score,
            'score_std': score_std
        }
    
    def _generate_combined_reasoning(self, score: float, confidence: float, 
                                   results: Dict[str, Any], consensus: Dict[str, Any]) -> str:
        """Generate combined reasoning from all agent results"""
        reasoning_parts = []
        
        # Overall assessment
        if score > 0.3:
            reasoning_parts.append("Multi-agent analysis suggests bullish conditions.")
        elif score < -0.3:
            reasoning_parts.append("Multi-agent analysis suggests bearish conditions.")
        else:
            reasoning_parts.append("Multi-agent analysis shows mixed signals.")
        
        # Consensus information
        consensus_level = consensus.get('level', 'unknown')
        consensus_direction = consensus.get('direction', 'unknown')
        
        if consensus_level == 'high':
            reasoning_parts.append(f"Strong consensus among agents points {consensus_direction}.")
        elif consensus_level == 'medium':
            reasoning_parts.append(f"Moderate consensus leans {consensus_direction}.")
        else:
            reasoning_parts.append("Agents show conflicting signals.")
        
        # Key supporting/opposing factors
        supporting_agents = [name for name, result in results.items() 
                           if result.get('score', 0) > 0.2]
        opposing_agents = [name for name, result in results.items() 
                         if result.get('score', 0) < -0.2]
        
        if supporting_agents:
            agent_names = [name.replace('Analysis', '').replace('Detection', '') 
                          for name in supporting_agents[:2]]
            reasoning_parts.append(f"Supported by {', '.join(agent_names)} analysis.")
        
        if opposing_agents:
            agent_names = [name.replace('Analysis', '').replace('Detection', '') 
                          for name in opposing_agents[:2]]
            reasoning_parts.append(f"Opposed by {', '.join(agent_names)} analysis.")
        
        # Confidence qualifier
        if confidence < 0.5:
            reasoning_parts.append("Low confidence suggests waiting for clearer signals.")
        elif confidence > 0.8:
            reasoning_parts.append("High confidence supports taking action.")
        
        return " ".join(reasoning_parts)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            agent_info = agent.get_info()
            perf = self.execution_stats['agent_performance'][name]
            agent_info.update({
                'success_count': perf['success'],
                'failure_count': perf['failure'],
                'success_rate': perf['success'] / (perf['success'] + perf['failure']) 
                              if (perf['success'] + perf['failure']) > 0 else 0.0
            })
            status[name] = agent_info
        
        return status
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def enable_agent(self, agent_name: str):
        """Enable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enable()
    
    def disable_agent(self, agent_name: str):
        """Disable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].disable()
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys())
