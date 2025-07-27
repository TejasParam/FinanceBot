"""
Performance Tracker for Agent Accuracy Improvement
Tracks individual agent performance and updates weights dynamically
"""

import json
import os
from typing import Dict, Any
from datetime import datetime
import numpy as np

class PerformanceTracker:
    """Tracks agent performance and calculates dynamic weights"""
    
    def __init__(self, save_path: str = "agent_performance.json"):
        self.save_path = save_path
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load existing performance data or initialize"""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                return json.load(f)
        
        # Initialize with default data
        return {
            'agents': {
                'TechnicalAnalysis': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'MarketTiming': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'PatternRecognition': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'IntermarketAnalysis': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'FundamentalAnalysis': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'MLPrediction': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'VolatilityAnalysis': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'SentimentAnalysis': {'correct': 0, 'total': 0, 'recent_accuracy': []},
                'RegimeDetection': {'correct': 0, 'total': 0, 'recent_accuracy': []},
            },
            'last_updated': str(datetime.now())
        }
    
    def update_agent_performance(self, agent_name: str, prediction: Dict[str, Any], actual_outcome: bool):
        """Update agent performance based on prediction outcome"""
        if agent_name not in self.performance_data['agents']:
            self.performance_data['agents'][agent_name] = {
                'correct': 0, 'total': 0, 'recent_accuracy': []
            }
        
        agent_data = self.performance_data['agents'][agent_name]
        agent_data['total'] += 1
        
        if actual_outcome:
            agent_data['correct'] += 1
        
        # Track recent accuracy (last 50 predictions)
        agent_data['recent_accuracy'].append(1 if actual_outcome else 0)
        if len(agent_data['recent_accuracy']) > 50:
            agent_data['recent_accuracy'].pop(0)
        
        # Save updated data
        self._save_performance_data()
    
    def get_dynamic_weights(self, market_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate dynamic weights based on agent performance"""
        weights = {}
        
        for agent_name, data in self.performance_data['agents'].items():
            # Base weight
            base_weight = 1.0 / len(self.performance_data['agents'])
            
            # Performance adjustment
            if data['total'] >= 20:  # Minimum samples
                # Use recent accuracy if available
                if data['recent_accuracy']:
                    accuracy = sum(data['recent_accuracy']) / len(data['recent_accuracy'])
                else:
                    accuracy = data['correct'] / data['total']
                
                # Weight = base * accuracy^2 (quadratic to emphasize better agents)
                performance_weight = accuracy ** 2
                
                # Confidence factor based on sample size
                confidence = min(1.0, data['total'] / 100)
                
                # Final weight
                weights[agent_name] = base_weight * performance_weight * confidence
            else:
                # Not enough data, use base weight
                weights[agent_name] = base_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply market context adjustments if provided
        if market_context:
            weights = self._adjust_weights_for_market(weights, market_context)
        
        return weights
    
    def _adjust_weights_for_market(self, weights: Dict[str, float], market_context: Dict[str, Any]) -> Dict[str, float]:
        """Adjust weights based on current market conditions"""
        trend = market_context.get('trend', 'unknown')
        volatility = market_context.get('volatility', 0.015)
        
        # In trending markets, emphasize momentum and timing
        if trend in ['strong_up', 'strong_down']:
            weights['MarketTiming'] *= 1.5
            weights['PatternRecognition'] *= 1.3
            weights['MLPrediction'] *= 1.2
            
            # Reduce fundamental weight in strong trends
            weights['FundamentalAnalysis'] *= 0.7
        
        # In high volatility, emphasize volatility analysis
        if volatility > 0.025:  # 2.5% daily vol
            weights['VolatilityAnalysis'] *= 2.0
            weights['RegimeDetection'] *= 1.5
            
            # Reduce pattern recognition in chaos
            weights['PatternRecognition'] *= 0.5
        
        # Renormalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific agent"""
        if agent_name not in self.performance_data['agents']:
            return {'error': 'Agent not found'}
        
        data = self.performance_data['agents'][agent_name]
        
        stats = {
            'total_predictions': data['total'],
            'correct_predictions': data['correct'],
            'overall_accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'recent_accuracy': sum(data['recent_accuracy']) / len(data['recent_accuracy']) 
                               if data['recent_accuracy'] else 0,
            'sample_size': len(data['recent_accuracy'])
        }
        
        return stats
    
    def get_all_agent_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all agents"""
        stats = {}
        for agent_name in self.performance_data['agents']:
            stats[agent_name] = self.get_agent_stats(agent_name)
        return stats
    
    def _save_performance_data(self):
        """Save performance data to file"""
        self.performance_data['last_updated'] = str(datetime.now())
        with open(self.save_path, 'w') as f:
            json.dump(self.performance_data, f, indent=2)
    
    def reset_performance_data(self):
        """Reset all performance data"""
        self.performance_data = self._load_performance_data()
        # Reset to initial state
        for agent_name in self.performance_data['agents']:
            self.performance_data['agents'][agent_name] = {
                'correct': 0, 'total': 0, 'recent_accuracy': []
            }
        self._save_performance_data()