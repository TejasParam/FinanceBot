"""
Deep Reinforcement Learning Strategy Selector
Uses PPO algorithm to dynamically select optimal agent weights
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for compatibility
    class nn:
        class Module:
            pass
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class MultiheadAttention:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            pass
        class Tanh:
            pass
        class Softmax:
            def __init__(self, *args, **kwargs):
                pass
    class torch:
        @staticmethod
        def FloatTensor(x):
            return x
        @staticmethod
        def LongTensor(x):
            return x
        @staticmethod
        def stack(x):
            return x
        @staticmethod
        def no_grad():
            return lambda: None
        class distributions:
            class Categorical:
                def __init__(self, *args):
                    pass
                def sample(self):
                    return 0
                def log_prob(self, x):
                    return 0
        
from collections import deque
from typing import Dict, List, Tuple, Any
import logging

class PolicyNetwork(nn.Module):
    """Neural network for policy (agent weight selection)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Deeper network for complex market patterns
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Attention mechanism for agent importance
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4)
        
        # Policy head (outputs agent weights)
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Value head (estimates state value)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Feature extraction
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Self-attention to focus on important features
        x_reshaped = x.unsqueeze(0)  # Add sequence dimension
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.squeeze(0)
        
        # Get policy (agent weights) and value
        policy = self.softmax(self.policy_head(x))
        value = self.value_head(x)
        
        return policy, value

class DRLStrategySelector:
    """
    Deep Reinforcement Learning for dynamic strategy/agent selection
    Learns optimal agent weights based on market conditions
    """
    
    def __init__(self, agents: List[str], learning_rate: float = 3e-4):
        self.agents = agents
        self.n_agents = len(agents)
        
        # Market features + agent recent performance
        self.state_dim = 50 + self.n_agents * 5  # 50 market features + 5 perf metrics per agent
        self.action_dim = self.n_agents  # Weight for each agent
        
        # Initialize policy network
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.gamma = 0.99  # Discount factor
        self.lambda_gae = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.entropy_coef = 0.01  # Entropy bonus
        self.value_loss_coef = 0.5
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.agent_performance = {agent: deque(maxlen=1000) for agent in agents}
        
        # Market state encoder
        self.market_features_buffer = deque(maxlen=100)
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        self.logger = logging.getLogger(__name__)
        
    def get_market_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract relevant market features for state representation"""
        
        features = []
        
        # 1. Price features (10 features)
        prices = market_data.get('prices', [])
        if len(prices) > 0:
            price_array = np.array(prices[-100:])  # Last 100 prices
            features.extend([
                np.mean(price_array),
                np.std(price_array),
                np.min(price_array),
                np.max(price_array),
                (price_array[-1] - price_array[0]) / price_array[0],  # Return
                np.mean(np.abs(np.diff(price_array))),  # Avg absolute change
                len(price_array[price_array > np.mean(price_array)]) / len(price_array),  # % above mean
                np.percentile(price_array, 25),
                np.percentile(price_array, 75),
                price_array[-1] / np.mean(price_array)  # Current vs mean
            ])
        else:
            features.extend([0] * 10)
        
        # 2. Volume features (5 features)
        volumes = market_data.get('volumes', [])
        if len(volumes) > 0:
            vol_array = np.array(volumes[-100:])
            features.extend([
                np.mean(vol_array),
                np.std(vol_array),
                vol_array[-1] / np.mean(vol_array),  # Current vs avg
                np.max(vol_array) / np.mean(vol_array),  # Max spike
                np.sum(vol_array[-10:]) / np.sum(vol_array)  # Recent concentration
            ])
        else:
            features.extend([0] * 5)
        
        # 3. Volatility features (5 features)
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            features.extend([
                np.std(returns),  # Historical volatility
                np.std(returns[-20:]) if len(returns) > 20 else 0,  # Recent vol
                np.std(returns[-5:]) if len(returns) > 5 else 0,  # Very recent vol
                np.percentile(np.abs(returns), 95) if len(returns) > 0 else 0,  # Tail risk
                len(returns[np.abs(returns) > 2*np.std(returns)]) / max(1, len(returns))  # Outlier frequency
            ])
        else:
            features.extend([0] * 5)
        
        # 4. Microstructure features (5 features)
        features.extend([
            market_data.get('spread', 0.0001),
            market_data.get('bid_ask_imbalance', 0),
            market_data.get('trade_intensity', 0),
            market_data.get('order_flow_imbalance', 0),
            market_data.get('tick_frequency', 0)
        ])
        
        # 5. Technical indicators (10 features)
        if len(prices) > 50:
            price_array = np.array(prices)
            # RSI approximation
            gains = np.maximum(np.diff(price_array), 0)
            losses = np.abs(np.minimum(np.diff(price_array), 0))
            avg_gain = np.mean(gains[-14:]) if len(gains) > 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 14 else 0
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
            
            features.extend([
                rsi / 100,
                (price_array[-1] - np.mean(price_array[-20:])) / np.std(price_array[-20:] + 1e-10),  # Z-score
                (np.mean(price_array[-5:]) - np.mean(price_array[-20:])) / np.mean(price_array[-20:]),  # MA crossover
                np.corrcoef(range(len(price_array[-20:])), price_array[-20:])[0, 1],  # Trend strength
                1 if price_array[-1] > np.mean(price_array[-50:]) else 0,  # Above MA50
                1 if price_array[-1] > np.max(price_array[-20:-1]) else 0,  # New high
                1 if price_array[-1] < np.min(price_array[-20:-1]) else 0,  # New low
                np.sum(price_array > price_array[-1]) / len(price_array),  # Percentile rank
                (np.max(price_array[-20:]) - np.min(price_array[-20:])) / np.mean(price_array[-20:]),  # Range
                0  # Placeholder
            ])
        else:
            features.extend([0] * 10)
        
        # 6. Market regime features (5 features)
        features.extend([
            market_data.get('vix', 20) / 100,  # Normalized VIX
            market_data.get('market_cap_ratio', 1),  # Large vs small cap
            market_data.get('sector_dispersion', 0),
            market_data.get('correlation_index', 0),
            market_data.get('liquidity_score', 1)
        ])
        
        # 7. Time features (5 features)
        features.extend([
            market_data.get('hour_of_day', 12) / 24,
            market_data.get('day_of_week', 3) / 7,
            market_data.get('day_of_month', 15) / 31,
            market_data.get('is_earnings_season', 0),
            market_data.get('is_fed_day', 0)
        ])
        
        # 8. Sentiment features (5 features)
        features.extend([
            market_data.get('news_sentiment', 0),
            market_data.get('social_sentiment', 0),
            market_data.get('options_sentiment', 0),
            market_data.get('analyst_sentiment', 0),
            market_data.get('insider_sentiment', 0)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_agent_performance_features(self) -> np.ndarray:
        """Get recent performance metrics for each agent"""
        
        features = []
        
        for agent in self.agents:
            agent_perfs = list(self.agent_performance[agent])
            
            if len(agent_perfs) > 0:
                recent_perfs = agent_perfs[-100:]  # Last 100 predictions
                
                # Calculate performance metrics
                accuracy = np.mean([p['correct'] for p in recent_perfs])
                avg_confidence = np.mean([p['confidence'] for p in recent_perfs])
                profit = np.sum([p['profit'] for p in recent_perfs])
                consistency = 1 - np.std([p['correct'] for p in recent_perfs])
                recent_trend = np.mean([p['correct'] for p in recent_perfs[-10:]]) - accuracy
                
                features.extend([accuracy, avg_confidence, profit / 100, consistency, recent_trend])
            else:
                features.extend([0.5, 0.5, 0, 0.5, 0])  # Default neutral values
        
        return np.array(features, dtype=np.float32)
    
    def get_state(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """Combine market state and agent performance into state vector"""
        
        market_features = self.get_market_state(market_data)
        agent_features = self.get_agent_performance_features()
        
        state = np.concatenate([market_features, agent_features])
        
        # Normalize state
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        
        return torch.FloatTensor(state)
    
    def select_action(self, state: torch.Tensor, explore: bool = True) -> Tuple[np.ndarray, float, float]:
        """Select agent weights using the policy network"""
        
        with torch.no_grad():
            policy, value = self.policy_net(state)
            
        # Add exploration noise if training
        if explore and self.exploration_noise > self.min_exploration:
            noise = torch.randn_like(policy) * self.exploration_noise
            policy = torch.softmax(policy + noise, dim=-1)
            self.exploration_noise *= self.exploration_decay
        
        # Sample action (agent weights)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Convert to weight vector
        weights = policy.numpy()
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        return weights, log_prob.item(), value.item()
    
    def update_agent_performance(self, agent: str, prediction_correct: bool, 
                               confidence: float, profit: float):
        """Update agent performance tracking"""
        
        self.agent_performance[agent].append({
            'correct': 1 if prediction_correct else 0,
            'confidence': confidence,
            'profit': profit
        })
    
    def store_transition(self, state: torch.Tensor, action: np.ndarray, 
                        reward: float, value: float, log_prob: float, done: bool):
        """Store transition for training"""
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        
        rewards = self.rewards + [next_value]
        values = self.values + [next_value]
        
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            advantage = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(self.values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, next_value: float = 0):
        """Perform PPO training step"""
        
        if len(self.states) == 0:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.stack(self.states)
        old_log_probs = torch.FloatTensor(self.log_probs)
        actions = torch.LongTensor(self.actions)
        
        # Multiple epochs of optimization
        for _ in range(4):
            # Get current policy
            policies, values = self.policy_net(states)
            dist = torch.distributions.Categorical(policies)
            
            # Compute log probs for taken actions
            # Since we used the full policy as action, we need to compute differently
            entropy = dist.entropy().mean()
            
            # For continuous weights, we use MSE loss instead of discrete action loss
            new_log_probs = -torch.sum((policies - torch.FloatTensor(np.array(self.actions)))**2, dim=1)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_optimal_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Get optimal agent weights for current market conditions"""
        
        state = self.get_state(market_data)
        weights, _, _ = self.select_action(state, explore=False)
        
        # Create weight dictionary
        weight_dict = {}
        for i, agent in enumerate(self.agents):
            weight_dict[agent] = float(weights[i])
        
        # Apply minimum weight threshold
        min_weight = 0.02
        for agent in weight_dict:
            if weight_dict[agent] < min_weight:
                weight_dict[agent] = 0
        
        # Renormalize
        total_weight = sum(weight_dict.values())
        if total_weight > 0:
            for agent in weight_dict:
                weight_dict[agent] /= total_weight
        else:
            # Fallback to equal weights
            for agent in weight_dict:
                weight_dict[agent] = 1.0 / len(weight_dict)
        
        return weight_dict
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_noise': self.exploration_noise,
            'episode_rewards': list(self.episode_rewards)
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_noise = checkpoint['exploration_noise']
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)