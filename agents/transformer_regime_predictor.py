"""
Transformer-based Market Regime Predictor
Uses attention mechanisms to predict market regimes before they fully develop
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy implementations
    class nn:
        class Module:
            pass
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class TransformerEncoderLayer:
            def __init__(self, *args, **kwargs):
                pass
        class TransformerEncoder:
            def __init__(self, *args, **kwargs):
                pass
        class Sequential:
            def __init__(self, *args):
                pass
        class LayerNorm:
            def __init__(self, *args):
                pass
        class ReLU:
            pass
        class Sigmoid:
            pass
    class F:
        @staticmethod
        def softmax(x, dim=-1):
            return x
        @staticmethod
        def cross_entropy(x, y):
            return 0
    class torch:
        @staticmethod
        def zeros(*args):
            return np.zeros(args)
        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args)
        @staticmethod
        def exp(x):
            return np.exp(x)
        @staticmethod
        def sin(x):
            return np.sin(x)
        @staticmethod
        def cos(x):
            return np.cos(x)
        @staticmethod
        def FloatTensor(x):
            return x
        @staticmethod
        def LongTensor(x):
            return x
        @staticmethod
        def no_grad():
            class NoGrad:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return NoGrad()
        @staticmethod
        def save(obj, path):
            pass
        @staticmethod
        def load(path):
            return {}
        class optim:
            class AdamW:
                def __init__(self, *args, **kwargs):
                    pass
            class lr_scheduler:
                class CosineAnnealingWarmRestarts:
                    def __init__(self, *args, **kwargs):
                        pass
                    def step(self):
                        pass
                        
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from collections import deque
import logging

class PositionalEncoding(nn.Module):
    """Add positional encoding to time series data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MarketTransformer(nn.Module):
    """Transformer model for market regime prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 1024, n_regimes: int = 7):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Regime prediction heads
        self.regime_classifier = nn.Linear(d_model, n_regimes)
        self.regime_transition = nn.Linear(d_model, n_regimes * n_regimes)
        self.volatility_predictor = nn.Linear(d_model, 3)  # Low, medium, high
        self.trend_predictor = nn.Linear(d_model, 3)  # Down, neutral, up
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Input projection and normalization
        x = self.input_projection(x)
        x = self.layer_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        transformer_out = self.transformer(x, src_key_padding_mask=mask)
        
        # Use the last output for predictions
        if len(transformer_out.shape) == 3:
            final_output = transformer_out[:, -1, :]  # [batch, d_model]
        else:
            final_output = transformer_out
        
        # Predictions
        regime_logits = self.regime_classifier(final_output)
        transition_logits = self.regime_transition(final_output)
        volatility_logits = self.volatility_predictor(final_output)
        trend_logits = self.trend_predictor(final_output)
        confidence = self.confidence_estimator(final_output)
        
        return {
            'regime': F.softmax(regime_logits, dim=-1),
            'transition': F.softmax(transition_logits.view(-1, 7, 7), dim=-1),
            'volatility': F.softmax(volatility_logits, dim=-1),
            'trend': F.softmax(trend_logits, dim=-1),
            'confidence': confidence
        }

class TransformerRegimePredictor:
    """
    Transformer-based market regime prediction system
    Detects regime changes before they fully develop
    """
    
    # Define market regimes
    REGIMES = {
        0: 'BULL_QUIET',      # Low volatility uptrend
        1: 'BULL_VOLATILE',   # High volatility uptrend
        2: 'BEAR_QUIET',      # Low volatility downtrend
        3: 'BEAR_VOLATILE',   # High volatility downtrend (crash)
        4: 'RANGING',         # Sideways market
        5: 'TRANSITION',      # Regime change in progress
        6: 'EXTREME'          # Black swan events
    }
    
    def __init__(self, lookback_window: int = 100, prediction_horizon: int = 20):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Model parameters
        self.input_dim = 20  # Features per timestep
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 6
        
        # Initialize transformer model
        self.model = MarketTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            n_regimes=len(self.REGIMES)
        )
        
        # Training components
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        
        # Data buffers
        self.feature_buffer = deque(maxlen=lookback_window * 2)
        self.regime_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)
        
        # Current regime tracking
        self.current_regime = 4  # Start with RANGING
        self.regime_confidence = 0.5
        self.regime_duration = 0
        
        # Performance metrics
        self.correct_predictions = 0
        self.total_predictions = 0
        
        self.logger = logging.getLogger(__name__)
        
    def extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data for transformer input"""
        
        features = []
        
        # Price features
        price = market_data.get('price', 100)
        features.append(price / 100)  # Normalized price
        
        # Returns at multiple scales
        if 'returns' in market_data:
            returns = market_data['returns']
            features.extend([
                returns.get('1min', 0),
                returns.get('5min', 0),
                returns.get('15min', 0),
                returns.get('1hour', 0),
                returns.get('1day', 0)
            ])
        else:
            features.extend([0] * 5)
        
        # Volume features
        volume = market_data.get('volume', 1000000)
        avg_volume = market_data.get('avg_volume', 1000000)
        features.append(volume / avg_volume)
        
        # Volatility features
        features.append(market_data.get('realized_vol', 0.02) / 0.02)
        features.append(market_data.get('implied_vol', 0.02) / 0.02)
        
        # Market microstructure
        features.append(market_data.get('spread', 0.0001) / 0.0001)
        features.append(market_data.get('bid_ask_imbalance', 0))
        features.append(market_data.get('trade_imbalance', 0))
        
        # Technical indicators
        features.append(market_data.get('rsi', 50) / 100)
        features.append(market_data.get('macd_signal', 0))
        features.append((market_data.get('bb_position', 0) + 1) / 2)  # Position in Bollinger Bands
        
        # Market breadth
        features.append(market_data.get('advance_decline', 0))
        features.append(market_data.get('new_highs_lows', 0))
        
        # Sentiment
        features.append(market_data.get('vix', 20) / 50)
        features.append(market_data.get('put_call_ratio', 1))
        features.append(market_data.get('sentiment_score', 0))
        
        # Ensure we have exactly input_dim features
        if len(features) < self.input_dim:
            features.extend([0] * (self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]
        
        return np.array(features, dtype=np.float32)
    
    def prepare_sequence(self) -> Optional[torch.Tensor]:
        """Prepare sequence data for transformer"""
        
        if len(self.feature_buffer) < self.lookback_window:
            return None
        
        # Get last lookback_window features
        sequence = list(self.feature_buffer)[-self.lookback_window:]
        sequence_array = np.array(sequence)
        
        # Normalize sequence
        mean = np.mean(sequence_array, axis=0, keepdims=True)
        std = np.std(sequence_array, axis=0, keepdims=True) + 1e-8
        sequence_normalized = (sequence_array - mean) / std
        
        # Convert to tensor and add batch dimension
        return torch.FloatTensor(sequence_normalized).unsqueeze(0)
    
    def predict_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict current and future market regime"""
        
        # Extract and store features
        features = self.extract_features(market_data)
        self.feature_buffer.append(features)
        
        # Need enough data for prediction
        if len(self.feature_buffer) < self.lookback_window:
            return {
                'current_regime': 'RANGING',
                'confidence': 0.5,
                'regime_probs': {regime: 1/7 for regime in self.REGIMES.values()},
                'transition_prob': 0.1,
                'volatility_forecast': 'MEDIUM',
                'trend_forecast': 'NEUTRAL',
                'early_warning': False
            }
        
        # Prepare sequence
        sequence = self.prepare_sequence()
        if sequence is None:
            return self._default_prediction()
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Extract predictions
        regime_probs = predictions['regime'][0].numpy()
        transition_matrix = predictions['transition'][0].numpy()
        volatility_probs = predictions['volatility'][0].numpy()
        trend_probs = predictions['trend'][0].numpy()
        confidence = predictions['confidence'][0].item()
        
        # Determine current regime
        predicted_regime_idx = np.argmax(regime_probs)
        predicted_regime = self.REGIMES[predicted_regime_idx]
        
        # Check for regime transition
        current_transitions = transition_matrix[self.current_regime]
        transition_prob = 1 - current_transitions[self.current_regime]
        
        # Early warning system
        early_warning = False
        warning_signals = []
        
        # Check for volatility spike warning
        if volatility_probs[2] > 0.6:  # High volatility likely
            early_warning = True
            warning_signals.append('Volatility spike expected')
        
        # Check for trend reversal warning
        current_trend = ['DOWN', 'NEUTRAL', 'UP'][np.argmax(trend_probs)]
        if self.current_regime in [0, 1] and trend_probs[0] > 0.4:  # Bull to bear
            early_warning = True
            warning_signals.append('Potential trend reversal (bearish)')
        elif self.current_regime in [2, 3] and trend_probs[2] > 0.4:  # Bear to bull
            early_warning = True
            warning_signals.append('Potential trend reversal (bullish)')
        
        # Check for extreme event warning
        if regime_probs[6] > 0.2:  # EXTREME regime
            early_warning = True
            warning_signals.append('Black swan event possible')
        
        # Update regime if confident
        if confidence > 0.7 and predicted_regime_idx != self.current_regime:
            self.regime_history.append({
                'from': self.REGIMES[self.current_regime],
                'to': predicted_regime,
                'confidence': confidence,
                'timestamp': market_data.get('timestamp', 0)
            })
            self.current_regime = predicted_regime_idx
            self.regime_duration = 0
        else:
            self.regime_duration += 1
        
        # Create regime probability dict
        regime_prob_dict = {self.REGIMES[i]: float(regime_probs[i]) for i in range(len(self.REGIMES))}
        
        # Volatility and trend forecasts
        volatility_forecast = ['LOW', 'MEDIUM', 'HIGH'][np.argmax(volatility_probs)]
        trend_forecast = ['DOWN', 'NEUTRAL', 'UP'][np.argmax(trend_probs)]
        
        result = {
            'current_regime': predicted_regime,
            'confidence': float(confidence),
            'regime_probs': regime_prob_dict,
            'transition_prob': float(transition_prob),
            'next_regime_probs': {self.REGIMES[i]: float(current_transitions[i]) 
                                 for i in range(len(self.REGIMES))},
            'volatility_forecast': volatility_forecast,
            'volatility_probs': {
                'LOW': float(volatility_probs[0]),
                'MEDIUM': float(volatility_probs[1]),
                'HIGH': float(volatility_probs[2])
            },
            'trend_forecast': trend_forecast,
            'trend_probs': {
                'DOWN': float(trend_probs[0]),
                'NEUTRAL': float(trend_probs[1]),
                'UP': float(trend_probs[2])
            },
            'early_warning': early_warning,
            'warning_signals': warning_signals,
            'regime_duration': self.regime_duration,
            'prediction_horizon': self.prediction_horizon
        }
        
        # Store prediction for evaluation
        self.prediction_history.append({
            'prediction': result,
            'timestamp': market_data.get('timestamp', 0)
        })
        
        return result
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Default prediction when insufficient data"""
        return {
            'current_regime': self.REGIMES[self.current_regime],
            'confidence': 0.5,
            'regime_probs': {regime: 1/7 for regime in self.REGIMES.values()},
            'transition_prob': 0.1,
            'next_regime_probs': {regime: 1/7 for regime in self.REGIMES.values()},
            'volatility_forecast': 'MEDIUM',
            'volatility_probs': {'LOW': 0.33, 'MEDIUM': 0.34, 'HIGH': 0.33},
            'trend_forecast': 'NEUTRAL',
            'trend_probs': {'DOWN': 0.33, 'NEUTRAL': 0.34, 'UP': 0.33},
            'early_warning': False,
            'warning_signals': [],
            'regime_duration': self.regime_duration,
            'prediction_horizon': self.prediction_horizon
        }
    
    def get_trading_adjustments(self, regime_prediction: Dict[str, Any]) -> Dict[str, float]:
        """Get trading parameter adjustments based on regime prediction"""
        
        adjustments = {
            'position_size_mult': 1.0,
            'stop_loss_mult': 1.0,
            'take_profit_mult': 1.0,
            'holding_period_mult': 1.0,
            'confidence_threshold': 0.6,
            'max_leverage': 1.0
        }
        
        regime = regime_prediction['current_regime']
        confidence = regime_prediction['confidence']
        
        # Adjust based on regime
        if regime == 'BULL_QUIET':
            adjustments['position_size_mult'] = 1.2
            adjustments['holding_period_mult'] = 1.5
            adjustments['max_leverage'] = 2.0
        
        elif regime == 'BULL_VOLATILE':
            adjustments['position_size_mult'] = 0.8
            adjustments['stop_loss_mult'] = 1.5
            adjustments['confidence_threshold'] = 0.7
        
        elif regime == 'BEAR_QUIET':
            adjustments['position_size_mult'] = 0.6
            adjustments['take_profit_mult'] = 0.8
            adjustments['max_leverage'] = 0.5
        
        elif regime == 'BEAR_VOLATILE':
            adjustments['position_size_mult'] = 0.3
            adjustments['stop_loss_mult'] = 2.0
            adjustments['confidence_threshold'] = 0.8
            adjustments['max_leverage'] = 0.2
        
        elif regime == 'RANGING':
            adjustments['holding_period_mult'] = 0.5
            adjustments['take_profit_mult'] = 0.7
        
        elif regime == 'TRANSITION':
            adjustments['position_size_mult'] = 0.5
            adjustments['confidence_threshold'] = 0.75
        
        elif regime == 'EXTREME':
            adjustments['position_size_mult'] = 0.1
            adjustments['stop_loss_mult'] = 3.0
            adjustments['confidence_threshold'] = 0.9
            adjustments['max_leverage'] = 0.1
        
        # Further adjust based on early warnings
        if regime_prediction['early_warning']:
            adjustments['position_size_mult'] *= 0.5
            adjustments['confidence_threshold'] += 0.1
            adjustments['stop_loss_mult'] *= 1.5
        
        # Scale by prediction confidence
        confidence_factor = min(1.0, confidence / 0.8)
        for key in ['position_size_mult', 'max_leverage']:
            adjustments[key] = 1.0 + (adjustments[key] - 1.0) * confidence_factor
        
        return adjustments
    
    def train_on_history(self, historical_data: pd.DataFrame, regime_labels: pd.Series):
        """Train the transformer on historical data with regime labels"""
        
        self.model.train()
        
        # Prepare training data
        sequences = []
        labels = []
        
        for i in range(self.lookback_window, len(historical_data) - self.prediction_horizon):
            # Extract sequence
            sequence_data = historical_data.iloc[i-self.lookback_window:i]
            sequence_features = []
            
            for _, row in sequence_data.iterrows():
                features = self.extract_features(row.to_dict())
                sequence_features.append(features)
            
            sequences.append(sequence_features)
            
            # Get label (future regime)
            future_regime = regime_labels.iloc[i + self.prediction_horizon]
            labels.append(future_regime)
        
        # Convert to tensors
        X = torch.FloatTensor(sequences)
        y = torch.LongTensor(labels)
        
        # Training loop
        n_epochs = 50
        batch_size = 32
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(len(X))
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Calculate losses
                regime_loss = F.cross_entropy(predictions['regime'], batch_y)
                
                # Add auxiliary losses
                total_loss = regime_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
            
            # Learning rate scheduling
            self.scheduler.step()
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / n_batches
                self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'current_regime': self.current_regime,
            'performance': {
                'correct': self.correct_predictions,
                'total': self.total_predictions
            }
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.current_regime = checkpoint['current_regime']
        self.correct_predictions = checkpoint['performance']['correct']
        self.total_predictions = checkpoint['performance']['total']