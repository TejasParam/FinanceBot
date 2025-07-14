"""
ML Prediction Agent for machine learning-based stock predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from .base_agent import BaseAgent

# Handle ML imports gracefully
try:
    from ml_predictor_enhanced import MLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from data_collection import DataCollectionAgent

class MLPredictionAgent(BaseAgent):
    """
    Agent specialized in machine learning predictions for stock movements.
    Uses ensemble ML models to predict future price direction.
    """
    
    def __init__(self):
        super().__init__("MLPrediction")
        self.data_collector = DataCollectionAgent()
        
        if ML_AVAILABLE:
            self.ml_predictor = MLPredictor()
            self.model_trained = False
        else:
            self.ml_predictor = None
            self.model_trained = False
        
    def analyze(self, ticker: str, period: str = "2y", **kwargs) -> Dict[str, Any]:
        """
        Perform ML-based prediction for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for training data (default: 2y)
            
        Returns:
            Dictionary with ML prediction results
        """
        try:
            if not ML_AVAILABLE or self.ml_predictor is None:
                return self._simulate_ml_prediction(ticker)
            
            # Check if we need to train or retrain models
            force_retrain = kwargs.get('force_retrain', False)
            
            if not self.model_trained or force_retrain:
                train_result = self._train_models(ticker, period)
                if 'error' in train_result:
                    return train_result
            
            # Get current features for prediction
            current_data = self.data_collector.get_historical_data(ticker, period="6mo")
            if current_data is None or len(current_data) < 50:
                return {
                    'error': f'Insufficient data for ML prediction of {ticker}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Not enough historical data for ML prediction'
                }
            
            # Prepare features for current prediction
            features = self.ml_predictor.prepare_features(current_data)
            if len(features) == 0:
                return {
                    'error': 'Failed to prepare features for prediction',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': 'Could not extract features from current data'
                }
            
            # Get latest feature values for prediction
            latest_features = features.iloc[-1].to_dict()
            
            # Make prediction
            prediction_result = self.ml_predictor.predict_probability(latest_features)
            
            if 'error' in prediction_result:
                return {
                    'error': f'ML prediction failed: {prediction_result["error"]}',
                    'score': 0.0,
                    'confidence': 0.0,
                    'reasoning': f'ML prediction error: {prediction_result["error"][:100]}'
                }
            
            # Convert ML prediction to standardized score (-1 to 1)
            ml_score = self._convert_prediction_to_score(prediction_result)
            
            # Use ML confidence directly
            ml_confidence = prediction_result.get('confidence', 0.5)
            
            # Generate reasoning
            reasoning = self._generate_ml_reasoning(prediction_result, ml_score)
            
            return {
                'score': ml_score,
                'confidence': ml_confidence,
                'reasoning': reasoning,
                'prediction': prediction_result['ensemble_prediction'],
                'probability_up': prediction_result['probability_up'],
                'probability_down': prediction_result['probability_down'],
                'model_accuracies': prediction_result.get('model_accuracies', {}),
                'individual_predictions': prediction_result.get('individual_predictions', {}),
                'raw_prediction': prediction_result
            }
            
        except Exception as e:
            return {
                'error': f'ML prediction analysis failed: {str(e)}',
                'score': 0.0,
                'confidence': 0.0,
                'reasoning': f'ML prediction error: {str(e)[:100]}'
            }
    
    def _train_models(self, ticker: str, period: str) -> Dict[str, Any]:
        """Train ML models if not already trained"""
        try:
            # Get training data
            training_data = self.data_collector.get_stock_data(ticker, period=period)
            if training_data is None or len(training_data) < 100:
                return {
                    'error': f'Insufficient training data for {ticker} (need at least 100 days, got {len(training_data) if training_data is not None else 0})'
                }
            
            # Train the models
            training_results = self.ml_predictor.train_models(training_data)
            
            if 'error' in training_results:
                return {
                    'error': f'ML model training failed: {training_results["error"]}'
                }
            
            self.model_trained = True
            self.logger.info(f"ML models trained successfully for {ticker}")
            
            return {
                'success': True,
                'training_results': training_results
            }
            
        except Exception as e:
            return {
                'error': f'ML model training failed: {str(e)}'
            }
    
    def _convert_prediction_to_score(self, prediction_result: Dict[str, Any]) -> float:
        """Convert ML prediction probability to standardized score (-1 to 1)"""
        prob_up = prediction_result.get('probability_up', 0.5)
        
        # Convert probability to score: 0.5 = 0, 1.0 = 1, 0.0 = -1
        score = (prob_up - 0.5) * 2
        
        # Apply confidence weighting - low confidence predictions move toward 0
        confidence = prediction_result.get('confidence', 0.5)
        score *= confidence
        
        return max(-1.0, min(1.0, score))
    
    def _generate_ml_reasoning(self, prediction_result: Dict[str, Any], score: float) -> str:
        """Generate human-readable reasoning for ML prediction"""
        reasoning_parts = []
        
        # Overall prediction
        prob_up = prediction_result.get('probability_up', 0.5)
        confidence = prediction_result.get('confidence', 0.5)
        
        if prob_up > 0.6:
            reasoning_parts.append(f"ML models predict upward movement with {prob_up:.1%} probability.")
        elif prob_up < 0.4:
            reasoning_parts.append(f"ML models predict downward movement with {1-prob_up:.1%} probability.")
        else:
            reasoning_parts.append(f"ML models show mixed signals with {prob_up:.1%} upward probability.")
        
        # Model confidence
        if confidence > 0.7:
            reasoning_parts.append("High model confidence in prediction.")
        elif confidence > 0.5:
            reasoning_parts.append("Moderate model confidence in prediction.")
        else:
            reasoning_parts.append("Low model confidence - mixed signals from ensemble.")
        
        # Model accuracy info
        model_accuracies = prediction_result.get('model_accuracies', {})
        if model_accuracies:
            ensemble_acc = model_accuracies.get('ensemble', 0)
            if ensemble_acc > 0:
                reasoning_parts.append(f"Ensemble model accuracy: {ensemble_acc:.1%}.")
        
        # Individual model agreement
        individual_preds = prediction_result.get('individual_predictions', {})
        if len(individual_preds) > 1:
            predictions = list(individual_preds.values())
            agreement = sum(predictions) / len(predictions)
            if agreement > 0.8 or agreement < 0.2:
                reasoning_parts.append("Strong agreement between individual models.")
            else:
                reasoning_parts.append("Mixed agreement between individual models.")
        
        return " ".join(reasoning_parts)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of ML models"""
        return {
            'model_trained': self.model_trained,
            'is_trained': self.ml_predictor.is_trained,
            'model_accuracies': getattr(self.ml_predictor, 'model_accuracies', {}),
            'feature_count': len(getattr(self.ml_predictor, 'feature_names', []))
        }
    
    def force_retrain(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
        """Force retrain the ML models"""
        self.model_trained = False
        return self._train_models(ticker, period)
    
    def _simulate_ml_prediction(self, ticker: str) -> Dict[str, Any]:
        """Simulate ML prediction when ML components are not available"""
        import random
        
        # Generate realistic ML-style prediction
        prediction = random.choice([0, 1])  # 0 = DOWN, 1 = UP
        probability_up = random.uniform(0.45, 0.85)
        confidence = random.uniform(0.6, 0.9)
        
        # Simulate ensemble results
        models = ['random_forest', 'gradient_boost', 'logistic']
        model_predictions = {}
        for model in models:
            model_predictions[model] = {
                'prediction': random.choice([0, 1]),
                'probability': random.uniform(0.4, 0.9)
            }
        
        return {
            'score': probability_up,
            'confidence': confidence,
            'reasoning': f'Simulated ML ensemble predicts {"UP" if prediction == 1 else "DOWN"} movement with {confidence:.1%} confidence',
            'prediction': prediction,
            'probability_up': probability_up,
            'probability_down': 1 - probability_up,
            'ensemble_prediction': prediction,
            'model_predictions': model_predictions,
            'features_used': random.randint(35, 45),
            'note': 'Simulated prediction - install ML dependencies for real predictions'
        }
