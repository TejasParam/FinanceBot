import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    """
    Machine Learning predictor for stock price movement direction.
    Uses multiple models and ensemble methods.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model from technical analysis data
        """
        features = pd.DataFrame()
        
        # Price-based features
        features['price_change_1d'] = data['Close'].pct_change(1)
        features['price_change_5d'] = data['Close'].pct_change(5)
        features['price_change_20d'] = data['Close'].pct_change(20)
        
        # Moving averages
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        features['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'], features['macd_signal'] = self._calculate_macd(data['Close'])
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_sma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
        
        # Price position features
        features['price_vs_sma20'] = data['Close'] / features['sma_20']
        features['price_vs_sma50'] = data['Close'] / features['sma_50']
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volatility
        features['volatility'] = data['Close'].rolling(20).std()
        
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def create_labels(self, data: pd.DataFrame, future_days: int = 5) -> pd.Series:
        """
        Create labels for prediction: 1 if price goes up in next N days, 0 otherwise
        """
        future_prices = data['Close'].shift(-future_days)
        current_prices = data['Close']
        return (future_prices > current_prices).astype(int)
    
    def train_models(self, price_data: pd.DataFrame, future_days: int = 5) -> Dict[str, Any]:
        """
        Train ML models on historical data
        """
        print("🤖 Preparing ML training data...")
        
        # Prepare features and labels
        features = self.prepare_features(price_data)
        labels = self.create_labels(price_data, future_days)
        
        # Align features and labels
        min_length = min(len(features), len(labels))
        features = features.iloc[:min_length]
        labels = labels.iloc[:min_length]
        
        # Remove any remaining NaN values
        valid_indices = ~(features.isnull().any(axis=1) | labels.isnull())
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        if len(features) < 100:
            return {"error": "Insufficient data for training (need at least 100 samples)"}
        
        self.feature_names = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("🤖 Training ML models...")
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = accuracy_score(y_train, model.predict(X_train))
            test_score = accuracy_score(y_test, model.predict(X_test))
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"    {name}: Test Accuracy = {test_score:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
        return results
    
    def predict_probability(self, current_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict probability of price increase using ensemble of models
        """
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([current_features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(feature_df)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]
                
                predictions[name] = pred
                probabilities[name] = {
                    'prob_down': prob[0],
                    'prob_up': prob[1]
                }
            
            # Ensemble prediction (majority vote)
            ensemble_pred = 1 if sum(predictions.values()) >= len(predictions) / 2 else 0
            
            # Average probability
            avg_prob_up = np.mean([prob['prob_up'] for prob in probabilities.values()])
            
            return {
                'ensemble_prediction': ensemble_pred,
                'probability_up': avg_prob_up,
                'probability_down': 1 - avg_prob_up,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'confidence': max(avg_prob_up, 1 - avg_prob_up)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if self.is_trained:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            print(f"✅ Models loaded from {filepath}")
            return True
        return False
