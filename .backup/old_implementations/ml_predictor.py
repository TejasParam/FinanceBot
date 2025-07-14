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
        # Optimized models with better hyperparameters
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=2000,
                C=1.0,
                solver='liblinear',
                penalty='l2'
            )
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_accuracies = {}  # Store model accuracies
        self.cv_scores = {}  # Store cross-validation scores
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced features for ML model from technical analysis data
        """
        features = pd.DataFrame()
        
        # Price-based features
        features['price_change_1d'] = data['Close'].pct_change(1)
        features['price_change_3d'] = data['Close'].pct_change(3)
        features['price_change_5d'] = data['Close'].pct_change(5)
        features['price_change_10d'] = data['Close'].pct_change(10)
        
        # Moving averages - reasonable periods for available data
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_10'] = data['Close'].rolling(10).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        features['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # Moving average crossovers (trend signals)
        features['sma5_vs_sma20'] = features['sma_5'] / features['sma_20']
        features['sma20_vs_sma50'] = features['sma_20'] / features['sma_50']
        features['ema12_vs_ema26'] = features['ema_12'] / features['ema_26']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(data['Close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)
        
        # Bollinger Bands
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(data['Close'])
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / features['sma_20']
        
        # Momentum indicators
        features['momentum_5'] = data['Close'] / data['Close'].shift(5)
        features['momentum_10'] = data['Close'] / data['Close'].shift(10)
        features['roc_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
        features['roc_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
        
        # Volume features (if available)
        if 'Volume' in data.columns:
            features['volume_sma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
            features['volume_momentum'] = data['Volume'] / data['Volume'].shift(5)
            # Price-volume relationship
            features['pv_trend'] = features['price_change_1d'] * features['volume_ratio']
        
        # Price position features
        features['price_vs_sma20'] = data['Close'] / features['sma_20']
        features['price_vs_sma50'] = data['Close'] / features['sma_50']
        features['price_vs_high_20'] = data['Close'] / data['High'].rolling(20).max()
        features['price_vs_low_20'] = data['Close'] / data['Low'].rolling(20).min()
        
        # Volatility features
        features['volatility'] = data['Close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(30).mean()
        features['atr'] = self._calculate_atr(data)
        
        # High-Low range features
        features['daily_range'] = (data['High'] - data['Low']) / data['Close']
        features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Drop rows with NaN values but be less aggressive
        # Only drop if more than 50% of features are NaN
        features_cleaned = features.dropna(thresh=len(features.columns) * 0.5)
        
        # For remaining NaN values, forward fill then backward fill
        features_cleaned = features_cleaned.ffill().bfill()
        
        return features_cleaned
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
        
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
        labels = (future_prices > current_prices).astype(int)
        
        # Remove the last 'future_days' entries since they don't have future data
        return labels[:-future_days]
    
    def train_models(self, price_data: pd.DataFrame, future_days: int = 5) -> Dict[str, Any]:
        """
        Train ML models on historical data with enhanced validation
        """
        print("🤖 Preparing ML training data...")
        
        # Prepare features and labels
        features = self.prepare_features(price_data)
        labels = self.create_labels(price_data, future_days)
        
        # Align features and labels by index (date) rather than position
        # Find common dates between features and labels
        common_dates = features.index.intersection(labels.index)
        
        if len(common_dates) == 0:
            return {"error": "No common dates between features and labels"}
        
        # Select data for common dates only
        features_aligned = features.loc[common_dates]
        labels_aligned = labels.loc[common_dates]
        
        print(f"📊 Found {len(common_dates)} common dates for training")
        
        # Remove any remaining NaN values
        valid_indices = ~(features_aligned.isnull().any(axis=1) | labels_aligned.isnull())
        features_final = features_aligned[valid_indices]
        labels_final = labels_aligned[valid_indices]
        
        if len(features_final) < 50:
            return {"error": f"Insufficient valid data for training (need at least 50 samples, got {len(features_final)})"}
        
        print(f"📊 Training with {len(features_final)} valid samples and {len(features_final.columns)} features")
        
        self.feature_names = features_final.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_final)
        
        # Split data with temporal awareness (earlier data for training)
        split_idx = int(len(features_scaled) * 0.8)
        X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
        y_train, y_test = labels_final.iloc[:split_idx], labels_final.iloc[split_idx:]
        
        print("🤖 Training ML models with cross-validation...")
        results = {}
        
        # Train each model with enhanced evaluation
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate with multiple metrics
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            
            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_samples': len(y_train),
                'test_samples': len(y_test)
            }
            
            # Store cross-validation scores
            self.cv_scores[name] = cv_scores
            
            print(f"    {name}:")
            print(f"      Test Accuracy: {test_score:.1%}")
            print(f"      CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
            print(f"      Train vs Test: {train_score:.1%} vs {test_score:.1%}")
            
            # Store accuracy for later use (use CV mean for more robust estimate)
            self.model_accuracies[name] = cv_scores.mean()
        
        # Calculate ensemble performance
        ensemble_pred = self._ensemble_predict(X_test)
        ensemble_score = accuracy_score(y_test, ensemble_pred)
        self.model_accuracies['ensemble'] = ensemble_score
        
        print(f"  🎯 Ensemble Accuracy: {ensemble_score:.1%}")
        
        self.is_trained = True
        return results
    
    def _ensemble_predict(self, X) -> np.ndarray:
        """Make ensemble predictions using weighted voting"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]  # Probability of price going up
            predictions.append(pred_proba)
            # Weight by model accuracy
            weight = self.model_accuracies.get(name, 0.5)
            weights.append(weight)
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        
        return (ensemble_proba > 0.5).astype(int)
    
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
            weighted_probs = []
            weights = []
            
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]
                
                predictions[name] = pred
                probabilities[name] = {
                    'prob_down': prob[0],
                    'prob_up': prob[1]
                }
                
                # Use model accuracy as weight
                weight = self.model_accuracies.get(name, 0.5)
                weighted_probs.append(prob[1])  # Probability of going up
                weights.append(weight)
            
            # Weighted ensemble prediction
            weights = np.array(weights) / np.sum(weights)  # Normalize weights
            ensemble_prob_up = np.average(weighted_probs, weights=weights)
            ensemble_pred = 1 if ensemble_prob_up > 0.5 else 0
            
            # Calculate confidence as distance from 0.5
            confidence = abs(ensemble_prob_up - 0.5) * 2
            
            return {
                'ensemble_prediction': ensemble_pred,
                'probability_up': ensemble_prob_up,
                'probability_down': 1 - ensemble_prob_up,
                'confidence': confidence,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'model_weights': dict(zip(self.models.keys(), weights)),
                'model_accuracies': self.model_accuracies.copy()
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
                'is_trained': self.is_trained,
                'model_accuracies': self.model_accuracies,
                'cv_scores': getattr(self, 'cv_scores', {})
            }
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
            self.model_accuracies = model_data.get('model_accuracies', {})
            self.cv_scores = model_data.get('cv_scores', {})
            print(f"✅ Models loaded from {filepath}")
            return True
        return False
