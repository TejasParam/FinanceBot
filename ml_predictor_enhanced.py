import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# For LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

class EnhancedMLPredictor:
    """
    Enhanced Machine Learning predictor with advanced models and techniques.
    Includes LSTM, ensemble methods, feature engineering, and optimization.
    """
    
    def __init__(self):
        # Traditional ML models with optimized hyperparameters
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.05,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.05,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.pca = None
        self.lstm_model = None
        self.is_trained = False
        self.feature_names = []
        self.selected_features = []
        self.model_accuracies = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced features with additional technical indicators and transformations
        """
        features = pd.DataFrame()
        
        # Price-based features with multiple timeframes
        for period in [1, 3, 5, 10, 20, 30]:
            features[f'price_change_{period}d'] = data['Close'].pct_change(period)
            features[f'volume_change_{period}d'] = data['Volume'].pct_change(period)
        
        # Enhanced moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = data['Close'] / features[f'sma_{period}']
        
        # Moving average crossovers and divergences
        features['golden_cross'] = (features['sma_50'] > features['sma_200']).astype(int)
        features['death_cross'] = (features['sma_50'] < features['sma_200']).astype(int)
        features['macd_divergence'] = features['sma_10'] - features['sma_30']
        
        # Enhanced technical indicators
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['rsi_ma'] = features['rsi'].rolling(10).mean()
        features['rsi_divergence'] = features['rsi'] - features['rsi_ma']
        
        # Stochastic RSI
        features['stoch_rsi'] = self._calculate_stochastic_rsi(data['Close'])
        
        # MACD with signal and histogram
        features['macd'], features['macd_signal'], features['macd_hist'] = self._calculate_macd_full(data['Close'])
        features['macd_cross'] = (features['macd'] > features['macd_signal']).astype(int)
        
        # Enhanced Bollinger Bands
        for period in [20, 30]:
            bb_data = self._calculate_bollinger_bands_enhanced(data['Close'], period)
            features[f'bb_upper_{period}'] = bb_data['upper']
            features[f'bb_lower_{period}'] = bb_data['lower']
            features[f'bb_width_{period}'] = bb_data['width']
            features[f'bb_position_{period}'] = bb_data['position']
            features[f'bb_squeeze_{period}'] = bb_data['squeeze']
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(data)
        
        # Commodity Channel Index (CCI)
        features['cci'] = self._calculate_cci(data)
        
        # Money Flow Index (MFI)
        features['mfi'] = self._calculate_mfi(data)
        
        # Average Directional Index (ADX)
        features['adx'] = self._calculate_adx(data)
        
        # Ichimoku Cloud
        ichimoku_data = self._calculate_ichimoku(data)
        for key, value in ichimoku_data.items():
            features[f'ichimoku_{key}'] = value
        
        # Volume indicators
        features['obv'] = self._calculate_obv(data)
        features['vwap'] = self._calculate_vwap(data)
        features['volume_ma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_ma']
        features['price_volume_trend'] = self._calculate_pvt(data)
        
        # Volatility features
        features['volatility'] = data['Close'].rolling(20).std()
        features['volatility_ma'] = features['volatility'].rolling(20).mean()
        features['volatility_ratio'] = features['volatility'] / features['volatility_ma']
        features['atr'] = self._calculate_atr(data)
        features['atr_ratio'] = features['atr'] / data['Close']
        
        # Parkinson volatility (using high-low range)
        features['parkinson_vol'] = self._calculate_parkinson_volatility(data)
        
        # Price patterns
        features['higher_high'] = (data['High'] > data['High'].shift(1)).astype(int)
        features['lower_low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
        features['inside_bar'] = ((data['High'] < data['High'].shift(1)) & 
                                  (data['Low'] > data['Low'].shift(1))).astype(int)
        
        # Support and Resistance levels
        features['distance_to_high_20'] = (data['High'].rolling(20).max() - data['Close']) / data['Close']
        features['distance_to_low_20'] = (data['Close'] - data['Low'].rolling(20).min()) / data['Close']
        
        # Market microstructure features
        features['high_low_spread'] = (data['High'] - data['Low']) / data['Close']
        features['close_to_high'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        features['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Trend strength indicators
        features['trend_strength'] = self._calculate_trend_strength(data['Close'])
        features['aroon_up'], features['aroon_down'] = self._calculate_aroon(data)
        
        # Fibonacci retracement levels
        fib_levels = self._calculate_fibonacci_levels(data)
        for level, value in fib_levels.items():
            features[f'distance_to_fib_{level}'] = (data['Close'] - value) / data['Close']
        
        # Statistical features
        for period in [5, 10, 20]:
            features[f'skewness_{period}'] = data['Close'].rolling(period).apply(lambda x: x.skew())
            features[f'kurtosis_{period}'] = data['Close'].rolling(period).apply(lambda x: x.kurtosis())
        
        # Handle NaN values more intelligently
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining rows with NaN
        features = features.dropna()
        
        return features
    
    def _calculate_stochastic_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI"""
        rsi = self._calculate_rsi(prices, period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        return stoch_rsi * 100
    
    def _calculate_macd_full(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD with signal and histogram"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_bollinger_bands_enhanced(self, prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate enhanced Bollinger Bands with additional metrics"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        width = upper - lower
        position = (prices - lower) / (upper - lower)
        squeeze = width / sma  # Normalized bandwidth
        
        return {
            'upper': upper,
            'lower': lower,
            'width': width,
            'position': position,
            'squeeze': squeeze
        }
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['High'].rolling(period).max()
        low_min = data['Low'].rolling(period).min()
        return -100 * (high_max - data['Close']) / (high_max - low_min)
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mad)
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_sum / negative_sum))
        return mfi
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = self._calculate_true_range(data)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line)
        high_9 = data['High'].rolling(9).max()
        low_9 = data['Low'].rolling(9).min()
        tenkan = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = data['High'].rolling(26).max()
        low_26 = data['Low'].rolling(26).min()
        kijun = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = data['High'].rolling(52).max()
        low_52 = data['Low'].rolling(52).min()
        senkou_b = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou = data['Close'].shift(-26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou,
            'cloud_thickness': np.abs(senkou_a - senkou_b)
        }
    
    def _calculate_pvt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend"""
        price_change = data['Close'].pct_change()
        pvt = (price_change * data['Volume']).cumsum()
        return pvt
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        log_hl = np.log(data['High'] / data['Low'])
        return np.sqrt(252 / (4 * np.log(2))) * log_hl.rolling(period).std()
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        def calculate_slope(x):
            if len(x) < 2:
                return 0
            indices = np.arange(len(x))
            slope = np.polyfit(indices, x, 1)[0]
            return slope / x.mean()  # Normalize by mean
        
        return prices.rolling(period).apply(calculate_slope)
    
    def _calculate_aroon(self, data: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon Up and Aroon Down"""
        aroon_up = data['High'].rolling(period + 1).apply(lambda x: x.argmax() / period * 100)
        aroon_down = data['Low'].rolling(period + 1).apply(lambda x: x.argmin() / period * 100)
        return aroon_up, aroon_down
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame, period: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        recent_data = data.tail(period)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        diff = high - low
        
        levels = {
            '0': high,
            '236': high - 0.236 * diff,
            '382': high - 0.382 * diff,
            '500': high - 0.5 * diff,
            '618': high - 0.618 * diff,
            '786': high - 0.786 * diff,
            '1000': low
        }
        
        return levels
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = self._calculate_true_range(data)
        return tr.rolling(period).mean()
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
        return obv
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> pd.DataFrame:
        """Select best features using multiple methods"""
        # Method 1: Statistical test (f_classif)
        selector_f = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector_f.fit(X, y)
        scores_f = pd.Series(selector_f.scores_, index=X.columns)
        
        # Method 2: Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        scores_rf = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Combine scores (average rank)
        combined_scores = (scores_f.rank() + scores_rf.rank()) / 2
        
        # Select top features
        top_features = combined_scores.nlargest(n_features).index.tolist()
        self.selected_features = top_features
        
        return X[top_features]
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Optional[Any]:
        """Create LSTM model for time series prediction"""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(30, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(20, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_data(self, features: pd.DataFrame, labels: pd.Series, 
                         sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(labels.iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_models(self, price_data: pd.DataFrame, future_days: int = 5) -> Dict[str, Any]:
        """
        Train enhanced ML models with feature selection and optimization
        """
        print("🤖 Preparing enhanced ML training data...")
        
        # Prepare enhanced features
        features = self.prepare_enhanced_features(price_data)
        labels = self.create_labels(price_data, future_days)
        
        # Align features and labels
        common_dates = features.index.intersection(labels.index)
        if len(common_dates) < 100:
            return {"error": "Insufficient data for training"}
        
        features_aligned = features.loc[common_dates]
        labels_aligned = labels.loc[common_dates]
        
        # Remove any NaN values
        valid_mask = ~(features_aligned.isnull().any(axis=1) | labels_aligned.isnull())
        features_final = features_aligned[valid_mask]
        labels_final = labels_aligned[valid_mask]
        
        print(f"📊 Initial features: {features_final.shape[1]}")
        
        # Feature selection
        features_selected = self.select_features(features_final, labels_final)
        print(f"📊 Selected features: {features_selected.shape[1]}")
        
        self.feature_names = features_selected.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_selected)
        
        # Train-test split (temporal)
        split_idx = int(len(features_scaled) * 0.8)
        X_train, X_test = features_scaled[:split_idx], features_scaled[split_idx:]
        y_train, y_test = labels_final.iloc[:split_idx], labels_final.iloc[split_idx:]
        
        print("🤖 Training enhanced ML models...")
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            try:
                # Hyperparameter tuning for some models
                if name in ['xgboost', 'lightgbm'] and len(X_train) > 1000:
                    # Quick grid search
                    param_grid = self._get_param_grid(name)
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    self.models[name] = model
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        self.feature_names, 
                        model.feature_importances_
                    ))
                
                results[name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_samples': len(y_train),
                    'test_samples': len(y_test)
                }
                
                self.cv_scores[name] = cv_scores
                self.model_accuracies[name] = cv_scores.mean()
                
                print(f"    {name}: Test Accuracy: {test_score:.1%}, CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
                
            except Exception as e:
                print(f"    Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Train LSTM if TensorFlow is available
        if TF_AVAILABLE and len(features_final) > 500:
            print("  Training LSTM model...")
            try:
                X_lstm, y_lstm = self.prepare_lstm_data(features_selected, labels_final)
                
                if len(X_lstm) > 100:
                    split_idx = int(len(X_lstm) * 0.8)
                    X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
                    y_lstm_train, y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]
                    
                    self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
                    
                    history = self.lstm_model.fit(
                        X_lstm_train, y_lstm_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_lstm_test, y_lstm_test),
                        callbacks=[early_stop, reduce_lr],
                        verbose=0
                    )
                    
                    lstm_test_score = self.lstm_model.evaluate(X_lstm_test, y_lstm_test, verbose=0)[1]
                    self.model_accuracies['lstm'] = lstm_test_score
                    
                    results['lstm'] = {
                        'test_accuracy': lstm_test_score,
                        'epochs_trained': len(history.history['loss'])
                    }
                    
                    print(f"    LSTM: Test Accuracy: {lstm_test_score:.1%}")
                    
            except Exception as e:
                print(f"    Error training LSTM: {e}")
        
        # Create ensemble model
        ensemble_models = [(name, model) for name, model in self.models.items() 
                          if name in ['xgboost', 'lightgbm', 'random_forest']]
        
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            weights=[0.4, 0.4, 0.2]  # Give more weight to gradient boosting models
        )
        
        self.ensemble_model.fit(X_train, y_train)
        ensemble_score = self.ensemble_model.score(X_test, y_test)
        self.model_accuracies['ensemble'] = ensemble_score
        
        print(f"  🎯 Ensemble Accuracy: {ensemble_score:.1%}")
        
        self.is_trained = True
        return results
    
    def _get_param_grid(self, model_name: str) -> Dict[str, List]:
        """Get parameter grid for hyperparameter tuning"""
        if model_name == 'xgboost':
            return {
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [200, 300],
                'subsample': [0.8, 0.9]
            }
        elif model_name == 'lightgbm':
            return {
                'max_depth': [8, 10, 12],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50, 70],
                'n_estimators': [200, 300]
            }
        return {}
    
    def create_labels(self, data: pd.DataFrame, future_days: int = 5) -> pd.Series:
        """Create labels for prediction"""
        future_prices = data['Close'].shift(-future_days)
        current_prices = data['Close']
        
        # Create multi-class labels: 0=down, 1=flat, 2=up
        returns = (future_prices - current_prices) / current_prices
        
        # For now, use binary classification
        labels = (returns > 0).astype(int)
        
        return labels[:-future_days]
    
    def predict_probability(self, current_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhanced prediction with multiple models and confidence estimation
        """
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            # Prepare features
            feature_df = pd.DataFrame([current_features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            feature_df = feature_df[self.feature_names]
            features_scaled = self.scaler.transform(feature_df)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name in self.model_accuracies:
                    pred = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0]
                    
                    predictions[name] = pred
                    probabilities[name] = {
                        'prob_down': prob[0],
                        'prob_up': prob[1]
                    }
            
            # Ensemble prediction
            ensemble_prob = self.ensemble_model.predict_proba(features_scaled)[0]
            ensemble_pred = self.ensemble_model.predict(features_scaled)[0]
            
            # LSTM prediction if available
            if self.lstm_model and TF_AVAILABLE:
                # Need sequence data for LSTM
                # For now, skip LSTM in single prediction
                pass
            
            # Calculate weighted probability
            weights = []
            weighted_probs = []
            
            for name, prob_dict in probabilities.items():
                weight = self.model_accuracies.get(name, 0.5)
                weights.append(weight)
                weighted_probs.append(prob_dict['prob_up'])
            
            # Add ensemble
            weights.append(self.model_accuracies.get('ensemble', 0.6) * 1.5)  # Give ensemble more weight
            weighted_probs.append(ensemble_prob[1])
            
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)
            final_prob_up = np.average(weighted_probs, weights=weights)
            
            # Calculate confidence based on model agreement and individual confidences
            prob_std = np.std(weighted_probs)
            confidence = 1.0 - min(1.0, prob_std * 2)  # Lower std = higher confidence
            
            # Adjust confidence based on probability extremity
            extremity = abs(final_prob_up - 0.5) * 2
            confidence = confidence * 0.7 + extremity * 0.3
            
            return {
                'ensemble_prediction': 1 if final_prob_up > 0.5 else 0,
                'probability_up': final_prob_up,
                'probability_down': 1 - final_prob_up,
                'confidence': confidence,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'model_weights': dict(zip(list(probabilities.keys()) + ['ensemble'], weights)),
                'model_accuracies': self.model_accuracies.copy(),
                'probability_std': prob_std,
                'top_features': self._get_top_features(current_features)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _get_top_features(self, features: Dict[str, float], n: int = 10) -> Dict[str, float]:
        """Get top contributing features for the prediction"""
        if not self.feature_importance:
            return {}
        
        # Average feature importance across models
        avg_importance = {}
        for model_features in self.feature_importance.values():
            for feature, importance in model_features.items():
                if feature in features:
                    avg_importance[feature] = avg_importance.get(feature, 0) + importance
        
        # Normalize
        n_models = len(self.feature_importance)
        for feature in avg_importance:
            avg_importance[feature] /= n_models
        
        # Get top features with their values
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:n]
        return {feature: features.get(feature, 0) for feature, _ in top_features}
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all models"""
        return self.feature_importance
    
    def save_models(self, filepath: str):
        """Save all trained models and metadata"""
        if self.is_trained:
            model_data = {
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'selected_features': self.selected_features,
                'is_trained': self.is_trained,
                'model_accuracies': self.model_accuracies,
                'cv_scores': self.cv_scores,
                'feature_importance': self.feature_importance
            }
            
            # Save LSTM separately if available
            if self.lstm_model and TF_AVAILABLE:
                lstm_path = filepath.replace('.pkl', '_lstm.h5')
                self.lstm_model.save(lstm_path)
                model_data['lstm_path'] = lstm_path
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            print(f"✅ Enhanced models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.ensemble_model = model_data.get('ensemble_model')
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.selected_features = model_data.get('selected_features', [])
            self.is_trained = model_data['is_trained']
            self.model_accuracies = model_data.get('model_accuracies', {})
            self.cv_scores = model_data.get('cv_scores', {})
            self.feature_importance = model_data.get('feature_importance', {})
            
            # Load LSTM if available
            if 'lstm_path' in model_data and TF_AVAILABLE:
                lstm_path = model_data['lstm_path']
                if os.path.exists(lstm_path):
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
            
            print(f"✅ Enhanced models loaded from {filepath}")
            return True
        return False