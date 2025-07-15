#!/usr/bin/env python3
"""
Final ML evaluation for FinanceBot - Testing model accuracy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def prepare_data(symbol, period='2y'):
    """Download and prepare stock data"""
    print(f"Downloading {symbol} data...")
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    if len(df) < 100:
        return None, None
    
    # Calculate basic features
    df['returns'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    
    # Price momentum
    df['momentum_3'] = df['Close'].pct_change(3)
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    # Simple moving averages
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    
    # Price position
    df['price_vs_sma10'] = (df['Close'] - df['sma_10']) / df['sma_10']
    df['price_vs_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target: Next day direction (1 = up, 0 = down)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Select features
    feature_cols = [
        'returns', 'volume_change', 'momentum_3', 'momentum_5', 'momentum_10',
        'price_vs_sma10', 'price_vs_sma20', 'volatility', 'rsi'
    ]
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 50:
        return None, None
        
    X = df[feature_cols].values[:-1]  # Remove last row (no target)
    y = df['target'].values[:-1]
    
    return X, y

def evaluate_models():
    """Evaluate ML models on multiple stocks"""
    print("=" * 80)
    print("FINANCEBOT ML MODEL ACCURACY EVALUATION")
    print("=" * 80)
    print("\nEvaluating machine learning models for stock price direction prediction")
    print("Target: Predict if next day's close price will be higher (1) or lower (0)\n")
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    all_results = []
    
    for symbol in stocks:
        print(f"\n{'-' * 60}")
        print(f"Analyzing {symbol}")
        print(f"{'-' * 60}")
        
        try:
            # Get data
            X, y = prepare_data(symbol)
            if X is None:
                print(f"✗ Insufficient data for {symbol}")
                continue
                
            # Split data (temporal split, no shuffle)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Dataset size: {len(X)} samples")
            print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
            print(f"Class distribution - Up: {sum(y)}, Down: {len(y)-sum(y)}")
            
            # Test multiple models
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
            }
            
            symbol_results = {'symbol': symbol}
            
            for name, model in models.items():
                print(f"\n{name} Results:")
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                print(f"  Test Accuracy: {accuracy:.2%}")
                print(f"  Precision: {precision:.2%}")
                print(f"  Recall: {recall:.2%}")
                print(f"  F1-Score: {f1:.2%}")
                print(f"  Cross-Val: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
                
                symbol_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_scores.mean()
                }
                
                # Feature importance for Random Forest
                if name == 'Random Forest':
                    feature_names = [
                        'returns', 'volume_change', 'momentum_3', 'momentum_5', 
                        'momentum_10', 'price_vs_sma10', 'price_vs_sma20', 
                        'volatility', 'rsi'
                    ]
                    importances = model.feature_importances_
                    top_features = sorted(
                        zip(feature_names, importances), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    print(f"\n  Top Features:")
                    for feat, imp in top_features:
                        print(f"    - {feat}: {imp:.3f}")
            
            all_results.append(symbol_results)
            
        except Exception as e:
            print(f"✗ Error analyzing {symbol}: {e}")
    
    # Overall Summary
    if all_results:
        print(f"\n{'=' * 80}")
        print("OVERALL PERFORMANCE SUMMARY")
        print(f"{'=' * 80}")
        
        # Calculate averages
        rf_accuracies = [r['Random Forest']['accuracy'] for r in all_results if 'Random Forest' in r]
        gb_accuracies = [r['Gradient Boosting']['accuracy'] for r in all_results if 'Gradient Boosting' in r]
        
        if rf_accuracies:
            avg_rf = np.mean(rf_accuracies)
            print(f"\nRandom Forest Average Accuracy: {avg_rf:.2%}")
            
        if gb_accuracies:
            avg_gb = np.mean(gb_accuracies)
            print(f"Gradient Boosting Average Accuracy: {avg_gb:.2%}")
        
        overall_avg = np.mean(rf_accuracies + gb_accuracies)
        print(f"\nOverall Average Accuracy: {overall_avg:.2%}")
        
        print(f"\n{'=' * 80}")
        print("ACCURACY INTERPRETATION")
        print(f"{'=' * 80}")
        
        print(f"\nYour FinanceBot ML models achieve approximately {overall_avg:.1%} accuracy")
        print("in predicting next-day stock price direction.\n")
        
        if overall_avg > 0.60:
            print("✅ EXCELLENT: 60%+ accuracy is very good for financial ML")
            print("   - Significantly better than random chance (50%)")
            print("   - Indicates strong predictive patterns in technical indicators")
            print("   - Can provide meaningful trading signals with proper risk management")
        elif overall_avg > 0.55:
            print("✅ GOOD: 55-60% accuracy provides a statistical edge")
            print("   - Better than random chance")
            print("   - Can be profitable with proper position sizing")
            print("   - Risk management is crucial for success")
        elif overall_avg > 0.52:
            print("⚠️  MODERATE: 52-55% accuracy shows slight predictive power")
            print("   - Marginally better than random")
            print("   - Requires very careful risk management")
            print("   - Consider additional features or data")
        else:
            print("❌ POOR: Below 52% is close to random chance")
            print("   - Models need improvement")
            print("   - Consider more features or different approaches")
        
        print(f"\nKEY INSIGHTS:")
        print("- Financial markets are inherently difficult to predict")
        print("- Even 55-60% accuracy can be very profitable with proper execution")
        print("- Success depends on risk management, not just prediction accuracy")
        print("- The bot combines ML predictions with other analysis for better results")

if __name__ == "__main__":
    evaluate_models()