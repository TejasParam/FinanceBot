#!/usr/bin/env python3
"""
Retrain ML models with current environment
"""

import os
import sys
from ml_predictor_enhanced import EnhancedMLPredictor
from data_collection import DataCollectionAgent

def retrain_models():
    """Retrain and save models with current numpy version"""
    print("Retraining ML models with current environment...")
    
    # Initialize
    predictor = EnhancedMLPredictor()
    data_agent = DataCollectionAgent()
    
    # Train on a few key stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        print(f"\nTraining on {symbol}...")
        try:
            # Get data
            data = data_agent.fetch_stock_data(symbol, period='1y')
            if data is not None and len(data) > 100:
                # Train models
                results = predictor.train_models(data)
                if results:
                    print(f"✓ {symbol}: Accuracy={results['test_accuracy']:.2%}")
        except Exception as e:
            print(f"✗ Error with {symbol}: {e}")
    
    # Save models
    try:
        # Backup old models
        if os.path.exists('models/portfolio_models.pkl'):
            os.rename('models/portfolio_models.pkl', 'models/portfolio_models_backup.pkl')
            print("\n✓ Backed up old models")
        
        # Save new models
        predictor.save_models('models/portfolio_models.pkl')
        print("✓ New models saved successfully!")
        print("\nModels are now compatible with your current environment.")
        
    except Exception as e:
        print(f"\n✗ Error saving models: {e}")

if __name__ == "__main__":
    retrain_models()