#!/bin/bash
# Install optional features for enhanced functionality

echo "Installing optional packages for enhanced features..."

# 1. TensorFlow for LSTM models (optional - adds deep learning)
echo "Installing TensorFlow (for LSTM models)..."
pip install tensorflow

# 2. Feedparser for RSS feeds (optional - adds more news sources)
echo "Installing feedparser (for RSS news feeds)..."
pip install feedparser

# 3. Social media libraries (optional - adds Reddit/Twitter sentiment)
echo "Installing social media libraries..."
pip install praw  # Reddit
pip install tweepy  # Twitter

# 4. Portfolio optimization libraries (optional - adds advanced optimization)
echo "Installing portfolio optimization libraries..."
pip install cvxpy
pip install PyPortfolioOpt

echo "✅ Optional features installation complete!"
echo ""
echo "Benefits of these packages:"
echo "- TensorFlow: Enables LSTM deep learning models for better predictions"
echo "- Feedparser: Adds RSS feed analysis for more news sources"
echo "- Social media: Analyzes Reddit/Twitter for sentiment"
echo "- CVXPY/PyPortfolioOpt: Advanced portfolio optimization algorithms"
echo ""
echo "Note: The bot works fine without these - they just add extra capabilities!"