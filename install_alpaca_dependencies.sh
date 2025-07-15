#!/bin/bash
# Install script for Alpaca trading bot dependencies

echo "Installing Alpaca Trading Bot dependencies..."

# Core trading dependencies
pip install alpaca-py
pip install schedule

# Web dashboard
pip install flask
pip install flask-cors

# Sentiment analysis (if not already installed)
pip install textblob
pip install vaderSentiment

# Download textblob corpora
python -m textblob.download_corpora

echo "Installation complete!"
echo "Note: If you see any errors, you may need to install some packages individually"