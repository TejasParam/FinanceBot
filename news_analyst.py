
import os
import json
import requests
from dotenv import find_dotenv, load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("alpha_vantage_api_key")
GEMINI_API_KEY = os.getenv("gemini_api_key")

def fetch_alpha_vantage_news(ticker="AAPL", limit=3):
    """Fetch real-time news for a ticker using Alpha Vantage."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("feed", [])
    except Exception as e:
        # Use logging in a real project; print for now
        print(f"Error fetching news: {e}")
        return []


# Load FinBERT model and tokenizer once
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_news(news_items):
    """Analyze news using FinBERT for sentiment. Uses Alpha Vantage summary as analysis."""
    results = []
    for news in news_items:
        try:
            text = news.get("summary", "") or news.get("title", "")
            inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = finbert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                sentiment_idx = torch.argmax(probs).item()
                sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
                sentiment = sentiment_map.get(sentiment_idx, "neutral")
            results.append({
                "title": news.get("title", ""),
                "summary": news.get("summary", ""),
                "analysis": news.get("summary", ""),  # Use Alpha Vantage summary as analysis
                "sentiment": sentiment
            })
        except Exception as e:
            print(f"Error analyzing news: {e}")
            results.append({
                "title": news.get("title", ""),
                "summary": news.get("summary", ""),
                "analysis": "Error during analysis.",
                "sentiment": "unknown"
            })
    return results



# NewsAnalyst class to be used by other modules
class news_analyst:
    def __init__(self):
        pass

    def fetch_news(self, ticker="AAPL", limit=3):
        return fetch_alpha_vantage_news(ticker, limit)

    def analyze_news(self, news_items):
        return analyze_news(news_items)

    def analyze_stock_news(self, ticker="AAPL", limit=3):
        news_items = self.fetch_news(ticker, limit)
        return self.analyze_news(news_items)


