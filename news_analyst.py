import os
import json
import requests
from dotenv import find_dotenv, load_dotenv

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

def analyze_news(news_items):
    """Analyze and summarize news using Gemini API. Extract sentiment as a separate field."""
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    results = []
    for news in news_items:
        try:
            prompt = f"Title: {news.get('title','')}\nSummary: {news.get('summary','')}\n\nSummarize this news and give its sentiment (positive, negative, or neutral):"
            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            }
            response = requests.post(
                GEMINI_API_URL + f"?key={GEMINI_API_KEY}",
                headers=headers,
                data=json.dumps(payload),
                timeout=20
            )
            response.raise_for_status()
            gemini_data = response.json()
            summary = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
            # Extract sentiment from the summary (look for the word 'positive', 'negative', or 'neutral')
            sentiment = "neutral"
            for s in ["positive", "negative", "neutral"]:
                if s in summary.lower():
                    sentiment = s
                    break
            results.append({
                "title": news.get("title", ""),
                "summary": news.get("summary", ""),
                "analysis": summary,
                "sentiment": sentiment
            })
        except Exception as e:
            # Use logging in a real project; print for now
            print(f"Error analyzing news: {e}")
            results.append({
                "title": news.get("title", ""),
                "summary": news.get("summary", ""),
                "analysis": "Error during analysis.",
                "sentiment": "unknown"
            })
    return results

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"
    news_items = fetch_alpha_vantage_news(ticker=ticker, limit=2)
    analyzed_news = analyze_news(news_items)
    for news in analyzed_news:
        print("Title:", news["title"])
        print("Summary:", news["summary"])
        print("Analysis:", news["analysis"])
        print("Sentiment:", news["sentiment"])
        print("-" * 40)
