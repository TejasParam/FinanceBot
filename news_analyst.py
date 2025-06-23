import openai
import requests

# Replace with your API keys
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

def fetch_alpha_vantage_news(ticker="AAPL", limit=3):
    """Fetch real-time news for a ticker using Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("feed", [])

def analyze_news(news_items):
    """Analyze and summarize news using GPT-4o mini."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    results = []
    for news in news_items:
        prompt = f"Title: {news['title']}\nSummary: {news['summary']}\n\nSummarize this news and give its sentiment (positive, negative, or neutral):"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a news analyst. Summarize the following news and provide its sentiment:"},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content
        results.append({
            "title": news["title"],
            "summary": news["summary"],
            "analysis": summary
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
        print("-" * 40)
