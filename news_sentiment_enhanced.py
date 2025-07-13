import os
import json
import requests
from dotenv import find_dotenv, load_dotenv
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
import re
from textblob import TextBlob
import yfinance as yf

# Advanced NLP imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModelForTokenClassification
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Advanced NLP features will be limited.")

# Social media imports
try:
    import praw
    import tweepy
    SOCIAL_AVAILABLE = True
except ImportError:
    SOCIAL_AVAILABLE = False
    print("Social media libraries not available. Reddit/Twitter analysis will be disabled.")

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# API keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

class EnhancedNewsSentimentAnalyzer:
    """
    Enhanced news and sentiment analyzer with multiple data sources,
    advanced NLP models, and entity-specific sentiment extraction.
    """
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
        # Initialize models
        if TRANSFORMERS_AVAILABLE:
            # Financial sentiment model
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # General sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            
            # Named Entity Recognition for company detection
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
        else:
            self.finbert_tokenizer = None
            self.finbert_model = None
            self.sentiment_pipeline = None
            self.ner_pipeline = None
        
        # Initialize social media clients
        self._init_social_clients()
        
        # News sources
        self.rss_feeds = {
            'reuters': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
            'wsj': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
            'ft': 'https://www.ft.com/rss/markets',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories'
        }
    
    def _init_social_clients(self):
        """Initialize social media API clients"""
        # Reddit client
        if SOCIAL_AVAILABLE and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            try:
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent="FinanceBot/1.0"
                )
                self.reddit_available = True
            except Exception as e:
                print(f"Reddit initialization failed: {e}")
                self.reddit_available = False
        else:
            self.reddit_available = False
        
        # Twitter client
        if SOCIAL_AVAILABLE and TWITTER_BEARER_TOKEN:
            try:
                self.twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
                self.twitter_available = True
            except Exception as e:
                print(f"Twitter initialization failed: {e}")
                self.twitter_available = False
        else:
            self.twitter_available = False
    
    def analyze_stock_sentiment(self, ticker: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis from multiple sources
        """
        # Check cache
        cache_key = f"{ticker}_{days_back}"
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data
        
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_sentiment': 0.0,
            'sentiment_momentum': 0.0,
            'confidence': 0.0,
            'key_events': [],
            'entity_sentiments': {},
            'sector_sentiment': 0.0
        }
        
        # Get company info
        company_info = self._get_company_info(ticker)
        company_name = company_info.get('name', ticker)
        sector = company_info.get('sector', 'Unknown')
        
        # Collect sentiment from multiple sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_news_sources, ticker, company_name, days_back): 'news',
                executor.submit(self._analyze_social_media, ticker, company_name): 'social',
                executor.submit(self._analyze_analyst_reports, ticker): 'analysts',
                executor.submit(self._analyze_sec_filings, ticker): 'sec_filings'
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    results['sources'][source] = future.result()
                except Exception as e:
                    print(f"Error analyzing {source}: {e}")
                    results['sources'][source] = {'error': str(e)}
        
        # Aggregate results
        results = self._aggregate_sentiments(results, sector)
        
        # Calculate sentiment momentum
        results['sentiment_momentum'] = self._calculate_sentiment_momentum(ticker, results['overall_sentiment'])
        
        # Extract key events
        results['key_events'] = self._extract_key_events(results['sources'])
        
        # Cache results
        self.sentiment_cache[cache_key] = (results, time.time())
        
        return results
    
    def _get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception:
            return {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown', 'description': ''}
    
    def _analyze_news_sources(self, ticker: str, company_name: str, days_back: int) -> Dict[str, Any]:
        """Analyze news from multiple RSS feeds and news APIs"""
        all_articles = []
        
        # Alpha Vantage news
        av_articles = self._fetch_alpha_vantage_news(ticker)
        all_articles.extend(av_articles)
        
        # RSS feeds
        for source, url in self.rss_feeds.items():
            try:
                feed_articles = self._parse_rss_feed(url, ticker, company_name, days_back)
                all_articles.extend(feed_articles)
            except Exception as e:
                print(f"Error parsing {source} RSS: {e}")
        
        # NewsAPI if available
        if NEWS_API_KEY:
            newsapi_articles = self._fetch_newsapi_articles(ticker, company_name, days_back)
            all_articles.extend(newsapi_articles)
        
        # Analyze sentiment for each article
        analyzed_articles = []
        for article in all_articles[:50]:  # Limit to 50 most recent
            sentiment_data = self._analyze_article_sentiment(article, ticker, company_name)
            analyzed_articles.append(sentiment_data)
        
        # Calculate aggregate metrics
        if analyzed_articles:
            sentiments = [a['sentiment_score'] for a in analyzed_articles if a['sentiment_score'] is not None]
            
            return {
                'article_count': len(analyzed_articles),
                'average_sentiment': np.mean(sentiments) if sentiments else 0.0,
                'sentiment_std': np.std(sentiments) if sentiments else 0.0,
                'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments) if sentiments else 0.0,
                'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments) if sentiments else 0.0,
                'articles': analyzed_articles[:10],  # Top 10 for detail
                'sentiment_distribution': self._calculate_sentiment_distribution(sentiments)
            }
        
        return {'article_count': 0, 'average_sentiment': 0.0, 'error': 'No articles found'}
    
    def _fetch_alpha_vantage_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": ALPHA_VANTAGE_API_KEY,
                "limit": 20
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("feed", []):
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'published': item.get('time_published', ''),
                    'source': item.get('source', 'Alpha Vantage'),
                    'relevance_score': float(item.get('overall_sentiment_score', 0))
                })
            
            return articles
        except Exception as e:
            print(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    def _parse_rss_feed(self, feed_url: str, ticker: str, company_name: str, days_back: int) -> List[Dict[str, Any]]:
        """Parse RSS feed for relevant articles"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:30]:  # Limit entries
                # Check if article is relevant
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                content = f"{title} {summary}".lower()
                
                if ticker.lower() in content or company_name.lower() in content:
                    # Parse date
                    published = entry.get('published_parsed')
                    if published:
                        pub_date = datetime.fromtimestamp(time.mktime(published))
                        if pub_date < cutoff_date:
                            continue
                    
                    articles.append({
                        'title': title,
                        'summary': summary,
                        'url': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': feed.feed.get('title', 'RSS Feed')
                    })
            
            return articles
        except Exception as e:
            print(f"Error parsing RSS feed: {e}")
            return []
    
    def _fetch_newsapi_articles(self, ticker: str, company_name: str, days_back: int) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{ticker} OR {company_name}",
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': NEWS_API_KEY,
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get('articles', []):
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'url': item.get('url', ''),
                    'published': item.get('publishedAt', ''),
                    'source': item.get('source', {}).get('name', 'NewsAPI')
                })
            
            return articles
        except Exception as e:
            print(f"Error fetching NewsAPI articles: {e}")
            return []
    
    def _analyze_article_sentiment(self, article: Dict[str, Any], ticker: str, company_name: str) -> Dict[str, Any]:
        """Analyze sentiment of a single article"""
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        
        sentiment_score = 0.0
        entity_sentiment = {}
        
        if TRANSFORMERS_AVAILABLE and self.finbert_model:
            # Use FinBERT for financial sentiment
            try:
                inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                    
                    # FinBERT: 0=positive, 1=negative, 2=neutral
                    sentiment_score = float(probs[0] - probs[1])
                    
                # Extract entity-specific sentiment
                entities = self._extract_entities(text)
                for entity in entities:
                    if entity['entity_group'] == 'ORG':
                        entity_text = entity['word']
                        # Get sentiment for sentences containing this entity
                        entity_sentiment[entity_text] = self._get_entity_sentiment(text, entity_text)
                        
            except Exception as e:
                print(f"FinBERT analysis error: {e}")
        
        # Fallback to TextBlob
        if sentiment_score == 0.0:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
        
        return {
            'title': article.get('title', ''),
            'source': article.get('source', ''),
            'published': article.get('published', ''),
            'url': article.get('url', ''),
            'sentiment_score': sentiment_score,
            'entity_sentiments': entity_sentiment,
            'relevance_score': article.get('relevance_score', self._calculate_relevance(text, ticker, company_name))
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                return entities
            except Exception:
                pass
        return []
    
    def _get_entity_sentiment(self, text: str, entity: str) -> float:
        """Get sentiment for specific entity mentions"""
        sentences = text.split('.')
        entity_sentences = [s for s in sentences if entity.lower() in s.lower()]
        
        if not entity_sentences:
            return 0.0
        
        sentiments = []
        for sentence in entity_sentences:
            if self.sentiment_pipeline:
                try:
                    result = self.sentiment_pipeline(sentence)[0]
                    # Convert to -1 to 1 scale
                    label = result['label']
                    score = result['score']
                    
                    if '1' in label or 'negative' in label.lower():
                        sentiment = -score
                    elif '5' in label or 'positive' in label.lower():
                        sentiment = score
                    else:
                        sentiment = 0.0
                    
                    sentiments.append(sentiment)
                except Exception:
                    blob = TextBlob(sentence)
                    sentiments.append(blob.sentiment.polarity)
            else:
                blob = TextBlob(sentence)
                sentiments.append(blob.sentiment.polarity)
        
        return np.mean(sentiments)
    
    def _calculate_relevance(self, text: str, ticker: str, company_name: str) -> float:
        """Calculate relevance score for article"""
        text_lower = text.lower()
        ticker_lower = ticker.lower()
        company_lower = company_name.lower()
        
        relevance = 0.0
        
        # Count mentions
        ticker_mentions = text_lower.count(ticker_lower)
        company_mentions = text_lower.count(company_lower)
        
        relevance += min(ticker_mentions * 0.2, 0.6)
        relevance += min(company_mentions * 0.15, 0.4)
        
        # Check for financial keywords
        financial_keywords = ['earnings', 'revenue', 'profit', 'growth', 'analyst', 'upgrade', 'downgrade',
                            'forecast', 'guidance', 'acquisition', 'merger', 'ipo', 'dividend']
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                relevance += 0.05
        
        return min(relevance, 1.0)
    
    def _analyze_social_media(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Analyze sentiment from social media sources"""
        results = {
            'reddit': {'error': 'Not available'},
            'twitter': {'error': 'Not available'},
            'overall_sentiment': 0.0,
            'volume': 0,
            'trending_score': 0.0
        }
        
        # Reddit analysis
        if self.reddit_available:
            reddit_data = self._analyze_reddit(ticker, company_name)
            results['reddit'] = reddit_data
        
        # Twitter analysis
        if self.twitter_available:
            twitter_data = self._analyze_twitter(ticker, company_name)
            results['twitter'] = twitter_data
        
        # Calculate overall social sentiment
        sentiments = []
        volumes = []
        
        for platform in ['reddit', 'twitter']:
            if 'error' not in results[platform]:
                sentiments.append(results[platform].get('average_sentiment', 0))
                volumes.append(results[platform].get('post_count', 0))
        
        if sentiments:
            # Weight by volume
            total_volume = sum(volumes)
            if total_volume > 0:
                weighted_sentiment = sum(s * v for s, v in zip(sentiments, volumes)) / total_volume
                results['overall_sentiment'] = weighted_sentiment
            else:
                results['overall_sentiment'] = np.mean(sentiments)
            
            results['volume'] = total_volume
            results['trending_score'] = self._calculate_trending_score(volumes)
        
        return results
    
    def _analyze_reddit(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Analyze Reddit sentiment"""
        try:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'StockMarket']
            all_posts = []
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for mentions
                for post in subreddit.search(f"{ticker} OR {company_name}", time_filter='week', limit=20):
                    all_posts.append({
                        'title': post.title,
                        'body': post.selftext,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created': datetime.fromtimestamp(post.created_utc)
                    })
            
            # Analyze sentiment
            sentiments = []
            for post in all_posts:
                text = f"{post['title']} {post['body']}"
                sentiment = self._get_text_sentiment(text)
                
                # Weight by engagement
                weight = np.log1p(post['score'] + post['num_comments'])
                sentiments.append((sentiment, weight))
            
            if sentiments:
                # Calculate weighted average
                total_weight = sum(w for _, w in sentiments)
                weighted_sentiment = sum(s * w for s, w in sentiments) / total_weight if total_weight > 0 else 0
                
                return {
                    'post_count': len(all_posts),
                    'average_sentiment': weighted_sentiment,
                    'sentiment_std': np.std([s for s, _ in sentiments]),
                    'total_engagement': sum(p['score'] + p['num_comments'] for p in all_posts),
                    'top_posts': sorted(all_posts, key=lambda x: x['score'], reverse=True)[:5]
                }
            
            return {'post_count': 0, 'average_sentiment': 0.0}
            
        except Exception as e:
            return {'error': f'Reddit analysis failed: {str(e)}'}
    
    def _analyze_twitter(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Analyze Twitter sentiment"""
        try:
            # Search for tweets
            query = f"${ticker} OR {company_name} -is:retweet lang:en"
            
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            if not tweets.data:
                return {'tweet_count': 0, 'average_sentiment': 0.0}
            
            # Analyze sentiment
            sentiments = []
            total_engagement = 0
            
            for tweet in tweets.data:
                sentiment = self._get_text_sentiment(tweet.text)
                
                # Get engagement metrics
                metrics = tweet.public_metrics
                engagement = metrics['retweet_count'] + metrics['like_count'] + metrics['reply_count']
                total_engagement += engagement
                
                # Weight by engagement
                weight = np.log1p(engagement)
                sentiments.append((sentiment, weight))
            
            # Calculate weighted average
            total_weight = sum(w for _, w in sentiments)
            weighted_sentiment = sum(s * w for s, w in sentiments) / total_weight if total_weight > 0 else 0
            
            return {
                'tweet_count': len(tweets.data),
                'average_sentiment': weighted_sentiment,
                'sentiment_std': np.std([s for s, _ in sentiments]),
                'total_engagement': total_engagement,
                'tweets_per_hour': len(tweets.data) / 24  # Last 24 hours
            }
            
        except Exception as e:
            return {'error': f'Twitter analysis failed: {str(e)}'}
    
    def _get_text_sentiment(self, text: str) -> float:
        """Get sentiment score for any text"""
        if TRANSFORMERS_AVAILABLE and self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text[:512])[0]  # Truncate for model
                
                # Convert to -1 to 1 scale
                label = result['label']
                score = result['score']
                
                if '1' in label or 'negative' in label.lower():
                    return -score
                elif '5' in label or 'positive' in label.lower():
                    return score
                else:
                    return 0.0
            except Exception:
                pass
        
        # Fallback to TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def _analyze_analyst_reports(self, ticker: str) -> Dict[str, Any]:
        """Analyze analyst recommendations and price targets"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get analyst recommendations
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                recent_recs = recommendations.tail(20)
                
                # Convert recommendations to sentiment scores
                rec_scores = {
                    'Strong Buy': 1.0,
                    'Buy': 0.5,
                    'Hold': 0.0,
                    'Sell': -0.5,
                    'Strong Sell': -1.0
                }
                
                sentiments = []
                for _, rec in recent_recs.iterrows():
                    grade = rec.get('To Grade', rec.get('Action', 'Hold'))
                    sentiment = rec_scores.get(grade, 0.0)
                    sentiments.append(sentiment)
                
                # Get price targets
                info = stock.info
                current_price = info.get('currentPrice', 0)
                target_price = info.get('targetMeanPrice', current_price)
                
                price_target_sentiment = (target_price - current_price) / current_price if current_price > 0 else 0
                
                return {
                    'recommendation_count': len(recent_recs),
                    'average_sentiment': np.mean(sentiments),
                    'recent_changes': len(recent_recs[recent_recs.index >= datetime.now() - timedelta(days=30)]),
                    'price_target_sentiment': price_target_sentiment,
                    'analyst_consensus': info.get('recommendationKey', 'none'),
                    'number_of_analysts': info.get('numberOfAnalystOpinions', 0)
                }
            
            return {'recommendation_count': 0, 'average_sentiment': 0.0}
            
        except Exception as e:
            return {'error': f'Analyst analysis failed: {str(e)}'}
    
    def _analyze_sec_filings(self, ticker: str) -> Dict[str, Any]:
        """Analyze recent SEC filings for sentiment"""
        # This would integrate with SEC EDGAR API
        # For now, return placeholder
        return {
            'filing_count': 0,
            'sentiment': 0.0,
            'material_events': []
        }
    
    def _aggregate_sentiments(self, results: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """Aggregate sentiments from all sources"""
        sentiments = []
        weights = []
        
        # Define source weights
        source_weights = {
            'news': 0.35,
            'social': 0.20,
            'analysts': 0.35,
            'sec_filings': 0.10
        }
        
        for source, weight in source_weights.items():
            if source in results['sources'] and 'error' not in results['sources'][source]:
                source_data = results['sources'][source]
                
                if source == 'news':
                    sentiment = source_data.get('average_sentiment', 0)
                elif source == 'social':
                    sentiment = source_data.get('overall_sentiment', 0)
                elif source == 'analysts':
                    sentiment = source_data.get('average_sentiment', 0)
                elif source == 'sec_filings':
                    sentiment = source_data.get('sentiment', 0)
                else:
                    sentiment = 0
                
                sentiments.append(sentiment)
                weights.append(weight)
        
        # Calculate weighted average
        if sentiments:
            total_weight = sum(weights)
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / total_weight
            results['overall_sentiment'] = weighted_sentiment
            
            # Calculate confidence based on source agreement
            sentiment_std = np.std(sentiments)
            results['confidence'] = max(0, 1 - sentiment_std)
        else:
            results['overall_sentiment'] = 0.0
            results['confidence'] = 0.0
        
        # Add sector sentiment (placeholder - would need sector-wide analysis)
        results['sector_sentiment'] = self._get_sector_sentiment(sector)
        
        return results
    
    def _calculate_sentiment_momentum(self, ticker: str, current_sentiment: float) -> float:
        """Calculate momentum of sentiment changes"""
        # Get historical sentiments from cache
        historical = []
        for days in [1, 3, 7, 14]:
            cache_key = f"{ticker}_{days}"
            if cache_key in self.sentiment_cache:
                data, _ = self.sentiment_cache[cache_key]
                historical.append((days, data.get('overall_sentiment', 0)))
        
        if len(historical) < 2:
            return 0.0
        
        # Calculate trend
        days_array = np.array([h[0] for h in historical])
        sentiment_array = np.array([h[1] for h in historical])
        
        # Linear regression for trend
        if len(days_array) > 1:
            slope = np.polyfit(days_array, sentiment_array, 1)[0]
            return slope * 10  # Scale for readability
        
        return 0.0
    
    def _extract_key_events(self, sources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key events from all sources"""
        events = []
        
        # Extract from news
        if 'news' in sources and 'articles' in sources['news']:
            for article in sources['news']['articles'][:5]:
                if abs(article.get('sentiment_score', 0)) > 0.5:
                    events.append({
                        'type': 'news',
                        'title': article.get('title', ''),
                        'date': article.get('published', ''),
                        'sentiment': article.get('sentiment_score', 0),
                        'source': article.get('source', '')
                    })
        
        # Extract from analyst changes
        if 'analysts' in sources and sources['analysts'].get('recent_changes', 0) > 0:
            events.append({
                'type': 'analyst_update',
                'title': f"Recent analyst recommendation changes",
                'date': datetime.now().isoformat(),
                'sentiment': sources['analysts'].get('average_sentiment', 0),
                'details': sources['analysts'].get('analyst_consensus', '')
            })
        
        # Sort by date and sentiment impact
        events.sort(key=lambda x: abs(x.get('sentiment', 0)), reverse=True)
        
        return events[:10]  # Top 10 events
    
    def _calculate_sentiment_distribution(self, sentiments: List[float]) -> Dict[str, float]:
        """Calculate distribution of sentiments"""
        if not sentiments:
            return {'very_negative': 0, 'negative': 0, 'neutral': 0, 'positive': 0, 'very_positive': 0}
        
        distribution = {
            'very_negative': sum(1 for s in sentiments if s < -0.6) / len(sentiments),
            'negative': sum(1 for s in sentiments if -0.6 <= s < -0.2) / len(sentiments),
            'neutral': sum(1 for s in sentiments if -0.2 <= s <= 0.2) / len(sentiments),
            'positive': sum(1 for s in sentiments if 0.2 < s <= 0.6) / len(sentiments),
            'very_positive': sum(1 for s in sentiments if s > 0.6) / len(sentiments)
        }
        
        return distribution
    
    def _calculate_trending_score(self, volumes: List[int]) -> float:
        """Calculate trending score based on volume patterns"""
        if not volumes:
            return 0.0
        
        # Simple trending score based on volume
        total_volume = sum(volumes)
        avg_volume = total_volume / len(volumes) if volumes else 0
        
        # Normalize to 0-1 scale (assuming 1000 posts/tweets is very high)
        return min(total_volume / 1000, 1.0)
    
    def _get_sector_sentiment(self, sector: str) -> float:
        """Get sentiment for entire sector (placeholder)"""
        # This would analyze sector ETFs or multiple stocks in sector
        # For now, return neutral
        return 0.0
    
    def get_realtime_alerts(self, ticker: str) -> List[Dict[str, Any]]:
        """Get real-time sentiment alerts"""
        current_sentiment = self.analyze_stock_sentiment(ticker, days_back=1)
        
        alerts = []
        
        # Check for extreme sentiment
        if current_sentiment['overall_sentiment'] > 0.7:
            alerts.append({
                'type': 'extreme_positive',
                'message': f"Very positive sentiment detected for {ticker}",
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            })
        elif current_sentiment['overall_sentiment'] < -0.7:
            alerts.append({
                'type': 'extreme_negative',
                'message': f"Very negative sentiment detected for {ticker}",
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for momentum
        if abs(current_sentiment['sentiment_momentum']) > 0.5:
            direction = 'improving' if current_sentiment['sentiment_momentum'] > 0 else 'deteriorating'
            alerts.append({
                'type': 'momentum_change',
                'message': f"Sentiment momentum {direction} rapidly for {ticker}",
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for high social media volume
        social_data = current_sentiment['sources'].get('social', {})
        if social_data.get('trending_score', 0) > 0.7:
            alerts.append({
                'type': 'trending',
                'message': f"{ticker} is trending on social media",
                'severity': 'medium',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts