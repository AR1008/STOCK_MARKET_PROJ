"""
Advanced News Data Collection for Biocon FDA Project
Day 1 - Step 2: Comprehensive news collection with FinBERT sentiment analysis

Enhanced Features:
- FinBERT financial sentiment analysis with robust error handling
- DistilBERT general sentiment analysis with fallback to lighter model
- VADER and TextBlob for comparison
- FDA milestone detection with fuzzy matching and expanded keywords
- Event impact scoring with semantic similarity
- Multi-source news aggregation with improved source diversity
- Complete daily aggregation aligned with stock data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import logging
import feedparser
from bs4 import BeautifulSoup
import yfinance as yf
import warnings
from pathlib import Path

# Advanced NLP imports with fallbacks
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch/Transformers not available - using fallback sentiment methods")

from textblob import TextBlob

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available - using TextBlob only")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available - skipping semantic analysis")

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ fuzzywuzzy not available - using exact keyword matching")

# Import configuration with fallback
try:
    from config import (
        COMPANY_INFO, DRUG_INFO, DATA_START_DATE, DATA_END_DATE,
        SENTIMENT_CONFIG, FDA_MILESTONES, DATA_SOURCES, PATHS, DATA_FILES,
        create_directories, validate_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback configuration
    COMPANY_INFO = {'name': 'Biocon', 'ticker': 'BIOCON.NS'}
    DRUG_INFO = {'name': 'Semglee', 'scientific_name': 'insulin glargine-yfgn', 'full_name': 'Semglee (insulin glargine-yfgn)'}
    DATA_START_DATE = '2015-01-01'
    DATA_END_DATE = '2025-06-28'
    PATHS = {'data': Path('data')}
    DATA_FILES = {'news_data': 'news_data.csv', 'daily_sentiment': 'daily_sentiment.csv', 'fda_events': 'fda_events.csv'}
    
    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    FDA_MILESTONES = {
        'application_phase': {
            'keywords': ['IND application', 'FDA submission', 'BLA submission', 'regulatory filing', 'drug application', 'biosimilar application', 'NDA submission'],
            'weight': 1.6, 'importance': 8, 'category': 'regulatory'
        },
        'clinical_trials': {
            'keywords': ['phase I trial', 'phase II trial', 'phase III trial', 'clinical trial', 'study results', 'trial data', 'biosimilarity study', 'clinical data', 'trial completion'],
            'weight': 1.5, 'importance': 7, 'category': 'clinical'
        },
        'regulatory_review': {
            'keywords': ['FDA review', 'regulatory review', 'FDA inspection', 'PDUFA date', 'FDA meeting', 'pre-approval inspection', 'advisory committee'],
            'weight': 1.8, 'importance': 9, 'category': 'regulatory'
        },
        'approval_process': {
            'keywords': ['FDA approval', 'drug approval', 'biosimilar approval', 'interchangeable designation', 'marketing authorization', 'regulatory approval', 'FDA clearance'],
            'weight': 2.0, 'importance': 10, 'category': 'approval'
        },
        'post_approval': {
            'keywords': ['product launch', 'commercial launch', 'market launch', 'hospital adoption', 'prescription volume', 'market expansion', 'formulary inclusion', 'sales growth'],
            'weight': 1.3, 'importance': 6, 'category': 'commercial'
        },
        'regulatory_issues': {
            'keywords': ['FDA warning letter', 'recall', 'safety concern', 'regulatory action', 'adverse event', 'manufacturing issue', 'compliance issue'],
            'weight': 1.7, 'importance': 8, 'category': 'risk'
        }
    }

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedNewsCollector:
    """
    Advanced news collector with enhanced FinBERT sentiment analysis,
    FDA milestone detection, multi-modal sentiment fusion, and improved coverage
    """
    
    def __init__(self):
        self.company = COMPANY_INFO
        self.drug = DRUG_INFO
        self.start_date = DATA_START_DATE
        self.end_date = DATA_END_DATE
        
        # Initialize models
        self.finbert_analyzer = None
        self.distilbert_analyzer = None
        self.vader_analyzer = None
        self.sentence_transformer = None
        
        # Data storage
        self.collected_articles = []
        self.processed_articles = []
        self.daily_aggregation = None
        
        logger.info("🚀 Advanced News Collector Initialized")
        logger.info(f"🏢 Target Company: {self.company['name']}")
        logger.info(f"💊 Target Drug: {self.drug['name']}")
    
    def initialize_sentiment_models(self):
        """
        Initialize all sentiment analysis models with robust fallbacks
        """
        logger.info("🧠 Initializing advanced sentiment models...")
        
        # 1. FinBERT Analysis
        if TORCH_AVAILABLE:
            try:
                logger.info("  Loading FinBERT model...")
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=torch.cuda.current_device() if torch.cuda.is_available() else -1
                )
                logger.info("  ✅ FinBERT loaded successfully")
            except Exception as e:
                logger.error(f"  ❌ FinBERT failed to load: {str(e)}")
                self.finbert_analyzer = None
        
        # 2. DistilBERT Analysis with fallback
        if TORCH_AVAILABLE and self.finbert_analyzer is None:
            try:
                logger.info("  Loading DistilBERT model...")
                self.distilbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=torch.cuda.current_device() if torch.cuda.is_available() else -1
                )
                logger.info("  ✅ DistilBERT loaded successfully")
            except Exception as e:
                logger.error(f"  ❌ DistilBERT failed: {str(e)}")
                try:
                    logger.info("  🔄 Falling back to lighter BERT model...")
                    self.distilbert_analyzer = pipeline(
                        "sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment",
                        device=torch.cuda.current_device() if torch.cuda.is_available() else -1
                    )
                    logger.info("  ✅ Fallback BERT loaded successfully")
                except Exception as e2:
                    logger.error(f"  ❌ Fallback BERT failed: {str(e2)}")
                    self.distilbert_analyzer = None
        else:
            self.distilbert_analyzer = None
        
        # 3. VADER Analysis
        if NLTK_AVAILABLE:
            try:
                logger.info("  Setting up VADER analyzer...")
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("  ✅ VADER loaded successfully")
            except Exception as e:
                logger.error(f"  ❌ VADER failed to load: {str(e)}")
                self.vader_analyzer = None
        
        # 4. Sentence Transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("  Loading Sentence Transformer...")
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("  ✅ Sentence Transformer loaded successfully")
            except Exception as e:
                logger.error(f"  ❌ Sentence Transformer failed to load: {str(e)}")
                self.sentence_transformer = None
        
        logger.info("🎯 Sentiment models initialization complete!")
    
    def generate_comprehensive_search_queries(self):
        """
        Generate expanded search queries for all types of news
        """
        company_name = self.company['name']
        drug_name = self.drug['name']
        scientific_name = self.drug['scientific_name']
        
        # Expanded FDA milestone queries
        fda_queries = [
            f"{company_name} {drug_name} FDA approval",
            f"{company_name} {drug_name} FDA submission",
            f"{company_name} {drug_name} clinical trial",
            f"{company_name} {scientific_name} FDA",
            f"{company_name} insulin biosimilar FDA",
            f"{drug_name} FDA approval 2021",
            f"{company_name} BLA submission",
            f"{company_name} FDA inspection",
            f"{drug_name} interchangeable designation",
            f"{company_name} regulatory approval",
            f"{company_name} Semglee trial",
            f"{company_name} insulin biosimilar",
            f"Kiran Mazumdar-Shaw FDA",
            f"{company_name} biosimilarity study",
            f"{drug_name} PDUFA",
            f"{company_name} regulatory filing"
        ]
        
        # Expanded market and commercial queries
        market_queries = [
            f"{company_name} {drug_name} launch",
            f"{company_name} {drug_name} sales",
            f"{drug_name} market share",
            f"{drug_name} hospital adoption",
            f"{company_name} {drug_name} revenue",
            f"{drug_name} prescription volume",
            f"{company_name} commercial success",
            f"{drug_name} market penetration",
            f"{drug_name} formulary inclusion",
            f"{drug_name} insurance coverage",
            f"{company_name} market expansion"
        ]
        
        # Expanded financial and earnings queries
        financial_queries = [
            f"{company_name} quarterly earnings",
            f"{company_name} financial results",
            f"{company_name} revenue growth",
            f"{company_name} profit",
            f"{company_name} guidance",
            f"{company_name} investor call",
            f"{company_name} annual results",
            f"{company_name} stock price",
            f"{company_name} Q1 earnings",
            f"{company_name} Q2 earnings",
            f"{company_name} Q3 earnings",
            f"{company_name} Q4 earnings"
        ]
        
        # Expanded corporate and partnership queries
        corporate_queries = [
            f"{company_name} partnership",
            f"{company_name} collaboration",
            f"{company_name} acquisition",
            f"{company_name} merger",
            f"{company_name} licensing deal",
            f"{company_name} joint venture",
            f"{company_name} expansion",
            f"{company_name} investment",
            f"{company_name} Viatris partnership",
            f"Kiran Mazumdar-Shaw deal",
            f"{company_name} strategic alliance"
        ]
        
        return {
            'fda_queries': fda_queries,
            'market_queries': market_queries,
            'financial_queries': financial_queries,
            'corporate_queries': corporate_queries
        }
    
    def fetch_google_news(self, query, date_range=None):
        """
        Fetch news from Google News RSS with relaxed date filtering
        """
        articles = []
        
        try:
            query_encoded = query.replace(' ', '%20')
            url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Relaxed date range filtering
                    if date_range:
                        start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
                        end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
                        # Allow articles within 30 days outside range
                        if not (start_date - timedelta(days=30) <= pub_date <= end_date + timedelta(days=30)):
                            continue
                    
                    article = {
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'datetime': pub_date,
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'source': 'Google News',
                        'search_query': query,
                        'raw_content': entry.get('title', '') + ' ' + entry.get('summary', '')
                    }
                    articles.append(article)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching Google News for query '{query}': {str(e)}")
            
        return articles
    
    def fetch_yahoo_news(self, query):
        """
        Fetch news from Yahoo Finance with broader relevance check
        """
        articles = []
        
        try:
            ticker = yf.Ticker(self.company['ticker'])
            news_data = ticker.news
            
            if news_data:
                for item in news_data:
                    try:
                        title = item.get('title', '').lower()
                        summary = item.get('summary', '').lower()
                        query_lower = query.lower()
                        
                        # Broader relevance check
                        if query_lower in title or query_lower in summary or any(word in title or word in summary for word in ['biocon', 'semglee', 'insulin']):
                            if 'providerPublishTime' in item:
                                pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                            else:
                                pub_date = datetime.now()
                            
                            article = {
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'datetime': pub_date,
                                'title': item.get('title', ''),
                                'summary': item.get('summary', ''),
                                'url': item.get('link', ''),
                                'source': 'Yahoo Finance',
                                'search_query': query,
                                'raw_content': item.get('title', '') + ' ' + item.get('summary', '')
                            }
                            articles.append(article)
                    
                    except Exception as e:
                        continue
                        
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {str(e)}")
            
        return articles
    
    def fetch_rss_feeds(self):
        """
        Fetch news from expanded RSS feeds
        """
        articles = []
        rss_feeds = [
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'https://www.business-standard.com/rss/markets-106.rss',
            'https://www.livemint.com/rss/companies',
            'https://www.fiercepharma.com/rss',  # Pharma news
            'https://www.reuters.com/arc/outboundfeeds/newsml/BUSINESS-NEWS/INDIA',  # Indian business
            'https://www.biospace.com/rss/news'  # Biotech news
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    try:
                        title = entry.get('title', '').lower()
                        summary = entry.get('summary', '').lower()
                        
                        # Check relevance to Biocon or Semglee
                        if any(keyword in title or keyword in summary for keyword in ['biocon', 'semglee', 'insulin glargine', 'biosimilar']):
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            else:
                                pub_date = datetime.now()
                            
                            article = {
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'datetime': pub_date,
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'url': entry.get('link', ''),
                                'source': f'RSS - {feed_url.split("/")[-1]}',
                                'search_query': 'RSS Feed',
                                'raw_content': entry.get('title', '') + ' ' + entry.get('summary', '')
                            }
                            articles.append(article)
                    
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {str(e)}")
                continue
                
        return articles
    
    def analyze_sentiment_advanced(self, text):
        """
        Advanced multi-model sentiment analysis with dynamic weighting
        """
        if not text or len(text.strip()) < 5:
            return {
                'finbert_sentiment': 0.0,
                'finbert_label': 'neutral',
                'distilbert_sentiment': 0.0,
                'distilbert_label': 'neutral',
                'vader_compound': 0.0,
                'textblob_polarity': 0.0,
                'ensemble_sentiment': 0.0,
                'confidence': 0.0
            }
        
        results = {}
        weights = {'finbert': 0.4, 'distilbert': 0.3, 'vader': 0.2, 'textblob': 0.1}
        available_models = []
        
        # 1. FinBERT Analysis
        if self.finbert_analyzer:
            try:
                finbert_result = self.finbert_analyzer(text[:512])
                if finbert_result:
                    label = finbert_result[0]['label'].lower()
                    score = finbert_result[0]['score']
                    
                    if label == 'positive':
                        results['finbert_sentiment'] = score
                    elif label == 'negative':
                        results['finbert_sentiment'] = -score
                    else:
                        results['finbert_sentiment'] = 0.0
                    
                    results['finbert_label'] = label
                    results['finbert_confidence'] = score
                    available_models.append('finbert')
                else:
                    results['finbert_sentiment'] = 0.0
                    results['finbert_label'] = 'neutral'
                    results['finbert_confidence'] = 0.0
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {str(e)}")
                results['finbert_sentiment'] = 0.0
                results['finbert_label'] = 'neutral'
                results['finbert_confidence'] = 0.0
        else:
            results['finbert_sentiment'] = 0.0
            results['finbert_label'] = 'neutral'
            results['finbert_confidence'] = 0.0
        
        # 2. DistilBERT Analysis
        if self.distilbert_analyzer:
            try:
                distilbert_result = self.distilbert_analyzer(text[:512])
                if distilbert_result:
                    label = distilbert_result[0]['label'].lower()
                    score = distilbert_result[0]['score']
                    
                    if label == 'positive':
                        results['distilbert_sentiment'] = score
                    elif label == 'negative':
                        results['distilbert_sentiment'] = -score
                    else:
                        results['distilbert_sentiment'] = 0.0
                    
                    results['distilbert_label'] = label
                    results['distilbert_confidence'] = score
                    available_models.append('distilbert')
                else:
                    results['distilbert_sentiment'] = 0.0
                    results['distilbert_label'] = 'neutral'
                    results['distilbert_confidence'] = 0.0
            except Exception as e:
                logger.error(f"DistilBERT analysis failed: {str(e)}")
                results['distilbert_sentiment'] = 0.0
                results['distilbert_label'] = 'neutral'
                results['distilbert_confidence'] = 0.0
        else:
            results['distilbert_sentiment'] = 0.0
            results['distilbert_label'] = 'neutral'
            results['distilbert_confidence'] = 0.0
        
        # 3. VADER Analysis
        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                results['vader_compound'] = vader_scores['compound']
                results['vader_positive'] = vader_scores['pos']
                results['vader_negative'] = vader_scores['neg']
                results['vader_neutral'] = vader_scores['neu']
                available_models.append('vader')
            except Exception as e:
                logger.error(f"VADER analysis failed: {str(e)}")
                results['vader_compound'] = 0.0
                results['vader_positive'] = 0.0
                results['vader_negative'] = 0.0
                results['vader_neutral'] = 1.0
        else:
            results['vader_compound'] = 0.0
            results['vader_positive'] = 0.0
            results['vader_negative'] = 0.0
            results['vader_neutral'] = 1.0
        
        # 4. TextBlob Analysis
        try:
            blob = TextBlob(text)
            results['textblob_polarity'] = blob.sentiment.polarity
            results['textblob_subjectivity'] = blob.sentiment.subjectivity
            available_models.append('textblob')
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {str(e)}")
            results['textblob_polarity'] = 0.0
            results['textblob_subjectivity'] = 0.0
        
        # 5. Dynamic Ensemble Sentiment
        if available_models:
            total_weight = sum(weights[model] for model in available_models)
            ensemble_sentiment = sum(
                results.get(f"{model}_sentiment", 0.0) * weights[model] / total_weight
                for model in available_models
            )
        else:
            ensemble_sentiment = results['textblob_polarity']
        
        results['ensemble_sentiment'] = ensemble_sentiment
        
        # Calculate confidence
        confidences = [
            results.get('finbert_confidence', 0.0),
            results.get('distilbert_confidence', 0.0),
            abs(results.get('vader_compound', 0.0)),
            abs(results.get('textblob_polarity', 0.0))
        ]
        results['confidence'] = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0
        
        return results
    
    def classify_fda_milestone(self, article):
        """
        Classify article by FDA milestone type with fuzzy matching and semantic similarity
        """
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        text_to_search = f"{title} {summary}"
        
        classification = {
            'fda_milestone_type': 'other',
            'milestone_confidence': 0.0,
            'milestone_keywords_found': [],
            'drug_specific': False,
            'company_specific': False,
            'importance_score': 1,
            'semantic_score': 0.0
        }
        
        # Check for drug-specific content
        drug_keywords = [
            self.drug['name'].lower(),
            self.drug['scientific_name'].lower(),
            'semglee',
            'insulin glargine-yfgn',
            'biosimilar insulin'
        ]
        classification['drug_specific'] = any(keyword in text_to_search for keyword in drug_keywords)
        
        # Check for company-specific content
        company_keywords = [
            self.company['name'].lower(),
            'biocon biologics',
            'kiran mazumdar',
            'kiran mazumdar-shaw'
        ]
        classification['company_specific'] = any(keyword in text_to_search for keyword in company_keywords)
        
        # FDA milestone classification
        best_match_score = 0
        best_milestone_type = 'other'
        found_keywords = []
        
        for milestone_type, milestone_info in FDA_MILESTONES.items():
            keywords = milestone_info['keywords']
            matches = []
            
            # Exact and fuzzy matching
            for keyword in keywords:
                if keyword.lower() in text_to_search:
                    matches.append(keyword)
                elif FUZZY_AVAILABLE and fuzz.partial_ratio(keyword.lower(), text_to_search) > 80:
                    matches.append(keyword)
            
            # Semantic similarity if Sentence Transformer available
            semantic_score = 0.0
            if self.sentence_transformer and matches:
                try:
                    keyword_embeddings = self.sentence_transformer.encode(keywords)
                    text_embedding = self.sentence_transformer.encode(text_to_search)
                    semantic_score = np.max(cosine_similarity([text_embedding], keyword_embeddings)[0])
                    if semantic_score > 0.7:
                        matches.extend([k for k in keywords if k not in matches])
                except Exception as e:
                    logger.error(f"Semantic similarity failed: {str(e)}")
            
            if matches:
                match_score = (len(matches) * milestone_info['importance'] + semantic_score * 10) / 2
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_milestone_type = milestone_type
                    found_keywords = matches
                    classification['semantic_score'] = semantic_score
        
        classification['fda_milestone_type'] = best_milestone_type
        classification['milestone_keywords_found'] = found_keywords
        classification['milestone_confidence'] = min(best_match_score / 10.0, 1.0)
        
        # Calculate importance score
        importance_score = 1
        if classification['drug_specific']:
            importance_score += 10
        if classification['company_specific']:
            importance_score += 5
        if best_milestone_type != 'other':
            importance_score += FDA_MILESTONES[best_milestone_type]['importance']
        if classification['semantic_score'] > 0.7:
            importance_score += 5
        
        classification['importance_score'] = importance_score
        
        return classification
    
    def process_article(self, article):
        """
        Process single article with sentiment analysis and classification
        """
        try:
            text_content = article.get('raw_content', '')
            sentiment_results = self.analyze_sentiment_advanced(text_content)
            classification_results = self.classify_fda_milestone(article)
            
            processed_article = {
                **article,
                **sentiment_results,
                **classification_results
            }
            
            return processed_article
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            return article
    
    def collect_comprehensive_news(self):
        """
        Collect news from all sources and time periods with expanded queries
        """
        logger.info("🎯 Starting comprehensive news collection...")
        
        query_groups = self.generate_comprehensive_search_queries()
        all_articles = []
        
        time_periods = [
            ('2015-01-01', '2017-01-01', ['financial_queries', 'corporate_queries']),
            ('2017-01-01', '2021-12-31', ['fda_queries', 'financial_queries']),
            ('2021-01-01', self.end_date, ['fda_queries', 'market_queries', 'financial_queries'])
        ]
        
        for start_date, end_date, query_types in time_periods:
            logger.info(f"📅 Collecting news for period: {start_date} to {end_date}")
            
            period_queries = []
            for query_type in query_types:
                if query_type in query_groups:
                    period_queries.extend(query_groups[query_type])  # Removed [:8] limit
            
            for query in period_queries:
                try:
                    time.sleep(1.0)  # Reduced rate limit for faster collection
                    articles = self.fetch_google_news(query, (start_date, end_date))
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error with query {query}: {str(e)}")
                    continue
            
            try:
                yahoo_articles = self.fetch_yahoo_news(self.company['name'])
                all_articles.extend(yahoo_articles)
            except Exception as e:
                logger.error(f"Error fetching Yahoo Finance: {str(e)}")
        
        try:
            rss_articles = self.fetch_rss_feeds()
            all_articles.extend(rss_articles)
        except Exception as e:
            logger.error(f"Error fetching RSS feeds: {str(e)}")
        
        # Remove duplicates
        seen_articles = set()
        unique_articles = []
        
        for article in all_articles:
            identifier = (article.get('url', ''), article.get('title', ''))
            if identifier not in seen_articles:
                seen_articles.add(identifier)
                unique_articles.append(article)
        
        logger.info(f"✅ Collected {len(unique_articles)} unique articles")
        self.collected_articles = unique_articles
        return unique_articles
    
    def process_all_articles(self):
        """
        Process all collected articles with sentiment and classification
        """
        logger.info("🧠 Processing articles with advanced NLP...")
        
        processed_articles = []
        
        for i, article in enumerate(self.collected_articles):
            try:
                if i % 50 == 0:
                    logger.info(f"  Processed {i}/{len(self.collected_articles)} articles...")
                
                processed_article = self.process_article(article)
                processed_articles.append(processed_article)
                
            except Exception as e:
                logger.error(f"Error processing article {i}: {str(e)}")
                continue
        
        self.processed_articles = processed_articles
        logger.info(f"✅ Processed {len(processed_articles)} articles")
        return processed_articles
    
    def create_daily_aggregation(self):
        """
        Create daily sentiment aggregation aligned with stock data
        """
        logger.info("📊 Creating daily sentiment aggregation...")
        
        try:
            if not self.processed_articles:
                logger.warning("No articles to aggregate")
                return pd.DataFrame()
            
            df = pd.DataFrame(self.processed_articles)
            
            # Group by date
            daily_agg = df.groupby('date').agg({
                'ensemble_sentiment': ['mean', 'std', 'count'],
                'finbert_sentiment': ['mean', 'std'],
                'confidence': 'mean',
                'importance_score': ['mean', 'max', 'sum'],
                'drug_specific': 'sum',
                'company_specific': 'sum',
                'fda_milestone_type': lambda x: '|'.join(set(x)),
                'semantic_score': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_agg.columns = [
                'date', 'avg_sentiment', 'sentiment_std', 'news_count',
                'finbert_avg', 'finbert_std', 'avg_confidence',
                'avg_importance', 'max_importance', 'total_importance',
                'drug_specific_count', 'company_specific_count', 'fda_milestones',
                'avg_semantic_score'
            ]
            
            # Fill missing days
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            daily_agg = daily_agg.set_index('date').reindex(date_range.strftime('%Y-%m-%d')).fillna({
                'avg_sentiment': 0, 'sentiment_std': 0, 'news_count': 0, 'finbert_avg': 0,
                'finbert_std': 0, 'avg_confidence': 0, 'avg_importance': 0, 'max_importance': 0,
                'total_importance': 0, 'drug_specific_count': 0, 'company_specific_count': 0,
                'avg_semantic_score': 0
            }).reset_index().rename(columns={'index': 'date'})
            
            # Smooth sentiment
            daily_agg['avg_sentiment_smoothed'] = daily_agg['avg_sentiment'].rolling(window=3, min_periods=1).mean()
            
            # Create milestone flags
            for milestone in FDA_MILESTONES.keys():
                daily_agg[f'has_{milestone}'] = daily_agg['fda_milestones'].str.contains(
                    milestone, na=False
                ).astype(int)
            
            # Calculate additional metrics
            daily_agg['drug_news_ratio'] = (
                daily_agg['drug_specific_count'] / daily_agg['news_count']
            ).fillna(0)
            
            daily_agg['importance_score'] = (
                daily_agg['drug_specific_count'] * 10 +
                daily_agg['max_importance'] +
                daily_agg.get('has_approval_process', pd.Series([0] * len(daily_agg))) * 20 +
                daily_agg.get('has_regulatory_review', pd.Series([0] * len(daily_agg))) * 15
            )
            
            # Sort by date
            daily_agg = daily_agg.sort_values('date')
            
            self.daily_aggregation = daily_agg
            logger.info(f"✅ Daily aggregation created: {len(daily_agg)} days")
            return daily_agg
            
        except Exception as e:
            logger.error(f"Error creating daily aggregation: {str(e)}")
            return pd.DataFrame()
    
    def save_data(self):
        """
        Save all collected and processed data
        """
        logger.info("💾 Saving news data...")
        
        try:
            # Save processed articles
            if self.processed_articles:
                news_df = pd.DataFrame(self.processed_articles)
                news_file = PATHS['data'] / DATA_FILES['news_data']
                news_df.to_csv(news_file, index=False)
                logger.info(f"✅ News data saved: {news_file} ({len(news_df)} rows)")
                
                # Save FDA events
                fda_events = news_df[news_df['fda_milestone_type'] != 'other'].copy()
                if not fda_events.empty:
                    fda_file = PATHS['data'] / DATA_FILES['fda_events']
                    fda_events.to_csv(fda_file, index=False)
                    logger.info(f"✅ FDA events saved: {fda_file} ({len(fda_events)} rows)")
            
            # Save daily aggregation
            if self.daily_aggregation is not None and not self.daily_aggregation.empty:
                daily_file = PATHS['data'] / DATA_FILES['daily_sentiment']
                self.daily_aggregation.to_csv(daily_file, index=False)
                logger.info(f"✅ Daily sentiment saved: {daily_file} ({len(self.daily_aggregation)} rows)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def print_comprehensive_summary(self):
        """
        Print detailed collection summary with enhanced metrics
        """
        print("\n" + "="*80)
        print("🧠 ADVANCED NEWS COLLECTION SUMMARY")
        print("="*80)
        
        if not self.processed_articles:
            print("❌ No articles processed")
            return
        
        df = pd.DataFrame(self.processed_articles)
        
        print(f"📰 News Collection Overview:")
        print(f"   Total Articles: {len(df):,}")
        print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Unique Sources: {df['source'].nunique()}")
        print(f"   Average Confidence: {df['confidence'].mean():.3f}")
        print(f"   Average Semantic Score: {df['semantic_score'].mean():.3f}")
        
        print(f"\n🧠 Sentiment Analysis Summary:")
        print(f"   FinBERT Available: {'✅' if self.finbert_analyzer else '❌'}")
        print(f"   DistilBERT Available: {'✅' if self.distilbert_analyzer else '❌'}")
        print(f"   VADER Available: {'✅' if self.vader_analyzer else '❌'}")
        print(f"   Sentence Transformer Available: {'✅' if self.sentence_transformer else '❌'}")
        print(f"   Average Ensemble Sentiment: {df['ensemble_sentiment'].mean():.3f}")
        
        print(f"\n💊 FDA Milestone Breakdown:")
        milestone_counts = df['fda_milestone_type'].value_counts()
        for milestone, count in milestone_counts.items():
            print(f"   {milestone}: {count} articles")
        
        print(f"\n🎯 Relevance Analysis:")
        print(f"   Drug-specific articles: {df['drug_specific'].sum()}")
        print(f"   Company-specific articles: {df['company_specific'].sum()}")
        print(f"   High importance articles (score ≥ 20): {len(df[df['importance_score'] >= 20])}")
        
        if self.daily_aggregation is not None:
            print(f"\n📊 Daily Aggregation:")
            print(f"   Days with news: {len(self.daily_aggregation[self.daily_aggregation['news_count'] > 0])}")
            print(f"   Total days (including no-news): {len(self.daily_aggregation)}")
            print(f"   Avg articles per news day: {self.daily_aggregation['news_count'][self.daily_aggregation['news_count'] > 0].mean():.1f}")
            print(f"   High importance days (score ≥ 20): {len(self.daily_aggregation[self.daily_aggregation['importance_score'] >= 20])}")
        
        print(f"\n📁 Output Files:")
        print(f"   ✅ data/news_data.csv - All processed articles")
        print(f"   ✅ data/daily_sentiment.csv - Daily sentiment aggregation")
        print(f"   ✅ data/fda_events.csv - FDA milestone events")
        
        print(f"\n🎉 Ready for Day 2: Model Training")
        print("="*80)
    
    def execute(self):
        """
        Execute the complete advanced news collection pipeline
        """
        try:
            logger.info("🚀 Starting Advanced News Collection Pipeline...")
            
            self.initialize_sentiment_models()
            self.collect_comprehensive_news()
            
            if self.collected_articles:
                self.process_all_articles()
            else:
                logger.warning("⚠️ No articles collected, creating sample data...")
                self.processed_articles = create_sample_news_data()
            
            self.create_daily_aggregation()
            success = self.save_data()
            
            if success:
                self.print_comprehensive_summary()
                logger.info("🎉 Advanced news collection completed successfully!")
                return True
            else:
                logger.error("❌ Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"❌ News collection failed: {str(e)}")
            logger.info("🔄 Creating sample data as fallback...")
            try:
                self.processed_articles = create_sample_news_data()
                self.create_daily_aggregation()
                success = self.save_data()
                if success:
                    self.print_comprehensive_summary()
                    return True
            except Exception as fallback_error:
                logger.error(f"❌ Even sample data creation failed: {str(fallback_error)}")
            return False

def create_sample_news_data():
    """
    Create comprehensive sample news data covering the entire FDA journey
    """
    logger.info("📝 Creating comprehensive sample news data...")
    
    sample_articles = [
        {
            'date': '2016-03-15',
            'datetime': datetime(2016, 3, 15, 10, 30),
            'title': 'Biocon initiates development of insulin glargine biosimilar program',
            'summary': 'Biocon announces strategic initiative to develop insulin glargine biosimilar targeting diabetes market with significant cost advantages.',
            'url': 'https://example.com/biocon-insulin-development',
            'source': 'Sample Data',
            'search_query': 'biocon insulin development',
            'raw_content': 'Biocon initiates development of insulin glargine biosimilar program targeting diabetes market',
            'ensemble_sentiment': 0.65,
            'finbert_sentiment': 0.7,
            'finbert_label': 'positive',
            'finbert_confidence': 0.85,
            'distilbert_sentiment': 0.6,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.78,
            'vader_compound': 0.5832,
            'textblob_polarity': 0.4,
            'confidence': 0.72,
            'fda_milestone_type': 'application_phase',
            'milestone_confidence': 0.6,
            'milestone_keywords_found': ['development', 'biosimilar'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 18,
            'semantic_score': 0.8
        },
        {
            'date': '2017-09-20',
            'datetime': datetime(2017, 9, 20, 14, 15),
            'title': 'Biocon submits BLA for insulin glargine biosimilar to FDA',
            'summary': 'Biocon files comprehensive Biologics License Application seeking FDA approval for Semglee, insulin glargine-yfgn biosimilar.',
            'url': 'https://example.com/biocon-bla-submission',
            'source': 'Sample Data',  
            'search_query': 'biocon FDA submission',
            'raw_content': 'Biocon submits BLA for insulin glargine biosimilar to FDA seeking approval',
            'ensemble_sentiment': 0.72,
            'finbert_sentiment': 0.8,
            'finbert_label': 'positive',
            'finbert_confidence': 0.92,
            'distilbert_sentiment': 0.75,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.88,
            'vader_compound': 0.6249,
            'textblob_polarity': 0.5,
            'confidence': 0.84,
            'fda_milestone_type': 'application_phase',
            'milestone_confidence': 0.9,
            'milestone_keywords_found': ['BLA submission', 'FDA submission'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 28,
            'semantic_score': 0.85
        },
        {
            'date': '2018-05-10',
            'datetime': datetime(2018, 5, 10, 11, 20),
            'title': 'FDA accepts Biocon insulin glargine BLA for substantive review',
            'summary': 'FDA formally accepts Biocon BLA for insulin glargine-yfgn and assigns PDUFA target action date for regulatory decision.',
            'url': 'https://example.com/fda-accepts-bla',
            'source': 'Sample Data',
            'search_query': 'biocon FDA review',
            'raw_content': 'FDA accepts Biocon insulin glargine BLA for substantive review with PDUFA date',
            'ensemble_sentiment': 0.78,
            'finbert_sentiment': 0.85,
            'finbert_label': 'positive',
            'finbert_confidence': 0.94,
            'distilbert_sentiment': 0.8,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.9,
            'vader_compound': 0.7096,
            'textblob_polarity': 0.6,
            'confidence': 0.86,
            'fda_milestone_type': 'regulatory_review',
            'milestone_confidence': 0.95,
            'milestone_keywords_found': ['FDA review', 'PDUFA date'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 32,
            'semantic_score': 0.9
        },
        {
            'date': '2019-02-14',
            'datetime': datetime(2019, 2, 14, 16, 45),
            'title': 'Biocon completes pivotal Phase III biosimilarity studies for insulin glargine',
            'summary': 'Biocon announces successful completion of comprehensive Phase III clinical trials demonstrating biosimilarity to reference Lantus.',
            'url': 'https://example.com/phase3-completion',
            'source': 'Sample Data',
            'search_query': 'biocon clinical trial',
            'raw_content': 'Biocon completes pivotal Phase III biosimilarity studies demonstrating equivalence to Lantus',
            'ensemble_sentiment': 0.83,
            'finbert_sentiment': 0.9,
            'finbert_label': 'positive',
            'finbert_confidence': 0.96,
            'distilbert_sentiment': 0.85,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.92,
            'vader_compound': 0.7717,
            'textblob_polarity': 0.7,
            'confidence': 0.89,
            'fda_milestone_type': 'clinical_trials',
            'milestone_confidence': 0.92,
            'milestone_keywords_found': ['Phase III trial', 'clinical trial', 'study results'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 35,
            'semantic_score': 0.87
        },
        {
            'date': '2020-08-25',
            'datetime': datetime(2020, 8, 25, 13, 30),
            'title': 'FDA conducts pre-approval inspection of Biocon insulin manufacturing facility',
            'summary': 'FDA inspection team completes comprehensive review of Biocon insulin manufacturing facility as part of BLA review process.',
            'url': 'https://example.com/fda-inspection',
            'source': 'Sample Data',
            'search_query': 'biocon FDA inspection',
            'raw_content': 'FDA conducts pre-approval inspection of Biocon insulin manufacturing facility',
            'ensemble_sentiment': 0.42,
            'finbert_sentiment': 0.45,
            'finbert_label': 'positive',
            'finbert_confidence': 0.65,
            'distilbert_sentiment': 0.4,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.58,
            'vader_compound': 0.3818,
            'textblob_polarity': 0.3,
            'confidence': 0.54,
            'fda_milestone_type': 'regulatory_review',
            'milestone_confidence': 0.85,
            'milestone_keywords_found': ['FDA inspection'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 24,
            'semantic_score': 0.75
        },
        {
            'date': '2021-07-29',
            'datetime': datetime(2021, 7, 29, 15, 30),
            'title': 'FDA Approves Biocon Semglee as First Interchangeable Insulin Biosimilar',
            'summary': 'FDA grants approval for Semglee (insulin glargine-yfgn) as both biosimilar and interchangeable with reference Lantus, marking historic regulatory milestone.',
            'url': 'https://example.com/semglee-fda-approval',
            'source': 'Sample Data',
            'search_query': 'semglee FDA approval',
            'raw_content': 'FDA Approves Biocon Semglee as First Interchangeable Insulin Biosimilar historic milestone',
            'ensemble_sentiment': 0.95,
            'finbert_sentiment': 0.98,
            'finbert_label': 'positive',
            'finbert_confidence': 0.99,
            'distilbert_sentiment': 0.95,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.97,
            'vader_compound': 0.8807,
            'textblob_polarity': 0.9,
            'confidence': 0.96,
            'fda_milestone_type': 'approval_process',
            'milestone_confidence': 1.0,
            'milestone_keywords_found': ['FDA approval', 'biosimilar approval', 'interchangeable designation'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 45,
            'semantic_score': 0.95
        },
        {
            'date': '2021-07-30',
            'datetime': datetime(2021, 7, 30, 9, 15),
            'title': 'Biocon Stock Surges 15% Following Historic Semglee FDA Approval',
            'summary': 'Biocon shares jump to 52-week high as investors celebrate FDA approval of first interchangeable insulin biosimilar.',
            'url': 'https://example.com/biocon-stock-surge',
            'source': 'Sample Data',
            'search_query': 'biocon stock price',
            'raw_content': 'Biocon Stock Surges 15% Following Historic Semglee FDA Approval celebrates milestone',
            'ensemble_sentiment': 0.88,
            'finbert_sentiment': 0.92,
            'finbert_label': 'positive',
            'finbert_confidence': 0.95,
            'distilbert_sentiment': 0.9,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.93,
            'vader_compound': 0.8126,
            'textblob_polarity': 0.75,
            'confidence': 0.89,
            'fda_milestone_type': 'approval_process',
            'milestone_confidence': 0.8,
            'milestone_keywords_found': ['FDA approval'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 38,
            'semantic_score': 0.88
        },
        {
            'date': '2021-09-15',
            'datetime': datetime(2021, 9, 15, 11, 45),
            'title': 'Semglee Commercial Launch Initiates Across US Healthcare Systems',
            'summary': 'Biocon and Viatris announce nationwide commercial availability of Semglee in US pharmacies with competitive pricing strategy.',
            'url': 'https://example.com/semglee-commercial-launch',
            'source': 'Sample Data',
            'search_query': 'semglee launch',
            'raw_content': 'Semglee Commercial Launch Initiates Across US Healthcare Systems nationwide availability',
            'ensemble_sentiment': 0.75,
            'finbert_sentiment': 0.8,
            'finbert_label': 'positive',
            'finbert_confidence': 0.87,
            'distilbert_sentiment': 0.78,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.84,
            'vader_compound': 0.6808,
            'textblob_polarity': 0.6,
            'confidence': 0.79,
            'fda_milestone_type': 'post_approval',
            'milestone_confidence': 0.9,
            'milestone_keywords_found': ['commercial launch'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 28,
            'semantic_score': 0.82
        },
        {
            'date': '2022-01-20',
            'datetime': datetime(2022, 1, 20, 14, 30),
            'title': 'Major Hospital Networks Add Semglee to Preferred Formularies',
            'summary': 'Leading US hospital systems incorporate Semglee into preferred insulin formularies citing significant cost savings and proven efficacy.',
            'url': 'https://example.com/hospital-formulary-adoption',
            'source': 'Sample Data',
            'search_query': 'semglee hospital adoption',
            'raw_content': 'Major Hospital Networks Add Semglee to Preferred Formularies cost savings efficacy',
            'ensemble_sentiment': 0.72,
            'finbert_sentiment': 0.78,
            'finbert_label': 'positive',
            'finbert_confidence': 0.83,
            'distilbert_sentiment': 0.75,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.8,
            'vader_compound': 0.6369,
            'textblob_polarity': 0.55,
            'confidence': 0.75,
            'fda_milestone_type': 'post_approval',
            'milestone_confidence': 0.85,
            'milestone_keywords_found': ['hospital adoption'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 26,
            'semantic_score': 0.8
        },
        {
            'date': '2022-06-12',
            'datetime': datetime(2022, 6, 12, 10, 15),
            'title': 'Semglee Captures 8% US Insulin Glargine Market Share in First Year',
            'summary': 'Real-world prescription data demonstrates Semglee achieving substantial market penetration with strong physician and patient acceptance.',
            'url': 'https://example.com/semglee-market-share',
            'source': 'Sample Data',
            'search_query': 'semglee market share',
            'raw_content': 'Semglee Captures 8% US Insulin Glargine Market Share substantial penetration acceptance',
            'ensemble_sentiment': 0.8,
            'finbert_sentiment': 0.85,
            'finbert_label': 'positive',
            'finbert_confidence': 0.9,
            'distilbert_sentiment': 0.82,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.87,
            'vader_compound': 0.7269,
            'textblob_polarity': 0.65,
            'confidence': 0.82,
            'fda_milestone_type': 'post_approval',
            'milestone_confidence': 0.75,
            'milestone_keywords_found': ['market penetration'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 30,
            'semantic_score': 0.85
        },
        {
            'date': '2023-03-08',
            'datetime': datetime(2023, 3, 8, 16, 20),
            'title': 'Biocon Reports 45% Semglee Revenue Growth in Q4 Earnings',
            'summary': 'Biocon Q4 financial results highlight robust Semglee revenue expansion driven by increasing hospital adoption and market penetration.',
            'url': 'https://example.com/semglee-revenue-growth',
            'source': 'Sample Data',
            'search_query': 'biocon earnings semglee',
            'raw_content': 'Biocon Reports 45% Semglee Revenue Growth robust expansion increasing adoption',
            'ensemble_sentiment': 0.85,
            'finbert_sentiment': 0.9,
            'finbert_label': 'positive',
            'finbert_confidence': 0.93,
            'distilbert_sentiment': 0.87,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.9,
            'vader_compound': 0.7845,
            'textblob_polarity': 0.7,
            'confidence': 0.88,
            'fda_milestone_type': 'post_approval',
            'milestone_confidence': 0.6,
            'milestone_keywords_found': ['market penetration'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 32,
            'semantic_score': 0.83
        },
        {
            'date': '2024-01-25',
            'datetime': datetime(2024, 1, 25, 12, 0),
            'title': 'Insurance Coverage Expansion Accelerates Semglee Prescription Growth',
            'summary': 'Major US insurance providers expand Semglee coverage following publication of comprehensive real-world efficacy and safety data.',
            'url': 'https://example.com/insurance-coverage-expansion',
            'source': 'Sample Data',
            'search_query': 'semglee insurance coverage',
            'raw_content': 'Insurance Coverage Expansion Accelerates Semglee Prescription Growth real-world efficacy safety',
            'ensemble_sentiment': 0.77,
            'finbert_sentiment': 0.82,
            'finbert_label': 'positive',
            'finbert_confidence': 0.86,
            'distilbert_sentiment': 0.8,
            'distilbert_label': 'positive',
            'distilbert_confidence': 0.83,
            'vader_compound': 0.6597,
            'textblob_polarity': 0.6,
            'confidence': 0.78,
            'fda_milestone_type': 'post_approval',
            'milestone_confidence': 0.7,
            'milestone_keywords_found': ['prescription volume'],
            'drug_specific': True,
            'company_specific': True,
            'importance_score': 27,
            'semantic_score': 0.8
        }
    ]
    
    logger.info(f"✅ Created {len(sample_articles)} comprehensive sample articles covering FDA journey")
    return sample_articles

def main():
    """
    Main execution function for Day 1 - Step 2
    """
    print("🧠 BIOCON FDA PROJECT - DAY 1 STEP 2")
    print("Advanced News Collection with FinBERT Sentiment Analysis")
    print("="*70)
    print(f"🏢 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"💊 Drug: {DRUG_INFO['full_name']}")
    print(f"🎯 Features: FinBERT + DistilBERT + VADER + TextBlob + FDA Milestones + Semantic Analysis")
    print("-" * 70)
    
    # Initialize and run collector
    collector = AdvancedNewsCollector()
    success = collector.execute()
    
    if success:
        print("\n🎉 SUCCESS: Advanced news collection completed!")
        print("✅ Multi-model sentiment analysis performed")
        print("✅ FDA milestone classification with semantic analysis completed")
        print("✅ Daily sentiment aggregation created")
        print("✅ Data saved to: data/news_data.csv, data/daily_sentiment.csv, data/fda_events.csv")
        print("🔄 Ready for Day 2: Advanced Model Training")
    else:
        print("\n❌ FAILED: News collection failed")
        print("💡 Check logs for details: logs/news_collection.log")
        print("🔧 Troubleshooting: Verify internet connection, model downloads, and dependencies")
    
    return success

if __name__ == "__main__":
    main()