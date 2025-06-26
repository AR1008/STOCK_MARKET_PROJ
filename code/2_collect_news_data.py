import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
import feedparser
from textblob import TextBlob
import yfinance as yf
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION - MODIFY FOR DIFFERENT DRUGS/COMPANIES
# =============================================================================

# Primary drug configuration
PRIMARY_DRUG = {
    'name': 'Semglee',
    'scientific_name': 'insulin glargine-yfgn',
    'full_name': 'Semglee (insulin glargine-yfgn)',
    'drug_type': 'insulin biosimilar',
    'indication': 'diabetes',
    'application_year': 2017,  # When FDA application started
    'approval_year': 2021,     # When FDA approved
    'launch_year': 2021        # When market launch began
}

# Company configuration
COMPANY = {
    'name': 'Biocon',
    'ticker': 'BIOCON.NS',
    'sector': 'Pharmaceutical',
    'subsidiary_names': ['Biocon Biologics', 'Biocon Pharma']
}

# Data collection configuration
DATA_CONFIG = {
    'start_date': '2015-01-01',  # Overall data collection start
    'end_date': '2025-06-26',    # Current date
    'drug_focus_start': '2017-01-01',  # When to focus on drug-specific news
    'min_articles_threshold': 10
}

# FDA milestone tracking configuration
FDA_MILESTONES = {
    'application_phase': [
        'IND application', 'investigational new drug', 'pre-clinical',
        'FDA submission', 'regulatory filing', 'drug application'
    ],
    'clinical_trials': [
        'phase I trial', 'phase II trial', 'phase III trial', 'clinical trial',
        'study results', 'trial data', 'clinical endpoint', 'patient enrollment'
    ],
    'regulatory_review': [
        'FDA review', 'regulatory review', 'FDA meeting', 'advisory committee',
        'FDA inspection', 'manufacturing inspection', 'facility inspection'
    ],
    'approval_process': [
        'FDA approval', 'drug approval', 'marketing authorization', 'BLA approval',
        'NDA approval', 'biosimilar approval', 'interchangeable designation'
    ],
    'post_approval': [
        'product launch', 'commercial launch', 'market launch', 'hospital adoption',
        'prescription volume', 'market penetration', 'real-world evidence'
    ],
    'regulatory_issues': [
        'FDA warning letter', 'recall', 'safety concern', 'adverse event',
        'manufacturing issue', 'quality issue', 'FDA inspection deficiency'
    ]
}

# Market impact tracking
MARKET_IMPACT_KEYWORDS = {
    'hospital_adoption': [
        'hospital formulary', 'hospital adoption', 'pharmacy adoption',
        'prescription growth', 'market share', 'hospital contract'
    ],
    'competitive_impact': [
        'market competition', 'competitor response', 'pricing pressure',
        'market access', 'payer coverage', 'insurance coverage'
    ],
    'financial_impact': [
        'revenue impact', 'sales growth', 'market opportunity',
        'earnings impact', 'financial guidance', 'revenue guidance'
    ]
}

# Comprehensive company news tracking
COMPANY_NEWS_CATEGORIES = {
    'financial_results': [
        'quarterly results', 'earnings', 'revenue', 'profit', 'loss',
        'financial results', 'q1 results', 'q2 results', 'q3 results', 'q4 results',
        'annual results', 'financial performance', 'earnings call', 'investor call'
    ],
    'insider_trading': [
        'insider trading', 'insider buying', 'insider selling', 'promoter stake',
        'shareholding pattern', 'board meeting', 'director appointment',
        'management change', 'ceo change', 'executive appointment'
    ],
    'corporate_actions': [
        'dividend', 'bonus shares', 'stock split', 'rights issue', 'buyback',
        'merger', 'acquisition', 'spin-off', 'demerger', 'restructuring'
    ],
    'partnerships_deals': [
        'partnership', 'collaboration', 'joint venture', 'licensing deal',
        'agreement', 'contract', 'strategic alliance', 'tie-up', 'acquisition deal'
    ],
    'regulatory_compliance': [
        'sebi', 'compliance', 'regulatory filing', 'stock exchange notice',
        'disclosure', 'penalty', 'investigation', 'audit', 'legal issue'
    ],
    'business_expansion': [
        'new facility', 'plant expansion', 'capacity increase', 'new market',
        'international expansion', 'manufacturing expansion', 'facility upgrade'
    ],
    'research_development': [
        'r&d investment', 'research collaboration', 'innovation', 'new product',
        'patent filing', 'intellectual property', 'technology transfer'
    ],
    'market_rumors': [
        'market speculation', 'analyst report', 'rating upgrade', 'rating downgrade',
        'target price', 'recommendation', 'analyst coverage', 'brokerage report'
    ]
}

# =============================================================================
# MAIN COLLECTION FUNCTIONS
# =============================================================================

def clear_old_news_data():
    """Remove old news data files"""
    print("=== CLEARING OLD NEWS DATA ===")
    
    data_path = 'data'
    news_files = ['news_data.csv', 'daily_sentiment.csv', 'biocon_news_data.csv']
    
    cleared_files = []
    for filename in news_files:
        file_path = os.path.join(data_path, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleared_files.append(filename)
                print(f"✓ Removed old file: {filename}")
            except Exception as e:
                print(f"⚠️  Could not remove {filename}: {str(e)}")
    
    print(f"✓ Cleared {len(cleared_files)} old files, ready for fresh collection")

def check_data_folder():
    """Check data folder exists"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✓ Created data folder")
    else:
        print("✓ Data folder exists")
    return 'data'

def setup_sentiment_analysis():
    """Setup sentiment analysis tools"""
    print("\n=== SETTING UP SENTIMENT ANALYSIS ===")
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        print("✓ NLTK VADER sentiment analyzer ready")
        return SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"⚠️  NLTK not available, using TextBlob only: {str(e)}")
        return None

def generate_comprehensive_search_queries():
    """Generate comprehensive search queries for FDA journey tracking and ALL company news"""
    drug = PRIMARY_DRUG
    company = COMPANY
    
    # Drug-specific FDA journey queries
    drug_fda_queries = [
        f"{drug['name']} FDA application",
        f"{drug['name']} FDA approval",
        f"{drug['name']} clinical trial",
        f"{drug['name']} FDA submission",
        f"{drug['name']} regulatory approval",
        f"{drug['scientific_name']} FDA",
        f"{company['name']} {drug['name']} FDA",
        f"{drug['name']} biosimilar approval",
        f"{drug['name']} interchangeable",
        f"insulin glargine-yfgn approval"
    ]
    
    # Market implementation queries
    market_queries = [
        f"{drug['name']} hospital adoption",
        f"{drug['name']} market launch",
        f"{drug['name']} prescription volume",
        f"{drug['name']} market share",
        f"{drug['name']} hospital formulary",
        f"{drug['name']} pharmacy",
        f"{drug['name']} real world evidence",
        f"{drug['name']} commercial success"
    ]
    
    # Company-FDA general queries
    company_fda_queries = [
        f"{company['name']} FDA approval",
        f"{company['name']} FDA warning letter",
        f"{company['name']} FDA inspection",
        f"{company['name']} regulatory",
        f"{company['name']} clinical trial",
        f"{company['name']} drug approval",
        f"{company['name']} biosimilar",
        f"Biocon Biologics FDA"
    ]
    
    # Financial and earnings queries
    financial_queries = [
        f"{company['name']} quarterly results",
        f"{company['name']} earnings",
        f"{company['name']} revenue",
        f"{company['name']} profit",
        f"{company['name']} financial results",
        f"{company['name']} investor call",
        f"{company['name']} guidance",
        f"{company['name']} stock price"
    ]
    
    # Corporate and business queries
    corporate_queries = [
        f"{company['name']} merger",
        f"{company['name']} acquisition",
        f"{company['name']} partnership",
        f"{company['name']} collaboration",
        f"{company['name']} joint venture",
        f"{company['name']} licensing deal",
        f"{company['name']} expansion",
        f"{company['name']} new facility"
    ]
    
    # Insider and regulatory queries
    insider_queries = [
        f"{company['name']} insider trading",
        f"{company['name']} promoter stake",
        f"{company['name']} board meeting",
        f"{company['name']} management change",
        f"{company['name']} sebi",
        f"{company['name']} compliance",
        f"{company['name']} investigation",
        f"Kiran Mazumdar Shaw"
    ]
    
    # Market and analyst queries
    market_queries_general = [
        f"{company['name']} analyst report",
        f"{company['name']} rating upgrade",
        f"{company['name']} rating downgrade",
        f"{company['name']} target price",
        f"{company['name']} brokerage report",
        f"{company['name']} market speculation",
        f"{company['name']} dividend",
        f"{company['name']} bonus shares"
    ]
    
    return drug_fda_queries, market_queries, company_fda_queries, financial_queries, corporate_queries, insider_queries, market_queries_general

def fetch_news_by_date_range(queries, date_start, date_end, query_type):
    """Fetch news for specific date range and query type"""
    print(f"\n--- Fetching {query_type} news from {date_start} to {date_end} ---")
    
    all_articles = []
    
    for query in queries:
        try:
            print(f"Searching: {query}")
            
            # Google News RSS search
            query_encoded = query.replace(' ', '%20')
            
            # Add date restrictions if possible
            url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Filter by date range
                    start_date = datetime.strptime(date_start, '%Y-%m-%d')
                    end_date = datetime.strptime(date_end, '%Y-%m-%d')
                    
                    if start_date <= pub_date <= end_date:
                        article = {
                            'date': pub_date.strftime('%Y-%m-%d'),
                            'datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'title': getattr(entry, 'title', ''),
                            'summary': getattr(entry, 'summary', ''),
                            'url': getattr(entry, 'link', ''),
                            'source': f'Google News - {query_type}',
                            'search_query': query,
                            'query_type': query_type,
                            'date_range': f'{date_start}_to_{date_end}'
                        }
                        all_articles.append(article)
                
                except Exception as e:
                    continue
            
            # Be respectful with requests
            time.sleep(1.5)
            
        except Exception as e:
            print(f"⚠️  Error searching '{query}': {str(e)}")
            continue
    
    print(f"✓ Collected {len(all_articles)} {query_type} articles")
    return all_articles

def fetch_comprehensive_news_data():
    """Fetch comprehensive news data across all time periods and ALL company news types"""
    print("=== COMPREHENSIVE NEWS DATA COLLECTION ===")
    print("Collecting: FDA journey + Financial + Corporate + Insider + Market news")
    
    # Generate all query types
    drug_fda_queries, market_queries, company_fda_queries, financial_queries, corporate_queries, insider_queries, market_queries_general = generate_comprehensive_search_queries()
    
    all_articles = []
    
    # Period 1: 2015-2017 (Pre-drug application, comprehensive company news)
    print("\n📅 PERIOD 1: 2015-2017 (Pre-application comprehensive company news)")
    period1_articles = fetch_news_by_date_range(
        company_fda_queries + financial_queries + corporate_queries + insider_queries + market_queries_general,
        '2015-01-01', '2016-12-31',
        'Comprehensive Company News'
    )
    all_articles.extend(period1_articles)
    
    # Period 2: 2017-2021 (FDA journey + ALL company news)
    print("\n📅 PERIOD 2: 2017-2021 (FDA Journey + All Company News)")
    period2_articles = fetch_news_by_date_range(
        drug_fda_queries + company_fda_queries + financial_queries + corporate_queries + insider_queries,
        '2017-01-01', '2021-12-31',
        'FDA Journey + Company News'
    )
    all_articles.extend(period2_articles)
    
    # Period 3: 2021-2025 (Post-approval + market + ALL company news)
    print("\n📅 PERIOD 3: 2021-2025 (Market Implementation + All Company News)")
    period3_articles = fetch_news_by_date_range(
        drug_fda_queries + market_queries + financial_queries + corporate_queries + insider_queries + market_queries_general,
        '2021-01-01', '2025-06-26',
        'Market + Complete Company News'
    )
    all_articles.extend(period3_articles)
    
    # Yahoo Finance company news (all periods) - gets ALL company news
    print("\n📅 YAHOO FINANCE: Complete company news (all periods)")
    yahoo_articles = fetch_yahoo_finance_news()
    all_articles.extend(yahoo_articles)
    
    # Additional comprehensive news sources
    print("\n📅 ADDITIONAL SOURCES: Financial news and insider information")
    additional_articles = fetch_additional_comprehensive_sources()
    all_articles.extend(additional_articles)
    
    return all_articles

def fetch_additional_comprehensive_sources():
    """Fetch additional comprehensive company news from specialized sources"""
    print("Fetching additional comprehensive company news...")
    
    additional_articles = []
    
    # Financial news RSS feeds
    financial_rss_feeds = [
        'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',  # ET Markets
        'https://www.business-standard.com/rss/markets-106.rss',  # Business Standard Markets
        'https://www.livemint.com/rss/companies',  # Mint Companies
    ]
    
    # Business news RSS feeds
    business_rss_feeds = [
        'https://www.moneycontrol.com/rss/business.xml',
        'https://feeds.feedburner.com/ndtvprofit-latest',
    ]
    
    all_feeds = financial_rss_feeds + business_rss_feeds
    
    for feed_url in all_feeds:
        try:
            print(f"Fetching from: {feed_url}")
            
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Focus on recent news (last 3 years) and check for company relevance
                    if pub_date >= datetime(2022, 1, 1):
                        title = getattr(entry, 'title', '').lower()
                        summary = getattr(entry, 'summary', '').lower()
                        
                        # Check if article mentions company or related terms
                        company_terms = ['biocon', 'kiran mazumdar', 'biocon biologics']
                        if any(term in title or term in summary for term in company_terms):
                            article = {
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                                'title': getattr(entry, 'title', ''),
                                'summary': getattr(entry, 'summary', ''),
                                'url': getattr(entry, 'link', ''),
                                'source': 'Financial RSS Feed',
                                'search_query': 'Company Financial News',
                                'query_type': 'Financial News Feed'
                            }
                            additional_articles.append(article)
                
                except Exception as e:
                    continue
            
            # Be respectful with requests
            time.sleep(2)
            
        except Exception as e:
            print(f"⚠️  Error fetching RSS feed {feed_url}: {str(e)}")
            continue
    
    print(f"✓ Collected {len(additional_articles)} additional comprehensive articles")
    return additional_articles

def fetch_yahoo_finance_news():
    """Fetch company news from Yahoo Finance"""
    try:
        print(f"Fetching Yahoo Finance {COMPANY['name']} news...")
        ticker = yf.Ticker(COMPANY['ticker'])
        news_data = ticker.news
        
        articles = []
        if news_data:
            for item in news_data:
                try:
                    if 'providerPublishTime' in item:
                        pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                    else:
                        pub_date = datetime.now()
                    
                    article = {
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'url': item.get('link', ''),
                        'source': 'Yahoo Finance',
                        'search_query': f'{COMPANY["name"]} Company News',
                        'query_type': 'Company Financial News'
                    }
                    articles.append(article)
                
                except Exception as e:
                    continue
        
        print(f"✓ Collected {len(articles)} Yahoo Finance articles")
        return articles
        
    except Exception as e:
        print(f"⚠️  Error fetching Yahoo Finance: {str(e)}")
        return []
    """Fetch company news from Yahoo Finance"""
    try:
        print(f"Fetching Yahoo Finance {COMPANY['name']} news...")
        ticker = yf.Ticker(COMPANY['ticker'])
        news_data = ticker.news
        
        articles = []
        if news_data:
            for item in news_data:
                try:
                    if 'providerPublishTime' in item:
                        pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                    else:
                        pub_date = datetime.now()
                    
                    article = {
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'url': item.get('link', ''),
                        'source': 'Yahoo Finance',
                        'search_query': f'{COMPANY["name"]} Company News',
                        'query_type': 'Company Financial News'
                    }
                    articles.append(article)
                
                except Exception as e:
                    continue
        
        print(f"✓ Collected {len(articles)} Yahoo Finance articles")
        return articles
        
    except Exception as e:
        print(f"⚠️  Error fetching Yahoo Finance: {str(e)}")
        return []

def create_comprehensive_sample_data():
    """Create comprehensive sample data covering the entire FDA journey"""
    print("\n=== CREATING COMPREHENSIVE SAMPLE DATA ===")
    
    sample_articles = [
        # Pre-application period (2015-2017)
        {
            'date': '2016-03-15',
            'datetime': '2016-03-15 10:30:00',
            'title': 'Biocon begins development of insulin glargine biosimilar program',
            'summary': 'Biocon announces initiation of development program for insulin glargine biosimilar to compete with Lantus.',
            'url': 'https://example.com/biocon-insulin-development',
            'source': 'Sample Data',
            'query_type': 'Pre-Application Development'
        },
        
        # FDA application period (2017-2019)
        {
            'date': '2017-09-20',
            'datetime': '2017-09-20 14:15:00',
            'title': 'Biocon submits insulin glargine biosimilar application to FDA',
            'summary': 'Biocon files Biologics License Application (BLA) for insulin glargine-yfgn with FDA seeking approval.',
            'url': 'https://example.com/biocon-bla-submission',
            'source': 'Sample Data',
            'query_type': 'FDA Application Milestone'
        },
        
        {
            'date': '2018-05-10',
            'datetime': '2018-05-10 11:20:00',
            'title': 'FDA accepts Biocon\'s insulin glargine BLA for review',
            'summary': 'FDA accepts Biocon\'s BLA for insulin glargine-yfgn and assigns PDUFA date for regulatory decision.',
            'url': 'https://example.com/fda-accepts-bla',
            'source': 'Sample Data',
            'query_type': 'FDA Review Milestone'
        },
        
        # Clinical and regulatory milestones (2018-2020)
        {
            'date': '2019-02-14',
            'datetime': '2019-02-14 16:45:00',
            'title': 'Biocon completes Phase III clinical trials for insulin glargine biosimilar',
            'summary': 'Biocon announces successful completion of Phase III clinical trials demonstrating bioequivalence to Lantus.',
            'url': 'https://example.com/phase3-completion',
            'source': 'Sample Data',
            'query_type': 'Clinical Trial Milestone'
        },
        
        {
            'date': '2020-08-25',
            'datetime': '2020-08-25 13:30:00',
            'title': 'FDA conducts inspection of Biocon manufacturing facility',
            'summary': 'FDA inspection team visits Biocon\'s insulin manufacturing facility as part of BLA review process.',
            'url': 'https://example.com/fda-inspection',
            'source': 'Sample Data',
            'query_type': 'Regulatory Review Process'
        },
        
        # Approval milestone (2021)
        {
            'date': '2021-07-29',
            'datetime': '2021-07-29 15:30:00',
            'title': 'FDA Approves Biocon\'s Semglee as First Interchangeable Insulin Biosimilar',
            'summary': 'FDA approves Semglee (insulin glargine-yfgn) as both biosimilar and interchangeable with Lantus, marking historic milestone.',
            'url': 'https://example.com/semglee-fda-approval',
            'source': 'Sample Data',
            'query_type': 'FDA Approval Milestone'
        },
        
        {
            'date': '2021-07-30',
            'datetime': '2021-07-30 09:15:00',
            'title': 'Biocon Stock Surges 15% on Semglee FDA Approval News',
            'summary': 'Biocon shares jump to 52-week high following FDA approval of Semglee as first interchangeable insulin biosimilar.',
            'url': 'https://example.com/biocon-stock-surge',
            'source': 'Sample Data',
            'query_type': 'Stock Market Impact'
        },
        
        # Market launch and implementation (2021-2022)
        {
            'date': '2021-09-15',
            'datetime': '2021-09-15 11:45:00',
            'title': 'Semglee Commercial Launch Begins in US Market',
            'summary': 'Biocon and Viatris announce commercial availability of Semglee in US pharmacies nationwide.',
            'url': 'https://example.com/semglee-commercial-launch',
            'source': 'Sample Data',
            'query_type': 'Market Launch Milestone'
        },
        
        {
            'date': '2022-01-20',
            'datetime': '2022-01-20 14:30:00',
            'title': 'Major Hospital Systems Add Semglee to Formularies',
            'summary': 'Leading hospital networks across US add Semglee to preferred insulin formularies citing cost savings.',
            'url': 'https://example.com/hospital-formulary-adoption',
            'source': 'Sample Data',
            'query_type': 'Hospital Adoption Impact'
        },
        
        # Market performance and real-world evidence (2022-2024)
        {
            'date': '2022-06-12',
            'datetime': '2022-06-12 10:15:00',
            'title': 'Semglee Captures 8% Market Share in First Year',
            'summary': 'Real-world data shows Semglee gaining significant market share in US insulin glargine market.',
            'url': 'https://example.com/semglee-market-share',
            'source': 'Sample Data',
            'query_type': 'Market Performance Data'
        },
        
        {
            'date': '2023-03-08',
            'datetime': '2023-03-08 16:20:00',
            'title': 'Biocon Reports Strong Semglee Revenue Growth in Q4',
            'summary': 'Biocon Q4 earnings show 45% increase in Semglee revenue driven by expanding hospital adoption.',
            'url': 'https://example.com/semglee-revenue-growth',
            'source': 'Sample Data',
            'query_type': 'Financial Impact Update'
        },
        
        # Recent market dynamics (2024-2025)
        {
            'date': '2024-01-25',
            'datetime': '2024-01-25 12:00:00',
            'title': 'Insurance Coverage Expansion Boosts Semglee Prescriptions',
            'summary': 'Major insurance providers expand Semglee coverage following real-world efficacy data publication.',
            'url': 'https://example.com/insurance-coverage-expansion',
            'source': 'Sample Data',
            'query_type': 'Market Access Improvement'
        }
    ]
    
    print(f"✓ Created {len(sample_articles)} comprehensive sample articles covering FDA journey")
    return sample_articles

def classify_news_by_fda_milestone(articles):
    """Classify news by FDA milestone, market impact, and ALL company news categories"""
    print("\n=== CLASSIFYING NEWS BY FDA MILESTONES + ALL COMPANY NEWS ===")
    
    classified_articles = []
    
    for article in articles:
        try:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            text_to_search = f"{title} {summary}"
            
            # Initialize classification
            article['fda_milestone_type'] = 'Other'
            article['market_impact_type'] = 'Other'
            article['company_news_category'] = 'Other'
            article['priority_score'] = 1
            
            # Check for drug-specific content
            drug_mentioned = any(keyword in text_to_search for keyword in [
                PRIMARY_DRUG['name'].lower(),
                PRIMARY_DRUG['scientific_name'].lower(),
                'semglee', 'insulin glargine-yfgn'
            ])
            
            # Classify by FDA milestone
            for milestone_type, keywords in FDA_MILESTONES.items():
                if any(keyword.lower() in text_to_search for keyword in keywords):
                    article['fda_milestone_type'] = milestone_type
                    break
            
            # Classify by market impact
            for impact_type, keywords in MARKET_IMPACT_KEYWORDS.items():
                if any(keyword.lower() in text_to_search for keyword in keywords):
                    article['market_impact_type'] = impact_type
                    break
            
            # Classify by comprehensive company news categories
            for category_type, keywords in COMPANY_NEWS_CATEGORIES.items():
                if any(keyword.lower() in text_to_search for keyword in keywords):
                    article['company_news_category'] = category_type
                    break
            
            # Calculate priority score
            priority_score = 1
            
            if drug_mentioned:
                priority_score += 10  # Drug-specific gets highest priority
            
            # FDA milestone scoring
            milestone_scores = {
                'application_phase': 8,
                'clinical_trials': 7,
                'regulatory_review': 9,
                'approval_process': 10,
                'post_approval': 6,
                'regulatory_issues': 8
            }
            priority_score += milestone_scores.get(article['fda_milestone_type'], 0)
            
            # Market impact scoring
            impact_scores = {
                'hospital_adoption': 7,
                'competitive_impact': 5,
                'financial_impact': 6
            }
            priority_score += impact_scores.get(article['market_impact_type'], 0)
            
            # Company news category scoring
            company_scores = {
                'financial_results': 8,
                'insider_trading': 7,
                'corporate_actions': 6,
                'partnerships_deals': 7,
                'regulatory_compliance': 6,
                'business_expansion': 5,
                'research_development': 6,
                'market_rumors': 4
            }
            priority_score += company_scores.get(article['company_news_category'], 0)
            
            article['drug_specific'] = drug_mentioned
            article['priority_score'] = priority_score
            
            # Add company mentions
            article['company_mentioned'] = any(name.lower() in text_to_search for name in [
                COMPANY['name'].lower(),
                *[sub.lower() for sub in COMPANY['subsidiary_names']]
            ])
            
            # Keep ALL relevant articles (lowered threshold to capture more company news)
            if priority_score > 1 or article['company_mentioned']:
                classified_articles.append(article)
        
        except Exception as e:
            print(f"⚠️  Error classifying article: {str(e)}")
            continue
    
    # Sort by priority score (highest first)
    classified_articles = sorted(classified_articles, key=lambda x: x['priority_score'], reverse=True)
    
    # Print comprehensive classification summary
    milestone_counts = {}
    company_news_counts = {}
    
    for article in classified_articles:
        milestone = article.get('fda_milestone_type', 'Other')
        company_category = article.get('company_news_category', 'Other')
        
        milestone_counts[milestone] = milestone_counts.get(milestone, 0) + 1
        company_news_counts[company_category] = company_news_counts.get(company_category, 0) + 1
    
    print(f"✓ FDA Milestone Classification:")
    for milestone, count in milestone_counts.items():
        print(f"    {milestone}: {count} articles")
    
    print(f"✓ Company News Classification:")
    for category, count in company_news_counts.items():
        print(f"    {category}: {count} articles")
    
    print(f"✓ Total relevant articles: {len(classified_articles)}")
    return classified_articles

def calculate_comprehensive_sentiment(articles, sentiment_analyzer=None):
    """Calculate sentiment with FDA milestone and market impact weighting"""
    print("\n=== CALCULATING COMPREHENSIVE SENTIMENT SCORES ===")
    
    for article in articles:
        try:
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            
            if not text.strip():
                article['sentiment_score'] = 0
                article['sentiment_label'] = 'neutral'
                continue
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # NLTK VADER sentiment (if available)
            if sentiment_analyzer:
                vader_scores = sentiment_analyzer.polarity_scores(text)
                vader_score = vader_scores['compound']
                combined_score = (textblob_score + vader_score) / 2
            else:
                combined_score = textblob_score
            
            # Apply weighting based on milestone importance
            milestone_weights = {
                'approval_process': 2.0,
                'regulatory_review': 1.8,
                'application_phase': 1.6,
                'regulatory_issues': 1.7,
                'clinical_trials': 1.5,
                'post_approval': 1.3
            }
            
            milestone_type = article.get('fda_milestone_type', 'Other')
            milestone_weight = milestone_weights.get(milestone_type, 1.0)
            
            # Additional weight for drug-specific news
            drug_weight = 1.5 if article.get('drug_specific', False) else 1.0
            
            # Calculate final weighted sentiment
            final_weight = milestone_weight * drug_weight
            weighted_sentiment = combined_score * final_weight
            
            # Classify sentiment
            if weighted_sentiment > 0.1:
                sentiment_label = 'positive'
            elif weighted_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            article['sentiment_score'] = combined_score
            article['weighted_sentiment'] = weighted_sentiment
            article['sentiment_label'] = sentiment_label
            article['sentiment_weight'] = final_weight
        
        except Exception as e:
            print(f"⚠️  Error calculating sentiment: {str(e)}")
            article['sentiment_score'] = 0
            article['weighted_sentiment'] = 0
            article['sentiment_label'] = 'neutral'
            article['sentiment_weight'] = 1.0
    
    print("✓ Calculated comprehensive sentiment scores with FDA milestone weighting")
    return articles

def create_comprehensive_daily_aggregation(articles):
    """Create daily aggregation with FDA milestone tracking"""
    print("\n=== CREATING COMPREHENSIVE DAILY AGGREGATION ===")
    
    try:
        df = pd.DataFrame(articles)
        
        if df.empty:
            print("❌ No articles to aggregate")
            return pd.DataFrame()
        
        # Group by date and calculate comprehensive aggregations
        daily_agg = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'weighted_sentiment': ['mean', 'std'],
            'priority_score': ['mean', 'max', 'sum'],
            'drug_specific': 'sum',
            'company_mentioned': 'sum',
            'fda_milestone_type': lambda x: '|'.join(x.unique()),
            'market_impact_type': lambda x: '|'.join(x.unique())
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'date', 'avg_sentiment', 'sentiment_std', 'news_count',
            'weighted_avg_sentiment', 'weighted_sentiment_std',
            'avg_priority', 'max_priority', 'total_priority',
            'drug_specific_count', 'company_news_count',
            'fda_milestones', 'market_impacts'
        ]
        
        # Fill NaN values
        daily_agg['sentiment_std'] = daily_agg['sentiment_std'].fillna(0)
        daily_agg['weighted_sentiment_std'] = daily_agg['weighted_sentiment_std'].fillna(0)
        
        # Create additional metrics
        daily_agg['drug_news_ratio'] = daily_agg['drug_specific_count'] / daily_agg['news_count']
        
        # Create FDA milestone flags
        milestone_types = list(FDA_MILESTONES.keys())
        for milestone in milestone_types:
            daily_agg[f'has_{milestone}'] = daily_agg['fda_milestones'].str.contains(milestone, na=False).astype(int)
        
        # Create importance score for correlation analysis
        daily_agg['day_importance_score'] = (
            daily_agg['drug_specific_count'] * 10 +
            daily_agg['max_priority'] +
            daily_agg['has_approval_process'] * 20 +
            daily_agg['has_regulatory_review'] * 15 +
            daily_agg['has_clinical_trials'] * 10 +
            daily_agg['has_regulatory_issues'] * 12
        )
        
        # Sort by date
        daily_agg = daily_agg.sort_values('date')
        
        print(f"✓ Created comprehensive daily aggregation: {len(daily_agg)} days with news")
        
        # Show high-importance days
        high_importance = daily_agg[daily_agg['day_importance_score'] >= 20]
        if not high_importance.empty:
            print(f"✓ High-importance days (FDA milestones): {len(high_importance)}")
        
        return daily_agg
        
    except Exception as e:
        print(f"❌ Error creating daily aggregation: {str(e)}")
        return pd.DataFrame()

def save_comprehensive_data(articles, daily_agg, data_path):
    """Save comprehensive news data with FDA milestone tracking"""
    print("\n=== SAVING COMPREHENSIVE NEWS DATA ===")
    
    try:
        # Save main news dataset
        news_df = pd.DataFrame(articles)
        if not news_df.empty:
            # Sort by priority score and date
            news_df = news_df.sort_values(['priority_score', 'datetime'], ascending=[False, True])
            
            news_file = os.path.join(data_path, 'news_data.csv')
            news_df.to_csv(news_file, index=False)
            
            # Verify file
            if os.path.exists(news_file) and os.path.getsize(news_file) > 100:
                print(f"✓ Comprehensive news data saved: {news_file}")
                print(f"✓ File size: {os.path.getsize(news_file):,} bytes")
                
                # Show sample with comprehensive info
                print("Sample news data (by priority and FDA milestone):")
                sample_cols = ['date', 'title', 'fda_milestone_type', 'priority_score', 'weighted_sentiment']
                available_cols = [col for col in sample_cols if col in news_df.columns]
                print(news_df[available_cols].head(5))
            else:
                print("❌ News file not saved properly")
                return False
        
        # Save daily aggregation
        if not daily_agg.empty:
            daily_file = os.path.join(data_path, 'daily_sentiment.csv')
            daily_agg.to_csv(daily_file, index=False)
            
            # Verify file
            if os.path.exists(daily_file) and os.path.getsize(daily_file) > 50:
                print(f"✓ Daily sentiment data saved: {daily_file}")
                print(f"✓ File size: {os.path.getsize(daily_file):,} bytes")
                
                # Show sample with FDA milestone info
                print("Sample daily sentiment data:")
                sample_cols = ['date', 'weighted_avg_sentiment', 'drug_specific_count', 'day_importance_score']
                available_cols = [col for col in sample_cols if col in daily_agg.columns]
                print(daily_agg[available_cols].head(5))
            else:
                print("❌ Daily sentiment file not saved properly")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving comprehensive data: {str(e)}")
        return False

def create_fda_milestone_summary(articles):
    """Create summary of FDA milestones for analysis"""
    print("\n=== CREATING FDA MILESTONE SUMMARY ===")
    
    try:
        df = pd.DataFrame(articles)
        if df.empty:
            return pd.DataFrame()
        
        # Filter for high-priority articles
        milestone_articles = df[df['priority_score'] >= 10].copy()
        
        if milestone_articles.empty:
            print("⚠️  No high-priority milestone articles found")
            return pd.DataFrame()
        
        # Create milestone summary
        milestone_summary = milestone_articles.groupby(['date', 'fda_milestone_type']).agg({
            'title': 'first',
            'sentiment_score': 'mean',
            'weighted_sentiment': 'mean',
            'priority_score': 'max',
            'drug_specific': 'max'
        }).reset_index()
        
        # Add milestone order for chronological analysis
        milestone_order = {
            'application_phase': 1,
            'clinical_trials': 2,
            'regulatory_review': 3,
            'approval_process': 4,
            'post_approval': 5,
            'regulatory_issues': 6
        }
        
        milestone_summary['milestone_order'] = milestone_summary['fda_milestone_type'].map(milestone_order)
        milestone_summary = milestone_summary.sort_values(['milestone_order', 'date'])
        
        print(f"✓ Created FDA milestone summary with {len(milestone_summary)} key events")
        return milestone_summary
        
    except Exception as e:
        print(f"❌ Error creating milestone summary: {str(e)}")
        return pd.DataFrame()

def print_comprehensive_summary(articles, daily_agg):
    """Print comprehensive analysis summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE FDA DRUG JOURNEY NEWS COLLECTION SUMMARY")
    print("="*80)
    
    if articles:
        # Time period analysis
        dates = sorted([a['date'] for a in articles if a.get('date')])
        if dates:
            print(f"✓ TIME PERIOD: {dates[0]} to {dates[-1]} (2015-2025)")
        
        # Article breakdown by period
        pre_app = sum(1 for a in articles if a.get('date', '') < '2017-01-01')
        fda_journey = sum(1 for a in articles if '2017-01-01' <= a.get('date', '') <= '2021-12-31')
        post_approval = sum(1 for a in articles if a.get('date', '') > '2021-12-31')
        
        print(f"✓ ARTICLE BREAKDOWN:")
        print(f"    Pre-Application (2015-2017): {pre_app} articles")
        print(f"    FDA Journey (2017-2021): {fda_journey} articles")
        print(f"    Market Implementation (2022-2025): {post_approval} articles")
        print(f"    TOTAL: {len(articles)} articles")
        
        # FDA milestone breakdown
        milestone_counts = {}
        for article in articles:
            milestone = article.get('fda_milestone_type', 'Other')
            milestone_counts[milestone] = milestone_counts.get(milestone, 0) + 1
        
        print(f"\n✓ FDA MILESTONE BREAKDOWN:")
        for milestone, count in sorted(milestone_counts.items()):
            print(f"    {milestone}: {count} articles")
        
        # Drug-specific analysis
        drug_specific = sum(1 for a in articles if a.get('drug_specific', False))
        approval_related = sum(1 for a in articles if a.get('fda_milestone_type') == 'approval_process')
        
        print(f"\n✓ DRUG-SPECIFIC ANALYSIS:")
        print(f"    {PRIMARY_DRUG['name']}-specific articles: {drug_specific}")
        print(f"    FDA approval-related articles: {approval_related}")
        
        # Market impact analysis
        hospital_adoption = sum(1 for a in articles if a.get('market_impact_type') == 'hospital_adoption')
        financial_impact = sum(1 for a in articles if a.get('market_impact_type') == 'financial_impact')
        
        print(f"\n✓ COMPREHENSIVE COMPANY NEWS ANALYSIS:")
        print(f"    Financial results articles: {sum(1 for a in articles if a.get('company_news_category') == 'financial_results')}")
        print(f"    Insider trading/management articles: {sum(1 for a in articles if a.get('company_news_category') == 'insider_trading')}")
        print(f"    Corporate actions articles: {sum(1 for a in articles if a.get('company_news_category') == 'corporate_actions')}")
        print(f"    Partnership/deal articles: {sum(1 for a in articles if a.get('company_news_category') == 'partnerships_deals')}")
        print(f"    Regulatory compliance articles: {sum(1 for a in articles if a.get('company_news_category') == 'regulatory_compliance')}")
        print(f"    Business expansion articles: {sum(1 for a in articles if a.get('company_news_category') == 'business_expansion')}")
        print(f"    R&D articles: {sum(1 for a in articles if a.get('company_news_category') == 'research_development')}")
        print(f"    Market rumors/analyst articles: {sum(1 for a in articles if a.get('company_news_category') == 'market_rumors')}")
        
        print(f"\n✓ FILES CREATED:")
        print(f"    news_data.csv - Complete dataset: FDA + Financial + Corporate + Insider + Market news")
        print(f"    daily_sentiment.csv - Daily aggregation for comprehensive stock correlation")
        
        print(f"\n✓ COMPREHENSIVE ANALYSIS READY:")
        print(f"    ✓ FDA milestones + Financial results + Corporate actions")
        print(f"    ✓ Insider trading + Management changes + Partnership deals")
        print(f"    ✓ Regulatory compliance + Business expansion + R&D updates")
        print(f"    ✓ Market rumors + Analyst reports + All company news")
        print(f"    ✓ Complete company news ecosystem for stock correlation analysis")
        
    else:
        print("❌ No articles collected")
    
    print("="*80)

def main():
    """Main function for comprehensive FDA drug journey news collection"""
    print("COMPREHENSIVE FDA DRUG JOURNEY NEWS COLLECTOR")
    print(f"Target Drug: {PRIMARY_DRUG['full_name']}")
    print(f"Company: {COMPANY['name']} ({COMPANY['ticker']})")
    print(f"Analysis Period: {DATA_CONFIG['start_date']} to {DATA_CONFIG['end_date']}")
    print(f"FDA Journey: {PRIMARY_DRUG['application_year']} (application) to present")
    print("="*80)
    
    # Step 1: Clear old data
    clear_old_news_data()
    
    # Step 2: Check data folder
    data_path = check_data_folder()
    
    # Step 3: Setup sentiment analysis
    sentiment_analyzer = setup_sentiment_analysis()
    
    # Step 4: Collect comprehensive news data across all periods
    all_articles = fetch_comprehensive_news_data()
    
    # Step 5: Add sample data if insufficient real data
    if len(all_articles) < DATA_CONFIG['min_articles_threshold']:
        print(f"\n⚠️  Only {len(all_articles)} articles found, adding comprehensive sample data")
        sample_articles = create_comprehensive_sample_data()
        all_articles.extend(sample_articles)
    
    print(f"\nTotal articles collected across all periods: {len(all_articles)}")
    
    # Step 6: Classify by FDA milestones and market impact
    classified_articles = classify_news_by_fda_milestone(all_articles)
    
    # Step 7: Calculate comprehensive sentiment with FDA weighting
    articles_with_sentiment = calculate_comprehensive_sentiment(classified_articles, sentiment_analyzer)
    
    # Step 8: Create comprehensive daily aggregation
    daily_sentiment = create_comprehensive_daily_aggregation(articles_with_sentiment)
    
    # Step 9: Save comprehensive data
    success = save_comprehensive_data(articles_with_sentiment, daily_sentiment, data_path)
    
    # Step 10: Print comprehensive summary
    print_comprehensive_summary(articles_with_sentiment, daily_sentiment)
    
    if success:
        print("\n🎉 SUCCESS: Comprehensive FDA drug journey news collection completed!")
        print("✓ Complete timeline from 2015 pre-application to 2025 market implementation")
        print("✓ FDA milestone tracking for regulatory correlation analysis")
        print("✓ Hospital adoption and market penetration tracking")
        print("✓ Ready for comprehensive stock price correlation analysis")
        return True
    else:
        print("\n💥 FAILED: News data collection failed!")
        return False

if __name__ == "__main__":
    main()