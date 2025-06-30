"""
Advanced Stock Data Collection for Biocon FDA Project
Day 1 - Step 1: Comprehensive stock data with technical indicators

Changes:
- Removed INDIA_VIX due to inconsistent data
- Added SENSEX (^BSESN) as an index
- Modified to continue pipeline if an index fails
- Disabled threading in nsepy.get_history to fix AttributeError
- Added DNS resolution check for NSE India
- Improved NSE India 401 error handling
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import talib
from pathlib import Path
import requests
from retry import retry
import time
from requests.exceptions import HTTPError, RequestException
from nsepy import get_history
import socket

# Import configuration
from config import (
    COMPANY_INFO, DATA_START_DATE, DATA_END_DATE, 
    DATA_SOURCES, FEATURE_CONFIG, PATHS, DATA_FILES,
    PROXY_CONFIG, create_directories, validate_config
)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stock_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedStockDataCollector:
    """
    Advanced stock data collector with comprehensive technical indicators,
    market relative metrics, FDA event alignment capabilities, and multiple data sources.
    """
    
    def __init__(self):
        self.start_date = DATA_START_DATE
        self.end_date = DATA_END_DATE
        self.symbols = DATA_SOURCES['stock_data']['symbols']
        self.collected_data = {}
        self.master_df = None
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://www.nseindia.com'
        }
        self.proxies = PROXY_CONFIG if PROXY_CONFIG else None
        
        # Initialize NSE session
        self._initialize_nse_session()
        
        # Create directories
        create_directories()
        validate_config()
        
        logger.info(f"🚀 Advanced Stock Data Collector Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"🏢 Target Company: {COMPANY_INFO['name']}")
        if self.proxies:
            logger.info(f"🛡️ Using proxies: {self.proxies}")
    
    def _check_dns_resolution(self, host):
        """
        Check if a host resolves correctly
        """
        try:
            socket.gethostbyname(host)
            return True
        except socket.gaierror as e:
            logger.error(f"❌ DNS resolution failed for {host}: {str(e)}")
            logger.info("💡 Try setting DNS servers to 8.8.8.8 or 1.1.1.1")
            return False
    
    def _initialize_nse_session(self):
        """
        Initialize session for NSE India with proper cookies
        """
        try:
            if not self._check_dns_resolution('www.nseindia.com'):
                raise Exception("DNS resolution failed for www.nseindia.com")
            
            endpoints = [
                'https://www.nseindia.com',
                'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY',
                'https://www.nseindia.com/api/market-data-pre-open',
                'https://www.nseindia.com/api/marketStatus'
            ]
            for url in endpoints:
                response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
                response.raise_for_status()
                time.sleep(1)  # Avoid rate limiting
            logger.info("✅ NSE session initialized with cookies")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize NSE session: {str(e)}")
            logger.info("ℹ️ Falling back to yfinance as primary source.")
            logger.info("💡 If outside India, use a VPN (e.g., NordVPN, ExpressVPN) with an Indian server.")
            logger.info("💡 Alternatively, set DNS to 8.8.8.8: sudo networksetup -setdnsservers Wi-Fi 8.8.8.8 1.1.1.1")
    
    def _validate_symbol(self, symbol, source):
        """
        Validate if a symbol is available in the data source
        """
        if source == 'Yahoo Finance':
            try:
                time.sleep(5)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1d", timeout=15)
                return not df.empty
            except Exception as e:
                logger.warning(f"⚠️ Symbol validation failed for {symbol} on {source}: {str(e)}")
                return False
        elif source == 'NSE India':
            try:
                if not self._check_dns_resolution('www.nseindia.com'):
                    raise Exception("DNS resolution failed for NSE India")
                nse_symbol = symbol.replace('.NS', '').replace('.BO', '')
                url = f"https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}"
                response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
                response.raise_for_status()
                return True
            except Exception as e:
                logger.warning(f"⚠️ Symbol validation failed for {symbol} on {source}: {str(e)}")
                return False
        return True
    
    @retry(tries=3, delay=10, backoff=2)
    def download_stock_data(self, symbol_group, symbol_name, allow_mock_data=False):
        """
        Download stock data with multiple fallback options
        """
        logger.info(f"📈 Downloading {symbol_name} data...")
        
        # Try Yahoo Finance first
        for symbol in symbol_group:
            if not self._validate_symbol(symbol, 'Yahoo Finance'):
                logger.warning(f"⚠️ Invalid symbol {symbol} for Yahoo Finance")
                continue
                
            try:
                logger.info(f"  Trying Yahoo Finance symbol: {symbol}")
                time.sleep(5)
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    prepost=True,
                    timeout=15
                )
                
                if df.empty:
                    logger.warning(f"  ⚠️ Empty data for {symbol} from Yahoo Finance")
                    continue
                
                if len(df) > 500:
                    df.reset_index(inplace=True)
                    df['Symbol'] = symbol
                    df['Company'] = symbol_name
                    df['Source'] = 'Yahoo Finance'
                    
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Daily_Return_Percent'] = df['Daily_Return'] * 100
                    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                    
                    logger.info(f"  ✅ Success: {len(df)} records for {symbol} from Yahoo Finance")
                    return df
                else:
                    logger.warning(f"  ⚠️ Insufficient data for {symbol} from Yahoo Finance")
                    
            except Exception as e:
                logger.error(f"  ❌ Error downloading {symbol} from Yahoo Finance: {str(e)}")
                continue
        
        # Try NSE India
        try:
            logger.info(f"  Trying NSE India for {symbol_name}")
            df = self.download_nse_data(symbol_name)
            if df is not None and not df.empty and len(df) > 500:
                logger.info(f"  ✅ Success: {len(df)} records from NSE India")
                return df
        except Exception as e:
            logger.error(f"  ❌ Error downloading from NSE India: {str(e)}")
        
        # Mock data only if explicitly allowed
        if allow_mock_data:
            logger.warning(f"⚠️ All real sources failed for {symbol_name}, generating mock data")
            return self._create_mock_data(symbol_name)
        else:
            logger.error(f"❌ All real sources failed for {symbol_name}. No data collected.")
            logger.info("💡 Troubleshooting: Verify symbols on https://finance.yahoo.com or https://www.nseindia.com")
            logger.info("💡 If outside India, use a VPN with an Indian server for NSE India access")
            logger.info("💡 Run with --allow-mock-data for testing with mock data")
            return None  # Return None instead of raising exception to continue pipeline
    
    def _create_mock_data(self, symbol_name):
        """
        Create mock data as a last resort
        """
        try:
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            df = pd.DataFrame({
                'Date': date_range,
                'Open': np.random.normal(100, 10, len(date_range)),
                'High': np.random.normal(105, 10, len(date_range)),
                'Low': np.random.normal(95, 10, len(date_range)),
                'Close': np.random.normal(100, 10, len(date_range)),
                'Volume': np.random.randint(100000, 1000000, len(date_range)),
                'Symbol': symbol_name,
                'Company': symbol_name,
                'Source': 'Mock Data'
            })
            df['Daily_Return'] = df['Close'].pct_change()
            df['Daily_Return_Percent'] = df['Daily_Return'] * 100
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            logger.info(f"  ✅ Generated mock data for {symbol_name}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"  ❌ Error creating mock data for {symbol_name}: {str(e)}")
            return None
    
    @retry(tries=3, delay=10, backoff=2)
    def download_nse_data(self, symbol_name):
        """
        Fetch stock data from NSE India with nsepy fallback
        """
        try:
            symbol = COMPANY_INFO['ticker'].replace('.NS', '') if symbol_name == 'BIOCON' else symbol_name
            symbol = symbol.replace('NIFTY50', 'NIFTY 50').replace('NIFTY_PHARMA', 'NIFTY PHARMA').replace('SENSEX', 'SENSEX')
            if not self._validate_symbol(symbol, 'NSE India'):
                raise ValueError(f"Invalid symbol {symbol} for NSE India")
                
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={self.start_date.strftime('%d-%m-%Y')}&to={self.end_date.strftime('%d-%m-%Y')}"
            
            self._initialize_nse_session()
            response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                logger.error("  ❌ Invalid response from NSE India")
                raise ValueError("Invalid NSE response")
                
            df = pd.DataFrame(data['data'])
            if df.empty:
                raise ValueError("Empty NSE data")
                
            df = df.rename(columns={
                'CH_TIMESTAMP': 'Date',
                'CH_OPENING_PRICE': 'Open',
                'CH_TRADE_HIGH_PRICE': 'High',
                'CH_TRADE_LOW_PRICE': 'Low',
                'CH_CLOSING_PRICE': 'Close',
                'CH_TOT_TRD_QTY': 'Volume',
                'CH_TOT_TRD_VAL': 'Value'
            })
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = df.loc[(df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)]
            
            df['Symbol'] = symbol
            df['Company'] = symbol_name
            df['Source'] = 'NSE India'
            
            df['Daily_Return'] = df['Close'].pct_change()
            df['Daily_Return_Percent'] = df['Daily_Return'] * 100
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Error in NSE India API download: {str(e)}")
            logger.info(f"  Trying nsepy for {symbol_name}")
            try:
                nsepy_symbol = {
                    'BIOCON': 'BIOCON',
                    'NIFTY50': 'NIFTY 50',
                    'SENSEX': 'SENSEX',
                    'NIFTY_PHARMA': 'NIFTY PHARMA'
                }.get(symbol_name, symbol_name)
                
                if not self._check_dns_resolution('www1.nseindia.com'):
                    raise Exception("DNS resolution failed for www1.nseindia.com")
                
                df = get_history(
                    symbol=nsepy_symbol,
                    start=self.start_date,
                    end=self.end_date,
                    index=(symbol_name != 'BIOCON'),
                    threads=False  # Disable threading to avoid AttributeError
                )
                
                if df.empty:
                    raise ValueError("Empty nsepy data")
                    
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume',
                    'Turnover': 'Value'
                })
                
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df = df.loc[(df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)]
                
                df['Symbol'] = nsepy_symbol
                df['Company'] = symbol_name
                df['Source'] = 'NSE India (nsepy)'
                
                df['Daily_Return'] = df['Close'].pct_change()
                df['Daily_Return_Percent'] = df['Daily_Return'] * 100
                df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                
                logger.info(f"  ✅ Success: {len(df)} records from NSE India (nsepy)")
                return df
                
            except Exception as e:
                logger.error(f"  ❌ Error downloading from nsepy: {str(e)}")
                return None
    
    def fetch_bse_announcements(self):
        """
        Scrape BSE India announcements and corporate actions
        """
        try:
            logger.info("📢 Fetching BSE announcements...")
            url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w?strCode={COMPANY_INFO['bse_code']}&strType=CA&strSearch=undefined&fromdt={self.start_date.strftime('%Y%m%d')}&todt={self.end_date.strftime('%Y%m%d')}"
            
            response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            announcements = []
            for item in data.get('Table', []):
                announcements.append({
                    'Date': pd.to_datetime(item.get('NEWS_SUBMISSION_DATE')),
                    'Purpose': item.get('PURPOSE', ''),
                    'Description': item.get('NEWSSUB', ''),
                    'Record_Date': pd.to_datetime(item.get('RecordDate')) if item.get('RecordDate') else None
                })
            
            df = pd.DataFrame(announcements)
            if not df.empty:
                df['Source'] = 'BSE India'
                logger.info(f"✅ Successfully fetched {len(df)} announcements from BSE India")
                return df
            else:
                logger.warning("⚠️ No announcements found on BSE India")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching BSE announcements: {str(e)}")
            return None
    
    def add_basic_technical_indicators(self, df):
        """
        Add comprehensive technical indicators using TA-Lib
        """
        logger.info("🔧 Adding technical indicators...")
        
        try:
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
            
            open_prices = df['Open'].values
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            volume = df['Volume'].values
            
            for arr, name in zip([open_prices, high, low, close, volume], numeric_columns):
                if not np.issubdtype(arr.dtype, np.floating):
                    raise ValueError(f"{name} array is not of floating type: {arr.dtype}")
                if np.any(np.isnan(arr)):
                    raise ValueError(f"{name} array contains NaN values after filling")

            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
            
            for period in [12, 26, 50]:
                df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
            
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            df['RSI_14'] = talib.RSI(close, timeperiod=14)
            df['RSI_21'] = talib.RSI(close, timeperiod=21)
            
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            df['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
            
            df['ROC_10'] = talib.ROC(close, timeperiod=10)
            df['ROC_20'] = talib.ROC(close, timeperiod=20)
            
            df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
            
            df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['ATR_21'] = talib.ATR(high, low, close, timeperiod=21)
            
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            for period in [10, 20, 30]:
                df[f'Volatility_{period}'] = df['Daily_Return'].rolling(window=period).std() * np.sqrt(252)
            
            df['OBV'] = talib.OBV(close, volume)
            
            df['VPT'] = talib.AD(high, low, close, volume)
            
            for period in [10, 20, 50]:
                df[f'Volume_SMA_{period}'] = talib.SMA(volume, timeperiod=period)
            
            df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
            
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            df['DOJI'] = talib.CDLDOJI(open_prices, high, low, close)
            df['HAMMER'] = talib.CDLHAMMER(open_prices, high, low, close)
            df['ENGULFING'] = talib.CDLENGULFING(open_prices, high, low, close)
            df['MORNING_STAR'] = talib.CDLMORNINGSTAR(open_prices, high, low, close)
            df['EVENING_STAR'] = talib.CDLEVENINGSTAR(open_prices, high, low, close)
            
            df['Support'] = df['Low'].rolling(window=20).min()
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
            df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
            
            logger.info("✅ Technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding technical indicators: {str(e)}")
            return self.add_simple_technical_indicators(df)
    
    def add_simple_technical_indicators(self, df):
        """
        Fallback method for technical indicators without TA-Lib
        """
        logger.info("🔧 Adding simple technical indicators (fallback)...")
        
        try:
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            for period in [12, 26]:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            for period in [10, 20]:
                df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
            
            df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            logger.info("✅ Simple technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding simple technical indicators: {str(e)}")
            return df
    
    def add_market_relative_metrics(self, biocon_df, nifty_df, sensex_df, pharma_df):
        """
        Calculate market-relative and sector-relative metrics
        """
        logger.info("📊 Calculating market relative metrics...")
        
        try:
            biocon_dates = biocon_df[['Date', 'Close', 'Daily_Return', 'Volume']].copy()
            biocon_dates.columns = ['Date', 'Biocon_Close', 'Biocon_Return', 'Biocon_Volume']
            
            # Merge Nifty 50 data
            if nifty_df is not None and 'Source' in nifty_df and nifty_df['Source'].iloc[0] != 'Mock Data':
                nifty_dates = nifty_df[['Date', 'Close', 'Daily_Return']].copy()
                nifty_dates.columns = ['Date', 'Nifty_Close', 'Nifty_Return']
                biocon_dates = pd.merge(biocon_dates, nifty_dates, on='Date', how='left')
                
                biocon_dates['Market_Relative_Return'] = biocon_dates['Biocon_Return'] - biocon_dates['Nifty_Return']
                
                def calculate_rolling_beta(stock_returns, market_returns, window=60):
                    return stock_returns.rolling(window).cov(market_returns) / market_returns.rolling(window).var()
                
                biocon_dates['Beta_60D'] = calculate_rolling_beta(
                    biocon_dates['Biocon_Return'], 
                    biocon_dates['Nifty_Return']
                )
                biocon_dates['Beta_252D'] = calculate_rolling_beta(
                    biocon_dates['Biocon_Return'], 
                    biocon_dates['Nifty_Return'], 
                    window=252
                )
                
                biocon_dates['Alpha_60D'] = biocon_dates['Biocon_Return'] - (
                    biocon_dates['Beta_60D'] * biocon_dates['Nifty_Return']
                )
                
                biocon_dates['Correlation_60D'] = biocon_dates['Biocon_Return'].rolling(60).corr(
                    biocon_dates['Nifty_Return']
                )
            
            # Merge Sensex data
            if sensex_df is not None and 'Source' in sensex_df and sensex_df['Source'].iloc[0] != 'Mock Data':
                sensex_dates = sensex_df[['Date', 'Close', 'Daily_Return']].copy()
                sensex_dates.columns = ['Date', 'Sensex_Close', 'Sensex_Return']
                biocon_dates = pd.merge(biocon_dates, sensex_dates, on='Date', how='left')
                
                biocon_dates['Sensex_Relative_Return'] = biocon_dates['Biocon_Return'] - biocon_dates['Sensex_Return']
                
                biocon_dates['Sensex_Beta_60D'] = calculate_rolling_beta(
                    biocon_dates['Biocon_Return'], 
                    biocon_dates['Sensex_Return']
                )
                biocon_dates['Sensex_Alpha_60D'] = biocon_dates['Biocon_Return'] - (
                    biocon_dates['Sensex_Beta_60D'] * biocon_dates['Sensex_Return']
                )
            
            # Merge Nifty Pharma data
            if pharma_df is not None and 'Source' in pharma_df and pharma_df['Source'].iloc[0] != 'Mock Data':
                pharma_dates = pharma_df[['Date', 'Close', 'Daily_Return']].copy()
                pharma_dates.columns = ['Date', 'Pharma_Close', 'Pharma_Return']
                biocon_dates = pd.merge(biocon_dates, pharma_dates, on='Date', how='left')
                
                biocon_dates['Sector_Relative_Return'] = biocon_dates['Biocon_Return'] - biocon_dates['Pharma_Return']
                biocon_dates['Sector_Beta_60D'] = calculate_rolling_beta(
                    biocon_dates['Biocon_Return'], 
                    biocon_dates['Pharma_Return']
                )
                biocon_dates['Sector_Alpha_60D'] = biocon_dates['Biocon_Return'] - (
                    biocon_dates['Sector_Beta_60D'] * biocon_dates['Pharma_Return']
                )
            
            merge_columns = [col for col in biocon_dates.columns if col not in ['Biocon_Close', 'Biocon_Return', 'Biocon_Volume']]
            biocon_df = pd.merge(biocon_df, biocon_dates[merge_columns], on='Date', how='left')
            
            logger.info("✅ Market relative metrics calculated successfully")
            return biocon_df
            
        except Exception as e:
            logger.error(f"❌ Error calculating market metrics: {str(e)}")
            return biocon_df
    
    def add_bse_announcements(self, df):
        """
        Add BSE announcements to the dataset
        """
        logger.info("📢 Adding BSE announcements to dataset...")
        
        try:
            announcements_df = self.fetch_bse_announcements()
            if announcements_df is not None and not announcements_df.empty:
                df['Is_Announcement'] = 0
                df['Is_Dividend'] = 0
                df['Is_Board_Meeting'] = 0
                df['Is_Result'] = 0
                
                for _, row in announcements_df.iterrows():
                    date_mask = df['Date'] == row['Date']
                    df.loc[date_mask, 'Is_Announcement'] = 1
                    
                    purpose = row['Purpose'].lower()
                    if 'dividend' in purpose:
                        df.loc[date_mask, 'Is_Dividend'] = 1
                    if 'board meeting' in purpose:
                        df.loc[date_mask, 'Is_Board_Meeting'] = 1
                    if 'result' in purpose or 'quarter' in purpose:
                        df.loc[date_mask, 'Is_Result'] = 1
                
                logger.info("✅ BSE announcements added successfully")
            else:
                logger.warning("⚠️ No announcements to add")
                df['Is_Announcement'] = 0
                df['Is_Dividend'] = 0
                df['Is_Board_Meeting'] = 0
                df['Is_Result'] = 0
                
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding BSE announcements: {str(e)}")
            df['Is_Announcement'] = 0
            df['Is_Dividend'] = 0
            df['Is_Board_Meeting'] = 0
            df['Is_Result'] = 0
            return df
    
    def add_advanced_features(self, df):
        """
        Add advanced feature engineering
        """
        logger.info("🧠 Adding advanced features...")
        
        try:
            df['Price_Momentum_5'] = df['Close'].pct_change(5)
            df['Price_Momentum_10'] = df['Close'].pct_change(10)
            df['Price_Momentum_20'] = df['Close'].pct_change(20)
            
            for period in [20, 50]:
                df[f'Price_Position_{period}'] = (
                    (df['Close'] - df['Low'].rolling(period).min()) / 
                    (df['High'].rolling(period).max() - df['Low'].rolling(period).min())
                )
            
            df['Volatility_Regime'] = np.where(
                df['Volatility_20'] > df['Volatility_20'].rolling(252).quantile(0.75), 
                'High', 
                np.where(
                    df['Volatility_20'] < df['Volatility_20'].rolling(252).quantile(0.25), 
                    'Low', 
                    'Medium'
                )
            )
            
            df['Volume_Spike'] = (df['Volume'] > df['Volume_SMA_20'] * 2).astype(int)
            df['Volume_Drought'] = (df['Volume'] < df['Volume_SMA_20'] * 0.5).astype(int)
            
            df['Gap_Up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
            df['Gap_Down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) < -0.02).astype(int)
            df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
            df['Is_Year_End'] = df['Date'].dt.is_year_end.astype(int)
            
            lag_features = ['Close', 'Volume', 'Daily_Return', 'RSI_14', 'MACD']
            for feature in lag_features:
                if feature in df.columns:
                    for lag in [1, 2, 3, 5]:
                        df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
            
            for horizon in [1, 3, 5, 10]:
                df[f'Forward_Return_{horizon}D'] = df['Close'].pct_change(-horizon)
                df[f'Forward_Direction_{horizon}D'] = (df[f'Forward_Return_{horizon}D'] > 0).astype(int)
            
            df['Max_Drawdown_20D'] = (df['Close'] / df['Close'].rolling(20).max() - 1)
            df['Upside_Capture'] = np.where(
                df['Nifty_Return'] > 0 if 'Nifty_Return' in df.columns else False,
                df['Daily_Return'] / df['Nifty_Return'] if 'Nifty_Return' in df.columns else np.nan,
                np.nan
            )
            df['Downside_Capture'] = np.where(
                df['Nifty_Return'] < 0 if 'Nifty_Return' in df.columns else False,
                df['Daily_Return'] / df['Nifty_Return'] if 'Nifty_Return' in df.columns else np.nan,
                np.nan
            )
            
            logger.info("✅ Advanced features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding advanced features: {str(e)}")
            return df
    
    def collect_all_data(self, allow_mock_data=False):
        """
        Collect data from all configured sources
        """
        logger.info("🎯 Starting comprehensive data collection...")
        
        collected_data = {}
        
        for symbol_name, symbol_group in self.symbols.items():
            logger.info(f"📈 Collecting {symbol_name} data...")
            try:
                data = self.download_stock_data(symbol_group, symbol_name, allow_mock_data)
                if data is not None:
                    collected_data[symbol_name] = data
                    logger.info(f"✅ {symbol_name}: {len(data)} records collected")
                else:
                    logger.warning(f"⚠️ Failed to collect {symbol_name} data, continuing with other symbols")
            except Exception as e:
                logger.error(f"❌ Error collecting {symbol_name} data: {str(e)}")
                logger.warning(f"⚠️ Continuing with other symbols")
        
        self.collected_data = collected_data
        logger.info(f"📊 Total datasets collected: {len(collected_data)}")
        return collected_data
    
    def create_master_dataset(self):
        """
        Create the comprehensive master dataset
        """
        logger.info("🏗️ Creating master dataset...")
        
        if 'BIOCON' not in self.collected_data:
            raise Exception("❌ No Biocon data collected - cannot proceed")
        
        master_df = self.collected_data['BIOCON'].copy()
        logger.info(f"📊 Base dataset: {len(master_df)} Biocon records")
        
        master_df = self.add_basic_technical_indicators(master_df)
        
        master_df = self.add_bse_announcements(master_df)
        
        nifty_data = self.collected_data.get('NIFTY50')
        sensex_data = self.collected_data.get('SENSEX')
        pharma_data = self.collected_data.get('NIFTY_PHARMA')
        master_df = self.add_market_relative_metrics(master_df, nifty_data, sensex_data, pharma_data)
        
        master_df = self.add_advanced_features(master_df)
        
        master_df = master_df.sort_values('Date').reset_index(drop=True)
        
        if nifty_data is not None and 'Source' in nifty_data and nifty_data['Source'].iloc[0] != 'Mock Data':
            nifty_subset = nifty_data[['Date', 'Close', 'Daily_Return']].copy()
            nifty_subset.columns = ['Date', 'Nifty50_Close', 'Nifty50_Return']
            master_df = pd.merge(master_df, nifty_subset, on='Date', how='left')
        
        if sensex_data is not None and 'Source' in sensex_data and sensex_data['Source'].iloc[0] != 'Mock Data':
            sensex_subset = sensex_data[['Date', 'Close', 'Daily_Return']].copy()
            sensex_subset.columns = ['Date', 'Sensex_Close', 'Sensex_Return']
            master_df = pd.merge(master_df, sensex_subset, on='Date', how='left')
        
        if pharma_data is not None and 'Source' in pharma_data and pharma_data['Source'].iloc[0] != 'Mock Data':
            pharma_subset = pharma_data[['Date', 'Close', 'Daily_Return']].copy()
            pharma_subset.columns = ['Date', 'NiftyPharma_Close', 'NiftyPharma_Return']
            master_df = pd.merge(master_df, pharma_subset, on='Date', how='left')
        
        self.master_df = master_df
        logger.info(f"✅ Master dataset created: {len(master_df)} records, {len(master_df.columns)} features")
        return master_df
    
    def save_data(self):
        """
        Save the comprehensive dataset
        """
        logger.info("💾 Saving stock data...")
        
        try:
            filepath = PATHS['data'] / DATA_FILES['stock_data']
            self.master_df.to_csv(filepath, index=False)
            
            if filepath.exists() and filepath.stat().st_size > 1000:
                logger.info(f"✅ Data saved successfully: {filepath}")
                logger.info(f"📁 File size: {filepath.stat().st_size / 1024:.1f} KB")
                
                tech_indicators = [col for col in self.master_df.columns 
                                 if any(indicator in col for indicator in 
                                       ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'ADX'])]
                
                tech_df = self.master_df[['Date', 'Close'] + tech_indicators].copy()
                tech_filepath = PATHS['data'] / DATA_FILES['technical_indicators']
                tech_df.to_csv(tech_filepath, index=False)
                
                announcement_cols = [col for col in self.master_df.columns if col.startswith('Is_')]
                if announcement_cols:
                    announcements_df = self.master_df[['Date'] + announcement_cols].copy()
                    announcements_filepath = PATHS['data'] / 'bse_announcements.csv'
                    announcements_df.to_csv(announcements_filepath, index=False)
                    logger.info(f"✅ Announcements saved: {announcements_filepath}")
                
                logger.info(f"✅ Technical indicators saved: {tech_filepath}")
                return True
            else:
                logger.error("❌ File not saved properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving data: {str(e)}")
            return False
    
    def print_comprehensive_summary(self):
        """
        Print detailed analysis summary
        """
        if self.master_df is None:
            logger.error("❌ No data to summarize")
            return
        
        df = self.master_df
        
        print("\n" + "="*80)
        print("🚀 BIOCON ADVANCED STOCK DATA COLLECTION SUMMARY")
        print("="*80)
        
        print(f"📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Features: {len(df.columns):,}")
        print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"   Trading Days: {len(df):,}")
        print(f"   Data Sources: {', '.join(df['Source'].unique())}")
        
        if 'Close' in df.columns:
            latest_price = df['Close'].iloc[-1]
            first_price = df['Close'].iloc[0]
            total_return = ((latest_price / first_price) - 1) * 100
            max_price = df['Close'].max()
            min_price = df['Close'].min()
            avg_volume = df['Volume'].mean()
            
            print(f"\n💰 Price Analysis:")
            print(f"   Current Price: ₹{latest_price:.2f}")
            print(f"   Total Return: {total_return:.1f}%")
            print(f"   Price Range: ₹{min_price:.2f} - ₹{max_price:.2f}")
            print(f"   Average Volume: {avg_volume:,.0f}")
        
        tech_indicators = [col for col in df.columns if any(indicator in col for indicator in 
                          ['RSI', 'MACD', 'BB', 'SMA', 'EMA', 'ATR'])]
        
        print(f"\n🔧 Technical Indicators ({len(tech_indicators)} total):")
        indicator_groups = {
            'Trend': [col for col in tech_indicators if any(x in col for x in ['SMA', 'EMA', 'BB'])],
            'Momentum': [col for col in tech_indicators if any(x in col for x in ['RSI', 'MACD', 'ROC'])],
            'Volatility': [col for col in tech_indicators if any(x in col for x in ['ATR', 'Volatility'])],
            'Volume': [col for col in tech_indicators if 'Volume' in col or 'OBV' in col]
        }
        
        for group, indicators in indicator_groups.items():
            print(f"   {group}: {len(indicators)} indicators")
        
        announcement_cols = [col for col in df.columns if col.startswith('Is_')]
        if announcement_cols:
            print(f"\n📢 Announcements:")
            for col in announcement_cols:
                count = df[col].sum()
                if count > 0:
                    print(f"   {col.replace('Is_', '')}: {count} events")
        
        missing_values = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        completeness = ((total_cells - missing_values) / total_cells) * 100
        
        print(f"\n📊 Data Quality:")
        print(f"   Completeness: {completeness:.1f}%")
        print(f"   Missing Values: {missing_values:,}")
        
        market_metrics = [col for col in df.columns if any(x in col for x in 
                         ['Beta', 'Alpha', 'Market_Relative', 'Sector_Relative', 'Sensex_Relative'])]
        
        if market_metrics:
            print(f"\n📈 Market Metrics:")
            if 'Beta_60D' in df.columns:
                current_beta = df['Beta_60D'].iloc[-1]
                if not pd.isna(current_beta):
                    print(f"   Current Nifty Beta (60D): {current_beta:.2f}")
            
            if 'Sensex_Beta_60D' in df.columns:
                current_sensex_beta = df['Sensex_Beta_60D'].iloc[-1]
                if not pd.isna(current_sensex_beta):
                    print(f"   Current Sensex Beta (60D): {current_sensex_beta:.2f}")
            
            if 'Market_Relative_Return' in df.columns:
                avg_relative_return = df['Market_Relative_Return'].mean() * 252 * 100
                if not pd.isna(avg_relative_return):
                    print(f"   Avg Nifty Relative Return: {avg_relative_return:.1f}% p.a.")
            
            if 'Sensex_Relative_Return' in df.columns:
                avg_sensex_relative = df['Sensex_Relative_Return'].mean() * 252 * 100
                if not pd.isna(avg_sensex_relative):
                    print(f"   Avg Sensex Relative Return: {avg_sensex_relative:.1f}% p.a.")
        
        feature_categories = {
            'Price Features': [col for col in df.columns if any(x in col for x in 
                             ['Close', 'Open', 'High', 'Low', 'Return', 'Momentum'])],
            'Technical Indicators': tech_indicators,
            'Volume Features': [col for col in df.columns if 'Volume' in col or 'OBV' in col],
            'Market Relative': market_metrics,
            'Time Features': [col for col in df.columns if any(x in col for x in 
                            ['Day_of_Week', 'Month', 'Quarter', 'Year'])],
            'Lag Features': [col for col in df.columns if 'Lag' in col],
            'Forward Features': [col for col in df.columns if 'Forward' in col],
            'Announcements': announcement_cols
        }
        
        print(f"\n🎯 Feature Engineering Summary:")
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
        print(f"\n📁 Output Files:")
        print(f"   ✅ data/stock_data.csv - Complete dataset ({len(df)} rows)")
        print(f"   ✅ data/technical_indicators.csv - Technical indicators only")
        print(f"   ✅ data/bse_announcements.csv - BSE announcements")
        
        print(f"\n🎉 Ready for Day 1 Step 2: News Data Collection")
        print("="*80)
    
    def execute(self, allow_mock_data=False):
        """
        Execute the complete advanced stock data collection pipeline
        """
        try:
            logger.info("🚀 Starting Advanced Stock Data Collection Pipeline...")
            
            collected_data = self.collect_all_data(allow_mock_data)
            
            if not collected_data or 'BIOCON' not in collected_data:
                raise Exception("❌ No Biocon data collected from any source")
            
            if any(df['Source'].iloc[0] == 'Mock Data' for df in collected_data.values() if df is not None):
                logger.warning("⚠️ Mock data used for one or more datasets")
                if not allow_mock_data:
                    raise Exception("❌ Mock data detected; use --allow-mock-data to proceed with mock data")
            
            master_df = self.create_master_dataset()
            
            success = self.save_data()
            
            if success:
                self.print_comprehensive_summary()
                logger.info("🎉 Stock data collection completed successfully!")
                return True
            else:
                logger.error("❌ Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stock data collection failed: {str(e)}")
            logger.info("💡 Troubleshooting steps:")
            logger.info("  1. Verify symbols on https://finance.yahoo.com/quote/BIOCON.NS")
            logger.info("  2. Check NSE India access: curl -I https://www.nseindia.com/api/quote-equity?symbol=BIOCON")
            logger.info("  3. Use a VPN (NordVPN/ExpressVPN) with an Indian server if outside India")
            logger.info("  4. Set DNS: sudo networksetup -setdnsservers Wi-Fi 8.8.8.8 1.1.1.1")
            logger.info("  5. Set proxies: export HTTP_PROXY='http://proxy.example.com:8080'")
            logger.info("  6. Run with --allow-mock-data for testing")
            return False
        finally:
            self.session.close()

def main(allow_mock_data=False):
    """
    Main execution function for Day 1 - Step 1
    """
    print("🚀 BIOCON FDA PROJECT - DAY 1 STEP 1")
    print("Advanced Stock Data Collection with Technical Indicators")
    print("="*60)
    print(f"🏢 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"📅 Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"🎯 Features: Technical Indicators + Market Metrics (Nifty 50, Sensex, Nifty Pharma) + Advanced Features + BSE Announcements")
    print("-" * 60)
    
    collector = AdvancedStockDataCollector()
    success = collector.execute(allow_mock_data=allow_mock_data)
    
    if success:
        print("\n🎉 SUCCESS: Advanced stock data collection completed!")
        print("✅ Comprehensive dataset with 100+ features created")
        print("✅ Technical indicators calculated using TA-Lib")
        print("✅ Market relative metrics computed (Nifty 50, Sensex, Nifty Pharma)")
        print("✅ Advanced features engineered")
        print("✅ BSE announcements integrated")
        print("✅ Data saved to: data/stock_data.csv")
        print("✅ Announcements saved to: data/bse_announcements.csv")
        print("🔄 Ready for Day 1 Step 2: News Data Collection")
    else:
        print("\n❌ FAILED: Stock data collection failed")
        print("💡 Check logs for details: logs/stock_collection.log")
        print("🔧 Troubleshooting:")
        print("  - Verify symbols on https://finance.yahoo.com/quote/BIOCON.NS")
        print("  - Check NSE India access: curl -I https://www.nseindia.com/api/quote-equity?symbol=BIOCON")
        print("  - Use a VPN (NordVPN/ExpressVPN) with an Indian server if outside India")
        print("  - Set DNS: sudo networksetup -setdnsservers Wi-Fi 8.8.8.8 1.1.1.1")
        print("  - Set proxies: export HTTP_PROXY='http://proxy.example.com:8080'")
        print("  - Run with --allow-mock-data for testing")
    
    return success

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Stock Data Collection for Biocon FDA Project")
    parser.add_argument('--allow-mock-data', action='store_true', help="Allow mock data for testing")
    args = parser.parse_args()
    
    main(allow_mock_data=args.allow_mock_data)