import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import requests
from bs4 import BeautifulSoup
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioconDataCollector:
    """
    Robust Biocon stock data collector with multiple data sources.
    Ensures data is properly collected and saved to CSV.
    """
    
    def __init__(self, start_date="2015-01-01", end_date="2025-06-26"):
        self.start_date = start_date
        self.end_date = end_date
        self.create_directories()
        
        # Target symbols
        self.symbols = {
            'BIOCON': {
                'primary': 'BIOCON.NS',
                'alternatives': ['BIOCON.BO', '532523.BO'],
                'name': 'Biocon Limited',
                'type': 'stock'
            },
            'NIFTY50': {
                'primary': '^NSEI',
                'alternatives': ['NIFTY50.NS'],
                'name': 'Nifty 50',
                'type': 'index'
            },
            'NIFTY_PHARMA': {
                'primary': '^CNXPHARMA',
                'alternatives': ['CNXPHARMA.NS', 'NIFTYPHARMA.NS'],
                'name': 'Nifty Pharma',
                'type': 'index'
            }
        }
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['data', 'logs']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def download_yfinance_data(self, symbol_info, symbol_key):
        """Download data using yfinance with fallback symbols"""
        symbols_to_try = [symbol_info['primary']] + symbol_info['alternatives']
        
        for symbol in symbols_to_try:
            try:
                logger.info(f"Attempting to download {symbol_info['name']} using {symbol}")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True
                )
                
                if not df.empty and len(df) > 500:  # Ensure substantial data
                    # Reset index and add metadata
                    df.reset_index(inplace=True)
                    df['Symbol'] = symbol
                    df['Company'] = symbol_info['name']
                    df['Type'] = symbol_info['type']
                    df['Source'] = 'Yahoo Finance'
                    
                    # Calculate basic returns
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Daily_Return_Percent'] = df['Daily_Return'] * 100
                    
                    logger.info(f"Successfully downloaded {len(df)} records for {symbol_info['name']}")
                    return df
                else:
                    logger.warning(f"Insufficient data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
                continue
        
        logger.error(f"Failed to download {symbol_info['name']} from all sources")
        return None
    
    def try_nse_api(self, symbol):
        """Attempt to get data from NSE (if available)"""
        try:
            # NSE API is often restricted, this is a placeholder
            logger.info(f"Attempting NSE API for {symbol}")
            # NSE API implementation would go here
            # For now, return None as NSE API requires specific setup
            return None
        except Exception as e:
            logger.warning(f"NSE API failed for {symbol}: {str(e)}")
            return None
    
    def try_bse_api(self, symbol):
        """Attempt to get data from BSE (if available)"""
        try:
            # BSE API is often restricted, this is a placeholder
            logger.info(f"Attempting BSE API for {symbol}")
            # BSE API implementation would go here
            # For now, return None as BSE API requires specific setup
            return None
        except Exception as e:
            logger.warning(f"BSE API failed for {symbol}: {str(e)}")
            return None
    
    def scrape_moneycontrol(self, symbol_name):
        """Scrape basic data from MoneyControl (simplified)"""
        try:
            logger.info(f"Attempting MoneyControl scraping for {symbol_name}")
            
            # MoneyControl URL pattern (simplified)
            # This is a basic example - full implementation would need detailed scraping
            url = f"https://www.moneycontrol.com/india/stockpricequote/{symbol_name.lower()}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # For now, return None as MoneyControl scraping requires careful implementation
            # to avoid being blocked and to handle their specific data structure
            logger.warning("MoneyControl scraping not implemented in this version")
            return None
            
        except Exception as e:
            logger.warning(f"MoneyControl scraping failed: {str(e)}")
            return None
    
    def add_technical_indicators(self, df):
        """Add essential technical indicators"""
        try:
            # Moving averages
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            
            # Volatility
            df['Volatility_20D'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Price momentum
            df['Price_Change_5D'] = df['Close'].pct_change(5) * 100
            df['Price_Change_20D'] = df['Close'].pct_change(20) * 100
            
            logger.info("Technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def calculate_market_metrics(self, biocon_df, nifty_df):
        """Calculate market-relative metrics"""
        try:
            if nifty_df is None:
                logger.warning("No Nifty data available for market metrics")
                return biocon_df
            
            # Merge datasets
            biocon_dates = biocon_df[['Date', 'Daily_Return']].copy()
            nifty_dates = nifty_df[['Date', 'Daily_Return']].copy()
            nifty_dates.columns = ['Date', 'Nifty_Return']
            
            merged = pd.merge(biocon_dates, nifty_dates, on='Date', how='inner')
            
            # Calculate beta (60-day rolling)
            def calculate_rolling_beta(stock_returns, market_returns, window=60):
                beta_values = []
                for i in range(len(stock_returns)):
                    if i < window:
                        beta_values.append(np.nan)
                    else:
                        stock_subset = stock_returns.iloc[i-window:i]
                        market_subset = market_returns.iloc[i-window:i]
                        
                        if len(stock_subset.dropna()) > 30 and len(market_subset.dropna()) > 30:
                            covariance = np.cov(stock_subset.dropna(), market_subset.dropna())[0][1]
                            market_variance = np.var(market_subset.dropna())
                            beta = covariance / market_variance if market_variance != 0 else np.nan
                            beta_values.append(beta)
                        else:
                            beta_values.append(np.nan)
                
                return pd.Series(beta_values, index=stock_returns.index)
            
            merged['Beta_60D'] = calculate_rolling_beta(merged['Daily_Return'], merged['Nifty_Return'])
            merged['Alpha'] = merged['Daily_Return'] - (merged['Beta_60D'] * merged['Nifty_Return'])
            merged['Market_Adjusted_Return'] = merged['Daily_Return'] - merged['Nifty_Return']
            
            # Merge back to original data
            biocon_df = pd.merge(
                biocon_df,
                merged[['Date', 'Beta_60D', 'Alpha', 'Market_Adjusted_Return']],
                on='Date',
                how='left'
            )
            
            logger.info("Market metrics calculated successfully")
            return biocon_df
            
        except Exception as e:
            logger.error(f"Error calculating market metrics: {str(e)}")
            return biocon_df
    
    def collect_all_data(self):
        """Collect data from all sources"""
        collected_data = {}
        
        for symbol_key, symbol_info in self.symbols.items():
            logger.info(f"Collecting data for {symbol_info['name']}")
            
            # Try multiple sources in order of preference
            data = None
            
            # 1. Try NSE API (if available)
            if symbol_key == 'BIOCON':
                data = self.try_nse_api(symbol_info['primary'])
            
            # 2. Try BSE API (if available)
            if data is None and symbol_key == 'BIOCON':
                data = self.try_bse_api(symbol_info['primary'])
            
            # 3. Try MoneyControl scraping (if available)
            if data is None and symbol_key == 'BIOCON':
                data = self.scrape_moneycontrol(symbol_info['name'])
            
            # 4. Use Yahoo Finance (most reliable)
            if data is None:
                data = self.download_yfinance_data(symbol_info, symbol_key)
            
            if data is not None:
                collected_data[symbol_key] = data
                logger.info(f"Successfully collected {symbol_info['name']} data")
            else:
                logger.error(f"Failed to collect {symbol_info['name']} data")
        
        return collected_data
    
    def create_master_dataset(self, collected_data):
        """Create the master dataset"""
        if 'BIOCON' not in collected_data:
            raise Exception("No Biocon data collected - cannot proceed")
        
        # Start with Biocon data
        master_df = collected_data['BIOCON'].copy()
        
        # Add technical indicators
        master_df = self.add_technical_indicators(master_df)
        
        # Add market metrics if available
        nifty_data = collected_data.get('NIFTY50')
        if nifty_data is not None:
            master_df = self.calculate_market_metrics(master_df, nifty_data)
        
        # Add benchmark data as separate columns
        for benchmark in ['NIFTY50', 'NIFTY_PHARMA']:
            if benchmark in collected_data:
                benchmark_df = collected_data[benchmark][['Date', 'Close', 'Daily_Return']].copy()
                benchmark_df.columns = ['Date', f'{benchmark}_Close', f'{benchmark}_Return']
                master_df = pd.merge(master_df, benchmark_df, on='Date', how='left')
        
        # Add date features
        master_df['Year'] = master_df['Date'].dt.year
        master_df['Month'] = master_df['Date'].dt.month
        master_df['Day_of_Week'] = master_df['Date'].dt.dayofweek
        master_df['Is_Month_End'] = master_df['Date'].dt.is_month_end
        master_df['Is_Quarter_End'] = master_df['Date'].dt.is_quarter_end
        
        # Sort by date
        master_df = master_df.sort_values('Date')
        
        return master_df
    
    def save_data(self, df):
        """Save data to CSV with proper validation"""
        try:
            # Ensure data directory exists
            if not os.path.exists('data'):
                os.makedirs('data')
            
            filepath = 'data/stock_data.csv'
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            # Validate file was created and has content
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                if file_size > 1000:  # File should be at least 1KB
                    logger.info(f"Data successfully saved to {filepath}")
                    logger.info(f"File size: {file_size/1024:.1f} KB")
                    
                    # Read back to verify
                    test_df = pd.read_csv(filepath)
                    logger.info(f"Verification: {len(test_df)} rows, {len(test_df.columns)} columns")
                    return True
                else:
                    logger.error(f"File created but appears empty ({file_size} bytes)")
                    return False
            else:
                logger.error("File was not created")
                return False
                
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def print_summary(self, df):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print("BIOCON STOCK DATA COLLECTION SUMMARY")
        print("="*70)
        
        # Basic info
        print(f"Total Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Data Source: {df['Source'].iloc[0] if 'Source' in df.columns else 'Multiple'}")
        
        # Price info
        if 'Close' in df.columns:
            latest_price = df['Close'].iloc[-1]
            first_price = df['Close'].iloc[0]
            total_return = ((latest_price / first_price) - 1) * 100
            
            print(f"\nPrice Analysis:")
            print(f"  Latest Price: ₹{latest_price:.2f}")
            print(f"  Highest Price: ₹{df['Close'].max():.2f}")
            print(f"  Lowest Price: ₹{df['Close'].min():.2f}")
            print(f"  Total Return: {total_return:.1f}%")
        
        # Data quality
        missing_values = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        completeness = ((total_cells - missing_values) / total_cells) * 100
        
        print(f"\nData Quality:")
        print(f"  Missing Values: {missing_values:,}")
        print(f"  Data Completeness: {completeness:.1f}%")
        
        # Available columns
        print(f"\nKey Columns Available:")
        key_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 
                      'MA_20', 'RSI', 'MACD', 'Volatility_20D']
        for col in key_columns:
            if col in df.columns:
                print(f"  ✓ {col}")
            else:
                print(f"  ✗ {col}")
        
        print(f"\nFile Location: data/stock_data.csv")
        print("="*70)
    
    def execute(self):
        """Execute the complete data collection process"""
        try:
            logger.info("Starting Biocon data collection process")
            
            # Collect data from all sources
            collected_data = self.collect_all_data()
            
            if not collected_data:
                raise Exception("No data collected from any source")
            
            # Create master dataset
            master_df = self.create_master_dataset(collected_data)
            
            # Save data
            success = self.save_data(master_df)
            
            if success:
                self.print_summary(master_df)
                logger.info("Data collection completed successfully")
                return True
            else:
                logger.error("Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    """Main execution function"""
    print("Starting Biocon Stock Data Collection")
    print("Target: Biocon (BIOCON.NS), Nifty 50, Nifty Pharma")
    print("Period: 2015-2025 (10 years)")
    print("Sources: NSE, BSE, MoneyControl, Yahoo Finance")
    print("-" * 50)
    
    collector = BioconDataCollector()
    success = collector.execute()
    
    if success:
        print("\n✓ SUCCESS: Biocon stock data collection completed!")
        print("✓ Data saved to: data/stock_data.csv")
        print("✓ Ready for news data collection")
    else:
        print("\n✗ FAILED: Stock data collection failed")
        print("✗ Check logs for details")
    
    return success

if __name__ == "__main__":
    main()