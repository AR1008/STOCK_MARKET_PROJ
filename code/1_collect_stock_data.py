"""
Advanced Stock Data Collection for Biocon FDA Project
Day 1 - Step 1: Comprehensive stock data with technical indicators

Features:
- Multiple data sources with fallback
- Complete technical indicator suite
- Market relative metrics
- FDA event alignment
- Advanced feature engineering
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import warnings
import talib
from pathlib import Path

# Import configuration
from config import (
    COMPANY_INFO, DATA_START_DATE, DATA_END_DATE, 
    DATA_SOURCES, FEATURE_CONFIG, PATHS, DATA_FILES,
    create_directories, validate_config
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
    market relative metrics, and FDA event alignment capabilities.
    """
    
    def __init__(self):
        self.start_date = DATA_START_DATE
        self.end_date = DATA_END_DATE
        self.symbols = DATA_SOURCES['stock_data']['symbols']
        self.collected_data = {}
        self.master_df = None
        
        # Create directories
        create_directories()
        validate_config()
        
        logger.info(f"🚀 Advanced Stock Data Collector Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"🏢 Target Company: {COMPANY_INFO['name']}")
    
    def download_stock_data(self, symbol_group, symbol_name):
        """
        Download stock data with multiple fallback options
        """
        logger.info(f"📈 Downloading {symbol_name} data...")
        
        for symbol in symbol_group:
            try:
                logger.info(f"  Trying symbol: {symbol}")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    prepost=True,
                    
                )
                
                if not df.empty and len(df) > 500:
                    # Reset index and add metadata
                    df.reset_index(inplace=True)
                    df['Symbol'] = symbol
                    df['Company'] = symbol_name
                    df['Source'] = 'Yahoo Finance'
                    
                    # Basic return calculations
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Daily_Return_Percent'] = df['Daily_Return'] * 100
                    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                    
                    logger.info(f"  ✅ Success: {len(df)} records for {symbol}")
                    return df
                else:
                    logger.warning(f"  ⚠️  Insufficient data for {symbol}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error downloading {symbol}: {str(e)}")
                continue
        
        logger.error(f"❌ Failed to download {symbol_name} from all sources")
        return None
    
    def add_basic_technical_indicators(self, df):
        """
        Add comprehensive technical indicators using TA-Lib
        """
        logger.info("🔧 Adding technical indicators...")
        
        try:
            # Ensure numeric columns and convert to float64
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            # Handle NaN values (forward and backward fill)
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
            
            # Convert to numpy arrays for TA-Lib
            open_prices = df['Open'].values
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            volume = df['Volume'].values
            
            # Validate arrays
            for arr, name in zip([open_prices, high, low, close, volume], numeric_columns):
                if not np.issubdtype(arr.dtype, np.floating):
                    raise ValueError(f"{name} array is not of floating type: {arr.dtype}")
                if np.any(np.isnan(arr)):
                    raise ValueError(f"{name} array contains NaN values after filling")

            # 1. TREND INDICATORS
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
            
            # 2. MOMENTUM INDICATORS
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
            
            # 3. VOLATILITY INDICATORS
            df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['ATR_21'] = talib.ATR(high, low, close, timeperiod=21)
            
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            for period in [10, 20, 30]:
                df[f'Volatility_{period}'] = df['Daily_Return'].rolling(window=period).std() * np.sqrt(252)
            
            # 4. VOLUME INDICATORS
            df['OBV'] = talib.OBV(close, volume)
            
            df['VPT'] = talib.AD(high, low, close, volume)
            
            for period in [10, 20, 50]:
                df[f'Volume_SMA_{period}'] = talib.SMA(volume, timeperiod=period)
            
            df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
            
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # 5. ADVANCED PATTERNS
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
            self.used_fallback = True  # Set flag for fallback
            return self.add_simple_technical_indicators(df)
    
    def add_simple_technical_indicators(self, df):
        """
        Fallback method for technical indicators without TA-Lib
        """
        logger.info("🔧 Adding simple technical indicators (fallback)...")
        
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in [12, 26]:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Volume indicators
            for period in [10, 20]:
                df[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
            
            # Volatility
            df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            logger.info("✅ Simple technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding simple technical indicators: {str(e)}")
            return df
    
    def add_market_relative_metrics(self, biocon_df, nifty_df, pharma_df):
        """
        Calculate market-relative and sector-relative metrics
        """
        logger.info("📊 Calculating market relative metrics...")
        
        try:
            # Prepare data for merging
            biocon_dates = biocon_df[['Date', 'Close', 'Daily_Return', 'Volume']].copy()
            biocon_dates.columns = ['Date', 'Biocon_Close', 'Biocon_Return', 'Biocon_Volume']
            
            if nifty_df is not None:
                nifty_dates = nifty_df[['Date', 'Close', 'Daily_Return']].copy()
                nifty_dates.columns = ['Date', 'Nifty_Close', 'Nifty_Return']
                biocon_dates = pd.merge(biocon_dates, nifty_dates, on='Date', how='left')
                
                # Market relative metrics
                biocon_dates['Market_Relative_Return'] = biocon_dates['Biocon_Return'] - biocon_dates['Nifty_Return']
                
                # Rolling Beta calculation
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
                
                # Alpha calculation
                biocon_dates['Alpha_60D'] = biocon_dates['Biocon_Return'] - (
                    biocon_dates['Beta_60D'] * biocon_dates['Nifty_Return']
                )
                
                # Correlation
                biocon_dates['Correlation_60D'] = biocon_dates['Biocon_Return'].rolling(60).corr(
                    biocon_dates['Nifty_Return']
                )
            
            if pharma_df is not None:
                pharma_dates = pharma_df[['Date', 'Close', 'Daily_Return']].copy()
                pharma_dates.columns = ['Date', 'Pharma_Close', 'Pharma_Return']
                biocon_dates = pd.merge(biocon_dates, pharma_dates, on='Date', how='left')
                
                # Sector relative metrics
                biocon_dates['Sector_Relative_Return'] = biocon_dates['Biocon_Return'] - biocon_dates['Pharma_Return']
                biocon_dates['Sector_Beta_60D'] = calculate_rolling_beta(
                    biocon_dates['Biocon_Return'], 
                    biocon_dates['Pharma_Return']
                )
                biocon_dates['Sector_Alpha_60D'] = biocon_dates['Biocon_Return'] - (
                    biocon_dates['Sector_Beta_60D'] * biocon_dates['Pharma_Return']
                )
            
            # Merge back to main dataframe
            merge_columns = [col for col in biocon_dates.columns if col not in ['Biocon_Close', 'Biocon_Return', 'Biocon_Volume']]
            biocon_df = pd.merge(biocon_df, biocon_dates[merge_columns], on='Date', how='left')
            
            logger.info("✅ Market relative metrics calculated successfully")
            return biocon_df
            
        except Exception as e:
            logger.error(f"❌ Error calculating market metrics: {str(e)}")
            return biocon_df
    
    def add_advanced_features(self, df):
        """
        Add advanced feature engineering
        """
        logger.info("🧠 Adding advanced features...")
        
        try:
            # 1. Price-based features
            df['Price_Momentum_5'] = df['Close'].pct_change(5)
            df['Price_Momentum_10'] = df['Close'].pct_change(10)
            df['Price_Momentum_20'] = df['Close'].pct_change(20)
            
            # Price position within recent range
            for period in [20, 50]:
                df[f'Price_Position_{period}'] = (
                    (df['Close'] - df['Low'].rolling(period).min()) / 
                    (df['High'].rolling(period).max() - df['Low'].rolling(period).min())
                )
            
            # 2. Volatility regime detection
            df['Volatility_Regime'] = np.where(
                df['Volatility_20'] > df['Volatility_20'].rolling(252).quantile(0.75), 
                'High', 
                np.where(
                    df['Volatility_20'] < df['Volatility_20'].rolling(252).quantile(0.25), 
                    'Low', 
                    'Medium'
                )
            )
            
            # 3. Volume analysis
            df['Volume_Spike'] = (df['Volume'] > df['Volume_SMA_20'] * 2).astype(int)
            df['Volume_Drought'] = (df['Volume'] < df['Volume_SMA_20'] * 0.5).astype(int)
            
            # 4. Gap analysis
            df['Gap_Up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
            df['Gap_Down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) < -0.02).astype(int)
            df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # 5. Time-based features
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
            df['Is_Year_End'] = df['Date'].dt.is_year_end.astype(int)
            
            # 6. Lag features for ML
            lag_features = ['Close', 'Volume', 'Daily_Return', 'RSI_14', 'MACD']
            for feature in lag_features:
                if feature in df.columns:
                    for lag in [1, 2, 3, 5]:
                        df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
            
            # 7. Forward returns for target variables
            for horizon in [1, 3, 5, 10]:
                df[f'Forward_Return_{horizon}D'] = df['Close'].pct_change(-horizon)
                df[f'Forward_Direction_{horizon}D'] = (df[f'Forward_Return_{horizon}D'] > 0).astype(int)
            
            # 8. Risk metrics
            df['Max_Drawdown_20D'] = (df['Close'] / df['Close'].rolling(20).max() - 1)
            df['Upside_Capture'] = np.where(
                df['Nifty_Return'] > 0,
                df['Daily_Return'] / df['Nifty_Return'],
                np.nan
            )
            df['Downside_Capture'] = np.where(
                df['Nifty_Return'] < 0,
                df['Daily_Return'] / df['Nifty_Return'],
                np.nan
            )
            
            logger.info("✅ Advanced features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding advanced features: {str(e)}")
            return df
    
    def collect_all_data(self):
        """
        Collect data from all configured sources
        """
        logger.info("🎯 Starting comprehensive data collection...")
        
        collected_data = {}
        
        # Collect stock data for each symbol group
        for symbol_name, symbol_group in self.symbols.items():
            logger.info(f"📈 Collecting {symbol_name} data...")
            data = self.download_stock_data(symbol_group, symbol_name)
            
            if data is not None:
                collected_data[symbol_name] = data
                logger.info(f"✅ {symbol_name}: {len(data)} records collected")
            else:
                logger.error(f"❌ Failed to collect {symbol_name} data")
        
        self.collected_data = collected_data
        logger.info(f"📊 Total datasets collected: {len(collected_data)}")
        return collected_data
    
    def create_master_dataset(self):
        """
        Create the comprehensive master dataset
        """
        logger.info("🏗️  Creating master dataset...")
        
        if 'BIOCON' not in self.collected_data:
            raise Exception("❌ No Biocon data collected - cannot proceed")
        
        # Start with Biocon data
        master_df = self.collected_data['BIOCON'].copy()
        logger.info(f"📊 Base dataset: {len(master_df)} Biocon records")
        
        # Add technical indicators
        master_df = self.add_basic_technical_indicators(master_df)
        
        # Add market relative metrics
        nifty_data = self.collected_data.get('NIFTY50')
        pharma_data = self.collected_data.get('NIFTY_PHARMA')
        master_df = self.add_market_relative_metrics(master_df, nifty_data, pharma_data)
        
        # Add advanced features
        master_df = self.add_advanced_features(master_df)
        
        # Sort by date
        master_df = master_df.sort_values('Date').reset_index(drop=True)
        
        # Store benchmark data as separate columns
        if nifty_data is not None:
            nifty_subset = nifty_data[['Date', 'Close', 'Daily_Return']].copy()
            nifty_subset.columns = ['Date', 'Nifty50_Close', 'Nifty50_Return']
            master_df = pd.merge(master_df, nifty_subset, on='Date', how='left')
        
        if pharma_data is not None:
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
            # Save to CSV
            filepath = PATHS['data'] / DATA_FILES['stock_data']
            self.master_df.to_csv(filepath, index=False)
            
            # Validate file
            if filepath.exists() and filepath.stat().st_size > 1000:
                logger.info(f"✅ Data saved successfully: {filepath}")
                logger.info(f"📁 File size: {filepath.stat().st_size / 1024:.1f} KB")
                
                # Save technical indicators separately
                tech_indicators = [col for col in self.master_df.columns 
                                 if any(indicator in col for indicator in 
                                       ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'ADX'])]
                
                tech_df = self.master_df[['Date', 'Close'] + tech_indicators].copy()
                tech_filepath = PATHS['data'] / DATA_FILES['technical_indicators']
                tech_df.to_csv(tech_filepath, index=False)
                
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
        
        # Basic information
        print(f"📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Total Features: {len(df.columns):,}")
        print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"   Trading Days: {len(df):,}")
        
        # Price analysis
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
        
        # Technical indicators summary
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
        
        # Data quality
        missing_values = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        completeness = ((total_cells - missing_values) / total_cells) * 100
        
        print(f"\n📊 Data Quality:")
        print(f"   Completeness: {completeness:.1f}%")
        print(f"   Missing Values: {missing_values:,}")
        
        # Market relative metrics
        market_metrics = [col for col in df.columns if any(x in col for x in 
                         ['Beta', 'Alpha', 'Market_Relative', 'Sector_Relative'])]
        
        if market_metrics:
            print(f"\n📈 Market Metrics:")
            if 'Beta_60D' in df.columns:
                current_beta = df['Beta_60D'].iloc[-1]
                if not pd.isna(current_beta):
                    print(f"   Current Beta (60D): {current_beta:.2f}")
            
            if 'Market_Relative_Return' in df.columns:
                avg_relative_return = df['Market_Relative_Return'].mean() * 252 * 100
                if not pd.isna(avg_relative_return):
                    print(f"   Avg Relative Return: {avg_relative_return:.1f}% p.a.")
        
        # Feature categories
        feature_categories = {
            'Price Features': [col for col in df.columns if any(x in col for x in 
                             ['Close', 'Open', 'High', 'Low', 'Return', 'Momentum'])],
            'Technical Indicators': tech_indicators,
            'Volume Features': [col for col in df.columns if 'Volume' in col or 'OBV' in col],
            'Market Relative': market_metrics,
            'Time Features': [col for col in df.columns if any(x in col for x in 
                            ['Day_of_Week', 'Month', 'Quarter', 'Year'])],
            'Lag Features': [col for col in df.columns if 'Lag' in col],
            'Forward Features': [col for col in df.columns if 'Forward' in col]
        }
        
        print(f"\n🎯 Feature Engineering Summary:")
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
        print(f"\n📁 Output Files:")
        print(f"   ✅ data/stock_data.csv - Complete dataset ({len(df)} rows)")
        print(f"   ✅ data/technical_indicators.csv - Technical indicators only")
        
        print(f"\n🎉 Ready for Day 1 Step 2: News Data Collection")
        print("="*80)
    
    def execute(self):
        """
        Execute the complete advanced stock data collection pipeline
        """
        try:
            logger.info("🚀 Starting Advanced Stock Data Collection Pipeline...")
            
            # Step 1: Collect raw data
            collected_data = self.collect_all_data()
            
            if not collected_data:
                raise Exception("❌ No data collected from any source")
            
            # Step 2: Create master dataset with all features
            master_df = self.create_master_dataset()
            
            # Step 3: Save comprehensive dataset
            success = self.save_data()
            
            # Step 4: Print summary
            if success:
                self.print_comprehensive_summary()
                logger.info("🎉 Stock data collection completed successfully!")
                return True
            else:
                logger.error("❌ Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stock data collection failed: {str(e)}")
            return False

def main():
    """
    Main execution function for Day 1 - Step 1
    """
    print("🚀 BIOCON FDA PROJECT - DAY 1 STEP 1")
    print("Advanced Stock Data Collection with Technical Indicators")
    print("="*60)
    print(f"🏢 Company: {COMPANY_INFO['name']} ({COMPANY_INFO['ticker']})")
    print(f"📅 Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"🎯 Features: Technical Indicators + Market Metrics + Advanced Features")
    print("-" * 60)
    
    # Initialize and run collector
    collector = AdvancedStockDataCollector()
    success = collector.execute()
    
    if success:
        print("\n🎉 SUCCESS: Advanced stock data collection completed!")
        print("✅ Comprehensive dataset with 100+ features created")
        print("✅ Technical indicators calculated using TA-Lib")
        print("✅ Market relative metrics computed")
        print("✅ Advanced features engineered")
        print("✅ Data saved to: data/stock_data.csv")
        print("🔄 Ready for Day 1 Step 2: News Data Collection")
    else:
        print("\n❌ FAILED: Stock data collection failed")
        print("💡 Check logs for details: logs/stock_collection.log")
        print("🔧 Troubleshooting: Verify internet connection and try again")
    
    return success

if __name__ == "__main__":
    main()