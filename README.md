# Biocon FDA Drug Journey Analysis

## 🎯 Project Overview

This project analyzes **Biocon's stock price movements** in relation to **FDA milestones** for their drug **Semglee (insulin glargine-yfgn)** - the first FDA-approved interchangeable insulin biosimilar. The analysis combines stock data with news sentiment analysis to predict future price movements using machine learning models.

### Key Objectives
- Track Biocon's complete FDA drug approval journey (2015-2025)
- Analyze correlation between FDA milestones and stock price movements  
- Predict stock price movements using news sentiment and FDA milestone data
- Create a comprehensive model for pharmaceutical stock analysis

## 📊 Project Structure

```
biocon_fda_project/
│
├── data/                             # All data files
│   ├── stock_data.csv               # Stock prices with technical indicators
│   ├── news_data.csv                # News articles with sentiment analysis
│   ├── daily_sentiment.csv          # Daily aggregated sentiment scores
│   └── combined_data.csv            # Combined stock + news dataset
│
├── code/                             # Python analysis scripts
│   ├── 1_collect_stock_data.py      # Day 1: Stock data collection
│   ├── 2_collect_news_data.py       # Day 1: News sentiment collection
│   ├── 3_train_model.py             # Day 2: Model training
│   ├── 4_test_model.py              # Day 3: Model testing & evaluation
│   └── 5_predict_new_drug.py        # Day 4: Future predictions
│
├── models/                           # Trained ML models
│   ├── final_model.pkl              # Best performing model
│   ├── lstm_model.pkl               # LSTM neural network model
│   ├── sentiment_model.pkl          # Sentiment analysis model
│   └── scalers.pkl                  # Data preprocessing scalers
│
├── results/                          # Analysis results
│   ├── model_performance.csv        # Model accuracy metrics
│   ├── predictions.csv              # Stock price predictions
│   ├── correlation_analysis.csv     # News-stock correlation data
│   └── charts/                      # Generated visualizations
│       ├── stock_price_chart.png    # Stock price timeline
│       ├── sentiment_chart.png      # Sentiment analysis charts
│       └── correlation_plot.png     # Correlation analysis plots
│
├── notebooks/                        # Jupyter notebooks
│   ├── data_exploration.ipynb       # Exploratory data analysis
│   └── model_testing.ipynb          # Model experimentation
│
├── requirements.txt                  # Python dependencies
├── config.py                        # Configuration settings
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for data collection

### Installation

1. **Clone or create the project structure:**
```bash
# Create project directory
mkdir biocon_fda_project
cd biocon_fda_project

# Create all subdirectories
mkdir -p data code models results/charts notebooks logs
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data (required for sentiment analysis):**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## 📈 Usage Guide

### Day 1: Data Collection

#### Step 1: Collect Stock Data
```bash
python code/1_collect_stock_data.py
```
**What it does:**
- Fetches Biocon stock data (2015-2025) from Yahoo Finance
- Downloads benchmark indices (Nifty 50, Nifty Pharma)
- Calculates technical indicators (RSI, MACD, Bollinger Bands)
- Generates comprehensive stock dataset

**Output:** `data/stock_data.csv`

#### Step 2: Collect News Data
```bash
python code/2_collect_news_data.py
```
**What it does:**
- Searches for FDA milestone news across multiple sources
- Collects comprehensive company news (earnings, partnerships, regulatory)
- Performs sentiment analysis on all articles
- Tracks drug-specific news vs general company news
- Creates daily sentiment aggregations

**Output:** `data/news_data.csv`, `data/daily_sentiment.csv`

### Day 2: Model Training

```bash
python code/3_train_model.py
```
**What it does:**
- Combines stock data with news sentiment
- Engineers 50+ features including:
  - Price momentum indicators
  - Sentiment moving averages
  - FDA milestone flags
  - Volume and volatility metrics
- Trains multiple models:
  - Linear Regression, Ridge, Lasso
  - Random Forest, Gradient Boosting
  - XGBoost, LightGBM
  - LSTM Neural Network
- Selects best performing model
- Saves trained models for testing

**Output:** `models/final_model.pkl`, `results/model_performance.csv`

### Day 3: Model Testing

```bash
python code/4_test_model.py
```
**What it does:**
- Loads trained models
- Performs comprehensive backtesting
- Evaluates prediction accuracy
- Generates performance visualizations
- Tests model robustness

**Output:** `results/predictions.csv`, various charts

### Day 4: Future Predictions

```bash
python code/5_predict_new_drug.py
```
**What it does:**
- Makes future stock price predictions
- Simulates FDA milestone scenarios
- Provides investment insights
- Creates prediction reports

**Output:** Future prediction reports and recommendations

## 🧪 Drug Information

### Semglee (insulin glargine-yfgn)
- **Type:** Interchangeable insulin biosimilar
- **Indication:** Type 1 and Type 2 diabetes
- **FDA Application:** 2017
- **FDA Approval:** July 2021 (Historic first interchangeable insulin)
- **Market Launch:** September 2021
- **Significance:** First FDA-approved interchangeable insulin biosimilar

### FDA Journey Timeline
- **2015-2017:** Pre-application development
- **2017:** BLA (Biologics License Application) submitted
- **2018-2021:** FDA review process, clinical trials, inspections
- **July 2021:** FDA approval granted
- **September 2021:** Commercial launch
- **2021-2025:** Market penetration and real-world evidence

## 📊 Key Features

### Stock Data Analysis
- **10 years of historical data** (2015-2025)
- **Technical indicators:** RSI, MACD, Bollinger Bands, Moving Averages
- **Market comparison:** Nifty 50 and Nifty Pharma benchmarking
- **Volatility analysis:** Rolling volatility and beta calculations

### News Sentiment Analysis
- **Comprehensive news collection** from multiple sources
- **FDA milestone tracking** across 6 categories:
  - Application Phase
  - Clinical Trials
  - Regulatory Review
  - Approval Process
  - Post-Approval Market Implementation
  - Regulatory Issues
- **Sentiment scoring** using TextBlob and NLTK VADER
- **Weighted sentiment** based on FDA milestone importance

### Machine Learning Models
- **Traditional ML:** Linear models, ensemble methods
- **Deep Learning:** LSTM neural networks for time series
- **Feature Engineering:** 50+ engineered features
- **Model Selection:** Automated best model selection

## 🔍 Analysis Results

### Expected Findings
Based on pharmaceutical industry patterns, the analysis should reveal:

1. **FDA Approval Impact:** Significant positive stock movement on approval announcement
2. **Clinical Trial Correlations:** Price volatility during trial result announcements
3. **Market Launch Effects:** Gradual price appreciation post-commercial launch
4. **Sentiment Predictive Power:** News sentiment as leading indicator of price movements

### Performance Metrics
Models are evaluated using:
- **RMSE (Root Mean Square Error):** Prediction accuracy
- **R² Score:** Variance explained by the model
- **MAE (Mean Absolute Error):** Average prediction error
- **Sharpe Ratio:** Risk-adjusted returns for trading strategies

## 📈 Sample Analysis Outputs

### Stock Price vs FDA Milestones
```
Date Range    Event Type           Price Change    Volume Change
2017-09-20    BLA Submission      +8.5%          +150%
2021-07-29    FDA Approval        +15.2%         +300%
2021-09-15    Market Launch       +6.8%          +120%
```

### Sentiment vs Returns Correlation
```
Sentiment Range    Avg Next-Day Return    Success Rate
> 0.5 (Positive)   +2.1%                 68%
-0.1 to 0.5        +0.3%                 52%
< -0.1 (Negative)  -1.8%                 35%
```

## 🛠️ Technical Implementation

### Data Sources
- **Stock Data:** Yahoo Finance API
- **News Data:** Google News RSS, Yahoo Finance News
- **Sentiment Analysis:** TextBlob + NLTK VADER
- **Technical Indicators:** Custom calculations

### Key Libraries Used
```python
# Data Processing
pandas, numpy, scipy

# Machine Learning
scikit-learn, xgboost, lightgbm, tensorflow

# Financial Data
yfinance, yahoo-fin

# Natural Language Processing
textblob, nltk, vaderSentiment

# Visualization
matplotlib, seaborn, plotly
```

### Model Architecture

#### Traditional ML Pipeline
1. **Data Collection** → Stock prices + News articles
2. **Feature Engineering** → 50+ technical and sentiment features
3. **Model Training** → 8 different algorithms
4. **Model Selection** → Best performer based on validation R²
5. **Backtesting** → Historical performance evaluation

#### LSTM Architecture
```python
Sequential([
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=True), 
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

## 🎯 Business Applications

### For Investors
- **Entry/Exit Timing:** Identify optimal buy/sell points
- **Risk Management:** Understand FDA milestone risks
- **Portfolio Optimization:** Biotech stock allocation strategies

### For Pharmaceutical Companies
- **IR Strategy:** Understand market reaction patterns
- **Timeline Planning:** Optimize announcement timing
- **Investor Communication:** Prepare for market volatility

### For Analysts
- **Price Target Setting:** Data-driven target price models
- **Research Reports:** Enhanced fundamental analysis
- **Risk Assessment:** Quantify regulatory risks

## 📊 Configuration Options

### Customizable Parameters (config.py)

#### Data Collection
```python
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2025-06-26"
COMPANY_TICKER = "BIOCON.NS"
DRUG_NAME = "Semglee"
```

#### Model Training
```python
TRAIN_TEST_SPLIT = [0.6, 0.2, 0.2]  # Train/Val/Test
SEQUENCE_LENGTH = 30  # LSTM lookback window
CROSS_VALIDATION_FOLDS = 5
```

#### Feature Engineering
```python
MOVING_AVERAGES = [5, 10, 20, 50, 200]
SENTIMENT_WINDOWS = [3, 7, 14]
TECHNICAL_INDICATORS = ['RSI', 'MACD', 'BB']
```

## 🚨 Important Disclaimers

### Financial Disclaimer
- **Not Investment Advice:** This is educational/research project only
- **Past Performance:** Does not guarantee future results
- **Risk Warning:** Stock investments carry significant risk
- **Professional Advice:** Consult financial advisors before investing

### Data Limitations
- **News Coverage:** May not capture all relevant events
- **Market Sentiment:** Human emotions not fully quantifiable
- **External Factors:** Model doesn't account for all market variables
- **Regulatory Changes:** FDA processes may evolve over time

## 🔧 Troubleshooting

### Common Issues

#### Data Collection Failures
```bash
# If Yahoo Finance fails
Error: "No data found for BIOCON.NS"
Solution: Check internet connection, try alternative tickers
```

#### Memory Issues
```bash
# If LSTM training fails
Error: "Out of memory"  
Solution: Reduce batch_size in config.py, use CPU instead of GPU
```

#### Missing Dependencies
```bash
# Install missing packages
pip install --upgrade -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Data Quality Checks
1. **Stock Data:** Verify 2000+ daily records
2. **News Data:** Ensure 100+ articles collected
3. **Sentiment Scores:** Check for reasonable distribution (-1 to +1)
4. **FDA Milestones:** Confirm key dates are captured

## 📚 Extended Documentation

### Jupyter Notebooks
- **`data_exploration.ipynb`:** Interactive data analysis
- **`model_testing.ipynb`:** Model experimentation

### Log Files
- **`logs/biocon_analysis.log`:** Complete execution logs
- **Error tracking:** All errors logged with timestamps

### Results Export
- **CSV Files:** All results exportable for Excel analysis
- **Charts:** High-resolution PNG files for presentations
- **Model Files:** Trained models saved for reuse

## 🔄 Future Enhancements

### Planned Features
1. **Real-time Monitoring:** Live stock price and news monitoring
2. **Multiple Drugs:** Extend to other Biocon drugs
3. **Competitor Analysis:** Compare with other pharma companies
4. **Options Pricing:** Volatility models for options trading
5. **ESG Integration:** Environmental and governance factors

### Advanced Models
- **Transformer Networks:** State-of-the-art NLP models
- **Ensemble Methods:** Combine multiple model predictions
- **Reinforcement Learning:** Automated trading strategies
- **Alternative Data:** Social media sentiment, patent filings

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd biocon_fda_project

# Create virtual environment
python -m venv biocon_env
source biocon_env/bin/activate  # Linux/Mac
# biocon_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Code Standards
- **PEP 8:** Python style guidelines
- **Documentation:** Comprehensive docstrings
- **Testing:** Unit tests for critical functions
- **Logging:** Proper error handling and logging

## 📞 Support

### Getting Help
1. **Check Logs:** Review `logs/biocon_analysis.log`
2. **Verify Data:** Ensure all CSV files are generated
3. **Dependencies:** Confirm all packages installed correctly
4. **Configuration:** Review `config.py` settings

### Common Solutions
- **Yahoo Finance Issues:** Use alternative data sources
- **Memory Problems:** Reduce data size or use sampling
- **Slow Performance:** Enable parallel processing in config
- **Accuracy Issues:** Adjust model parameters or add features

## 📄 License

This project is for educational and research purposes. Not for commercial use without proper licensing.

## 🏆 Acknowledgments

- **Data Sources:** Yahoo Finance, Google News
- **Libraries:** scikit-learn, TensorFlow, pandas communities
- **Research:** Based on academic studies in financial ML
- **FDA Information:** Publicly available regulatory data

---

## 📋 Quick Start Checklist

- [ ] Create project directory structure
- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Download NLTK data for sentiment analysis
- [ ] Run Day 1: `python code/1_collect_stock_data.py`
- [ ] Run Day 1: `python code/2_collect_news_data.py`
- [ ] Verify data files created in `data/` folder
- [ ] Run Day 2: `python code/3_train_model.py`
- [ ] Check model performance in `results/model_performance.csv`
- [ ] Run Day 3: `python code/4_test_model.py`
- [ ] Run Day 4: `python code/5_predict_new_drug.py`
- [ ] Review all results and charts

**Expected Runtime:** 2-4 hours total (depending on data collection speed)

**Success Indicators:**
✅ Stock data: 2000+ records collected
✅ News data: 100+ articles with sentiment scores  
✅ Models trained: 8+ algorithms with performance metrics
✅ Predictions generated: Future price forecasts available

---

*Last Updated: June 2025*
*Project Version: 1.0*
*Python Version: 3.8+*