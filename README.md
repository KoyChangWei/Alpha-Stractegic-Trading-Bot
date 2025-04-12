# Alpha Trading Strategy - HMM + XGBoost Model

![屏幕截图 2025-04-12 142140](https://github.com/user-attachments/assets/019ba5cf-8b84-4763-866d-7c5295839e2f)

## 🚀 Project Overview

This project is a sophisticated cryptocurrency trading model that combines Hidden Markov Models (HMM) for market regime detection with XGBoost for signal generation. Developed during a hackathon, this prototype demonstrates the potential of combining traditional time series analysis with modern machine learning techniques for cryptocurrency trading.

## 🎯 Key Features

- Real-time BTC/USDT price monitoring
- Advanced market regime detection using HMM
- Signal generation with XGBoost
- Interactive candlestick chart visualization
- Performance metrics tracking
- Automated trading signals with confidence scores

## 🔬 Technical Implementation

### Market Regime Detection (HMM)
- Uses Hidden Markov Models with 4 states to identify different market regimes
- Features used for regime detection:
  - Returns
  - Volatility (20-period)
  - Volume ratio
  - RSI (14-period)
  - Momentum (5-day)
  - Trend strength indicators

### Signal Generation (XGBoost)
- XGBoost classifier for trade signal generation
- Feature engineering includes:
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Market regime information from HMM
  - Volume metrics
  - Price momentum indicators
  - Support/resistance levels
- Hyperparameters optimized for risk-adjusted returns

### Performance Metrics
- Model Accuracy: ~79%
- Sharpe Ratio: Target ≥ 1.8
- Maximum Drawdown: Target ≥ -40%
- Trade Frequency: Target ≥ 3%

## 🛠️ Technology Stack

### Backend (Python)
- Flask for API endpoints
- pandas for data manipulation
- numpy for numerical computations
- hmmlearn for Hidden Markov Models
- XGBoost for machine learning
- scikit-learn for data preprocessing

### Frontend
- HTML5/CSS3 for structure and styling
- Chart.js for interactive charts
- JavaScript for dynamic updates
- Modern responsive design

### Data Sources
- CyboTrade API for market data
- Features:
  - Price data (OHLCV)
  - On-chain metrics
  - Market sentiment indicators
  - Funding rates
  - Liquidation data

## 📊 Algorithm Details

### Data Preprocessing
1. Feature engineering with technical indicators
2. Volatility normalization
3. Missing value handling
4. Outlier detection and treatment

### HMM Implementation
1. State detection using multivariate Gaussian HMM
2. Features normalized using StandardScaler
3. Multiple initialization attempts for optimal convergence
4. State characteristics analysis for regime identification

### XGBoost Model
1. Time-based cross-validation
2. Probability threshold optimization
3. Position sizing based on confidence scores
4. Risk management rules integration

### Risk Management
- Dynamic position sizing based on:
  - Model confidence
  - Volatility ranking
  - Trend alignment
  - Market regime
- Transaction cost consideration
- Maximum drawdown controls

## 🚦 Performance Requirements

The model aims to achieve:
- Sharpe Ratio ≥ 1.8 (Risk-adjusted returns)
- Maximum Drawdown ≥ -40% (Risk management)
- Trade Frequency ≥ 3% (Activity threshold)

## 📈 Live Dashboard

The dashboard provides:
- Real-time price updates
- Candlestick charts
- Current trading signals
- Performance metrics
- Market regime indicators

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alpha-trading-strategy.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your CyboTrade API key:
```python
CYBOTRADE_API_KEY = 'your_api_key_here'
```

4. Run the application:
```bash
python app.py
```

## 🤝 Contributing

This is a hackathon prototype and we welcome contributions! Please feel free to submit pull requests or open issues for improvements.

## ⚠️ Disclaimer

This is a prototype developed during a hackathon and should not be used for actual trading without proper risk management and further testing. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors.

## 📝 License

MIT License - feel free to use this code for your own projects. 
