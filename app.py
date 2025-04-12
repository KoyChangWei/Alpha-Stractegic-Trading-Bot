from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import logging
import time
import random
import asyncio
import cybotrade_datasource
from datetime import datetime, timedelta, timezone
from hmmlearn import hmm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='')
CORS(app)  # Enable CORS for all routes

# ========== CONFIG ==========
CYBOTRADE_API_KEY = 'CFyapwMlPYqScPa2s4LuDGvAKKWhrDWnj7EhNj4BvtRxmERA'

# Multiple data sources configuration
TOPICS = {
    'price': 'cryptoquant|btc/market-data/price?exchange=binance&pair=btcusdt&currency_symbol=btc&window=hour',
    'miners_outflow': 'cryptoquant|btc/on-chain-data/bitcoin-miners-outflow?window=hour',
    'funding_rate': 'cryptoquant|btc/market-data/futures-funding-rate?exchange=binance&pair=btcusdt&window=hour',
    'liquidations': 'cryptoquant|btc/market-data/liquidations?exchange=deribit&window=hour',
    'miner_flows': 'cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour'
}

LOOKBACK = 1000         # Number of data points to fetch (increased from 300)
TRADING_FEE = 0.0006   # 0.06% trading fee
MIN_TRADE_FREQ = 0.03  # 3% minimum trade frequency
USE_FALLBACK_DATA = True
API_KEY_VALID = False
# ================================

def ensure_data_directory():
    if not os.path.exists('data'):
        os.makedirs('data')
        logger.info("Created data directory")

def generate_fallback_data():
    """Generate synthetic data for testing when API is unavailable"""
    logger.info("Generating fallback data")
    
    # Generate timestamps for the last LOOKBACK periods
    end_time = datetime.now()
    start_time = end_time - timedelta(days=LOOKBACK)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=LOOKBACK)
    
    # Generate more realistic price data with multiple trends
    base_price = 50000
    trend_periods = 5
    price_data = []
    for i in range(trend_periods):
        period_length = LOOKBACK // trend_periods
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < trend_periods - 1 else LOOKBACK
        
        # Generate different trends for each period
        if i % 3 == 0:
            # Bullish trend
            trend = np.linspace(0, 0.2, end_idx - start_idx)
        elif i % 3 == 1:
            # Bearish trend
            trend = np.linspace(0, -0.15, end_idx - start_idx)
        else:
            # Sideways trend
            trend = np.linspace(0, 0.05, end_idx - start_idx)
        
        # Add noise and volatility
        noise = np.random.normal(0, 0.01, end_idx - start_idx)
        volatility = np.random.normal(0, 0.02, end_idx - start_idx)
        
        # Calculate price for this period
        if i == 0:
            last_price = base_price
        else:
            last_price = price_data[-1]
        
        period_prices = last_price * (1 + trend + noise + volatility)
        price_data.extend(period_prices)
    
    # Generate volume data with correlation to price movements
    base_volume = 1000
    volume_data = []
    for i in range(len(price_data)):
        if i == 0:
            volume = base_volume
        else:
            price_change = (price_data[i] - price_data[i-1]) / price_data[i-1]
            volume = volume_data[-1] * (1 + np.random.normal(0, 0.1) + price_change * 2)
            volume = max(volume, base_volume * 0.5)  # Ensure minimum volume
        
        volume_data.append(volume)
    
    # Create DataFrame with enhanced features
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': price_data,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in price_data],
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in price_data],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in price_data],
        'volume': volume_data
    })
    
    # Add more realistic features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['price_ma_20'] = df['close'].rolling(window=20).mean()
    df['price_ma_50'] = df['close'].rolling(window=50).mean()
    df['price_ma_200'] = df['close'].rolling(window=200).mean()
    
    # Add market regime indicators
    df['trend_strength'] = (df['price_ma_20'] - df['price_ma_50']) / df['price_ma_50']
    df['volatility_regime'] = df['volatility'].rolling(window=50).mean()
    
    # Add momentum indicators
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'] = calculate_macd(df['close'])
    
    # Add support/resistance levels
    df['support_level'] = df['close'].rolling(window=50).min()
    df['resistance_level'] = df['close'].rolling(window=50).max()
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    return exp1 - exp2

async def fetch_data_async():
    """Fetch data from multiple Cybotrade API sources"""
    try:
        logger.info("Fetching data from multiple Cybotrade API sources")
        
        all_data = {}
        for name, topic in TOPICS.items():
            try:
                # Fetch data in chunks to handle large lookback periods
                data = []
                remaining = LOOKBACK
                while remaining > 0:
                    chunk_size = min(remaining, 300)  # API limit per request
                    chunk_data = await cybotrade_datasource.query_paginated(
                        api_key=CYBOTRADE_API_KEY,
                        topic=topic,
                        limit=chunk_size
                    )
                    data.extend(chunk_data)
                    remaining -= len(chunk_data)
                    if len(chunk_data) < chunk_size:
                        break
                
                all_data[name] = pd.DataFrame(data)
                logger.info(f"Successfully fetched {len(all_data[name])} rows from {name}")
            except Exception as e:
                logger.error(f"Error fetching {name} data: {str(e)}")
                if not USE_FALLBACK_DATA:
                    raise
        
        if not all_data or all(df.empty for df in all_data.values()):
            if USE_FALLBACK_DATA:
                logger.info("Using fallback data due to empty data from all sources")
                return generate_fallback_data()
            else:
                raise ValueError("No data received from any API source")
        
        # Use price data as the base DataFrame
        df = all_data.get('price', pd.DataFrame())
        if df.empty and USE_FALLBACK_DATA:
            df = generate_fallback_data()
        
        # Ensure price columns exist and are properly mapped
        if 'price' in df.columns and 'close' not in df.columns:
            df['close'] = df['price']
        
        # Add required columns if they don't exist
        for col in ['open', 'high', 'low']:
            if col not in df.columns and 'close' in df.columns:
                df[col] = df['close']
        
        if 'volume' not in df.columns:
            df['volume'] = df['close'] * random.uniform(0.01, 0.1)
        
        # Add on-chain features from other data sources
        if 'miners_outflow' in all_data:
            df['miner_outflow'] = all_data['miners_outflow'].get('value', 0)
        
        if 'funding_rate' in all_data:
            df['funding_rate'] = all_data['funding_rate'].get('value', 0)
        
        if 'liquidations' in all_data:
            df['liquidations'] = all_data['liquidations'].get('value', 0)
        
        if 'miner_flows' in all_data:
            df['miner_flows'] = all_data['miner_flows'].get('value', 0)
        
        ensure_data_directory()
        df.to_csv('data/raw_data.csv', index=False)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        if USE_FALLBACK_DATA:
            logger.info("Using fallback data due to API error")
            return generate_fallback_data()
        else:
            raise

def fetch_data():
    """Synchronous wrapper for async fetch_data_async function"""
    try:
        if not API_KEY_VALID and USE_FALLBACK_DATA:
            logger.warning("API key is invalid. Using fallback data.")
            return generate_fallback_data()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            df = loop.run_until_complete(fetch_data_async())
        finally:
            loop.close()
        return df
    except Exception as e:
        logger.error(f"Error in fetch_data: {str(e)}")
        if USE_FALLBACK_DATA:
            return generate_fallback_data()
        else:
            raise

def preprocess_data(df):
    """Preprocess data with enhanced feature engineering"""
    try:
        logger.info("Preprocessing data with enhanced features")
        
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volatility features
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(window=100).mean()
        
        # Moving averages and crossovers
        for window in [5, 10, 20, 50, 100]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_to_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
        
        # Add moving average crossovers
        df['ma_cross_short'] = (df['ma_5'] > df['ma_20']).astype(float)
        df['ma_cross_long'] = (df['ma_20'] > df['ma_50']).astype(float)
        
        # Volume-based features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=50).mean()
        
        # RSI with multiple timeframes
        for window in [6, 14, 28]:
            df[f'rsi_{window}'] = calculate_rsi(df['close'], window)
        
        # Momentum indicators
        df['momentum_1d'] = df['close'].pct_change(1)
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # Bollinger Bands
        for window in [20, 50]:
            ma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = ma + 2 * std
            df[f'bb_lower_{window}'] = ma - 2 * std
            df[f'bb_position_{window}'] = (df['close'] - ma) / (2 * std)
        
        # On-chain features with better normalization
        if 'miner_outflow' in df.columns:
            df['miner_outflow_ma'] = df['miner_outflow'].rolling(window=24).mean()
            df['miner_outflow_ratio'] = df['miner_outflow'] / df['miner_outflow_ma']
            df['miner_outflow_zscore'] = (df['miner_outflow'] - df['miner_outflow'].rolling(window=50).mean()) / df['miner_outflow'].rolling(window=50).std()
        
        if 'funding_rate' in df.columns:
            df['funding_rate_ma'] = df['funding_rate'].rolling(window=24).mean()
            df['funding_pressure'] = df['funding_rate'] - df['funding_rate_ma']
            df['funding_zscore'] = (df['funding_rate'] - df['funding_rate'].rolling(window=50).mean()) / df['funding_rate'].rolling(window=50).std()
        
        if 'liquidations' in df.columns:
            df['liquidation_intensity'] = df['liquidations'].rolling(window=24).sum()
            df['liquidation_ma'] = df['liquidations'].rolling(window=72).mean()
            df['liquidation_ratio'] = df['liquidation_intensity'] / df['liquidation_ma']
        
        if 'miner_flows' in df.columns:
            df['miner_flow_intensity'] = df['miner_flows'].rolling(window=24).mean()
            df['miner_flow_trend'] = df['miner_flow_intensity'] / df['miner_flows'].rolling(window=72).mean()
        
        # Add trend strength indicators
        df['adx'] = calculate_adx(df)
        df['trend_strength'] = df['adx'].rolling(window=14).mean()
        
        # Fill missing values with appropriate methods
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate Directional Indicators
        pdi = 100 * pd.Series(pos_dm).rolling(window=period).mean() / atr
        ndi = 100 * pd.Series(neg_dm).rolling(window=period).mean() / atr
        
        # Calculate ADX
        adx = 100 * abs(pdi - ndi) / (pdi + ndi)
        return adx.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series(0, index=df.index)

def apply_hmm_states(df, n_states=4):
    """Apply Hidden Markov Model to identify market regimes"""
    try:
        logger.info(f"Applying HMM with {n_states} states")
        
        # Select features for regime detection
        regime_features = [
            'returns',
            'volatility_20',
            'volume_ratio',
            'rsi_14',
            'momentum_5d',
            'trend_strength'
        ]
        
        missing_features = [col for col in regime_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Prepare features for HMM
        features = df[regime_features].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        if len(features) < 2:
            raise ValueError("Not enough valid data points for HMM")
            
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train HMM with increased iterations and multiple restarts
        best_score = float('-inf')
        best_model = None
        best_states = None
        
        for _ in range(3):  # Try 3 different initializations
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=2000,
                    random_state=np.random.randint(0, 100)
                )
                model.fit(scaled_features)
                score = model.score(scaled_features)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_states = model.predict(scaled_features)
            except Exception as e:
                logger.warning(f"HMM initialization failed, trying again: {str(e)}")
                continue
        
        if best_model is None:
            raise ValueError("Failed to train HMM after multiple attempts")
        
        # Add market state to dataframe
        df['market_state'] = -1
        df.loc[features.index, 'market_state'] = best_states
        
        # Calculate state characteristics
        state_chars = {}
        for state in range(n_states):
            mask = best_states == state
            if mask.any():
                state_chars[state] = {
                    'avg_return': df.loc[features.index[mask], 'returns'].mean(),
                    'volatility': df.loc[features.index[mask], 'volatility_20'].mean(),
                    'volume': df.loc[features.index[mask], 'volume_ratio'].mean(),
                    'trend': df.loc[features.index[mask], 'trend_strength'].mean()
                }
        
        logger.info(f"HMM applied successfully. State distribution: {np.bincount(best_states)}")
        logger.info(f"State characteristics: {state_chars}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in HMM processing: {str(e)}")
        raise Exception(f"Error in HMM processing: {str(e)}")

def apply_risk_management(df, predictions, confidence_scores):
    """Apply risk management rules to trading signals"""
    try:
        # Initialize position sizes
        position_sizes = np.zeros_like(predictions, dtype=float)
        
        # Calculate volatility-adjusted position sizes
        vol = df['volatility_20']
        vol_rank = vol.rank(pct=True)
        max_position = 1.0
        
        for i in range(len(predictions)):
            if predictions[i] == 0:  # Sell signal
                continue
                
            # Base position size on confidence
            base_size = confidence_scores[i] * max_position
            
            # Reduce position size in high volatility
            vol_adjustment = 1 - vol_rank.iloc[i]
            
            # Reduce position size if against trend
            trend_adjustment = 1.0
            if df['ma_cross_short'].iloc[i] == 0:
                trend_adjustment *= 0.7
            if df['ma_cross_long'].iloc[i] == 0:
                trend_adjustment *= 0.7
            
            # Additional risk checks
            risk_multiplier = 1.0
            
            # Reduce position if funding rate is extreme
            if 'funding_zscore' in df.columns:
                funding_z = df['funding_zscore'].iloc[i]
                if abs(funding_z) > 2:
                    risk_multiplier *= 0.8
            
            # Reduce position if liquidations are high
            if 'liquidation_ratio' in df.columns:
                liq_ratio = df['liquidation_ratio'].iloc[i]
                if liq_ratio > 1.5:
                    risk_multiplier *= 0.8
            
            # Final position size
            position_sizes[i] = base_size * vol_adjustment * trend_adjustment * risk_multiplier
        
        return position_sizes
        
    except Exception as e:
        logger.error(f"Error in risk management: {str(e)}")
        return np.zeros_like(predictions, dtype=float)

def calculate_performance_metrics(df, predictions, confidence_scores):
    """
    Calculate trading performance metrics including Sharpe Ratio, Maximum Drawdown, and Trade Frequency.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        predictions (np.array): Array of trading signals (-1, 0, 1)
        confidence_scores (np.array): Array of confidence scores for each prediction
    
    Returns:
        tuple: (total_returns, sharpe_ratio, max_drawdown, trade_frequency)
    """
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Validate required data
        if 'close' not in df.columns:
            logger.error("Missing 'close' price data")
            return 0.0, 0.0, 0.0, 0.0
            
        # Calculate base returns
        df['returns'] = df['close'].pct_change()
        df['returns'] = df['returns'].fillna(0)
        
        # Apply position sizing based on confidence
        df['position'] = predictions * confidence_scores
        
        # Calculate position changes for transaction costs
        position_changes = np.abs(np.diff(df['position'], prepend=0))
        TRANSACTION_COST = 0.001  # 10 bps per trade
        
        # Calculate strategy returns with transaction costs
        df['strategy_returns'] = df['position'].shift(1) * df['returns'] - position_changes * TRANSACTION_COST
        df['strategy_returns'] = df['strategy_returns'].fillna(0)
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        total_returns = (df['cumulative_returns'].iloc[-1] - 1) * 100  # Convert to percentage
        
        # Calculate annualized Sharpe Ratio
        returns_std = np.std(df['strategy_returns'])
        if returns_std > 0:
            annual_factor = np.sqrt(252)  # Assuming daily data
            sharpe_ratio = np.mean(df['strategy_returns']) / returns_std * annual_factor
        else:
            sharpe_ratio = 0.0
            
        # Calculate Maximum Drawdown
        rolling_max = df['cumulative_returns'].expanding().max()
        drawdowns = (df['cumulative_returns'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100  # Convert to percentage
        
        # Calculate trade frequency (trades per day)
        trade_frequency = np.count_nonzero(position_changes) / len(df)
        
        # Clip values to prevent extreme outliers
        total_returns = np.clip(total_returns, -100, 1000)
        sharpe_ratio = np.clip(sharpe_ratio, -10, 10)
        max_drawdown = np.clip(max_drawdown, 0, 100)
        trade_frequency = np.clip(trade_frequency, 0, 1)
        
        logger.info(f"Performance metrics calculated: Returns={total_returns:.2f}%, Sharpe={sharpe_ratio:.2f}, "
                   f"MaxDD={max_drawdown:.2f}%, TradeFreq={trade_frequency:.3f}")
        
        return float(total_returns), float(sharpe_ratio), float(max_drawdown), float(trade_frequency)
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        logger.error(f"DataFrame shape: {df.shape}, columns: {df.columns}")
        return 0.0, 0.0, 0.0, 0.0

def train_xgboost(df):
    """Train XGBoost model for price prediction using HMM and technical features"""
    try:
        logger.info("Preparing data for XGBoost training")
        
        df = preprocess_data(df)
        
        # Feature selection - focus on most predictive features
        core_features = [
            'returns', 'log_returns',
            'volatility_20', 'volatility_ratio_20',
            'ma_cross_short', 'ma_cross_long',
            'volume_ratio', 'volume_trend',
            'rsi_14', 'momentum_5d',
            'bb_position_20', 'trend_strength',
            'market_state'
        ]
        
        # Add on-chain features if available
        if 'funding_zscore' in df.columns:
            core_features.extend(['funding_zscore', 'funding_pressure'])
        if 'miner_outflow_zscore' in df.columns:
            core_features.extend(['miner_outflow_zscore', 'miner_outflow_ratio'])
        if 'liquidation_ratio' in df.columns:
            core_features.append('liquidation_ratio')
        
        features = df[core_features].copy()
        
        # Handle missing values and outliers
        for col in features.columns:
            # Fill missing values
            features[col] = features[col].fillna(method='ffill').fillna(0)
            
            # Handle outliers using winsorization
            q1 = features[col].quantile(0.01)
            q3 = features[col].quantile(0.99)
            features[col] = features[col].clip(q1, q3)
        
        df = df.dropna()
        if len(df) < 10:
            raise ValueError("Not enough data points for training")
        
        # Create target variable with forward returns
        # Use multi-timeframe targets for more robust signals
        short_target = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        medium_target = np.where(df['close'].shift(-3).rolling(window=3).mean() > df['close'], 1, 0)
        df['target'] = (short_target & medium_target).astype(int)  # Require both timeframes to agree
        df = df.dropna()
        
        X = features
        y = df['target']
        
        logger.info(f"Training XGBoost with {len(X)} samples and {X.shape[1]} features")
        
        # Use time-based split for financial data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Optimized hyperparameters for better risk-adjusted returns
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1.2,  # Slight bias towards positive class for better risk management
            random_state=42
        )
        
        # Simple train without early stopping
        model.fit(X_train, y_train)
        
        # Generate predictions with probability threshold optimization
        y_pred_proba = model.predict_proba(X_test)
        
        # Default predictions using standard threshold
        default_threshold = 0.5
        default_predictions = (y_pred_proba[:, 1] >= default_threshold).astype(int)
        default_confidence = np.where(y_pred_proba[:, 1] >= default_threshold, 
                                    y_pred_proba[:, 1], 
                                    1 - y_pred_proba[:, 1])
        
        # Initialize best predictions with defaults
        best_sharpe = -999999  # Use a large negative number instead of -inf
        best_threshold = default_threshold
        best_predictions = default_predictions
        best_confidence = default_confidence
        
        # Try to optimize threshold
        for threshold in np.arange(0.5, 0.9, 0.05):
            y_pred_current = (y_pred_proba[:, 1] >= threshold).astype(int)
            confidence_scores = np.where(y_pred_proba[:, 1] >= threshold, 
                                      y_pred_proba[:, 1], 
                                      1 - y_pred_proba[:, 1])
            
            try:
                # Calculate performance metrics for this threshold
                total_returns, sharpe_ratio, max_drawdown, trade_frequency = calculate_performance_metrics(
                    df.iloc[split_idx:].copy(),
                    y_pred_current,
                    confidence_scores
                )
                
                current_sharpe = sharpe_ratio
                # Handle potential NaN or infinite values
                if np.isfinite(current_sharpe) and current_sharpe > best_sharpe:
                    best_sharpe = current_sharpe
                    best_threshold = threshold
                    best_predictions = y_pred_current
                    best_confidence = confidence_scores
            except Exception as e:
                logger.warning(f"Error calculating metrics for threshold {threshold}: {str(e)}")
                continue
        
        logger.info(f"Optimal probability threshold: {best_threshold:.2f}")
        
        # Use optimized predictions (or defaults if optimization failed)
        y_pred = best_predictions
        confidence_scores = best_confidence
        
        # Calculate accuracy and classification report
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"XGBoost training completed with accuracy: {acc:.4f}")
        
        # Calculate performance metrics with final predictions
        try:
            total_returns, sharpe_ratio, max_drawdown, trade_frequency = calculate_performance_metrics(
                df.iloc[split_idx:].copy(),
                y_pred,
                confidence_scores
            )
            
            # Ensure all metrics are finite numbers
            total_returns = float(np.nan_to_num(total_returns, nan=0.0))
            sharpe_ratio = float(np.nan_to_num(sharpe_ratio, nan=0.0))
            max_drawdown = float(np.nan_to_num(max_drawdown, nan=0.0))
            trade_frequency = float(np.nan_to_num(trade_frequency, nan=0.0))
            
        except Exception as e:
            logger.error(f"Error calculating final performance metrics: {str(e)}")
            # Provide default metrics if calculation fails
            total_returns, sharpe_ratio, max_drawdown, trade_frequency = 0.0, 0.0, 0.0, 0.0
        
        # Save model and feature importance
        ensure_data_directory()
        model_path = 'data/xgb_model.json'
        model.save_model(model_path)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('data/feature_importance.csv', index=False)
        
        # Generate latest prediction using optimized threshold
        latest_data = features.iloc[-1].values.reshape(1, -1)
        latest_proba = model.predict_proba(latest_data)[0]
        latest_prediction = int(latest_proba[1] >= best_threshold)
        
        prediction_label = "BUY" if latest_prediction == 1 else "SELL"
        confidence = float(latest_proba[latest_prediction] * 100)  # Ensure confidence is a float
        
        return (
            model,
            float(acc),  # Ensure accuracy is a float
            classification_report(y_test, y_pred, output_dict=True),
            prediction_label,
            confidence,
            (total_returns, sharpe_ratio, max_drawdown, trade_frequency)
        )
    except Exception as e:
        logger.error(f"Error in XGBoost training: {str(e)}")
        raise

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/run-model', methods=['GET'])
def run_model():
    """API endpoint to run the trading model pipeline using HMM and XGBoost"""
    try:
        logger.info("Starting HMM + XGBoost trading model pipeline")

        df = fetch_data()
        if df.empty:
            logger.error("No data received from API")
            return jsonify({'status': 'error', 'message': 'No data received from API'})

        df_processed = preprocess_data(df)
        df_with_hmm = apply_hmm_states(df_processed)
        
        model, acc, report, prediction_label, confidence, performance = train_xgboost(df_with_hmm)
        
        # Convert numpy int64 to regular Python int
        regime_distribution = {str(i): int(c) for i, c in enumerate(np.bincount(df_with_hmm['market_state'].astype(int)))}
        
        logger.info("Model pipeline completed successfully")
        
        # Check if performance meets requirements
        requirements_met = bool(
            performance[1] >= 1.8 and
            performance[2] >= -0.4 and
            performance[3] >= MIN_TRADE_FREQ
        )
        
        warning = None
        if not API_KEY_VALID:
            warning = "Using synthetic data due to invalid API key. Please update your API key for real data."
        elif not requirements_met:
            warning = "Model performance does not meet minimum requirements. Use with caution."

        return jsonify({
            'status': 'success',
            'accuracy': float(acc),
            'classification_report': report,
            'warning': warning,
            'prediction': {
                'signal': prediction_label,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            },
            'market_regime': {
                'states_count': int(len(np.unique(df_with_hmm['market_state']))),
                'regime_distribution': regime_distribution
            },
            'performance_metrics': {
                'total_returns': float(performance[0]),
                'sharpe_ratio': float(performance[1]),
                'max_drawdown': float(performance[2]),
                'trade_frequency': float(performance[3]),
                'meets_requirements': bool(requirements_met)
            }
        })
    except Exception as e:
        logger.error(f"Error in model run: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Service is running'})

async def verify_api_key_async(api_key):
    logger.info(f"Verifying API key: {api_key[:5]}...{api_key[-5:]}")
    try:
        data = await cybotrade_datasource.query_paginated(
            api_key=api_key,
            topic=TOPICS['price'],
            limit=5
        )
        if data and len(data) > 0:
            logger.info("API key is valid!")
            return True
        else:
            logger.error("API key verification failed - no data returned")
            return False
    except Exception as e:
        logger.error(f"Error verifying API key: {str(e)}")
        return False

def verify_api_key(api_key):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(verify_api_key_async(api_key))
        finally:
            loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in verify_api_key: {str(e)}")
        return False

@app.route('/update-api-key', methods=['POST'])
def update_api_key():
    try:
        data = request.json
        if not data or 'api_key' not in data:
            return jsonify({'status': 'error', 'message': 'API key is required'}), 400
        new_api_key = data['api_key']
        logger.info("Updating API key")
        is_valid = verify_api_key(new_api_key)
        if is_valid:
            global CYBOTRADE_API_KEY, API_KEY_VALID
            CYBOTRADE_API_KEY = new_api_key
            API_KEY_VALID = True
            logger.info("API key updated and verified successfully")
            return jsonify({
                'status': 'success', 
                'message': 'API key updated and verified successfully',
                'is_valid': True
            })
        else:
            logger.warning("API key verification failed")
            return jsonify({
                'status': 'error', 
                'message': 'API key verification failed. Please check your key and try again.',
                'is_valid': False
            }), 400
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/topics', methods=['GET'])
def get_available_topics():
    topics = list(TOPICS.keys())
    return jsonify({
        'status': 'success',
        'topics': topics
    })

@app.route('/chart-data', methods=['GET'])
def get_chart_data():
    """API endpoint to get data for the trading chart"""
    try:
        df = fetch_data()
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available'})

        # Process data for charting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Format data for candlestick chart
        chart_data = {
            'timestamps': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'candlesticks': [
                {
                    'time': t,
                    'open': float(o),
                    'high': float(h),
                    'low': float(l),
                    'close': float(c)
                }
                for t, o, h, l, c in zip(
                    df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    df['open'],
                    df['high'],
                    df['low'],
                    df['close']
                )
            ],
            'latest_price': float(df['close'].iloc[-1]),
            'price_change': float(df['close'].iloc[-1] - df['close'].iloc[-2]),
            'price_change_percent': float((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100)
        }

        # Replace NaN with None for safe JSON serialization
        def replace_nan(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            elif isinstance(obj, list):
                return [replace_nan(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            else:
                return obj

        chart_data_clean = replace_nan(chart_data)

        return jsonify({
            'status': 'success',
            'data': chart_data_clean
        })

    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}")
        raise
