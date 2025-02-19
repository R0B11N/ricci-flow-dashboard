import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def fetch_stock_data(tickers: List[str], start: str = "2020-01-01", end: str = "2024-01-01") -> Optional[Dict]:
    """
    Fetch historical data and additional metrics for given tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing prices and additional metrics, or None if error occurs
    """
    try:
        # Initialize metrics dictionaries
        metrics = {
            'prices': None,
            'market_caps': {},
            'volumes': {},
            'volatilities': {},
            'sectors': {},
            'historical_curvatures': pd.DataFrame()
        }
        
        # Download historical price data
        data = yf.download(
            tickers, 
            start=start, 
            end=end, 
            progress=False,
            auto_adjust=True
        )
        
        # Handle single vs multiple tickers case
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data['Close'] if 'Close' in data.columns else data.xs('Close', axis=1, level=1)
            
        if len(prices) == 0:
            logger.error("Empty dataset received from yfinance")
            return None
            
        # Calculate returns and rolling correlations (historical curvatures)
        returns = prices.pct_change().dropna()
        
        # Calculate historical curvatures using rolling windows
        window_size = 30  # 30-day rolling window
        historical_curvatures = pd.DataFrame()
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                pair = f"{tickers[i]}-{tickers[j]}"
                # Calculate rolling correlation as a proxy for historical curvature
                rolling_corr = returns[tickers[i]].rolling(window=window_size).corr(returns[tickers[j]])
                historical_curvatures[pair] = rolling_corr
        
        metrics['historical_curvatures'] = historical_curvatures
        
        # Fetch individual stock info
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Market Cap
                metrics['market_caps'][ticker] = info.get('marketCap', 0)
                
                # Average Daily Volume
                metrics['volumes'][ticker] = info.get('averageVolume', 0)
                
                # Sector
                metrics['sectors'][ticker] = info.get('sector', 'Unknown')
                
            except Exception as e:
                logger.error(f"Error fetching info for {ticker}: {str(e)}")
                metrics['market_caps'][ticker] = 0
                metrics['volumes'][ticker] = 0
                metrics['sectors'][ticker] = 'Unknown'
        
        # Calculate annualized volatility
        volatility = returns.std() * np.sqrt(252)  # 252 trading days in a year
        for ticker in tickers:
            metrics['volatilities'][ticker] = float(volatility[ticker])
        
        # Store prices
        metrics['prices'] = prices
        
        logger.info(f"Successfully fetched data with shape {returns.shape}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None
