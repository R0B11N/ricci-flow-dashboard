import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta

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
        if not tickers:
            logger.error("No tickers provided")
            return None
            
        logger.info(f"Fetching data for tickers: {tickers}")
        
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
        
        if data.empty:
            logger.error("No data returned from Yahoo Finance")
            return None
        
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

class DataFetcher:
    def __init__(self):
        # Updated sector names to use underscores
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE', 'CSCO', 'ORCL', 'IBM',
                'QCOM', 'TXN', 'NOW', 'AMAT', 'MU'],
            'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V',
                'MA', 'USB', 'PNC', 'TFC', 'COF'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'DHR', 'BMY', 'AMGN', 'LLY',
                'CVS', 'ISRG', 'GILD', 'REGN', 'VRTX'],
            'ConsumerCyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'MAR',
                'F', 'GM', 'BKNG', 'ROST', 'BBY'],
            'Communication': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'ATVI', 'EA', 'DIS',
                'CHTR', 'WBD', 'PARA', 'DISH', 'LUMN'],
            'Industrial': ['UPS', 'HON', 'CAT', 'DE', 'BA', 'GE', 'MMM', 'RTX', 'LMT', 'UNP',
                'FDX', 'EMR', 'ETN', 'ITW', 'CSX'],
            'ConsumerDefensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'EL', 'CL', 'GIS',
                'K', 'HSY', 'KMB', 'SYY', 'KR'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
                'DVN', 'HAL', 'KMI', 'WMB', 'BKR'],
            'BasicMaterials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'ECL', 'NUE', 'CTVA',
                'CF', 'ALB', 'FMC', 'VMC', 'MLM'],
            'RealEstate': ['PLD', 'AMT', 'EQIX', 'PSA', 'O', 'WY', 'SPG', 'WELL', 'AVB', 'EQR',
                'DLR', 'VTR', 'ARE', 'BXP', 'HST'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'ED', 'EXC', 'PCG',
                'WEC', 'ES', 'ETR', 'PEG', 'AEE']
        }

    def get_sector_indices(self, sectors):
        try:
            # Get tickers for requested sectors
            tickers = []
            for sector in sectors:
                if sector not in self.sector_tickers:
                    logger.error(f"Invalid sector requested: {sector}")
                    raise ValueError(f"Invalid sector: {sector}")
                tickers.extend(self.sector_tickers[sector])
            
            logger.info(f"Fetching data for sectors: {sectors}")
            logger.info(f"Total tickers to fetch: {len(tickers)}")
            
            # Fetch data using yfinance with error handling
            try:
                data = yf.download(
                    tickers,
                    start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    progress=False,
                    auto_adjust=True  # This ensures we get adjusted prices
                )
                
                # Handle the data structure
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close']
                else:
                    prices = pd.DataFrame(data['Close'])
                
            except Exception as e:
                logger.error(f"Error downloading data: {str(e)}")
                return None
            
            if prices.empty:
                logger.error("No data retrieved from Yahoo Finance")
                return None
                
            # Calculate sector indices (equal-weighted)
            sector_indices = pd.DataFrame()
            for sector in sectors:
                sector_stocks = self.sector_tickers[sector]
                available_stocks = [s for s in sector_stocks if s in prices.columns]
                
                if not available_stocks:
                    logger.error(f"No data available for sector: {sector}")
                    return None
                    
                sector_data = prices[available_stocks].mean(axis=1)
                sector_indices[sector] = sector_data
                logger.info(f"Calculated index for {sector} using {len(available_stocks)} stocks")
            
            return sector_indices
            
        except Exception as e:
            logger.error(f"Error fetching sector indices: {str(e)}")
            logger.exception("Full traceback:")
            return None
            
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data
        """
        try:
            returns = prices.pct_change().fillna(0)
            return returns
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
            
    def calculate_curvature(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market curvature from returns
        """
        try:
            # Use rolling windows to calculate curvature
            window = 20
            rolling_std = returns.rolling(window=window).std()
            rolling_mean = returns.rolling(window=window).mean()
            
            # Simple curvature metric
            curvature = (returns - rolling_mean) / (rolling_std + 1e-6)
            return curvature.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating curvature: {str(e)}")
            return pd.DataFrame()
