from fastapi import FastAPI, HTTPException
from typing import List, Dict
from datetime import datetime, timedelta
import logging
from ricci_flow import compute_ricci_curvature
from data_fetcher import fetch_stock_data, DataFetcher
from fastapi.middleware.cors import CORSMiddleware
from market_analyzer import MarketAnalyzer
from regime_analyzer import RegimeAnalyzer
import pandas as pd
from lead_lag_analyzer import LeadLagAnalyzer
import numpy as np
import yfinance as yf
from pydantic import BaseModel
from market_regime_analyzer import MarketRegimeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Market Ricci Curvature API",
             description="API for computing Ricci curvature of stock correlation networks",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

market_analyzer = MarketAnalyzer()
regime_analyzer = RegimeAnalyzer()
lead_lag_analyzer = LeadLagAnalyzer()

# Correct initialization
data_fetcher = DataFetcher()

# Get the sector list from DataFetcher
VALID_SECTORS = list(data_fetcher.sector_tickers.keys())

# Initialize the analyzer
market_regime_analyzer = MarketRegimeAnalyzer()

def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

@app.get("/ricci-curvature/")
async def get_ricci_curvature(tickers: str, start: str = "2020-01-01", end: str = "2024-01-01"):
    try:
        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        logger.info(f"Fetching data for tickers: {ticker_list}")
        # Fetch stock data
        stock_data = fetch_stock_data(ticker_list, start, end)
        
        if stock_data is None:
            logger.error("Data fetching returned None")
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
            
        if stock_data['prices'] is None or len(stock_data['prices']) == 0:
            logger.error("Empty dataset returned")
            raise HTTPException(status_code=404, detail="No data found for given tickers")
        
        logger.info("Computing Ricci curvature")
        ricci_data = compute_ricci_curvature(stock_data['prices'])
        
        return {
            "status": "success",
            "curvature": ricci_data.to_dict(),
            "metadata": {
                "tickers": ticker_list,
                "start_date": start,
                "end_date": end,
                "data_points": len(stock_data['prices']),
                "market_caps": stock_data['market_caps'],
                "volumes": stock_data['volumes'],
                "sectors": stock_data['sectors'],
                "volatilities": stock_data['volatilities']
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/market-analysis/")
async def analyze_market(tickers: str, start: str = "2020-01-01", end: str = "2024-01-01"):
    try:
        stock_data = fetch_stock_data(tickers.split(','), start, end)
        if not stock_data:
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
            
        # Calculate returns
        returns = stock_data['prices'].pct_change().dropna()
        
        ricci_data = compute_ricci_curvature(stock_data['prices'])
        
        optimal_pairs = market_analyzer.optimize_portfolio(
            ricci_data.to_dict(),
            stock_data,
            returns,
            target_risk=0.5
        )
        
        return {
            "status": "success",
            "optimal_pairs": optimal_pairs,
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "num_pairs_analyzed": len(ricci_data)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regime-analysis/")
async def analyze_regimes(tickers: str, start: str, end: str):
    try:
        # Add input validation
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        time_delta = end_date - start_date
        
        logger.info(f"Analyzing regimes for {tickers} from {start} to {end}")
        logger.info(f"Time span: {time_delta.days} days")
        
        # Fetch data
        stock_data = fetch_stock_data(tickers.split(','), start, end)
        if not stock_data or 'prices' not in stock_data:
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
        
        # Log data shape
        prices_df = stock_data['prices']
        logger.info(f"Data shape: {prices_df.shape}")
        
        if len(prices_df) < 2:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data points. Need at least 2 data points."
            )
        
        # Calculate returns
        returns = prices_df.pct_change().dropna()
        
        # Use the new market regime analyzer
        regime_results = market_regime_analyzer.analyze_market_regimes(returns)
        
        return regime_results
        
    except ValueError as e:
        logger.error(f"Value error in regime analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in regime analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lead-lag-analysis/")
async def analyze_lead_lag(tickers: str, start: str = "2020-01-01", end: str = None):
    try:
        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        if len(ticker_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 tickers are required for lead-lag analysis")
        
        # Set end date to today if not provided
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
            
        # Fetch stock data
        stock_data = fetch_stock_data(ticker_list, start, end)
        if not stock_data or 'prices' not in stock_data:
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
            
        # Calculate returns
        returns = stock_data['prices'].pct_change().dropna()
        
        if len(returns) < 30:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data points. At least 30 days of data required."
            )
        
        # Create curvature history DataFrame
        curvature_history = pd.DataFrame()
        window_size = 30  # 30-day rolling window
        
        # Calculate rolling correlations for each pair
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                pair = f"{ticker_list[i]}-{ticker_list[j]}"
                rolling_corr = returns[ticker_list[i]].rolling(window=window_size).corr(returns[ticker_list[j]])
                curvature_history[pair] = rolling_corr
        
        if curvature_history.empty:
            raise HTTPException(
                status_code=404,
                detail="Failed to compute correlations between stocks"
            )
        
        # Create metadata dictionary
        metadata = {
            'sectors': stock_data.get('sectors', {}),
            'market_caps': stock_data.get('market_caps', {}),
            'volumes': stock_data.get('volumes', {})
        }
        
        # Log the data being passed to analyze_lead_lag
        logger.info(f"Curvature history shape: {curvature_history.shape}")
        logger.info(f"Returns shape: {returns.shape}")
        logger.info(f"Metadata keys: {metadata.keys()}")
        
        # Perform lead-lag analysis
        results = lead_lag_analyzer.analyze_lead_lag(
            curvature_history,
            returns,
            metadata
        )
        
        if not results or not results.get('lead_lag_relationships'):
            return {
                "status": "success",
                "lead_lag_relationships": [],
                "sector_leaders": {},
                "message": "No significant lead-lag relationships found"
            }
        
        return {
            "status": "success",
            "lead_lag_relationships": results.get('lead_lag_relationships', []),
            "sector_leaders": results.get('sector_leaders', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in lead-lag analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze lead-lag relationships: {str(e)}"
        )

@app.get("/predictions/")
async def get_predictions(tickers: str, start: str = "2020-01-01", end: str = None):
    try:
        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        if len(ticker_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 tickers are required for predictions")
        
        # Set end date to today if not provided
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
            
        # Fetch stock data
        stock_data = fetch_stock_data(ticker_list, start, end)
        if not stock_data or 'prices' not in stock_data:
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
            
        # Calculate returns
        returns = stock_data['prices'].pct_change().dropna()
        
        # Generate predictions for each pair
        predictions = []
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                stock1, stock2 = ticker_list[i], ticker_list[j]
                pair = f"{stock1}-{stock2}"
                
                # Calculate correlation and volatility
                correlation = returns[stock1].corr(returns[stock2])
                vol1 = returns[stock1].std() * np.sqrt(252)  # Annualized volatility
                vol2 = returns[stock2].std() * np.sqrt(252)
                
                # Simple prediction based on recent performance and correlation
                recent_returns1 = returns[stock1].tail(5).mean()
                recent_returns2 = returns[stock2].tail(5).mean()
                
                # Determine direction and confidence
                if abs(correlation) > 0.5:
                    # For highly correlated pairs
                    combined_return = (recent_returns1 + recent_returns2) / 2
                    direction = 'up' if combined_return > 0 else 'down'
                    confidence = min(0.9, abs(correlation) * (1 - abs(vol1 - vol2)))
                else:
                    # For less correlated pairs
                    direction = 'up' if recent_returns1 > recent_returns2 else 'down'
                    confidence = min(0.7, abs(recent_returns1 - recent_returns2) / max(vol1, vol2))
                
                predictions.append({
                    'pair': pair,
                    'direction': direction,
                    'confidence': float(confidence),  # Ensure JSON serializable
                    'horizon': 5,  # 5-day prediction horizon
                    'correlation': float(correlation),
                    'volatility': float((vol1 + vol2) / 2)
                })
        
        # Calculate risk-return metrics for pairs
        risk_return_data = []
        risk_free_rate = 0.0425  # Current 10-year Treasury yield
        
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                stock1, stock2 = ticker_list[i], ticker_list[j]
                pair = f"{stock1}-{stock2}"
                
                # Calculate daily returns and volatility
                returns1 = returns[stock1]
                returns2 = returns[stock2]
                
                # Calculate portfolio returns (equal weight)
                pair_returns = (returns1 + returns2) / 2
                annual_return = pair_returns.mean() * 252
                
                # Calculate portfolio volatility including correlation
                correlation = returns1.corr(returns2)
                vol1 = returns1.std() * np.sqrt(252)
                vol2 = returns2.std() * np.sqrt(252)
                portfolio_vol = np.sqrt(
                    0.25 * (vol1**2 + vol2**2 + 2 * correlation * vol1 * vol2)
                )
                
                # Calculate Sharpe Ratio
                sharpe_ratio = (annual_return - risk_free_rate/100) / portfolio_vol
                
                risk_return_data.append({
                    'pair': pair,
                    'return': float(annual_return),
                    'risk': float(portfolio_vol),
                    'sharpe_ratio': float(sharpe_ratio),
                    'correlation': float(correlation)
                })

        # Sort pairs by Sharpe ratio
        risk_return_data.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return {
            'predictions': predictions,
            'risk_return_analysis': risk_return_data,
            'analysis_details': {
                'risk_free_rate': risk_free_rate,
                'time_period': f"{start} to {end}",
                'total_trading_days': len(returns)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate predictions: {str(e)}"
        )

class RegimeAnalysisRequest(BaseModel):
    sectors: List[str]
    timeframe: int

@app.post("/future-regime-analysis/")
async def analyze_future_regime(request: RegimeAnalysisRequest):
    try:
        # Validate sectors against the list from DataFetcher
        invalid_sectors = [s for s in request.sectors if s not in VALID_SECTORS]
        if invalid_sectors:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid sectors: {invalid_sectors}. Valid sectors are: {VALID_SECTORS}"
            )
        
        logger.info(f"Analyzing sectors: {request.sectors}")
        
        # Get sector data using the sector tickers from DataFetcher
        sector_indices = data_fetcher.get_sector_indices(request.sectors)
        if sector_indices is None:
            raise HTTPException(status_code=500, detail="Failed to fetch sector data")
            
        # Calculate returns
        returns = sector_indices.pct_change().fillna(0)
        
        # Calculate curvature
        curvature = data_fetcher.calculate_curvature(returns)
        
        # Analyze regime
        regime_analysis = regime_analyzer.analyze_regime(
            sector_indices=sector_indices,
            returns=returns,
            curvature_history=curvature,
            timeframe=request.timeframe
        )
        
        if regime_analysis is None:
            raise HTTPException(status_code=500, detail="Failed to analyze regime")
            
        return {"regime_analysis": regime_analysis}
        
    except Exception as e:
        logger.error(f"Error in analyze_future_regime: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors/")
async def get_valid_sectors():
    return {"sectors": VALID_SECTORS}

async def get_sector_representatives(sectors: List[str]) -> List[str]:
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC'],
        'Financial': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
        'Consumer_Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
        'Communication': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA'],
        'Industrial': ['CAT', 'BA', 'HON', 'UPS', 'GE'],
        'Consumer_Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'Basic_Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX'],
        'Real_Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
    }
    
    selected_stocks = []
    for sector in sectors:
        if sector in sector_mapping:
            selected_stocks.extend(sector_mapping[sector][:2])  # Take top 2 from each sector
    
    return selected_stocks

def create_sector_indices(prices_df: pd.DataFrame, sector_stocks: List[str]) -> pd.DataFrame:
    sector_indices = pd.DataFrame(index=prices_df.index)
    
    # Get market cap data for weighting
    market_caps = {}
    for ticker in sector_stocks:
        try:
            stock = yf.Ticker(ticker)
            market_caps[ticker] = stock.info.get('marketCap', 1e9)  # default to 1B if not found
        except:
            market_caps[ticker] = 1e9  # default to 1B on error
    
    # Create sector indices
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC'],
        'Financial': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
        'Consumer_Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
        'Communication': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA'],
        'Industrial': ['CAT', 'BA', 'HON', 'UPS', 'GE'],
        'Consumer_Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'Basic_Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX'],
        'Real_Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
    }
    
    # Create reverse mapping from stock to sector
    stock_to_sector = {}
    for sector, stocks in sector_mapping.items():
        for stock in stocks:
            stock_to_sector[stock] = sector
    
    # Calculate sector indices
    for sector in set(stock_to_sector[stock] for stock in sector_stocks):
        sector_stocks_list = [s for s in sector_stocks if stock_to_sector[s] == sector]
        if sector_stocks_list:
            # Get market caps for sector stocks
            sector_caps = {s: market_caps[s] for s in sector_stocks_list}
            total_cap = sum(sector_caps.values())
            weights = {s: cap/total_cap for s, cap in sector_caps.items()}
            
            # Calculate weighted index
            sector_index = sum(prices_df[stock] * weights[stock] 
                             for stock in sector_stocks_list)
            sector_indices[sector] = sector_index
    
    return sector_indices
