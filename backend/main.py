from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime
import logging
from ricci_flow import compute_ricci_curvature
from data_fetcher import fetch_stock_data
from fastapi.middleware.cors import CORSMiddleware
from market_analyzer import MarketAnalyzer
from regime_analyzer import RegimeAnalyzer
import pandas as pd
from lead_lag_analyzer import LeadLagAnalyzer
import numpy as np

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

def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

@app.get("/ricci-curvature/")
async def get_ricci_curvature(tickers: str, start: str = "2020-01-01", end: str = "2024-01-01"):
    """
    Compute Ricci curvature for selected stocks.
    
    Args:
        tickers: Comma-separated list of stock tickers (e.g., "AAPL,GOOGL,MSFT")
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
    """
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
async def analyze_regimes(tickers: str, start: str = "2020-01-01", end: str = None):
    try:
        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        # Set end date to today if not provided
        if not end:
            end = datetime.now().strftime("%Y-%m-%d")
            
        # Fetch stock data
        stock_data = fetch_stock_data(ticker_list, start, end)
        if not stock_data:
            raise HTTPException(status_code=404, detail="Failed to fetch stock data")
            
        # Calculate returns and prepare curvature history
        returns = stock_data['prices'].pct_change().dropna()
        
        # Create curvature history DataFrame
        curvature_history = pd.DataFrame()
        window_size = 30  # 30-day rolling window
        
        # Calculate rolling correlations for each pair
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                pair = f"{ticker_list[i]}-{ticker_list[j]}"
                rolling_corr = returns[ticker_list[i]].rolling(window=window_size).corr(returns[ticker_list[j]])
                curvature_history[pair] = rolling_corr
                
        # Detect regimes
        regime_results = regime_analyzer.detect_regimes(curvature_history, returns)
        
        if regime_results is None:
            raise HTTPException(status_code=500, detail="Failed to detect market regimes")
            
        return {
            "status": "success",
            "regimes": regime_results['regimes'],
            "regime_stats": regime_results['regime_stats'],
            "regime_types": regime_results['regime_types'],
            "transition_matrix": regime_results['transition_matrix'],
            "stability": regime_results['stability'],
            "dates": regime_results['dates']
        }
        
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
