from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_stock_info(ticker_list):
    """Fetch additional stock information."""
    info = {
        "market_caps": {},
        "volumes": {},
        "sectors": {},
        "volatilities": {}
    }
    
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            info_data = stock.info
            
            # Market Cap
            info["market_caps"][ticker] = info_data.get("marketCap", 0)
            
            # Average Volume
            info["volumes"][ticker] = info_data.get("averageVolume", 0)
            
            # Sector
            info["sectors"][ticker] = info_data.get("sector", "Unknown")
            
            # Volatility (beta)
            info["volatilities"][ticker] = info_data.get("beta", 0)
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {str(e)}")
            info["market_caps"][ticker] = 0
            info["volumes"][ticker] = 0
            info["sectors"][ticker] = "Unknown"
            info["volatilities"][ticker] = 0
            
    return info

@app.get("/ricci-curvature/")
async def get_ricci_curvature(tickers: str, start: str = "2020-01-01", end: str = "2024-01-01"):
    try:
        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        # Download price data
        data = yf.download(ticker_list, start=start, end=end, progress=False)
        
        # Basic correlation calculation
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data['Close'] if len(ticker_list) == 1 else data.xs('Close', axis=1, level=1)
            
        returns = prices.pct_change().dropna()
        correlation = returns.corr()
        
        # Get additional stock information
        stock_info = get_stock_info(ticker_list)
        
        # Calculate daily volatility from returns
        volatilities = returns.std() * np.sqrt(252)  # Annualized volatility
        for ticker in ticker_list:
            if stock_info["volatilities"][ticker] == 0:
                stock_info["volatilities"][ticker] = float(volatilities[ticker])
        
        # Convert correlation to curvature-like metric
        curvature = {}
        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                key = f"{ticker_list[i]}-{ticker_list[j]}"
                curvature[key] = float(correlation.iloc[i, j])
        
        return {
            "status": "success",
            "curvature": curvature,
            "metadata": {
                "tickers": ticker_list,
                "start_date": start,
                "end_date": end,
                "data_points": len(returns),
                "market_caps": stock_info["market_caps"],
                "volumes": stock_info["volumes"],
                "sectors": stock_info["sectors"],
                "volatilities": stock_info["volatilities"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 