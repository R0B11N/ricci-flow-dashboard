# Ricci Flow Dashboard

A sophisticated financial analysis platform that uses Ricci curvature to analyze stock market network dynamics and predict market behavior.

## Overview

The Ricci Flow Dashboard combines differential geometry concepts with financial market analysis to provide insights into market structure and stock relationships. It uses the mathematical concept of Ricci curvature to analyze the interconnectedness of stocks and predict market behavior.

## Features

### Market Analysis
- Real-time stock data fetching via Yahoo Finance API
- Network visualization of stock relationships
- Ricci curvature calculation for market structure analysis
- Portfolio optimization based on network metrics

### Prediction Capabilities
- Market regime detection
- Lead-lag relationship analysis
- Stock pair correlation analysis
- Volatility and risk assessment

### Technical Analysis Tools
- Anomaly detection using Z-scores
- Sector-based analysis
- Market cap and volume consideration
- Beta and volatility metrics

## Technical Architecture

### Backend (`/backend`)
- **FastAPI Framework**: High-performance API server
- **Key Components**:
  - `app.py`: Core application logic and stock data fetching
  - `data_fetcher.py`: Yahoo Finance integration
  - `ricci_flow.py`: Ricci curvature calculations
  - `market_analyzer.py`: Market analysis algorithms
  - `regime_analyzer.py`: Market regime detection
  - `lead_lag_analyzer.py`: Lead-lag relationship analysis

### Frontend (`/frontend`)
- **React Framework**: Modern UI implementation
- **Key Features**:
  - Interactive stock selection
  - Real-time data visualization
  - Network graph rendering
  - Analysis results display

## Installation

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker and Docker Compose (optional)

### Local Development Setup

1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 5000
OR
python -m uvicorn main:app --reload
```

2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

## API Endpoints

### Market Analysis
- `GET /ricci-curvature/`: Calculate Ricci curvature for selected stocks
- `GET /market-analysis/`: Comprehensive market structure analysis
- `GET /regime-analysis/`: Detect market regimes
- `GET /lead-lag-analysis/`: Analyze lead-lag relationships
- `GET /predictions/`: Generate market predictions

## Technology Stack

### Backend
- FastAPI (API Framework)
- pandas (Data Processing)
- numpy (Numerical Operations)
- yfinance (Stock Data)
- networkx (Network Analysis)
- GraphRicciCurvature (Curvature Calculations)
- scikit-learn (Machine Learning)
- hmmlearn (Hidden Markov Models)

### Frontend
- React
- Plotly.js (Visualization)
- Axios (API Client)
- React-Plotly.js (Interactive Charts)

## Mathematical Foundation

The dashboard implements Ricci curvature analysis on stock market networks:
- Network construction using stock correlations
- Ricci curvature calculation for edge weights
- Regime detection using curvature dynamics
- Portfolio optimization using network metrics

## Usage Examples

### Basic Market Analysis
```bash
# Fetch and analyze stocks
GET /ricci-curvature/?tickers=AAPL,MSFT,GOOGL&start=2023-01-01&end=2024-01-01

# Detect market regimes
GET /regime-analysis/?tickers=AAPL,MSFT,GOOGL&start=2023-01-01

# Analyze lead-lag relationships
GET /lead-lag-analysis/?tickers=AAPL,MSFT,GOOGL&start=2023-01-01
```
