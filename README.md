# Ricci Flow Dashboard: A Geometric Approach to Market Analysis

## Project Overview
A sophisticated financial analysis platform leveraging differential geometry concepts, specifically Ricci curvature, to analyze stock market network dynamics and predict market behavior.

## Mathematical Foundation: Ricci Flow and Financial Networks

### Theoretical Background
This project builds upon the groundbreaking work of Sandhu et al. (2016) in "Graph Curvature and the Robustness of Complex Networks" and extends it to financial market analysis. The fundamental insight is that Ricci curvature, a concept from differential geometry, can be adapted to discrete networks to measure network robustness and structural characteristics.

### Ollivier-Ricci Curvature
The core mathematical concept we employ is Ollivier-Ricci curvature, which is a discrete analog of Ricci curvature from Riemannian geometry. For a financial network:

#### Network Construction
- **Vertices (V):** Individual stocks
- **Edges (E):** Correlations between stocks
- **Weight function w:** E → [0,1]

#### Curvature Definition
For any two vertices \(x,y \in V\), the Ollivier-Ricci curvature \( \kappa(x,y) \) is defined as:

```math
\kappa(x,y) = 1 - \frac{W_1(\mu_x, \mu_y)}{d(x,y)}
```
where:

- \( W_1 \) is the Wasserstein-1 (transportation) distance
- \( \mu_x, \mu_y \) are probability measures
- \( d(x,y) \) is the geodesic distance

### Application to Financial Markets

#### Market Network Construction
- **Nodes:** Represent individual stocks
- **Edge weights:** \( w_{ij} = \rho_{ij} \) (correlation coefficient)
- **Price returns:** \( r_i(t) = \log(P_i(t)/P_i(t-1)) \)

#### Network Measures
- **Correlation Matrix:**
  
  ```math
  \rho_{ij} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \sigma_j}
  ```

- **Local Curvature:**
  
  ```math
  \kappa(i) = \frac{\sum_j w_{ij} \kappa(i,j)}{\sum_j w_{ij}}
  ```

### Market Regimes
The scalar curvature flow equation:

```math
\frac{\partial g}{\partial t} = -2 Ric(g)
```

governs the evolution of market structure.

#### Key Theoretical Results

##### Network Stability
- **Positive curvature:** More robust network
- **Negative curvature:** More fragile network
- \( \kappa > 0 \) indicates market stability
- \( \kappa < 0 \) suggests potential market stress

##### Regime Detection
The temporal evolution of Ricci curvature helps identify market regimes:

```math
R(t) = \frac{\sum_{ij} \kappa_{ij}(t) w_{ij}(t)}{\sum_{ij} w_{ij}(t)}
```

### Portfolio Optimization
Using curvature as a risk measure:

```math
\min_w w^T \Sigma w \quad \text{s.t.} \quad w^T \mu \geq r_0, \quad \sum_i w_i = 1, \quad \kappa(w) \geq \kappa_0
```

where:
- \( w \): Portfolio weights
- \( \Sigma \): Covariance matrix
- \( \mu \): Expected returns
- \( \kappa_0 \): Minimum curvature threshold

### Why It Works
#### Geometric Interpretation
- Captures higher-order relationships beyond pairwise correlations
- Measures local interaction strength and global network structure

#### Topological Stability
- Ricci curvature bounds entropy changes
- Provides early warning signals for market stress

#### Scale Invariance
- Results are robust to market size
- Applicable across different time scales

### Implementation Details
#### Curvature Computation

```math
\kappa(x,y) \approx 1 - \frac{d_W(\mu_x, \mu_y)}{d(x,y)}
```

#### Optimal Transport
- Uses network simplex algorithm
- Approximates Wasserstein distance

#### Regime Detection
- Hidden Markov Models on curvature time series
- Maximum likelihood estimation for regime parameters


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
# OR
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

