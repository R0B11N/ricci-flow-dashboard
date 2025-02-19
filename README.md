# Ricci Flow Dashboard: A Geometric Approach to Market Analysis
## Project Overview
A sophisticated financial analysis platform leveraging differential geometry concepts, specifically Ricci curvature, to analyze stock market network dynamics and predict market behavior.

![ricciimage](https://github.com/user-attachments/assets/c370969a-488c-4e11-8234-ec0db0d6a9e6)

## Mathematical Foundation: Ricci Flow and Financial Networks

### Theoretical Background
This project builds upon the groundbreaking work of Sandhu et al. (2016) in "Graph Curvature and the Robustness of Complex Networks" and extends it to financial market analysis. The fundamental insight is that Ricci curvature, a concept from differential geometry, can be adapted to discrete networks to measure network viability and structural characteristics.

### Ollivier-Ricci Curvature
The core mathematical concept of note is Ollivier-Ricci curvature, which is a discrete analog of Ricci curvature from Riemannian geometry. For a financial network:

#### Network Construction
- **Vertices (V):** Individual stocks
- **Edges (E):** Correlations between stocks
- **Weight function w:** E → [0,1]

![snetworkimage](https://github.com/user-attachments/assets/94b3c8df-df69-4f6d-988c-2118d02f5554)

#### Curvature Definition
For any two vertices \(x,y ∈ V\), the Ollivier-Ricci curvature κ(x,y) is defined as:
```math
\kappa(x,y) = 1 - \frac{W_1(\mu_x, \mu_y)}{d(x,y)}
```
where:

-  W<sub>1</sub> is the Wasserstein-1 (transportation) distance
-  μ<sub>x</sub>, μ<sub>y</sub> are probability measures
-  d(x,y) is the geodesic distance

### Application to Financial Markets

#### Market Network Construction
- **Nodes:** Represent individual stocks
- **Edge weights:** \( w<sub>ij,/sub >= ρ<sub>ij</sub> \) (correlation coefficient)
- **Price returns:** \( r<sub>i(t)</sub> = log(P<sub>i(t)</sub> | r<sub>i(t-1)</sub>) \)

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
- **Positive curvature:** More sturdy network
- **Negative curvature:** More fragile network
- \( κ > 0 \) indicates market stability
- \( κ < 0 \) suggests potential market stress

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
- \( Σ \): Covariance matrix
- \( μ \): Expected returns
- \( κ<sub>0</sub> \): Minimum curvature threshold

### Why Does it Work?

It works because it goes beyond simple pairwise correlations, capturing the higher-order relationships that influence the system.

This approach measures both local interaction strength and the overall network structure, providing a clearer picture of how elements are connected.

Additionally, it ensures topological stability by using Ricci curvature to bound entropy changes, offering valuable early warning signals for potential market stress. 

![toptradingpairsimage](https://github.com/user-attachments/assets/02711bda-097c-4c68-8e6e-dab7d002991f)

![performingpairsimage](https://github.com/user-attachments/assets/27918d76-0ff0-4184-bcbb-fa2183c9fbeb)

## Features

### Market Analysis
- Real-time stock data fetching via Yahoo Finance API
- Network visualization of stock relationships
- Ricci curvature calculation for market structure analysis
- Portfolio optimization based on network metrics
  
![marketsimage](https://github.com/user-attachments/assets/a557e742-03ba-4373-89a4-26158db5a15e)


### Prediction Capabilities Utilizing HMMs and the Sharpe Ratio:
- Market regime detection
- Lead-lag relationship analysis
- Stock pair correlation analysis
- Volatility and risk-return assessment
  
![marketimage](https://github.com/user-attachments/assets/7d1affe8-86f3-47f4-9137-f5295a5c6c29)

## Technical Architecture

### Backend (`/backend`)
- **FastAPI Framework**: High-performance API server
- **Key Components**:
  - `app.py`: Core application logic and stock data fetching
  - `data_fetcher.py`: Yahoo Finance integration
  - `ricci_flow.py`: Ricci curvature calculations
  - `market_analyzer.py`: Market analysis algorithms
  - `regime_analyzer.py`: Market regime detection
  - `lead_lag_analyzer.py`: Lead-lag relationship analysis (WIP)

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
- Docker and Docker Compose (optional) (still a WIP)

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

# If you're interested in adding any cool tidbits or your own work, create a Pull Request and give it a go!
