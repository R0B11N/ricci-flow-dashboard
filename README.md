# Ricci Flow Dashboard: A Geometric Approach to Market Analysis

![riccistonksimage](https://github.com/user-attachments/assets/e8e016b9-69bc-466f-94ba-3685ca74c5a4)

---

A financial analysis platform that applies differential geometry, particularly Ricci curvature, to model stock market dynamics and predict future trends.

## Mathematical Foundation: Ricci Flow and Financial Networks

---

### Abstract

Imagine the stock market visualized as a flexible geometric surface (like a plastic sheet!) where each point represents a collection of stock prices, the shape of the surface tells us about market behavior, and curvature measures how this shape deviates from being flat. Consider the outputs in backend/market_visualizer/output to demonstrate how we model our stock market.

Representing the market as a geometric surface where peaks and valleys show how stocks move together or apart - similar to the plastic sheet we referred to that has connected points that rise when stocks do well (creating mountains) and sink when they struggle (forming valleys), while the lines between stocks show how closely they move together, giving us a 3D map of market behavior and relationships. 

On this geometric representation (one of many), we can observe different market regimes or states - like how during a bull market the surface forms mostly peaks showing widespread optimism, or during a crisis it creates deep valleys indicating market-wide stress.

The bending or curvature of this surface (mathematically called Ricci curvature) helps us measure market stability - a highly curved surface suggests market stress, while a flatter surface indicates more stable conditions. Therein- this visual approach makes complex market analysis more notable, helping us see patterns and relationships that might be hidden in traditional stock charts and numbers.

---

![peaksimage](https://github.com/user-attachments/assets/b3b2bfdc-c0fd-48f1-b1f6-d619b35e102e)

---

![marketmanifoldimage](https://github.com/user-attachments/assets/aa86bb7a-0927-4548-a186-b2977c224225)

---

![alt-text](https://github.com/R0B11N/ricci-flow-dashboard/blob/main/backend/market_flow_2024.gif)

---

### Theoretical Background
This project builds upon the groundbreaking work of Sandhu et al. (2016) in "Graph Curvature and the Robustness of Complex Networks" and extends it to financial market analysis. The fundamental insight is that Ricci curvature, a concept from differential geometry, can be adapted to discrete networks to measure network viability and structural characteristics. Essentially, curvature measures how interconnected and resilient a financial network is- positive curvature indicates stability, where stocks move cohesively, while negative curvature signals fragility, with sectors behaving inconsistently. By detecting these shifts early, curvature provides a powerful tool for predicting volatility and identifying regime transitions before they fully develop. Therein, using the geometric surface metaphor, we can do some cool operations with Riemannian Geometry to create real market predictions.

While the approach is inspired by Ricci flow and differential geometry, it's important to note that the foundation is built off of a discretized version adapted for financial markets. Specifically:

- Reducing Dimensionality using Multi-Dimensional Scaling (so it can actually function)
- Using Ollivier-Ricci curvature to measure local market structure and stability
- The implementation captures market relationships through a network-based approximation of geometric concepts
- The connection to pure differential geometry serves as a mathematical inspiration rather than a strict theoretical framework

This distinction helps frame our results in their proper context: as practical tools for market analysis rather than pure geometric theory.

---

### Ollivier-Ricci Curvature
The core mathematical concept of note is Ollivier-Ricci curvature, which is a discrete analog of Ricci curvature from Riemannian geometry. For a financial network:

---

#### Network Construction
- **Vertices (V):** Individual stocks
- **Edges (E):** Correlations between stocks
- **Weight function w:** E → [0,1]

---

![snetworkimage](https://github.com/user-attachments/assets/fc77f466-e143-4c3a-a12e-addba1c8cb86)

---

#### Curvature Definition
For any two vertices \(x,y ∈ V\), the Ollivier-Ricci curvature κ(x,y) is defined as:
```math
\kappa(x,y) = 1 - \frac{W_1(\mu_x, \mu_y)}{d(x,y)}
```
where:

-  W<sub>1</sub> is the Wasserstein-1 (transportation) distance
-  μ<sub>x</sub>, μ<sub>y</sub> are probability measures representing local mass distributions
-  d(x,y) is the shortest-path distance in the weighted network

---

### Application to Financial Markets

#### Market Network Construction
- **Nodes:** Represent individual stocks
- **Edge weights:** \( w<sub>ij</sub> ≥ ρ<sub>ij</sub> \) (correlation coefficient)
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

---

### Market Regimes
The scalar curvature flow equation:

```math
\frac{\partial g}{\partial t} = -2 Ric(g)
```

governs the evolution of market structure.

---

#### Key Theoretical Results

##### Network Stability

Therein, we can deduce that:

- **Positive curvature:** More sturdy network
- **Negative curvature:** More fragile network
- \( κ > 0 \) indicates market stability
- \( κ < 0 \) suggests potential market stress

---

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

---

### Math TL;DR: Curvature helps analyze market structure, predict regime shifts, and optimize portfolios for stability.

Within the model:
- Ollivier-Ricci Curvature: Measures how "tight" or "spread out" connections between stocks are, helping us detect strong or weak market structures.
- Market Analysis: Stocks form a network where curvature measures how interconnected and resilient the market is. High curvature means strong, stable relationships, while low curvature signals potential instability.
- Regime Predictions: The evolution of curvature over time helps detect shifts between stable and volatile market conditions. If curvature trends negative, stress is increasing, signaling a potential downturn.
- Portfolio Optimization: Curvature is used as a risk constraint, ensuring investments stay in stable regions by selecting assets that maintain a minimum curvature threshold, reducing exposure to fragile market conditions.

---
![toptradingpairsimage](https://github.com/user-attachments/assets/02711bda-097c-4c68-8e6e-dab7d002991f)

---

![performingpairsimage](https://github.com/user-attachments/assets/27918d76-0ff0-4184-bcbb-fa2183c9fbeb)

---

## Features

### Market Analysis
- Real-time stock data fetching via Yahoo Finance API
- Network visualization of stock relationships
- Ricci curvature calculation for market structure analysis
- Portfolio optimization based on network metrics

---

![marketsfimage](https://github.com/user-attachments/assets/6632b956-531f-467b-b7ee-41822f5e2308)

---

### Prediction Capabilities Utilizing Hidden Markov Models and the Sharpe Ratio:
- Market regime detection
- Lead-lag relationship analysis
- Stock pair correlation analysis
- Volatility and risk-return assessment

---

![bearlolimage](https://github.com/user-attachments/assets/fde57249-26c4-4a17-b97c-b3d9f610fb13)

---

## Applicability

## Future Regime Predictor
--- 

![futuresimage](https://github.com/user-attachments/assets/2d83ff3d-ae71-46a5-b97a-375eb376dcad)

---

![futuresiiimage](https://github.com/user-attachments/assets/04832ed3-12a8-420f-b295-447baec0b5f9)

---

The future regime predictor employs a hybrid approach combining geometric flow analysis with Hidden Markov Models. The system analyzes market topology through sectoral Ricci curvature and geodesic flows.

The sectoral Ricci tensor computation (referenced in `geometric_regime_analyzer.py`) follows the standard formula for our geodesic flow analysis, where the market evolution is governed by the modified Ricci flow equation:

```math
\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \nabla_i X_j + \nabla_j X_i
```

where:
- \( g<sub>ij</sub> \) is the metric tensor
- \( R<sub>ij</sub> \) is the Ricci tensor
- \( X<sub>i</sub> \) represents the sectoral velocity field
- \( ∇<sub>ij</sub> \) represents the covariant derivative w.r.t coordinate i,j

---

#### Regime Transition Dynamics

The regime transition probability is computed using a combination of geometric and statistical features:

The system incorporates:

1. **Sectoral Velocity Fields**: Derived from the Ricci flow equation
   
2. **Market State Tensors**: Constructed from:
   - Rolling volatility metrics
   - Cross-sectional dispersion
   - Correlation structure evolution

---

#### Metric Evolution

The metric tensor evolution follows a modified Hamilton-DeTurck flow:

```math
\frac{\partial}{\partial t} g_{\alpha\beta} = -2R_{\alpha\beta} + \mathcal{L}_W g_{\alpha\beta}
```

where:
```math
\mathcal{L}_W g_{\alpha\beta} = \nabla_{\alpha} W_{\beta} + \nabla_{\beta} W_{\alpha}
```
represents the Lie derivative along the chosen vector field W<sub>β</sub> ensuring gauge-fixing.

If X<sub>i</sub> and V<sub>α</sub> are meant to represent sectoral velocity fields, we unify notation as follows:
```math
\frac{\partial g_{ij}}{\partial t} = -2R_{ij} + \nabla_{i} W_{j} + \nabla_{j} W_{i}
```
to maintain consistency across formulations.

---

### Math TL;DR: We use Ricci flow to track how the market's structure evolves over time, like watching how a financial network "bends" or "flattens" under stress.

- Ricci Flow Equation: Adjusts the market’s "shape" by smoothing out irregularities, helping detect whether sectors are moving in sync (stable) or pulling apart (unstable).
- Sectoral Velocity Fields: Measure how fast different parts of the market are shifting, acting like a "wind map" for sector movements.
- Hamilton-DeTurck Flow: Ensures our model stays consistent by adjusting for distortions, similar to keeping a map properly aligned over time.
- Hidden Markov Model: Uses these features to predict the probability of market regime changes, like detecting early warning signs of crashes or booms.
  
#### All of which are necessary to forecast a market shift before it happens.

---

## Technical Architecture

### Backend (`/backend`)
- **FastAPI Framework**: High-performance API server
- **Key Components**:
  - `app.py`: Core application logic and stock data fetching
  - `data_fetcher.py`: Yahoo Finance integration
  - `ricci_flow.py`: Ricci curvature calculations
  - `market_analyzer.py`: Market analysis algorithms
  - `market_regime_analyzer.py`: Stock Pair - Time Regime Analysis
  - `regime_analyzer.py`: Future market regime detection
  -  `geometric_regime_analyzer.py`: Advanced regime detection using Hidden Markov Models (HMM) and geometric features:
    - Multi-dimensional feature extraction from market data
    - Regime classification (Stable/Volatile Bull/Bear)
    - Transition probability matrices
    - Stability metrics based on eigenvalue analysis
  - `lead_lag_analyzer.py`: Lead-lag relationship analysis (WIP)
    

### Frontend (`/frontend`)
- **React Framework**: Modern UI implementation
- **Key Features**:
  - Interactive stock selection
  - Real-time data visualization
  - Network graph rendering
  - Analysis results display

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker and Docker Compose (optional) (still a WIP)

### Local Development Setup

1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
# ONLY THEN YOU CAN DO:
python -m uvicorn main:app --reload
# Any dependency issues can be adjusted in the requirements file. YFinance on a trial run had to be updated.
# If YF doesn't work, do:
pip install yfinance --upgrade
```
![backendimage](https://github.com/user-attachments/assets/bddb3f21-5a59-430a-a43f-12727b5be6d5)

2. Frontend Setup
```bash
cd frontend
npm install
npm start
# wait a bit, make sure the backend is running first
```
![hooplaimage](https://github.com/user-attachments/assets/20815c0c-e364-4990-accf-8785f967f08d)
![WOOHOOimage](https://github.com/user-attachments/assets/d1a68762-9ca6-4ad6-b3ad-b093eee65ca9)

3. Additional Notes

If you're prompted with no optimal trading partners or the graph doesn't generate, try using less stock comparisons, or refresh the application.

---

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

---

### Analytics

Monte Carlo simulations were conducted on the Ricci Flow Curvature, and the analytics can be found under /notebooks -> simulations.ipynb

#### If you're interested in adding any cool tidbits or your own work, create a Pull Request and give it a go! If something's broken, please make the pull request and let me know of the issue.
