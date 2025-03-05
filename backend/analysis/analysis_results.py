import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import logging
import yfinance as yf

# Setup path to import backend modules
BACKEND_DIR = Path(__file__).parent.parent
sys.path.append(str(BACKEND_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'results' / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your existing modules
from data_fetcher import fetch_stock_data, DataFetcher
from ricci_flow import compute_ricci_curvature
from market_analyzer import MarketAnalyzer
from regime_analyzer import RegimeAnalyzer

class MarketAnalysisResults:
    def __init__(self, stocks=None, start_date="2020-01-01", end_date=None):
        self.stocks = stocks or ["MSFT", "GOOGL", "NVDA", "AAPL", "TSLA", "NFLX", "GME"]
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # Setup directories
        self.results_dir = Path(__file__).parent / 'results'
        self.viz_dir = self.results_dir / "visualizations"
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.market_analyzer = MarketAnalyzer()
        self.regime_analyzer = RegimeAnalyzer()
        
        logger.info(f"Analysis initialized for {len(self.stocks)} stocks")

    def validate_data(self, data):
        if data is None or not isinstance(data, dict):
            return False
        if 'prices' not in data or data['prices'] is None:
            return False
        if data['prices'].empty:
            return False
        return True

    def fetch_and_prepare_data(self):
        logger.info("Fetching market data...")
        try:
            # Direct yfinance download for prices
            data = yf.download(
                self.stocks,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                self.prices = data['Close']
            else:
                self.prices = pd.DataFrame(data['Close'], columns=self.stocks)
            
            # Calculate returns
            self.returns = self.prices.pct_change().dropna()
            
            # Calculate volatilities
            self.volatilities = self.returns.std() * np.sqrt(252)
            
            logger.info(f"Successfully fetched data with shape {self.returns.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return False

    def analyze_network_structure(self):
        logger.info("Analyzing network structure...")
        
        try:
            # Create correlation matrix
            self.corr_matrix = self.returns.corr()
            
            # Build network
            self.G = nx.Graph()
            for i in range(len(self.stocks)):
                for j in range(i + 1, len(self.stocks)):
                    correlation = self.corr_matrix.iloc[i, j]
                    self.G.add_edge(self.stocks[i], self.stocks[j], weight=abs(correlation))
            
            # Calculate network metrics
            self.network_metrics = {
                "density": nx.density(self.G),
                "clustering_coefficient": nx.average_clustering(self.G, weight="weight"),
                "avg_degree": np.mean([d for n, d in self.G.degree()]),
                "centrality": nx.eigenvector_centrality(self.G, weight="weight")
            }
            
            # Save network metrics
            pd.DataFrame(self.network_metrics).to_csv(
                self.results_dir / "network_metrics.csv"
            )
            logger.info("Network analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in network analysis: {str(e)}")
            return False

    def analyze_regimes(self):
        logger.info("Analyzing market regimes...")
        
        try:
            # Prepare sector indices (using returns as proxy)
            sector_indices = self.returns
            
            # Create historical curvatures
            historical_curvatures = pd.DataFrame(index=self.returns.index)
            for i in range(len(self.stocks)):
                for j in range(i + 1, len(self.stocks)):
                    pair = f"{self.stocks[i]}-{self.stocks[j]}"
                    historical_curvatures[pair] = self.returns[self.stocks[i]].rolling(30).corr(
                        self.returns[self.stocks[j]]
                    )
            
            # Run regime analysis
            self.regime_results = self.regime_analyzer.analyze_regime(
                sector_indices=sector_indices,
                returns=self.returns,
                curvature_history=historical_curvatures,
                timeframe=30
            )
            
            if self.regime_results:
                pd.DataFrame(self.regime_results).to_csv(
                    self.results_dir / "regime_results.csv"
                )
                logger.info("Regime analysis completed")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            return False

    def generate_visualizations(self):
        logger.info("Generating visualizations...")
        
        try:
            # 1. Correlation Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.corr_matrix, 
                       annot=True, 
                       cmap='coolwarm',
                       vmin=-1, 
                       vmax=1)
            plt.title("Stock Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(self.viz_dir / "correlation_heatmap.png")
            plt.close()
            
            # 2. Network Graph
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.G)
            nx.draw(self.G, pos,
                   node_color='lightblue',
                   node_size=[v * 3000 for v in self.network_metrics['centrality'].values()],
                   width=[self.G[u][v]['weight'] * 2 for u, v in self.G.edges()],
                   with_labels=True,
                   font_size=10)
            plt.title("Stock Market Network")
            plt.savefig(self.viz_dir / "network_graph.png")
            plt.close()
            
            # 3. Returns Distribution
            plt.figure(figsize=(12, 6))
            for stock in self.stocks:
                sns.kdeplot(data=self.returns[stock], label=stock)
            plt.title("Returns Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.viz_dir / "returns_distribution.png")
            plt.close()
            
            logger.info("Visualizations completed")
            return True
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return False

    def run_statistical_tests(self):
        logger.info("Running statistical tests...")
        
        try:
            stats_results = {
                'normality_tests': {},
                'correlation_tests': {},
                'volatility': {}
            }
            
            # Normality tests
            for stock in self.stocks:
                stat, p_value = stats.normaltest(self.returns[stock].dropna())
                stats_results['normality_tests'][stock] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
                
                # Add volatility
                stats_results['volatility'][stock] = self.volatilities[stock]
            
            # Correlation tests
            for i in range(len(self.stocks)):
                for j in range(i + 1, len(self.stocks)):
                    pair = f"{self.stocks[i]}-{self.stocks[j]}"
                    corr, p_value = stats.pearsonr(
                        self.returns[self.stocks[i]].dropna(),
                        self.returns[self.stocks[j]].dropna()
                    )
                    stats_results['correlation_tests'][pair] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'is_significant': p_value < 0.05
                    }
            
            # Save results
            with open(self.results_dir / "statistical_tests.txt", 'w') as f:
                f.write("Statistical Analysis Results\n")
                f.write("==========================\n\n")
                
                f.write("Normality Tests:\n")
                for stock, result in stats_results['normality_tests'].items():
                    f.write(f"{stock}: p-value = {result['p_value']:.4f} ")
                    f.write(f"({'normal' if result['is_normal'] else 'non-normal'})\n")
                
                f.write("\nCorrelation Tests:\n")
                for pair, result in stats_results['correlation_tests'].items():
                    f.write(f"{pair}: corr = {result['correlation']:.4f}, ")
                    f.write(f"p-value = {result['p_value']:.4f}\n")
                
                f.write("\nAnnualized Volatility:\n")
                for stock, vol in stats_results['volatility'].items():
                    f.write(f"{stock}: {vol:.4f}\n")
            
            logger.info("Statistical tests completed")
            return stats_results
            
        except Exception as e:
            logger.error(f"Error in statistical tests: {str(e)}")
            return None

    def generate_summary_report(self, stats_results=None):
        logger.info("Generating summary report...")
        
        try:
            with open(self.results_dir / "summary_report.txt", 'w') as f:
                f.write("Market Analysis Summary Report\n")
                f.write("============================\n\n")
                
                if hasattr(self, 'network_metrics'):
                    f.write("Network Metrics:\n")
                    f.write(str(self.network_metrics) + "\n\n")
                
                if hasattr(self, 'regime_results'):
                    f.write("Regime Analysis:\n")
                    f.write(str(self.regime_results) + "\n\n")
                
                if stats_results:
                    f.write("Statistical Analysis:\n")
                    f.write("Highest Correlation: ")
                    corr_tests = stats_results['correlation_tests']
                    max_corr = max(corr_tests.items(), key=lambda x: abs(x[1]['correlation']))
                    f.write(f"{max_corr[0]}: {max_corr[1]['correlation']:.4f}\n\n")
                    
                    f.write("Most Volatile Stock: ")
                    max_vol = max(stats_results['volatility'].items(), key=lambda x: x[1])
                    f.write(f"{max_vol[0]}: {max_vol[1]:.4f}\n")
            
            logger.info("Summary report generated")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")

    def run_analysis(self):
        logger.info("Starting full analysis...")
        
        try:
            if not self.fetch_and_prepare_data():
                raise ValueError("Data preparation failed")
            
            if not self.analyze_network_structure():
                raise ValueError("Network analysis failed")
            
            if not self.analyze_regimes():
                raise ValueError("Regime analysis failed")
            
            if not self.generate_visualizations():
                raise ValueError("Visualization generation failed")
            
            stats_results = self.run_statistical_tests()
            if stats_results is None:
                raise ValueError("Statistical tests failed")
            
            self.generate_summary_report(stats_results)
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    try:
        analyzer = MarketAnalysisResults()
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 