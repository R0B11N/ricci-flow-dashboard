import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class MarketSurfaceGenerator:
    def __init__(self, save_path="./visualizations"):
        self.save_path = save_path
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        
    def fetch_real_data(self, tickers, start_date, end_date):
        try:
            logger.info(f"Fetching data for {tickers}")
            # Increase timeout and add retries
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date,
                timeout=20,  # Increased timeout
                progress=False  # Reduce output noise
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                # Handle multiple stocks (returns 'Close' prices for each)
                returns = data['Close'].pct_change().dropna()
            else:
                # Handle single stock
                returns = data['Close'].pct_change().dropna()
                returns = pd.DataFrame(returns, columns=[tickers[0]])
            
            # Check if we got data for all tickers
            missing_tickers = [t for t in tickers if t not in returns.columns]
            if missing_tickers:
                logger.warning(f"Missing data for: {missing_tickers}")
                # Remove missing tickers from the list
                tickers = [t for t in tickers if t not in missing_tickers]
            
            if returns.empty:
                logger.error("No data returned from Yahoo Finance")
                return None
            
            return returns
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None

    def generate_synthetic_data(self, tickers, days=30, regime='normal'):
        # Synthetic data for the sake of representation
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        returns_data = {}
        
        # Define sector groups
        sectors = {
            'tech': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META'],
            'finance': ['JPM', 'GS', 'BAC'],
            'payments': ['V', 'MA'],
            'retail': ['AMZN', 'COST'],
            'entertainment': ['DIS', 'NFLX'],
            'ev': ['TSLA']
        }
        
        regimes = {
            'bull': {
                'tech': {'mean': 0.003, 'std': 0.015},
                'finance': {'mean': 0.002, 'std': 0.01},
                'default': {'mean': 0.001, 'std': 0.01}
            },
            'bear': {
                'tech': {'mean': -0.003, 'std': 0.02},
                'finance': {'mean': -0.002, 'std': 0.015},
                'default': {'mean': -0.002, 'std': 0.015}
            },
            'volatile': {
                'tech': {'mean': 0, 'std': 0.04},
                'finance': {'mean': 0, 'std': 0.03},
                'default': {'mean': 0, 'std': 0.03}
            },
            'sector_rotation': {
                'tech': {'mean': -0.002, 'std': 0.02},
                'finance': {'mean': 0.002, 'std': 0.015},
                'default': {'mean': 0, 'std': 0.02}
            },
            'crisis': {
                'tech': {'mean': -0.004, 'std': 0.05},
                'finance': {'mean': -0.005, 'std': 0.06},
                'default': {'mean': -0.003, 'std': 0.04}
            },
            'normal': {
                'tech': {'mean': 0.001, 'std': 0.01},
                'finance': {'mean': 0.001, 'std': 0.01},
                'default': {'mean': 0.001, 'std': 0.01}
            }
        }
        
        regime_params = regimes.get(regime, regimes['normal'])
        
        # Generate returns with sector correlations
        for ticker in tickers:
            # Find which sector the ticker belongs to
            sector = next((s for s, stocks in sectors.items() if ticker in stocks), 'default')
            params = regime_params.get(sector, regime_params['default'])
            
            # Generate base returns
            base_returns = np.random.normal(params['mean'], params['std'], len(dates))
            
            # Add sector-specific trends
            trend = np.linspace(0, params['mean'] * len(dates), len(dates))
            wave = np.sin(np.linspace(0, 4*np.pi, len(dates))) * params['std']
            
            returns_data[ticker] = base_returns + trend + wave
            
        return pd.DataFrame(returns_data, index=dates)

    def create_surface_visualization(self, returns, stocks, frame_idx=0, regime_info=None):

        plt.close('all')
        
        # Create figure with extra space for text
        fig = plt.figure(figsize=(20, 16))
        
        # Create main plot and description text area
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])  # Ratio between plot and text
        ax = fig.add_subplot(gs[0], projection='3d')
        text_ax = fig.add_subplot(gs[1])
        text_ax.axis('off')  # Hide axes for text area
        
        # Validate inputs
        if returns is None or returns.empty:
            logger.error("No valid return data provided")
            return None, None
        
        # Filter stocks to only those we have data for
        available_stocks = [s for s in stocks if s in returns.columns]
        if not available_stocks:
            logger.error("No valid stocks found in return data")
            return None, None
        
        # Use only available stocks
        stocks = available_stocks
        
        # Generate surface data
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Arrange stocks in a circle
        n_stocks = len(stocks)
        angles = np.linspace(0, 2*np.pi, n_stocks, endpoint=False)
        radius = 2.0  # Slightly smaller radius
        stock_positions = {}
        
        # Calculate positions
        for i, (stock, angle) in enumerate(zip(stocks, angles)):
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)
            stock_positions[stock] = (x_pos, y_pos)
        
        # Create market surface
        for stock, (x_pos, y_pos) in stock_positions.items():
            stock_return = returns[stock].iloc[frame_idx]
            Z += stock_return * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / 0.3)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn',
                              linewidth=0, antialiased=True,
                              alpha=0.7)
        
        # Add stock markers and labels
        for stock, (x_pos, y_pos) in stock_positions.items():
            z_pos = returns[stock].iloc[frame_idx]
            
            # Plot marker
            ax.scatter([x_pos], [y_pos], [z_pos], 
                      c='black', marker='o', s=100)
            
            # Add label just slightly above the point
            label_offset = 0.005  # Small fixed offset
            ax.text(x_pos, y_pos, z_pos + label_offset, 
                    stock,
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8,
                        pad=2
                    ),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    zorder=100)  # Ensure label is on top
            
            # Draw correlation lines
            for other_stock, (other_x, other_y) in stock_positions.items():
                if stock < other_stock:
                    other_z = returns[other_stock].iloc[frame_idx]
                    correlation = returns[stock].corr(returns[other_stock])
                    
                    if abs(correlation) > 0.5:
                        color = plt.cm.RdYlGn((correlation + 1) / 2)
                        ax.plot([x_pos, other_x], 
                               [y_pos, other_y], 
                               [z_pos, other_z],
                               color=color, linewidth=1.5, alpha=0.5)
        
        # Add title and description if regime info is provided
        if regime_info:
            title = regime_info['title']
            description = regime_info['description']
            
            # Add main title
            ax.set_title(f'Market Surface Visualization - {returns.index[frame_idx].date()}\n{title}',
                        fontsize=14, pad=20)
            
            # Add description text
            text_ax.text(0.5, 0.6, description,
                        wrap=True,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        bbox=dict(
                            facecolor='white',
                            edgecolor='lightgray',
                            alpha=0.8,
                            pad=10,
                            boxstyle='round'
                        ))
        else:
            # Default title for real data
            ax.set_title(f'Market Surface Visualization - {returns.index[frame_idx].date()}\n'
                        f'Real Market Data',
                        fontsize=14, pad=20)
            
            # Add description for real data
            text_ax.text(0.5, 0.6,
                        'Actual market data showing real-world relationships and movements between stocks. '
                        'The surface represents current market conditions and correlations.',
                        wrap=True,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        bbox=dict(
                            facecolor='white',
                            edgecolor='lightgray',
                            alpha=0.8,
                            pad=10,
                            boxstyle='round'
                        ))
        
        # Customize plot
        ax.set_xlabel('Market Space X', fontsize=12)
        ax.set_ylabel('Market Space Y', fontsize=12)
        ax.set_zlabel('Returns', fontsize=12)
        
        # Add correlation colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                  norm=plt.Normalize(-1, 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('Stock Correlation', fontsize=10)
        
        # Adjust view
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        return fig, ax

    def generate_animation(self, returns, stocks, save_path=None):

        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # Close any existing figures to prevent memory issues
        plt.close('all')

        # Create figure and axis
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        def animate(frame):
            ax.clear()
            # Generate the surface for this frame
            self.create_surface_visualization(returns, stocks, frame)
            return ax.get_children()

        logger.info(f"Creating animation with {len(returns)} frames")
        
        # Create animation with proper settings
        anim = FuncAnimation(
            fig,
            animate,
            frames=len(returns),
            interval=200,
            repeat=True,
            blit=False
        )

        if save_path:
            try:
                logger.info("Saving animation with PillowWriter...")
                writer = PillowWriter(
                    fps=5,  # Slower fps for better viewing
                    metadata=dict(artist='Market Visualizer')
                )
                anim.save(save_path, writer=writer)
                logger.info(f"Successfully saved animation to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save animation AUGHHH: {str(e)}")
                raise

        plt.close(fig)  # Clean up
        return anim

    def save_visualization(self, fig, filename):

        try:
            full_path = f"{self.save_path}/{filename}"
            fig.savefig(full_path)
            logger.info(f"Saved visualization to {full_path}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}") 