from surface_generator import MarketSurfaceGenerator
import os
import matplotlib.pyplot as plt

def run_demo():
    # Create generator
    generator = MarketSurfaceGenerator(save_path="./output")
    
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)
    
    # Define stocks with sector grouping
    stocks = [
        'AAPL',  # Tech
        'GOOGL', 
        'MSFT',
        'NVDA',
        'META',
        'AMZN',
        'TSLA',  # EV/Tech
        'JPM',   # Finance
        'GS',
        'BAC',
        'V',     # Payments
        'MA',
        'DIS',   # Entertainment
        'NFLX',
        'COST'   # Retail
    ]
    
    # Generate different regime visualizations with descriptions
    regimes = {
        'bull': {
            'title': 'Bull Market - Tech Rally',
            'description': 'Strong positive correlation among tech stocks, overall market optimism. '
                         'Characterized by rising peaks and positive returns.'
        },
        'bear': {
            'title': 'Bear Market - Risk Off',
            'description': 'Negative correlation between defensive and growth stocks. '
                         'Shows overall market pessimism with pronounced valleys.'
        },
        'normal': {
            'title': 'Normal Market Conditions',
            'description': 'Balanced market state with moderate correlations. '
                         'Mix of positive and negative movements across sectors.'
        },
        'volatile': {
            'title': 'High Volatility Regime',
            'description': 'Extreme peaks and valleys showing market stress. '
                         'High variability in returns and unstable correlations.'
        },
        'sector_rotation': {
            'title': 'Sector Rotation',
            'description': 'Some sectors rise while others fall. '
                         'Shows money flowing between different market segments.'
        },
        'crisis': {
            'title': 'Market Stress/Crisis',
            'description': 'High correlation across all stocks (usually negative). '
                         'Deep valleys showing market-wide stress and panic selling.'
        }
    }
    
    print("\nGenerating market surface visualizations...")
    
    for regime, info in regimes.items():
        print(f"\nCreating {info['title']}")
        
        # Generate synthetic data
        returns = generator.generate_synthetic_data(stocks, regime=regime)
        
        # Create static visualization with regime info
        fig, ax = generator.create_surface_visualization(
            returns, 
            stocks, 
            regime_info=info
        )
        generator.save_visualization(fig, f'market_surface_{regime}.png')
        plt.close(fig)
    
    # Real data visualization (no regime info needed)
    print("\nFetching real market data...")
    real_returns = generator.fetch_real_data(
        stocks,
        '2024-01-01',
        '2024-02-01'
    )
    
    if real_returns is not None and not real_returns.empty:
        print("Creating real market visualization...")
        available_stocks = [s for s in stocks if s in real_returns.columns]
        print(f"Using available stocks: {available_stocks}")
        
        fig, ax = generator.create_surface_visualization(
            real_returns, 
            available_stocks
        )
        if fig is not None:
            generator.save_visualization(fig, 'market_surface_real.png')
            plt.close(fig)
    else:
        print("Could not generate real market visualization due to data fetch issues")

    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    run_demo() 