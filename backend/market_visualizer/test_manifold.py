from market_manifold_3d import MarketManifold3D
from datetime import datetime, timedelta

def main():
    visualizer = MarketManifold3D()
    
    # Generate animation for 2024
    start_date = "2024-11-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    visualizer.generate_animation(
        start_date=start_date,
        end_date=end_date,
        output_file="market_flow_2024.gif"
    )

if __name__ == "__main__":
    main()