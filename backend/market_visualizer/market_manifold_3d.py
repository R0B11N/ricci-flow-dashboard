import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import MDS
import yfinance as yf
from datetime import datetime, timedelta
import imageio
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

class MarketManifold3D:
    def __init__(self):
        self.frames = []
        self.sector_colors = {
            'TECH': '#00ff88',
            'FINANCE': '#4169E1',
            'HEALTH': '#FF69B4',
            'CONSUMER': '#FFD700',
            'ENERGY': '#FF4500',
            'INDUSTRIAL': '#8B4513',
            'TELECOM': '#9370DB',
        }
    
    def fetch_data(self, start_date, end_date):
        stocks_by_sector = {
            'TECH': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC'],
            'FINANCE': ['JPM', 'BAC', 'GS', 'MS', 'V'],
            'HEALTH': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
            'CONSUMER': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
            'ENERGY': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'INDUSTRIAL': ['CAT', 'BA', 'HON', 'UPS', 'RTX'],
            'TELECOM': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA']
        }
        
        self.stocks = []
        self.stock_sectors = {}
        for sector, symbols in stocks_by_sector.items():
            for symbol in symbols:
                self.stocks.append(symbol)
                self.stock_sectors[symbol] = sector

        print(f"Fetching data from {start_date} to {end_date}...")
        df = yf.download(self.stocks, start=start_date, end=end_date)['Close']
        self.returns = df.pct_change().dropna()
        self.prices = df
        self.dates = self.returns.index
        
        return self.returns

    def _process_frame(self, frame_data):
        i, window_data, prices = frame_data
        
        try:
            # Handle NaN values in window_data
            window_data = np.nan_to_num(window_data, nan=0.0)
            
            # Compute correlations and handle NaN values
            corr = np.corrcoef(window_data.T)
            corr = np.nan_to_num(corr, nan=0.0)  # Replace NaN correlations with 0
            
            # Calculate returns as a dictionary mapping stock symbols to values
            returns_dict = {}
            for idx, stock in enumerate(self.stocks):
                try:
                    returns_dict[stock] = ((prices.iloc[-1][stock] / prices.iloc[0][stock]) - 1) * 100
                except:
                    returns_dict[stock] = 0.0  # Default if calculation fails
            
            # Create distance matrix and ensure no NaN values
            dist_matrix = 1 - np.abs(corr)
            dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
            np.fill_diagonal(dist_matrix, 0)
            
            positions = MDS(n_components=3, 
                          dissimilarity='precomputed',
                          normalized_stress='auto',
                          max_iter=100,
                          n_init=1,
                          random_state=42).fit_transform(dist_matrix)
            
            return i, positions, corr, returns_dict
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            return None
    def _create_frame(self, positions, correlations, returns_dict, frame_idx):
        x, y, z = positions.T
        
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            opacity=0.8,
            color='rgba(70, 70, 255, 0.8)',
            alphahull=2,
            flatshading=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.9,
                roughness=0.1,
                fresnel=0.2
            )
        ))
        
        # Add date annotation
        current_date = self.dates[min(frame_idx, len(self.dates)-1)].strftime('%Y-%m-%d')
        fig.add_annotation(
            text=f"Date: {current_date}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=24, color='white'),
            bgcolor='rgba(0,0,0,0.5)'
        )
        
        # Add points colored by sector
        for sector in self.sector_colors:
            sector_stocks = [s for s in self.stocks if self.stock_sectors[s] == sector]
            if not sector_stocks:
                continue
                
            indices = [self.stocks.index(s) for s in sector_stocks]
            sector_returns = [returns_dict[s] for s in sector_stocks]
            
            # Scale marker size by absolute return
            sizes = [max(8, min(20, 8 + abs(ret))) for ret in sector_returns]
            
            # Color intensity based on return
            colors = [self.sector_colors[sector] if ret >= 0 else '#FF0000' for ret in sector_returns]
            
            fig.add_trace(go.Scatter3d(
                x=x[indices], y=y[indices], z=z[indices],
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                text=[f"{s}: {ret:.1f}%" for s, ret in zip(sector_stocks, sector_returns)],
                textposition="top center",
                name=sector,
                hoverinfo='text'
            ))
        
        # Add correlation lines
        for i in range(len(self.stocks)):
            for j in range(i+1, len(self.stocks)):
                corr = correlations[i,j]
                if abs(corr) > 0.7:
                    fig.add_trace(go.Scatter3d(
                        x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                        mode='lines',
                        line=dict(
                            color='red' if corr > 0 else 'blue',
                            width=2 * abs(corr)
                        ),
                        opacity=0.7
                    ))
        
        # Very slow camera movement
        theta = frame_idx / 200 * 2 * np.pi
        r = 2.5
        fig.update_layout(
            scene = dict(
                camera=dict(
                    eye=dict(
                        x=r * np.cos(theta),
                        y=r * np.sin(theta),
                        z=1.8 + 0.3 * np.sin(theta/3)
                    ),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube',
                bgcolor='black'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0.5)'
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='black'
        )
        
        return fig

    def generate_animation(self, start_date="2024-11-01", end_date="2025-03-27", output_file="market_flow.gif"):
        try:
            print("\nFetching market data...")
            returns = self.fetch_data(start_date, end_date)
            
            # Define all frame-related variables at the start
            total_days = len(returns)
            n_keyframes = 60  # Target number of keyframes
            n_interp = 2      # Number of interpolation frames between keyframes
            n_frames = n_keyframes * n_interp
            frame_step = max(1, total_days // n_keyframes)
            
            # Only process the number of frames we need
            frame_indices = range(0, total_days, frame_step)[:n_keyframes]
            
            print(f"\nProcessing {n_keyframes} keyframes across {total_days} trading days...")
            
            # Process frames in parallel with more workers (TESTING)
            results = []
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
                futures = []
                for i in frame_indices:  # Use frame_indices instead of range
                    window_start = max(0, i-30)
                    frame_data = (
                        i,
                        self.returns.iloc[window_start:i+1].values,
                        self.prices.iloc[window_start:i+1]
                    )
                    futures.append(executor.submit(self._process_frame, frame_data))
                
                for i, future in enumerate(futures, 1):
                    try:
                        result = future.result(timeout=30)
                        if result is not None:
                            results.append(result)
                            print(f"✓ Keyframe {i}/{n_keyframes} processed")
                        else:
                            print(f"✗ Keyframe {i}/{n_keyframes} failed")
                    except Exception as e:
                        print(f"✗ Keyframe {i}/{n_keyframes} error: {str(e)}")
            
            results.sort(key=lambda x: x[0])
            
            # Generate frames with optimized settings
            print("\nGenerating frames:")
            frames = []
            
            # Reducing image quality slightly for faster processing (I HATE YOU RAHHHH)
            image_width = 1280   # Reduced from 1920
            image_height = 720   # Reduced from 1080
            image_scale = 1.5    # Reduced from 2
            
            for i in range(len(results)-1):
                _, pos1, corr1, rets1 = results[i]
                _, pos2, corr2, rets2 = results[i+1]
                
                for t in range(n_interp):
                    alpha = t / n_interp
                    smooth_alpha = alpha * alpha * (3 - 2 * alpha)
                    
                    pos = pos1 * (1 - smooth_alpha) + pos2 * smooth_alpha
                    corr = corr1 * (1 - smooth_alpha) + corr2 * smooth_alpha
                    
                    rets = {}
                    for stock in self.stocks:
                        rets[stock] = rets1[stock] * (1 - smooth_alpha) + rets2[stock] * smooth_alpha
                    
                    frame_idx = i * n_interp + t
                    fig = self._create_frame(pos, corr, rets, frame_idx)
                    
                    temp_file = f'temp_frame_{frame_idx:04d}.png'
                    fig.write_image(temp_file, 
                                  width=image_width, 
                                  height=image_height,
                                  scale=image_scale)
                    
                    frames.append(imageio.imread(temp_file))
                    os.remove(temp_file)
                    print(f"✓ Frame {frame_idx+1}/{n_frames} generated")
            
            # Add final keyframe
            _, pos, corr, rets = results[-1]
            fig = self._create_frame(pos, corr, rets, n_frames-1)
            temp_file = f'temp_frame_final.png'
            fig.write_image(temp_file, width=image_width, height=image_height, scale=image_scale)
            frames.append(imageio.imread(temp_file))
            os.remove(temp_file)
            
            print("\nSaving animation...")
            imageio.mimsave(output_file, 
                          frames,
                          fps=5,
                          quality=90,  # Slightly reduced quality but we ball nonetheless
                          loop=0)
            print(f"✓ Animation saved as {output_file}")
            
        except Exception as e:
            print(f"\n✗ Fatal error: {str(e)}")
            raise