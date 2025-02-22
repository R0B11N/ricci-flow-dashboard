import networkx as nx
import numpy as np
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_ricci_curvature(stock_data):
    """Modified for short time periods"""
    if (stock_data.index[-1] - stock_data.index[0]).days < 5:
        # Use simple correlation for very short periods
        corr_matrix = stock_data.corr()
        curvature_dict = {}
        for i, stock1 in enumerate(corr_matrix.columns):
            for j, stock2 in enumerate(corr_matrix.columns):
                if i < j:
                    key = f"{stock1}-{stock2}"
                    # Convert correlation directly to curvature
                    curvature_dict[key] = 2 * abs(corr_matrix.loc[stock1, stock2]) - 1
        return pd.Series(curvature_dict)
    
    if stock_data.shape[1] < 2:
        raise ValueError("Need at least 2 stocks to compute correlation")
        
    # Compute correlation matrix
    corr_matrix = stock_data.corr()
    
    # Build a graph where edges are weighted by correlation
    G = nx.Graph()
    for i, stock1 in enumerate(corr_matrix.columns):
        for j, stock2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicate edges
                correlation = corr_matrix.loc[stock1, stock2]
                # Convert correlation to distance (1 - correlation)
                G.add_edge(stock1, stock2, weight=max(0.01, 1 - abs(correlation)))

    try:
        # Initialize OllivierRicci with single process
        orc = OllivierRicci(
            G,
            alpha=0.5,
            verbose="ERROR",
            proc=0,  # Set to 0 to disable multiprocessing
            method="OTD",
            enable_logging=False
        )
        
        # Compute Ricci curvature
        orc.compute_ricci_curvature()
        
        # Extract Ricci curvature values
        curvature_dict = {}
        for edge in G.edges:
            key = f"{edge[0]}-{edge[1]}"
            value = G.edges[edge].get('ricciCurvature', 0.0)
            curvature_dict[key] = value
        
        return pd.Series(curvature_dict)
        
    except Exception as e:
        print(f"Error in Ricci computation: {str(e)}")
        # Return a simple correlation-based measure as fallback
        fallback_dict = {}
        for i, stock1 in enumerate(corr_matrix.columns):
            for j, stock2 in enumerate(corr_matrix.columns):
                if i < j:
                    key = f"{stock1}-{stock2}"
                    fallback_dict[key] = corr_matrix.loc[stock1, stock2]
        return pd.Series(fallback_dict)
