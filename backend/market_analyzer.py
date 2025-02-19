import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.curvature_history = {}
        self.volatility_threshold = 2.0  # Z-score threshold for anomalies
        
    def detect_anomalies(self, 
                        curvature_data: Dict[str, float], 
                        historical_curvatures: pd.DataFrame) -> List[Dict]:
        """
        Detect market anomalies based on sudden changes in Ricci curvature.
        
        Args:
            curvature_data: Current curvature values for each edge
            historical_curvatures: DataFrame of historical curvature values
        """
        try:
            anomalies = []
            
            # Calculate z-scores of curvature changes
            for edge, current_value in curvature_data.items():
                if edge in historical_curvatures.columns:
                    historical_values = historical_curvatures[edge].dropna()
                    if len(historical_values) > 0:
                        # Calculate z-score of current value
                        z_score = (current_value - historical_values.mean()) / historical_values.std()
                        
                        if abs(z_score) > self.volatility_threshold:
                            anomalies.append({
                                'edge': edge,
                                'z_score': z_score,
                                'current_value': current_value,
                                'historical_mean': historical_values.mean(),
                                'severity': 'high' if abs(z_score) > 3 else 'medium'
                            })
            
            return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return []

    def optimize_portfolio(self, 
                         curvature_data: Dict[str, float],
                         metadata: Dict,
                         returns_data: pd.DataFrame,
                         min_sharpe: float = 0.5,  # Lowered from default
                         target_risk: float = 0.2,  # Adjusted target risk
                         max_pairs: int = 10) -> List[Dict]:
        """
        Find optimal trading pairs based on curvature and market metrics.
        """
        try:
            pairs = []
            
            for edge, curvature in curvature_data.items():
                stock1, stock2 = edge.split('-')
                
                # Get returns for both stocks
                if stock1 not in returns_data.columns or stock2 not in returns_data.columns:
                    continue
                    
                pair_returns = returns_data[[stock1, stock2]]
                
                # Skip if we don't have enough data
                if len(pair_returns.dropna()) < 30:
                    continue
                    
                # Calculate volatilities
                vol1 = returns_data[stock1].std() * np.sqrt(252)  # Annualized
                vol2 = returns_data[stock2].std() * np.sqrt(252)
                avg_volatility = (vol1 + vol2) / 2
                
                # Calculate Sharpe ratio (using 0% risk-free rate for simplicity)
                avg_returns = pair_returns.mean().mean() * 252  # Annualized
                avg_sharpe = avg_returns / avg_volatility if avg_volatility > 0 else 0
                
                # More lenient filtering conditions
                if avg_sharpe < min_sharpe:
                    continue
                    
                # Calculate market cap compatibility (more lenient)
                market_cap_ratio = min(
                    metadata['market_caps'].get(stock1, 0),
                    metadata['market_caps'].get(stock2, 0)
                ) / max(
                    metadata['market_caps'].get(stock1, 0),
                    metadata['market_caps'].get(stock2, 0)
                ) if max(
                    metadata['market_caps'].get(stock1, 0),
                    metadata['market_caps'].get(stock2, 0)
                ) > 0 else 0
                
                # Calculate volume compatibility (more lenient)
                volume_ratio = min(
                    metadata['volumes'].get(stock1, 1),
                    metadata['volumes'].get(stock2, 1)
                ) / max(
                    metadata['volumes'].get(stock1, 1),
                    metadata['volumes'].get(stock2, 1)
                )
                
                # Enhanced scoring system (adjusted weights)
                score = (
                    0.35 * abs(curvature) +      # Increased weight for stability
                    0.15 * market_cap_ratio +    # Reduced weight for size
                    0.15 * volume_ratio +        # Reduced weight for liquidity
                    0.25 * avg_sharpe +          # Increased weight for returns
                    0.10 * (1 - abs(avg_volatility - target_risk))  # Risk matching
                )
                
                # Calculate hedge ratio
                hedge_ratio = vol1 / vol2 if vol2 != 0 else 1.0
                
                pairs.append({
                    'pair': edge,
                    'score': score,
                    'curvature': curvature,
                    'avg_volatility': avg_volatility,
                    'sharpe_ratio': avg_sharpe,
                    'market_cap_ratio': market_cap_ratio,
                    'volume_ratio': volume_ratio,
                    'hedge_ratio': hedge_ratio,
                    'sectors': [
                        metadata['sectors'].get(stock1, 'Unknown'),
                        metadata['sectors'].get(stock2, 'Unknown')
                    ],
                    'expected_return': avg_returns,
                    'risk_score': 1 - abs(avg_volatility - target_risk)
                })
            
            # Sort by score and return top pairs
            return sorted(pairs, key=lambda x: x['score'], reverse=True)[:max_pairs]
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return [] 