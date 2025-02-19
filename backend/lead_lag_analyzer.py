import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from scipy.signal import correlate
import logging

logger = logging.getLogger(__name__)

class LeadLagAnalyzer:
    def __init__(self, max_lag: int = 10):
        self.max_lag = max_lag

    def analyze_lead_lag(self, curvature_history: pd.DataFrame, returns: pd.DataFrame, metadata: Dict) -> Dict:
        try:
            lead_lag_relationships = []
            sector_leaders = {}

            # Get list of stocks
            stocks = list(returns.columns)

            # Calculate lead-lag relationships for each pair
            for i in range(len(stocks)):
                for j in range(i + 1, len(stocks)):
                    stock1, stock2 = stocks[i], stocks[j]

                    # Calculate cross-correlation
                    returns1 = returns[stock1].fillna(0)
                    returns2 = returns[stock2].fillna(0)
                    
                    # Normalize returns
                    returns1_norm = (returns1 - returns1.mean()) / returns1.std()
                    returns2_norm = (returns2 - returns2.mean()) / returns2.std()
                    
                    # Calculate cross-correlation
                    cross_corr = correlate(returns1_norm, returns2_norm, mode='full')
                    lags = np.arange(-(len(returns1)-1), len(returns1))
                    
                    # Find the lag with maximum correlation
                    max_corr_idx = np.argmax(np.abs(cross_corr))
                    lag_days = lags[max_corr_idx]
                    max_correlation = cross_corr[max_corr_idx]
                    
                    # Only include significant relationships
                    if abs(max_correlation) > 0.3 and abs(lag_days) <= self.max_lag:
                        leader = stock1 if lag_days < 0 else stock2
                        follower = stock2 if lag_days < 0 else stock1
                        actual_lag = abs(lag_days)
                        
                        relationship = {
                            'leader': leader,
                            'follower': follower,
                            'lag_days': int(actual_lag),
                            'correlation': float(max_correlation),
                            'significance': float(abs(max_correlation) * (1 - actual_lag/self.max_lag)),
                            'leader_sector': metadata.get('sectors', {}).get(leader, 'Unknown'),
                            'follower_sector': metadata.get('sectors', {}).get(follower, 'Unknown')
                        }
                        
                        lead_lag_relationships.append(relationship)
                        
                        # Update sector leaders
                        sector = relationship['leader_sector']
                        if sector not in sector_leaders or \
                           relationship['significance'] > sector_leaders[sector]['significance']:
                            sector_leaders[sector] = {
                                'stock': leader,
                                'significance': relationship['significance'],
                                'avg_lag': actual_lag
                            }

            # Sort relationships by significance
            lead_lag_relationships.sort(key=lambda x: x['significance'], reverse=True)
            
            return {
                'lead_lag_relationships': lead_lag_relationships,
                'sector_leaders': sector_leaders
            }
            
        except Exception as e:
            logger.error(f"Error in lead-lag analysis: {str(e)}")
            return {
                'lead_lag_relationships': [],
                'sector_leaders': {}
            }
        __all__ = ['LeadLagAnalyzer']