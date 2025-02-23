import numpy as np
from hmmlearn import hmm
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Initially remade the regime_analyzer py file, but it was not working, took an old version and re-grafted it to work

class MarketRegimeAnalyzer:
    def __init__(self):
        self.hmm = hmm.GaussianHMM(
            n_components=3,  # Three regimes: stable bull, volatile bear, stable bear
            covariance_type="full",
            n_iter=100,
            random_state=42,
            init_params=""  # Don't initialize parameters automatically
        )
        
        # Set initial parameters manually
        self.hmm.startprob_ = np.array([0.6, 0.3, 0.1])  # Initial state probabilities
        self.hmm.transmat_ = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])

    def analyze_market_regimes(self, returns: pd.DataFrame) -> Dict:
        try:
            # Calculate rolling statistics properly
            mean_returns = returns.mean(axis=1)
            std_returns = returns.std(axis=1)
            rolling_mean = returns.rolling(5).mean().mean(axis=1)
            rolling_std = returns.rolling(5).std().mean(axis=1)
            lower_quantile = returns.quantile(0.1, axis=1)
            upper_quantile = returns.quantile(0.9, axis=1)
            
            # Calculate average pairwise correlation
            correlations = []
            for i in returns.columns:
                for j in returns.columns:
                    if i < j:
                        corr = returns[i].rolling(10).corr(returns[j])
                        correlations.append(corr)
            avg_correlation = pd.concat(correlations, axis=1).mean(axis=1)
            
            # Stack features with proper alignment
            features = np.column_stack([
                mean_returns,
                std_returns,
                rolling_mean,
                rolling_std,
                lower_quantile,
                upper_quantile,
                avg_correlation
            ])
            
            # Handle NaN values
            features = pd.DataFrame(features).fillna(method='ffill').fillna(method='bfill').values
            
            # Fit HMM and predict
            self.hmm.fit(features)
            hidden_states = self.hmm.predict(features)
            
            # Calculate regime statistics
            regime_stats = {}
            for i in range(self.hmm.n_components):
                mask = (hidden_states == i)
                regime_returns = returns.mean(axis=1).values[mask]
                
                if len(regime_returns) > 0:
                    avg_return = float(np.mean(regime_returns) * 100)
                    volatility = float(np.std(regime_returns) * 100)
                    
                    regime_type = "STABLE_BULL" if avg_return > 0 and volatility < 15 else \
                                "VOLATILE_BULL" if avg_return > 0 else \
                                "STABLE_BEAR" if volatility < 15 else "VOLATILE_BEAR"
                    
                    regime_stats[i] = {
                        'avg_return': avg_return,
                        'volatility': volatility,
                        'duration': int(np.sum(mask)),
                        'transitions': int(np.sum(np.diff(hidden_states) == i)),
                        'name': regime_type  # Add name to stats
                    }

            # Match the expected format from the original analyzer
            return {
                'regimes': hidden_states.tolist(),
                'regime_stats': regime_stats,
                'regime_types': {i: stats['name'] for i, stats in regime_stats.items()},
                'transition_matrix': self.hmm.transmat_.tolist(),
                'stability': np.diag(self.hmm.transmat_).tolist(),
                'dates': returns.index.strftime('%Y-%m-%d').tolist()
            }

        except Exception as e:
            logger.error(f"Error in market regime analysis: {str(e)}")
            raise ValueError(f"Failed to analyze market regimes: {str(e)}")