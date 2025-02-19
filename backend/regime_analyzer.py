import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RegimeAnalyzer:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, curvature_history: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for regime detection.
        """
        try:
            # First, ensure indexes match
            common_index = curvature_history.index.intersection(returns.index)
            curvature_history = curvature_history.loc[common_index]
            returns = returns.loc[common_index]

            # Create features DataFrame
            features = pd.concat([
                curvature_history.mean(axis=1),  # Average curvature
                curvature_history.std(axis=1),   # Curvature dispersion
                returns.mean(axis=1),            # Market return
                returns.std(axis=1),             # Market volatility
                curvature_history.quantile(0.1, axis=1),  # Lower tail behavior
                curvature_history.quantile(0.9, axis=1),  # Upper tail behavior
                curvature_history.rolling(5).mean().mean(axis=1),  # 5-day trend
            ], axis=1)

            # Handle NaN values
            features = features.fillna(method='ffill')  # Forward fill
            features = features.fillna(method='bfill')  # Backward fill for any remaining NaNs
            
            # Drop any remaining rows with NaN values
            features = features.dropna()

            if len(features) == 0:
                raise ValueError("No valid data points after cleaning")

            # Scale the features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features

        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            raise ValueError(f"Failed to prepare features: {str(e)}")

    def detect_regimes(self, curvature_history: pd.DataFrame, returns: pd.DataFrame) -> Dict:
        """
        Detect market regimes using HMM.
        """
        try:
            # Ensure we have enough data
            if len(curvature_history) < 30 or len(returns) < 30:
                logger.error("Not enough data for regime detection")
                raise ValueError("Insufficient data for regime detection (minimum 30 data points required)")

            # Remove any NaN values
            curvature_history = curvature_history.fillna(method='ffill').fillna(method='bfill')
            returns = returns.fillna(method='ffill').fillna(method='bfill')

            # Prepare features
            features = self.prepare_features(curvature_history, returns)
            
            if len(features) == 0:
                raise ValueError("Failed to prepare features for regime detection")

            # Fit HMM and predict regimes
            self.hmm.fit(features)
            regimes = self.hmm.predict(features)
            
            # Calculate regime characteristics
            regime_stats = {}
            for i in range(self.n_regimes):
                mask = (regimes == i)
                if np.any(mask):  # Only calculate stats if we have data points for this regime
                    regime_stats[i] = {
                        'avg_curvature': float(np.mean(features[mask, 0])),
                        'avg_dispersion': float(np.mean(features[mask, 1])),
                        'avg_return': float(np.mean(features[mask, 2])),
                        'volatility': float(np.std(features[mask, 2])),
                        'duration': int(np.sum(mask)),
                        'transitions': int(np.sum(np.diff(regimes) == i))
                    }
                else:
                    logger.warning(f"No data points found for regime {i}")
                    regime_stats[i] = {
                        'avg_curvature': 0.0,
                        'avg_dispersion': 0.0,
                        'avg_return': 0.0,
                        'volatility': 0.0,
                        'duration': 0,
                        'transitions': 0
                    }
            
            # Classify regimes
            regime_types = self._classify_regimes(regime_stats)
            
            # Get transition probabilities
            transitions = self.hmm.transmat_
            
            # Calculate regime stability
            stability = np.diag(transitions)
            
            # Convert dates to string format for JSON serialization
            dates = curvature_history.index.strftime('%Y-%m-%d').tolist()
            
            return {
                'regimes': regimes.tolist(),
                'regime_stats': regime_stats,
                'regime_types': regime_types,
                'transition_matrix': transitions.tolist(),
                'stability': stability.tolist(),
                'dates': dates
            }
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            raise ValueError(f"Failed to detect market regimes: {str(e)}")

    def _classify_regimes(self, regime_stats: Dict) -> Dict[int, str]:
        """
        Classify regimes as bull, bear, or transition based on characteristics.
        """
        classifications = {}
        
        for regime_id, stats in regime_stats.items():
            # Classify based on return and curvature characteristics
            if stats['avg_return'] > 0 and stats['avg_curvature'] > 0:
                regime_type = 'bull'
            elif stats['avg_return'] < 0 and stats['avg_curvature'] < 0:
                regime_type = 'bear'
            else:
                regime_type = 'transition'
                
            classifications[regime_id] = regime_type
            
        return classifications

    def get_current_regime(self, 
                          current_features: np.ndarray, 
                          regime_history: List[int]) -> Dict:
        """
        Analyze current market regime and predict potential transitions.
        """
        try:
            # Get current regime
            current_regime = self.hmm.predict([current_features])[-1]
            
            # Get transition probabilities for current regime
            transition_probs = self.hmm.transmat_[current_regime]
            
            # Calculate regime duration
            current_duration = 1
            for i in reversed(regime_history[:-1]):
                if i == current_regime:
                    current_duration += 1
                else:
                    break
            
            return {
                'current_regime': int(current_regime),
                'duration': current_duration,
                'transition_probs': transition_probs.tolist(),
                'stability': float(transition_probs[current_regime])
            }
            
        except Exception as e:
            logger.error(f"Error in current regime analysis: {str(e)}")
            return None

    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over regimes.
        """
        try:
            return self.hmm.predict_proba(features)
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {str(e)}")
            return None 